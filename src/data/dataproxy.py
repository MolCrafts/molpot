# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-09-21
# version: 0.0.1
import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, List, Dict, Any, Iterable, Union, Tuple

import torch
import copy
from ase import Atoms
from ase.db import connect
from pathlib import Path

import molpot as mpot
import molpot.keywords as kw

logger = logging.getLogger(__name__)

class AtomsDataFormat(Enum):
    """Enumeration of data formats"""

    ASE = "ase"


class AtomsDataError(Exception):
    pass


extension_map = {AtomsDataFormat.ASE: ".db"}


class DataProxy(ABC):
    """
    Base mixin class for atomistic data. Use together with PyTorch Dataset or
    IterableDataset to implement concrete data formats.
    """

    def __init__(
        self,
        load_properties: Optional[List[str]] = None,
        load_structure: bool = True,
        transforms: Optional[List[mpot.Transform]] = None,
        subset_idx: Optional[List[int]] = None,
    ):
        """
        Args:
            load_properties: Set of properties to be loaded and returned.
                If None, all properties in the ASE dB will be returned.
            load_structure: If True, load structure properties.
            transforms: preprocessing transforms (see schnetpack.data.transforms)
            subset: List of data indices.
        """
        self._transform_module = None
        self.load_properties = load_properties
        self.load_structure = load_structure
        self.transforms = transforms
        self.subset_idx = subset_idx

    def __len__(self) -> int:
        raise NotImplementedError

    @property
    def transforms(self):
        return self._transforms

    @transforms.setter
    def transforms(self, value: Optional[List[mpot.Transform]]):
        self._transforms = []
        self._transform_module = None

        if value is not None:
            for tf in value:
                self._transforms.append(tf)
            self._transform_module = torch.nn.Sequential(*self._transforms)

    def subset(self, subset_idx: List[int]):
        assert (
            subset_idx is not None
        ), "Indices for creation of the subset need to be provided!"
        ds = copy.copy(self)
        if ds.subset_idx:
            ds.subset_idx = [ds.subset_idx[i] for i in subset_idx]
        else:
            ds.subset_idx = subset_idx
        return ds

    @property
    @abstractmethod
    def available_properties(self) -> List[str]:
        """Available properties in the dataset"""
        pass

    @property
    @abstractmethod
    def units(self) -> Dict[str, str]:
        """Property to unit dict"""
        pass

    @property
    def load_properties(self) -> List[str]:
        """Properties to be loaded"""
        if self._load_properties is None:
            return self.available_properties
        else:
            return self._load_properties

    @load_properties.setter
    def load_properties(self, val: List[str]):
        if val is not None:
            props = self.available_properties
            assert all(
                [p in props for p in val]
            ), "Not all given properties are available in the dataset!"
        self._load_properties = val

    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Global metadata"""
        pass

    @property
    @abstractmethod
    def atomrefs(self) -> Dict[str, torch.Tensor]:
        """Single-atom reference values for properties"""
        pass

    @abstractmethod
    def update_metadata(self, **kwargs):
        pass

    @abstractmethod
    def iter_properties(
        self,
        indices: Union[int, Iterable[int]] = None,
        load_properties: List[str] = None,
        load_structure: Optional[bool] = None,
    ):
        pass

    @staticmethod
    @abstractmethod
    def create(
        datapath: str,
        position_unit: str,
        property_unit_dict: Dict[str, str],
        atomrefs: Dict[str, List[float]],
        **kwargs,
    ) -> "DataProxy":
        pass

    @abstractmethod
    def add_systems(
        self,
        property_list: List[Dict[str, Any]],
        atoms_list: Optional[List[Atoms]] = None,
    ):
        pass

    @abstractmethod
    def add_system(self, atoms: Optional[Atoms] = None, **properties):
        pass

class ASEDataProxy(DataProxy):

    def __init__(self, data_dir, distance_unit: str, property_unit_map: dict[str, str]):
        self.data_dir = Path(data_dir)
        self.distance_unit = distance_unit
        self.property_unit_map = property_unit_map
        self._check_db()
        self.conn = connect(self.data_dir, use_lock_file=False)

        # initialize units
        md = self.metadata
        if "_distance_unit" not in md.keys():
            raise AtomsDataError(
                "Dataset does not have a distance unit set. Please add units to the "
                + "dataset using `spkconvert`!"
            )
        if "_property_unit_dict" not in md.keys():
            raise AtomsDataError(
                "Dataset does not have a property units set. Please add units to the "
                + "dataset using `spkconvert`!"
            )

        if distance_unit:
            self.distance_conversion = mpot.units.convert_units(
                md["_distance_unit"], distance_unit
            )
            self.distance_unit = distance_unit
        else:
            self.distance_conversion = 1.0
            self.distance_unit = md["_distance_unit"]

        self._units = md["_property_unit_dict"]
        self.conversions = {prop: 1.0 for prop in self._units}
        if property_unit_map is not None:
            for prop, unit in property_unit_map.items():
                self.conversions[prop] = mpot.units.convert_units(
                    self._units[prop], unit
                )
                self._units[prop] = unit
    def __len__(self) -> int:
        if self.subset_idx is not None:
            return len(self.subset_idx)

        with connect(self.datapath, use_lock_file=False) as conn:
            return conn.count()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.subset_idx is not None:
            idx = self.subset_idx[idx]

        props = self._get_properties(
            self.conn, idx, self.load_properties, self.load_structure
        )
        props = self._apply_transforms(props)

        return props

    def _apply_transforms(self, props):
        if self._transform_module is not None:
            props = self._transform_module(props)
        return props

    def _check_db(self):
        if not os.path.exists(self.datapath):
            raise AtomsDataError(f"ASE DB does not exists at {self.datapath}")

        if self.subset_idx:
            with connect(self.datapath, use_lock_file=False) as conn:
                n_structures = conn.count()

            assert max(self.subset_idx) < n_structures

    def iter_properties(
        self,
        indices: Union[int, Iterable[int]] = None,
        load_properties: List[str] = None,
        load_structure: Optional[bool] = None,
    ):
        """
        Return property dictionary at given indices.

        Args:
            indices: data indices
            load_properties (sequence or None): subset of available properties to load
            load_structure: load and return structure

        Returns:
            properties (dict): dictionary with molecular properties

        """
        if load_properties is None:
            load_properties = self.load_properties
        load_structure = load_structure or self.load_structure

        if self.subset_idx:
            if indices is None:
                indices = self.subset_idx
            elif type(indices) is int:
                indices = [self.subset_idx[indices]]
            else:
                indices = [self.subset_idx[i] for i in indices]
        else:
            if indices is None:
                indices = range(len(self))
            elif type(indices) is int:
                indices = [indices]

        # read from ase db
        with connect(self.datapath, use_lock_file=False) as conn:
            for i in indices:
                yield self._get_properties(
                    conn,
                    i,
                    load_properties=load_properties,
                    load_structure=load_structure,
                )

    def _get_properties(
        self, conn, idx: int, load_properties: List[str], load_structure: bool
    ):
        row = conn.get(idx + 1)

        # extract properties
        # TODO: can the copies be avoided?
        properties = {}
        properties[kw.idx] = torch.tensor([idx])
        for pname in load_properties:
            properties[pname] = (
                torch.tensor(row.data[pname].copy()) * self.conversions[pname]
            )

        Z = row["numbers"].copy()
        properties[kw.n_atoms] = torch.tensor([Z.shape[0]])

        if load_structure:
            properties[kw.Z] = torch.tensor(Z, dtype=torch.long)
            properties[kw.position] = (
                torch.tensor(row["positions"].copy()) * self.distance_conversion
            )
            properties[kw.cell] = (
                torch.tensor(row["cell"][None].copy()) * self.distance_conversion
            )
            properties[kw.pbc] = torch.tensor(row["pbc"])

        return properties

    # Metadata

    @property
    def metadata(self):
        with connect(self.datapath) as conn:
            return conn.metadata

    def _set_metadata(self, val: Dict[str, Any]):
        with connect(self.datapath) as conn:
            conn.metadata = val

    def update_metadata(self, **kwargs):
        assert all(
            key[0] != 0 for key in kwargs
        ), "Metadata keys starting with '_' are protected!"

        md = self.metadata
        md.update(kwargs)
        self._set_metadata(md)

    @property
    def available_properties(self) -> List[str]:
        md = self.metadata
        return list(md["_property_unit_dict"].keys())

    @property
    def units(self) -> Dict[str, str]:
        """Dictionary of properties to units"""
        return self._units

    @property
    def atomrefs(self) -> Dict[str, torch.Tensor]:
        md = self.metadata
        arefs = md["atomrefs"]
        arefs = {k: self.conversions[k] * torch.tensor(v) for k, v in arefs.items()}
        return arefs

    ## Creation

    @staticmethod
    def create(
        datapath: str,
        distance_unit: str,
        property_unit_dict: Dict[str, str],
        atomrefs: Optional[Dict[str, List[float]]] = None,
        **kwargs,
    ) -> "ASEDataProxy":
        """

        Args:
            datapath: Path to ASE DB.
            distance_unit: unit of atom positions and cell
            property_unit_dict: Defines the available properties of the datasetseta and
                provides units for ALL properties of the dataset. If a property is
                unit-less, you can pass "arb. unit" or `None`.
            atomrefs: dictionary mapping properies (the keys) to lists of single-atom
                reference values of the property. This is especially useful for
                extensive properties such as the energy, where the single atom energies
                contribute a major part to the overall value.
            kwargs: Pass arguments to init.

        Returns:
            newly created ASEAtomsData

        """
        if not datapath.endswith(".db"):
            raise AtomsDataError(
                "Invalid datapath! Please make sure to add the file extension '.db' to "
                "your dbpath."
            )

        if os.path.exists(datapath):
            raise AtomsDataError(f"Dataset already exists: {datapath}")

        atomrefs = atomrefs or {}

        with connect(datapath) as conn:
            conn.metadata = {
                "_property_unit_dict": property_unit_dict,
                "_distance_unit": distance_unit,
                "atomrefs": atomrefs,
            }

        return ASEDataProxy(datapath, **kwargs)

    # add systems
    def add_system(self, atoms: Optional[Atoms] = None, **properties):
        """
        Add atoms data to the dataset.

        Args:
            atoms: System composition and geometry. If Atoms are None,
                the structure needs to be given as part of the property dict
                (using structure.Z, structure.R, structure.cell, structure.pbc)
            **properties: properties as key-value pairs. Keys have to match the
                `available_properties` of the dataset.

        """
        with connect(self.datapath) as conn:
            self._add_system(conn, atoms, **properties)

    def add_systems(
        self,
        property_list: List[Dict[str, Any]],
        atoms_list: Optional[List[Atoms]] = None,
    ):
        """
        Add atoms data to the dataset.

        Args:
            atoms_list: System composition and geometry. If Atoms are None,
                the structure needs to be given as part of the property dicts
                (using structure.Z, structure.R, structure.cell, structure.pbc)
            property_list: Properties as list of key-value pairs in the same
                order as corresponding list of `atoms`.
                Keys have to match the `available_properties` of the dataset
                plus additional structure properties, if atoms is None.
        """
        if atoms_list is None:
            atoms_list = [None] * len(property_list)

        with connect(self.datapath) as conn:
            for at, prop in zip(atoms_list, property_list):
                self._add_system(conn, at, **prop)

    def _add_system(self, conn, atoms: Optional[Atoms] = None, **properties):
        """Add systems to DB"""
        if atoms is None:
            try:
                Z = properties[kw.Z]
                R = properties[kw.R]
                cell = properties[kw.cell]
                pbc = properties[kw.pbc]
                atoms = Atoms(numbers=Z, positions=R, cell=cell, pbc=pbc)
            except KeyError as e:
                raise AtomsDataError(
                    "Property dict does not contain all necessary structure keys"
                ) from e

        # add available properties to database
        valid_props = set().union(
            conn.metadata["_property_unit_dict"].keys(),
            [
                kw.Z,
                kw.R,
                kw.cell,
                kw.pbc,
            ],
        )
        for prop in properties:
            if prop not in valid_props:
                logger.warning(
                    f"Property `{prop}` is not a defined property for this dataset and "
                    + f"will be ignored. If it should be included, it has to be "
                    + f"provided together with its unit when calling "
                    + f"AseAtomsData.create()."
                )

        data = {}
        for pname in conn.metadata["_property_unit_dict"].keys():
            try:
                data[pname] = properties[pname]
            except:
                raise AtomsDataError("Required property missing:" + pname)

        conn.write(atoms, data=data)


def create_dataproxy(data_dir, distance_unit: str, property_unit_map: dict[str, str], format="ase"):
    if format == "ase":
        return ASEDataProxy(data_dir, distance_unit, property_unit_map)