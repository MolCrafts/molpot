# dataset.py – Internal Dataset Interface for MolPot

This module defines the core `Dataset` class and its utilities used in the MolPot pipeline. It is responsible for managing data download, caching, loading, and applying a sequence of processing steps to molecular data prior to training.

## Overview

The module includes:

- `Dataset`: a base class implementing data loading and preprocessing logic.
- `IterStyleDataset`: for iterable-style dataset handling.
- `MapStyleDataset`: for map-style datasets that apply transformations on access.
- `DATASET_LOADER_MAP`: a registry mapping dataset names to their loader classes.

---

## Class: `Dataset`

### Constructor

```python
Dataset(name: str, save_dir: Path | None = None, device: str = "cpu")
