from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes import functional_datapipe
import molpy as mp
from molpot import kw

__all__ = [
    "XYZReader",
]


@functional_datapipe("read_xyz")
class XYZReader(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe, keywords: mp.Keywords):
        super().__init__()
        self.source_dp = source_dp
        self.keywords = keywords

    def __iter__(self):
        local_kw = self.keywords
        for d in self.source_dp:
            frame = mp.Frame()

            lines = d[1].readlines()
            frame[kw.natoms] = int(lines[0])
            props_line = lines[1].split()[1:]
            frame["index"] = int(props_line[0])
            for prop, p in zip(local_kw.get_aliases(), props_line[1:]):
                
                if prop in local_kw:
                    src_unit = local_kw.get_unit(prop)
                    dst_unit = kw.get_unit(prop)
                    frame[prop] = mp.units.convert(float(p), src_unit, dst_unit)
                else:
                    frame[prop] = float(p)

            frame.atoms[kw.xyz] = [
                [i.replace("*^", "E") for i in line.split()[1:4]]
                for line in lines[2:-3]
            ]
            frame.atoms[kw.Z] = [
                mp.Element.get_atomic_number_by_symbol(line.split()[0])
                for line in lines[2:-3]
            ]

            yield frame
