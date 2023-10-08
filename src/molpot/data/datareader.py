from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes import functional_datapipe
import molpy as mp
from molpot import kw

__all__ = ["XYZReader", ]

@functional_datapipe("read_xyz")
class XYZReader(IterDataPipe):

    def __init__(self, source_dp: IterDataPipe):
        super().__init__()
        self.source_dp = source_dp

    def __iter__(self):

        for d in self.source_dp:
            frame = mp.Frame()

            lines = d.readlines()
            frame[kw.natoms] = int(lines[0])
            props_line = lines[1].split()[1:]
            frame['index'] = int(props_line[0])
            for prop, p in zip(self.frame, props_line[1:]):
                src_unit = self.keywords.get_unit(prop)
                if prop in kw:
                    dst_unit = kw.get_unit(prop)
                    frame[prop] = mp.units.convert(
                        float(p), src_unit, dst_unit
                    )
                else:
                    frame[prop] = float(p)
            
            frame.atoms[kw.xyz] = [[i.replace('*^', 'E') for i in l.split()[1:4]]
                for l in lines[2:-3]]
            frame.atoms[kw.Z] = [mp.Element.get_atomic_number_by_symbol(l.split()[0]) for l in lines[2:-3]]

            yield frame