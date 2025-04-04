from molpot import alias
from ..event import MDEvents
from ..handler import MDHandler
import torch


class Initializer(MDHandler):

    def __init__(
        self,
        name: str
    ):
        super().__init__(name, events={MDEvents.STARTED})

class TemperatureInitializer(Initializer):

    def __init__(
        self,
        name: str,
        T: float,
        remove_center_of_mass: bool = True,
        remove_translation: bool = True,
        # remove_rotation: bool = True
    ):
        super().__init__(name)
        self.T = T
        self.remove_center_of_mass = remove_center_of_mass
        self.remove_translation = remove_translation
        # self.remove_rotation = remove_rotation

class UniformTemperature(TemperatureInitializer):

    def __init__(
        self,
        T: float,
        remove_center_of_mass: bool = True,
        remove_translation: bool = True,
        # remove_rotation: bool = True
    ):
        super().__init__("uniform_temperature", T, remove_center_of_mass, remove_translation)
        self.T = T
        self.remove_center_of_mass = remove_center_of_mass
        self.remove_translation = remove_translation

    def on_started(self, engine):
        
        frame = engine.state.frame
        atom_momentum = frame[alias.atom_momentum] if alias.atom_momentum in frame else torch.randn_like(frame[alias.R]) * frame[alias.atom_mass]

        scaling = torch.sqrt(self.T / engine.state.thermo["T"])
        atom_momentum *= scaling