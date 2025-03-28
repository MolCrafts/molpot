import pint


class Unit(pint.UnitRegistry):

    UNIT_STYLE = {
        "real": {
            "mass": "g/mol",
            "distance": "angstrom",
            "time": "femtosecond",
            "energy": "kcal/mol",
            "velocity": "angstrom / femtosecond",
            "force": "kcal / mol / angstrom",
            "temperature": "kelvin",
            "pressure": "atmosphere",
            "charge": "e",
            "dipole": "e * angstrom",
            "efield": "volt / angstrom",
            "density": "g / cm ** 3",
        },
        "electron": {
            "mass": "amu",
            "distance": "bohr",
            "time": "femtosecond",
            "energy": "hartree",
            "velocity": "bohr / femtosecond",
            "force": "hartree / bohr",
            "temperature": "kelvin",
            "pressure": "pascal",
            "charge": "e",
            "dipole": "debye",
            "efield": "volt / cm",
        },
    }

    def __init__(self, style: str = "real", **kwargs):
        super().__init__(**kwargs)
        self._style = style

    def _after_init(self):
        super()._after_init()

        style = Unit.UNIT_STYLE[self._style]
        for name, unit in style.items():
            self.define(f"{name} = {unit}")

    @property
    def kB(self) -> float:
        return (
            (1 * self.boltzmann_constant * self.temperature)
            .to(self.energy, "energy")
            .magnitude
        )

    # Return conversion factor for given units
    def convert_unit(self, src, dest):
        return self.Quantity(1, src).to(dest).magnitude


_units = {"real": Unit("real"), "electron": Unit("electron")}


def get_unit(style: str = "real") -> Unit:
    if style not in _units:
        _units[style] = Unit(style)
    return _units[style]
