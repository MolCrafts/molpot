from ..potential.base import Potential

class PotentialInspector:

    def __init__(self, potential: Potential):

        self.potential = potential

    def plot1d(self, inputs, y_key, axis='x'):

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        outputs = {}

        if axis == 'x':
            x = inputs['xyz'][:, 0]
        elif axis == 'y':
            x = inputs['xyz'][:, 1]
        elif axis == 'z':
            x = inputs['xyz'][:, 2]

        inputs, outputs = self.potential(inputs, outputs)

        ax.plot(x, outputs[y_key])
        return fig, ax