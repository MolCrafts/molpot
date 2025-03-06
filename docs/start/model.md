# Assemble the model like *Lego*™

The most important feature of MolPot is its modular design, which allows users to combine different types of potential functions like building blocks seamlessly. The core of MolPot is the `Potential` class, which represents a single potential function. With leveraging the [tensordict](https://pytorch.org/tensordict/stable/index.html), we can specify the input and output tensors' names.

Let's start with a simple example of combining a neural network potential with a coulomb potential, as well as energy and force readout modules.

``` py
import molpot as mpot
# define PiNet potential
pinet = mpot.potential.nnp.PiNet2(
    depth=4,
    basis_fn=mpot.potential.nnp.radial.GaussianRBF(10, 5.0),
    cutoff_fn=mpot.potential.nnp.cutoff.CosineCutoff(5.0),
    pi_nodes=[64, 64],
    ii_nodes=[64, 64, 64, 64],
    pp_nodes=[64, 64, 64, 64],
    activation=torch.nn.Tanh(),
)
# define long-range coulomb potential
coulomb = mpot.potential.classic.PME(
    gridx, gridy, gridz, order, alpha, coulomb, exclusions
)
# predict energy from nnp features
e_readout = mpot.potential.nnp.readout.Atomwise(
    n_neurons=[64, 64, 1],
    in_keys=[alias.atom_batch, ("pinet", "p1")],
    out_keys=[("predicts", "energy")],
    reduce="sum",
)
# predict forces from nnp features
f_readout = mpot.potential.nnp.readout.PairForce(
    in_keys=("predicts", "energy"),
    dx_key=alias.pair_diff,
    out_keys=("predicts", "forces"),
    create_graph=True,
)
potential = mpot.potential.PotentialSeq(pinet, coulomb, e_readout, f_readout)
```

After above code, we have a `potential` instance that can calculate the energy and forces of a given atomic configuration. The `PotentialSeq` class is a subclass of `TensorDictSequential`. It ensures that the input and output tensors of each potential function are correctly connected. 

It's quite easy to implement new potential without intrude the existing code. Just inherit the `Potential` class and implement the `forward` method, and also specify the input and output tensors' names. 

``` py
class FixedKeysPotential(mpot.Potential):

    in_keys = ["positions", "charges"]
    out_keys = ["energy"]

    def __init__(self, some_args):
        super().__init__()
        self.some_args = some_args

    def forward(self, positions, charges):
        # do something
        return energy

    def cite(self):
        return "Some paper"  # don't forget to cite your work
```

sometimes the potential is too flexible you can't specify the input and output tensors' names in advance, you can also assign it when initilize the potential.

``` py
class DynamicKeysPotential(mpot.Potential):

    def __init__(self, in_keys, out_keys, some_args):
        super().__init__()
        self.in_keys = in_keys
        self.out_keys = out_keys
        self.some_args = some_args

    def forward(self, **kwargs):
        # do something
        return energy
```
