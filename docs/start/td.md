# Data container and its alias

## Data container bases on `TensorDict`

Given a bunch of arbitrary potential functions organized in some inexplicable order, managing the intermediate variables will be a problem. To solve this problem, we use the [`TensorDict`](https://pytorch.org/tensordict/stable/index.html) class to manage namespace and variabel. `TensorDict` is a nested dictionary-like class that inherits properties from tensors, such as indexing, shape operations, casting to device etc.

The `TensorDict` instance we use flows through the potentials has a protocol:

```
td - 
 | - atoms
 | - pairs
 | - labels
 | - predicts
 | - bonds
 | - ...
```
For example, per-atom features like positions, charges, and atom types are stored in the `atoms`. `bonds` and `pairs` store the information of the bond and non-bonded interactions. The target value of a dataset like `energy` and `forces` are stored in the `labels`. The predicted value of the model is stored in the `predicts`. `TensorDict` behaves more than a dictionary, it can be indexed by multiple strings or tuple of string: 

``` py
from tensordict import TensorDict
td = TensorDict({
    "atoms": {
        "positions": torch.randn(10, 3),
        "charges": torch.randn(10),
    },
    "pairs": {
        "diff": torch.randn(10, 10, 3),
        "i": torch.randint(0, 10, (10,)),
        "j": torch.randint(0, 10, (10,)),
    },
    "labels": {
        "energy": torch.randn(10),
        "forces": torch.randn(10, 3),
    }
})

assert td["atoms", "positions"].shape == (10, 3)
assert td[("pairs", "diff")].shape == (10, 10, 3)

```

??? note "customized namespace"

    Most built-in modules follow the protocol mentioned above, but you can still define your own—just ensure consistency.

## Alias for nested keys

You may find the key of `TensorDict` is a bit verbose, either without intelligence or too long. To make the code more readable, we provide a set of aliases for the keys. The aliases are defined in the `molpot.alias` module. For example, the `R` or `xyz` alias is defined as `("atoms", "R")`. You can use the alias to index the `TensorDict` instance:

``` py
from molpot import alias

assert td[alias.R].shape == (10, 3)
assert td[alias.pair_diff].shape == (10, 10, 3)
```
???+ note "Why use alias? - part 2"
    Another reason to introduce the alias is trying to make internal units / shape consistent, as well as comments and documentation. But it's little bit hard to find a elegant solution. 
