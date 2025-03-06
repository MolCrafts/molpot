# data structure and its alias

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
The `td` instance is a `TensorDict` instance

