This project includes code obtained from various sources. Below is a summary of the sources and appropriate citations:

## locality
### neighbor

- **Source**: [](https://github.com/openmm/NNPOps/tree/master/src/pytorch/neighbors)
- **Modified**: Yes
    * use CppFunction::makeFallthrough() to avoid warning
- **License**: MIT License

## pot
### PME

- **Source**: [](https://github.com/openmm/NNPOps/tree/master/src/pytorch/pme)
- **Modified**: Yes
    * change class name from PME to PME kernel;
    * reuse neighborlist results;
- **License**: MIT License

## scatter

- **Source**: [](https://github.com/rusty1s/pytorch_scatter)
- **Modified**: Yes
    * add helper function for batch calculation;
- **License**: MIT License