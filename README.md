# Molecular Potential Training and Deploying Framework

## Abstract

In the realm of materials science and molecular dynamics, the development of accurate and efficient force fields is pivotal for simulating complex interactions at the atomic and molecular levels. Researchers demand models that are easy to train for specific systems and seamlessly embeddable in molecular dynamics (MD) or other simulation packages.

Over the past five years, significant progress has been made in the development of data-driven or neural network potentials, achieving commendable accuracy. Despite this progress, challenges persist. Architectures such as message passing neural networks struggle to capture long-range interactions, and the inherent opacity of neural network potentials impedes interpretation with physical meaning. This abstract proposes a novel approach by introducing a physics-driven potential. This allows training classical models like neural network potentials and replacing any part with a neural network kernel.

The platform's distinctive feature lies in its incorporation of a physics-driven potential, ensuring a foundation rooted in fundamental physical principles. This unique integration of neural networks and physics-based models strikes a delicate balance between flexibility and accuracy. This dual approach enables the force field to adeptly capture both intricate nuances and overarching trends in molecular interactions, making it applicable across a diverse spectrum of materials and scenarios.

Notably, the platform excels in performance. Leveraging optimized neighbor-list algorithms and highly efficient C++ embedding code, the trained model seamlessly integrates into various MD packages. The entire procedure is automated through scripts, simplifying the setup of custom force fields for researchers with just a few commands. This user-friendly automation enhances accessibility and encourages broader adoption of the proposed platform for advancing molecular dynamics simulations in materials science research.

## Feature

* Hybrid and Modular Neural Network and Classical Potential;
* One-stop training and deployment solution;
* Active learning framework;
* Error metrices;
* High performance embedding API for MD engine;
* Monitoring and profiling auxiliary tools;
* Model inspector and visualizer;
* Datapipeline based on torchdata.

## Roadmap
- [ ] Implement PiNet and PaiNN neural network potential and test-related tools;
- [ ] Compile & train NNP and profiling performace;
- [ ] Embedding model into LAMMPS and test;
- [ ] Forcefield management and Topolygy support;
- [ ] Classical potential implement;
- [ ] Simple typification and parameterization;
- [ ] Native MD code for quick test;
- [ ] Distributed and integrated with deepspeed;

## Vailidation Instances

### NNP
- [x] QM9 and rMD17 test;

### Classical Potential
- [ ] Coarse-grained polyethylene total energy training;
- [ ] All-atom polybutadiene decomposition energy training;
- [ ] Long-range PEO training;

### Hybrid Potential
- [ ] Long-range point charge and short-range NNP potential for PEO;

