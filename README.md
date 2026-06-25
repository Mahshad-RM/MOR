# Model Order Reduction

This repository contains lab and assignment material for the **Model Order Reduction** course in the master's program in Mechanical Engineering at Politecnico di Milano.

The material focuses on reduced-order modeling techniques for parametrized PDE problems, combining Full Order Models (FOMs), Proper Orthogonal Decomposition (POD), Galerkin projection, and data-driven neural-network based reduced models.

## Repository Structure

```text
labs/
  part_1_overview/

assignments/
  exercise_1_chemical_dispenser/
  exercise_2_linear_elasticity/
```

## Labs

- **Part 1 overview**: Introductory Colab/Jupyter notebook for the first part of the course assignments. It gives an overview of the available model order reduction case studies and includes the chemical dispenser setup used later in Exercise 1.

## Assignments

- **Exercise 1 - Designing a chemical dispenser**: Convection-diffusion problem for a chemical dispenser. The work builds a reduced-order model using POD-Galerkin, starting from FEniCS-based full-order simulations in `dispenser.py`. The notebook studies snapshot generation, singular-value decay, reduced basis construction, ROM accuracy, computational speedup, and bottom-outflow optimization.

- **Exercise 2 - Linear elasticity**: Material design problem for a stunt training/playground floor. The notebooks explore reduced modeling approaches for a parametrized linear elasticity problem, including POD-NN, DL-ROM, POD-DL-ROM, autoencoder-based reduction, and SINDy-style latent dynamics.

## Notes

- The notebooks were prepared to run in Google Colab.
- Some cells install or import external packages such as `dlroms`, FEniCS-related tools, PyTorch, NumPy, SciPy, and Matplotlib.
- `dispenser.py` is placed next to the notebooks that import it, so those notebooks can run after uploading/opening the folder content in Colab.
