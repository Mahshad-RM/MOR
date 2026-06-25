# Model Order Reduction

Course material and assignments for the **Model Order Reduction** master's course at **Politecnico di Milano**, Mechanical Engineering program.

The repository is organized in two main folders:

```text
labs/
assignment/
```

## Labs

- **Lab 1 - Introduction to Python**: Python basics and introductory numerical exercises used before the model order reduction material.
- **Lab 2 - Finite Elements**: Finite element discretization workflow and first FEniCS/DL-ROMs finite element examples.
- **Lab 3 - Reduced Basis Part 1**: Introduction to reduced basis methods and projection-based reduced models.
- **Lab 4 - Reduced Basis Part 2**: Extended reduced basis techniques, offline/online decomposition, and error-oriented ROM workflow.
- **Lab 5 - Neural networks**: Neural network basics used for non-intrusive reduced-order modeling.
- **Lab 6 - POD-NN**: Proper Orthogonal Decomposition combined with neural networks for parametric ROMs.
- **Lab 7 - DL-ROMs**: Deep Learning Reduced Order Models using nonlinear latent representations.
- **Lab 8 - MINNs and PCA-net**: Manifold-informed neural networks and PCA-based neural reduced models.
- **Lab 9 - AE latent dynamics with SINDy**: Autoencoder latent-space dynamics and SINDy-based reduced dynamics.

## Assignment

- **Exercise 1 - Designing a chemical dispenser**: Part 1 assignment on a convection-diffusion chemical dispenser problem. The work uses `dispenser.py` in Google Colab/FEniCS, generates FOM snapshots, applies POD-Galerkin reduction, evaluates ROM accuracy and speed, and studies the bottom-outflow objective.

- **Exercise 2 - Stunt training facility**: Part 2 assignment on time-dependent linear elasticity for material design of a two-layer stunt training floor. The work uses data-driven ROM approaches such as POD-NN, DL-ROM, POD-DL-ROM, and AE+SINDy to approximate floor deformation and analyze maximum displacement.

## Notes

- Notebooks were prepared for Google Colab/Jupyter.
- Some notebooks install or use external packages such as `dlroms`, FEniCS-related tools, PyTorch, NumPy, SciPy, and Matplotlib.
- Assignment helper scripts are kept in the same folder as the notebooks that import them.
