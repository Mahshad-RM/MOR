import numpy as np
import sys
import os
import warnings
from matplotlib import pyplot as plt

# Suppress MPI warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    from dispenser import stokes_solver, FOMsolver, Vh, Vb
    from dlroms import fe
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please make sure you have installed all required dependencies:")
    print("1. numpy")
    print("2. matplotlib")
    print("3. fenics-dolfin")
    print("4. mshr")
    print("5. torch")
    print("6. imageio")
    print("7. dlroms (from GitHub)")
    sys.exit(1)

def plot_solution(solution, title, filename):
    """Plot solution as a heatmap"""
    plt.figure(figsize=(10, 6))
    plt.imshow(solution.reshape(-1, solution.shape[-1]).T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Value')
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Node')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    try:
        # Compute basis functions
        print("Computing basis functions...")
        b1 = stokes_solver(1, 0, 0)
        b2 = stokes_solver(0, 1, 0)
        b3 = stokes_solver(0, 0, 1)

        print("\nBasis functions shapes:")
        print(f"b1 shape: {b1.shape}")
        print(f"b2 shape: {b2.shape}")
        print(f"b3 shape: {b3.shape}")

        # Plot basis functions
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.plot(b1)
        plt.title('Basis Function b1')
        plt.subplot(132)
        plt.plot(b2)
        plt.title('Basis Function b2')
        plt.subplot(133)
        plt.plot(b3)
        plt.title('Basis Function b3')
        plt.tight_layout()
        plt.savefig('basis_functions.png')
        plt.close()

        # Compute solutions
        print("\nComputing solutions...")
        u = FOMsolver(40, 20, 30)
        u_hom = FOMsolver(40, 20, 30)

        print("\nSolution shapes:")
        print(f"u shape: {u.shape}")
        print(f"u_hom shape: {u_hom.shape}")

        # Plot solutions
        print("\nPlotting solutions...")
        plot_solution(u, 'Full Solution (u)', 'u_evolution.png')
        plot_solution(u_hom, 'Homogenized Solution (u_hom)', 'u_hom_evolution.png')

        print("\nPlots have been saved as:")
        print("1. basis_functions.png")
        print("2. u_evolution.png")
        print("3. u_hom_evolution.png")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check the following:")
        print("1. Make sure FEniCS is properly installed")
        print("2. Check if the mesh generation is working correctly")
        print("3. Verify that all required Python packages are installed")
        sys.exit(1)

if __name__ == "__main__":
    main() 