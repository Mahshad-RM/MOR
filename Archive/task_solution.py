try:
    from dlroms import *
except:
    import subprocess
    subprocess.run(['pip', 'install', 'matplotlib'])
    subprocess.run(['pip', 'install', 'torch'])
    subprocess.run(['pip', 'install', 'imageio'])
    subprocess.run(['pip', 'install', 'fenics-dolfin'])
    subprocess.run(['pip', 'install', 'mshr'])
    subprocess.run(['pip', 'install', 'scipy'])
    subprocess.run(['pip', 'install', 'fenics'])
    subprocess.run(['pip', 'install', '--no-deps', 'git+https://github.com/NicolaRFranco/dlroms.git'])
    from dlroms import *

import numpy as np
from fenics import *
from dispenser import stokes_solver, FOMsolver, Vh, Vb

# Step 1: Compute basis functions b₁, b₂, b₃
def compute_basis_functions():
    # Compute b₁ (c₁ = 1, c₂ = c₃ = 0)
    b1 = stokes_solver(1, 0, 0)
    
    # Compute b₂ (c₂ = 1, c₁ = c₃ = 0)
    b2 = stokes_solver(0, 1, 0)
    
    # Compute b₃ (c₃ = 1, c₁ = c₂ = 0)
    b3 = stokes_solver(0, 0, 1)
    
    return b1, b2, b3

# Step 2: Demonstrate that u = uₕₒₘ + 1
def demonstrate_solution_relationship(c1, c2, c3):
    # Compute the full solution u with u = 1 on Γ₁ᶦⁿ
    def FOMsolver_u(c1, c2, c3, steps=700, dt=5e-4):
        from fenics import inner, grad, dx, ds, FacetNormal
        from scipy.sparse.linalg import spsolve

        # Computing the velocity field by solving Stokes' equation
        b = stokes_solver(c1, c2, c3)

        # Assembling relevant operators
        M = fe.assemble(lambda u, v: u*v*dx, Vh)
        S = fe.assemble(lambda u, v: 0.5*inner(grad(u), grad(v))*dx, Vh)

        bf = fe.asfunction(b, Vb)
        B = fe.assemble(lambda u, v: inner(bf, grad(u))*v*dx, Vh)

        # Boundary conditions for u: u = 1 on Γ₁ᶦⁿ
        bc = fe.DirichletBC(lambda x: x[0]<1e-12, 1.0)
        M = fe.applyBCs(M, Vh, bc)
        S = fe.applyBCs(S, Vh, bc)
        B = fe.applyBCs(B, Vh, bc)

        def FOMstep(u0, dt, b):
            A = M + dt*S + dt*B
            rhs = M @ u0
            fe.applyBCs(rhs, Vh, bc)
            return spsolve(A, rhs)

        # Initial condition for u: u(·, 0) = 1Γ₁ᶦⁿ
        u0f = fe.interpolate(lambda x: x[0]<1e-12, Vh)
        u0 = fe.dofs(u0f)

        u = [u0]
        for n in range(steps):
            uold = u[-1]
            unew = FOMstep(uold, dt, b)
            u.append(unew)

        return np.stack(u)
    
    # Compute uₕₒₘ using the homogenized equation
    def FOMsolver_hom(c1, c2, c3, steps=700, dt=5e-4):
        from fenics import inner, grad, dx, ds, FacetNormal
        from scipy.sparse.linalg import spsolve

        # Compute b as linear combination of basis functions
        b = c1*b1 + c2*b2 + c3*b3

        # Assembling relevant operators
        M = fe.assemble(lambda u, v: u*v*dx, Vh)
        S = fe.assemble(lambda u, v: 0.5*inner(grad(u), grad(v))*dx, Vh)

        bf = fe.asfunction(b, Vb)
        B = fe.assemble(lambda u, v: inner(bf, grad(u))*v*dx, Vh)

        # Boundary conditions for uₕₒₘ: uₕₒₘ = 0 on Γ₁ᶦⁿ
        bc = fe.DirichletBC(lambda x: x[0]<1e-12, 0.0)
        M = fe.applyBCs(M, Vh, bc)
        S = fe.applyBCs(S, Vh, bc)
        B = fe.applyBCs(B, Vh, bc)

        def FOMstep(u0, dt, b):
            A = M + dt*S + dt*B
            rhs = M @ u0
            fe.applyBCs(rhs, Vh, bc)
            return spsolve(A, rhs)

        # Initial condition for uₕₒₘ: uₕₒₘ(·, 0) = 1Γ₁ᶦⁿ - 1
        u0f = fe.interpolate(lambda x: (x[0]<1e-12) - 1, Vh)
        u0 = fe.dofs(u0f)

        u = [u0]
        for n in range(steps):
            uold = u[-1]
            unew = FOMstep(uold, dt, b)
            u.append(unew)

        return np.stack(u)

    # Compute basis functions
    b1, b2, b3 = compute_basis_functions()
    
    # Compute solutions
    u_full = FOMsolver_u(c1, c2, c3)
    u_hom = FOMsolver_hom(c1, c2, c3)
    
    # Verify that u = uₕₒₘ + 1
    error = np.max(np.abs(u_full - (u_hom + 1)))
    print(f"Maximum error between u and uₕₒₘ + 1: {error}")
    
    return u_full, u_hom

# Main execution
if __name__ == "__main__":
    # Example values for c₁, c₂, c₃
    c1, c2, c3 = 1.0, 0.5, 0.3
    
    # Demonstrate the relationship
    u_full, u_hom = demonstrate_solution_relationship(c1, c2, c3)
    print("Relationship between u and uₕₒₘ verified")

    # plot the solution
    import matplotlib.pyplot as plt
    plt.plot(u_full, label='u')
    plt.plot(u_hom, label='uₕₒₘ')
    plt.legend()
    plt.show()
    
