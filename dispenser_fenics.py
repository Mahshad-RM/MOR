# Simplified dispenser.py to use basic FEniCS without dlroms
import numpy as np
import fenics as fe
from fenics import *

# Global variables that will be initialized
mesh = None
Vh = None
Vb = None

def initialize_mesh_and_spaces():
    """Initialize mesh and function spaces"""
    global mesh, Vh, Vb
    
    if mesh is None:
        # Create a simple mesh
        mesh = UnitSquareMesh(32, 32)
        
        # Transform the unit square to approximate the dispenser domain
        coords = mesh.coordinates()
        coords[:, 0] = coords[:, 0] * 10 - 1  # Scale and translate x: [0,1] -> [-1,9]
        coords[:, 1] = coords[:, 1] * 4 - 3   # Scale and translate y: [0,1] -> [-3,1]
        
        # Create function spaces
        Vh = FunctionSpace(mesh, 'CG', 1)  # FE space for the advection-diffusion eq.
        
        # Simplified vector space (without bubble elements for now)
        Vb = VectorFunctionSpace(mesh, 'CG', 1)  # FE space for Stokes' equation
        
        print("Simplified mesh and function spaces created")
    
    return mesh, Vh, Vb

def stokes_solver(c1, c2, c3):
    """Simplified Stokes' solver using Taylor-Hood elements"""
    mesh, Vh, Vb = initialize_mesh_and_spaces()
    
    # Use Taylor-Hood elements (P2-P1) for stability
    V = VectorFunctionSpace(mesh, 'CG', 2)  # Velocity
    Q = FunctionSpace(mesh, 'CG', 1)        # Pressure
    W = V * Q
    
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    # Define the variational form
    a = (inner(grad(u), grad(v)) - div(v)*p - q*div(u)) * dx
    L = inner(Constant((-10.0, 0.0)), v) * dx

    # Define boundary conditions
    def inflow1(x, on_boundary):
        return on_boundary and near(x[0], -1, 0.1)
    
    def inflow2(x, on_boundary):
        return on_boundary and near(x[1], -3, 0.1) and x[0] > -0.5
    
    def walls(x, on_boundary):
        return on_boundary and not (near(x[0], 9, 0.1) or inflow1(x, on_boundary) or inflow2(x, on_boundary))

    # Apply boundary conditions
    bc1 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), walls)
    bc2 = DirichletBC(W.sub(0), Constant((c1, 0.0)), inflow1)
    bc3 = DirichletBC(W.sub(0), Constant((c2, c3)), inflow2)
    bcs = [bc1, bc2, bc3]

    # Solve
    up = Function(W)
    solve(a == L, up, bcs)
    
    # Extract velocity field
    u_sol = up.sub(0, deepcopy=True)
    
    # Interpolate to Vb space and return vector
    ub = interpolate(u_sol, Vb)
    return ub.vector()[:]

def compute_basis_functions():
    """Compute basis functions b₁, b₂, b₃"""
    b1 = stokes_solver(1, 0, 0)
    b2 = stokes_solver(0, 1, 0)
    b3 = stokes_solver(0, 0, 1)
    return b1, b2, b3

def FOMsolver(c1, c2, c3, steps=700, dt=5e-4):
    """FOM of the advection-diffusion equation"""
    mesh, Vh, Vb = initialize_mesh_and_spaces()
    
    b = stokes_solver(c1, c2, c3)
    
    # Create a function from the velocity field
    bf = Function(Vb)
    bf.vector()[:] = b
    
    u = TrialFunction(Vh)
    v = TestFunction(Vh)
    
    M = assemble(u * v * dx)
    S = assemble(0.5 * inner(grad(u), grad(v)) * dx)
    B = assemble(inner(bf, grad(u)) * v * dx)
    
    # Apply boundary conditions
    def inflow_boundary(x, on_boundary):
        return on_boundary and near(x[0], -1, 1e-12)
    
    bc = DirichletBC(Vh, Constant(1.0), inflow_boundary)
    
    # Apply BCs to matrices
    bc.apply(M)
    bc.apply(S) 
    bc.apply(B)
    
    def FOMstep(u0, dt):
        A = M + dt * S + dt * B
        rhs = M * u0
        bc.apply(rhs)
        u_new = Function(Vh)
        solve(A, u_new.vector(), rhs)
        return u_new.vector()[:]
    
    # Initial condition
    u0_func = interpolate(Expression('x[0] < -0.9 ? 1.0 : 0.0', degree=1), Vh)
    u0 = u0_func.vector()[:]
    
    u = [u0]
    for _ in range(steps):
        uold = u[-1]
        unew = FOMstep(uold, dt)
        u.append(unew)
    
    return np.stack(u)

def assemble_FOM_matrices():
    """Assemble FOM matrices for reduced basis method"""
    mesh, Vh, Vb = initialize_mesh_and_spaces()
    
    b1, b2, b3 = compute_basis_functions()
    
    b1f = Function(Vb)
    b1f.vector()[:] = b1
    b2f = Function(Vb) 
    b2f.vector()[:] = b2
    b3f = Function(Vb)
    b3f.vector()[:] = b3
    
    u = TrialFunction(Vh)
    v = TestFunction(Vh)
    
    B1 = assemble(inner(b1f, grad(u)) * v * dx)
    B2 = assemble(inner(b2f, grad(u)) * v * dx)
    B3 = assemble(inner(b3f, grad(u)) * v * dx)
    
    M = assemble(u * v * dx)
    S = assemble(0.5 * inner(grad(u), grad(v)) * dx)
    
    # Apply homogeneous boundary conditions
    def inflow_boundary(x, on_boundary):
        return on_boundary and near(x[0], -1, 1e-12)
    
    bch = DirichletBC(Vh, Constant(0.0), inflow_boundary)
    bch.apply(M)
    bch.apply(S)
    bch.apply(B1)
    bch.apply(B2) 
    bch.apply(B3)
    
    return M.array(), S.array(), B1.array(), B2.array(), B3.array()

def FOMsolverhom(c1, c2, c3, steps=700, dt=5e-4):
    """Homogenized FOM using precomputed basis"""
    mesh, Vh, Vb = initialize_mesh_and_spaces()
    
    M, S, B1, B2, B3 = assemble_FOM_matrices()
    
    def inflow_boundary(x, on_boundary):
        return on_boundary and near(x[0], -1, 1e-12)
    
    bch = DirichletBC(Vh, Constant(0.0), inflow_boundary)
    
    def FOMstep(u0, dt):
        A = M + dt * S + dt * (c1 * B1 + c2 * B2 + c3 * B3)
        rhs = M @ u0
        bch.apply(rhs)
        return np.linalg.solve(A, rhs)
    
    u0_func = interpolate(Expression('x[0] < -0.9 ? 1.0 : 0.0', degree=1), Vh)
    u0_func.vector()[:] -= 1.0  # Subtract 1 for homogeneous version
    u0 = u0_func.vector()[:]
    
    u = [u0]
    for _ in range(steps):
        uold = u[-1]
        unew = FOMstep(uold, dt)
        u.append(unew)
    
    return np.stack(u), u0

def bottomOutflow(u):
    """Compute bottom outflow functional"""
    mesh, Vh, Vb = initialize_mesh_and_spaces()
    
    # Find bottom boundary DOFs
    boundary_dofs = []
    for i, coord in enumerate(mesh.coordinates()):
        if near(coord[0], 9, 1e-12) and coord[1] < -2:
            boundary_dofs.append(i)
    
    if len(boundary_dofs) == 0:
        return 0.0
    
    # Sort by y-coordinate
    boundary_coords = mesh.coordinates()[boundary_dofs]
    ys = boundary_coords[:, 1]
    isort = np.argsort(ys)
    ys = ys[isort]
    
    u_out = u[-1][boundary_dofs][isort]
    if len(u_out) > 1:
        line_integral = (0.5 * (u_out[:-1] + u_out[1:]) * np.diff(ys)).sum()
    else:
        line_integral = u_out[0] if len(u_out) > 0 else 0.0
    
    return line_integral

# Initialize everything when module is imported
mesh, Vh, Vb = initialize_mesh_and_spaces()

print("Pure FEniCS dispenser module loaded successfully")
