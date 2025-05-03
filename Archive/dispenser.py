from IPython.display import clear_output as clc
import numpy as np
from dlroms import*

## Construction of the geometry
points = [np.array([0.0, 0.0])]
def step(z):
  points.append(points[-1] + [z.real, z.imag])
def arch(r, thetas, knots = 12):
  rho = np.abs(r)
  center = points[-1] + [r.real, r.imag]
  th = np.linspace(*thetas, knots)[1:]
  shift = rho*np.exp(1j*th)
  vals = np.stack([center[0] + shift.real, center[1]+shift.imag], axis = 1)
  for v in vals:
    points.append(v)
      
step(-1j)
step(1)
step(-2j)
arch(1, thetas = (np.pi, np.pi*3/2))
step(8)
step(1j)
step(-1)
arch(1j, thetas = (np.pi*3/2, np.pi))
arch(1, thetas = (np.pi, np.pi/2))
step(1)
step(1j)
step(-10)
points = np.stack(points)

domain = fe.polygon(points)
mesh = fe.mesh(domain, stepsize = 0.15)
Vh = fe.space(mesh, 'CG', 1) # FE space for the advection-diffusion eq.
Vb = fe.space(mesh, 'CG', 1, vector_valued = True, bubble = True) # FE space for Stokes' equation
clc()


## Definition of the Full Order Model (FOM)

## Auxiliary Stokes' solver
def stokes_solver(c1, c2, c3):
  from fenics import FiniteElement, NodalEnrichedElement, FunctionSpace, VectorElement, TrialFunctions, TestFunctions
  from fenics import inner, grad, dx, div, assemble, DirichletBC, Constant, Expression
  from scipy.sparse.linalg import spsolve
  from scipy.sparse import csr_matrix

  # Defining the mixed dimensional space for velocity and pressure
  pP1  = FiniteElement("CG", mesh.ufl_cell(), 1)
  vP1B = VectorElement(NodalEnrichedElement(FiniteElement("CG",       mesh.ufl_cell(), 1),
                                            FiniteElement("Bubble",   mesh.ufl_cell(), mesh.topology().dim() + 1)))

  pspace, vspace = pP1, vP1B
  W = FunctionSpace(mesh, vspace * pspace)
  (b, p) = TrialFunctions(W)
  (v, q) = TestFunctions(W)

  space = fe.space(mesh, "CG", 1, vector_valued = True, bubble = True)

  f = fe.interpolate(lambda x: [-10.0, 0.0], space)

  a = inner(grad(b), grad(v))*dx - div(v)*p*dx - q*div(b)*dx
  L = inner(f, v)*dx

  def outflow(x):
    return 10-x[0]<1e-12

  def inflow1(x):
    return x[0]<1e-12

  def inflow2(x):
    return 3+x[1]<1e-12 and 2-x[0]>0

  def inflow(x):
    return inflow1(x) or inflow2(x)

  def walls(x):
    return not(outflow(x) or inflow(x))

  noslip = DirichletBC(W.sub(0), Constant((0.0, 0.0)), lambda x, on: on and walls(x))
  in1 = DirichletBC(W.sub(0), Constant((c1, 0.0)), lambda x, on: on and inflow1(x))
  in2 = DirichletBC(W.sub(0), Constant((c2, c3)), lambda x, on: on and inflow2(x))

  A = assemble(a)
  F = assemble(L)

  for bc in [noslip, in1, in2]:
    bc.apply(A)
    bc.apply(F)

  A = csr_matrix(A.array())
  F = F[:]

  bp = spsolve(A, F)
  bp_f = fe.asfunction(bp, W)
  bp_f.set_allow_extrapolation(True)

  xdofs = fe.dofs(fe.space(mesh, 'CG', 1, bubble = True))
  b = np.stack([bp_f(x0)[:2] for x0 in xdofs]).reshape(-1)
  clc()

  return b


## FOM of the advection-diffusion equation
def FOMsolver(c1, c2, c3, steps = 700, dt = 5e-4):
  from fenics import inner, grad, dx, ds, FacetNormal
  from scipy.sparse.linalg import spsolve

  # Computing the velocity field by solving Stokes' equation
  b = stokes_solver(c1, c2, c3)

  # Assembling relevant operators
  M = fe.assemble(lambda u, v: u*v*dx, Vh) # mass matrix
  S = fe.assemble(lambda u, v: 0.5*inner(grad(u), grad(v))*dx, Vh) # stiffness (diffusion) matrix

  bf = fe.asfunction(b, Vb)
  B = fe.assemble(lambda u, v: inner(bf, grad(u))*v*dx, Vh) # transport matrix

  bc = fe.DirichletBC(lambda x: x[0]<1e-12, 1.0)
  M = fe.applyBCs(M, Vh, bc)
  S = fe.applyBCs(S, Vh, bc)
  B = fe.applyBCs(B, Vh, bc)

  # Time-stepping scheme
  def FOMstep(u0, dt, b):
      A = M + dt*S + dt*B
      rhs = M @ u0
      fe.applyBCs(rhs, Vh, bc)
      return spsolve(A, rhs)


  # Initial condition
  u0f = fe.interpolate(lambda x: x[0]<1e-12, Vh)
  u0 = fe.dofs(u0f)

  u = [u0] # list of states in time

  # Time loop
  for n in range(steps):
      uold = u[-1]
      unew = FOMstep(uold, dt, b)
      u.append(unew)

  clc()
  return np.stack(u)



## Auxiliary functions for Task (iv)
bottom_out = fe.dofs(fe.interpolate(lambda x: (10-x[0]<1e-12)*(x[1]<-2), Vh))

ys = fe.dofs(Vh)[bottom_out>0][:, 1]
isort = np.argsort(ys)
ys = ys[isort]

def bottomOutflow(u):
  # Extracting values of u at outflow (bottom channel) at final time
  u_out = u[-1, bottom_out > 0][isort]

  # Trapezoidal rule
  line_integral = (0.5*(u_out[:-1]+u_out[1:])*np.diff(ys)).sum(axis = -1)

  return line_integral