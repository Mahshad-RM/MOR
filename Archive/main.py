from dlroms import *
from dispenser import FOMsolver, Vh
from IPython.display import clear_output as clc
import numpy as np
from dlroms import*




u = FOMsolver(40, 20, 30)
u.shape

fe.animate(u[::10], Vh)
