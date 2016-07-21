import pdb

import time
import math
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from pyamg.gallery import poisson
from pyamg.gallery.diffusion import diffusion_stencil_2d
from pyamg.gallery.stencil import stencil_grid
from pyamg.util.utils import symmetric_rescaling
from pyamg.aggregation.trace_min import trace_min_solver

# from poisson import get_poisson

# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

# General multilevel parameters
# -----------------------------
max_levels 		   = 20 		# Max levels in hierarchy
max_coarse 		   = 20 		# Max points allowed on coarse grid
tol 			   = 1e-8		# Residual convergence tolerance
is_pdef 		   = True		# Assume matrix positive definite (only for aSA)
keep_levels 	   = False		# Also store SOC, aggregation, and tentative P operators
diagonal_dominance = False		# Avoid coarsening diagonally dominant rows 
coarse_solver = 'pinv'
accel = 'cg'
keep = False


strength = ('classical', {'theta': 0.25} )
# strength = ('evolution', {'k': 2, 'epsilon': 5.0})

split_agg = [ ['RS', None], [None, 'standard'] ]

trace_min={'deg': 1, 'maxiter': 5,
           'tol': 1e-8, 'debug': False,
           'get_tau': 3.0}

relaxation = ('jacobi', {'omega': 4.0/3.0, 'iterations': 1} )
# relaxation = ('gauss_seidel', {'sweep': 'symmetric', 'iterations': 1} )

# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

# Problem parameters and variables
# ---------------------------------

N 			= 1000
epsilon 	= 0.00
theta 		= [math.pi/16.0, 3.0*math.pi/16.0, math.pi/4.0]


# 2d Poisson
grid_dims = [N,N]
stencil = diffusion_stencil_2d(epsilon,theta)
A = stencil_grid(stencil, grid_dims, format='csr')

# # Vectors and additional variables
[d,d,A] = symmetric_rescaling(A)
vec_size = np.prod(grid_dims)


b = np.zeros((vec_size,1))
x0 = np.random.rand(vec_size,1)


# Dolfin?
# -------
# eps = 1
# theta= 1.0
# n0 = 100
# A, b = get_poisson(n=n0,eps=eps,theta=theta,rand=True)
# x0 = np.random.rand(A.shape[0],1)

# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

# Trace-min solver
# -------------------
residuals = []
start = time.clock()
ml = trace_min_solver(A=A, strength=strength, splitting=splitting, aggregate=aggregate,
					  trace_min=trace_min, presmoother=relaxation, postsmoother=relaxation,
                      max_levels=max_levels, max_coarse=max_coarse, keep=keep)
end = time.clock()
setup_time = end-start;

start = time.clock()
sol = ml.solve(b, x0, tol, residuals=residuals, accel=accel)
end = time.clock()
solve_time = end-start

OC = ml.operator_complexity()
CC = ml.cycle_complexity()
SC = ml.setup_complexity()

conv_factors = np.zeros((len(residuals)-1,1))
for i in range(1,len(residuals)-1):
	conv_factors[i] = residuals[i]/residuals[i-1]

CF = np.mean(conv_factors)

print "Trace-min, problem size ",A.shape[0]," x ",A.shape[0],", ",A.nnz," nonzeros"
print "\tSetup time 		 = ",setup_time
print "\tSolve time 		 = ",solve_time
print "\tSetup complexity 	 = ",SC
print "\tOperator complexity 	 = ",OC
print "\tCycle complexity 	 = ",CC
print "\tConvergence factor 	 = ",CF


pdb.set_trace()

