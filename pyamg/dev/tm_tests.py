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
from pyamg.util.utils import symmetric_rescaling, symmetric_rescaling_sa
from pyamg.aggregation.trace_min import trace_min_solver

from poisson import get_poisson
from elasticity_bar import get_elasticity_bar

# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

# General multilevel parameters
# -----------------------------
max_levels 		   = 20 		# Max levels in hierarchy
max_coarse 		   = 20 		# Max points allowed on coarse grid
tol 			   = 1e-8		# Residual convergence tolerance
diagonal_dominance = False		# Avoid coarsening diagonally dominant rows 
coarse_solver = 'pinv'
accel = 'cg'
keep = False


# strength = ('symmetric', {'theta': 0.25} )
# strength = ('classical', {'theta': 0.35} )
strength = ('evolution', {'k': 2, 'epsilon': 3.0})

# split_agg = [ ['RS', None], [None, 'standard'] ]
split_agg = ['RS', None]
# split_agg = [None, 'standard']

trace_min={'deg': 2, 'maxiter': 5,
           'tol': 1e-8, 'prefilter': {'theta': 0.3},
           'get_tau': 0.0001, 'precondition': True,
           'debug': False}

# relaxation1 = ('jacobi', {'omega': 4.0/3.0, 'iterations': 1} )
# relaxation2 = ('jacobi', {'omega': 4.0/3.0, 'iterations': 1} )
# improve_candidates = ('jacobi', {'omega': 4.0/3.0, 'iterations': 4})

relaxation1 = ('gauss_seidel', {'sweep': 'forward', 'iterations': 1} )
relaxation2 = ('gauss_seidel', {'sweep': 'backward', 'iterations': 1} )
improve_candidates = ('gauss_seidel', {'sweep': 'forward', 'iterations': 4})
# GS_NR/NE seem to be unstable with CG
# relaxation1 = ('gauss_seidel_nr', {'sweep': 'forward', 'iterations': 1} )
# relaxation2 = ('gauss_seidel_nr', {'sweep': 'backward', 'iterations': 1} )
# improve_candidates = ('gauss_seidel_nr', {'sweep': 'forward', 'iterations': 4})


# ----------------------------------------------------------------------------- #

# N 			= 1000
# epsilon 	= 0.00
# theta		= 3.0*math.pi/16.0
# # theta 		= [math.pi/16.0, 3.0*math.pi/16.0, math.pi/4.0]

# # 2d Poisson
# grid_dims = [N,N]
# stencil = diffusion_stencil_2d(epsilon,theta)
# A = stencil_grid(stencil, grid_dims, format='csr')

# # # Vectors and additional variables
# [d,d,A] = symmetric_rescaling(A)
# vec_size = np.prod(grid_dims)


# b = np.zeros((vec_size,1))
# x0 = np.random.rand(vec_size,1)


# Dolfin?
# -------
eps = 0.00
theta = 4.0*math.pi/16.0
n0 = 1500
A, b = get_poisson(n=n0,eps=eps,theta=theta,rand=False)
x0 = np.random.rand(A.shape[0],1)
[d,d,A] = symmetric_rescaling(A)
B = None

# nx = 10
# ny = 100
# nz = 10
# A, b, B = get_elasticity_bar(nx=nx, ny=ny, nz=nz)
# x0 = np.random.rand(A.shape[0],1)


# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

# Trace-min solver
# -------------------
residuals = []
start = time.clock()
ml = trace_min_solver(A=A, B=B, strength=strength, splitting=split_agg[0], aggregate=split_agg[1],
					  trace_min=trace_min, presmoother=relaxation1, postsmoother=relaxation2,
                      max_levels=max_levels, max_coarse=max_coarse, keep=keep)
end = time.clock()
setup_time = end-start;

print "Solver built"

start = time.clock()
sol = ml.solve(b, x0, tol, residuals=residuals, accel=accel, maxiter=200)
end = time.clock()
solve_time = end-start

OC = ml.operator_complexity()
CC = ml.cycle_complexity()
SC = ml.setup_complexity()

conv_factors = np.zeros((len(residuals)-1,1))
for i in range(1,len(residuals)-1):
	conv_factors[i] = residuals[i]/residuals[i-1]

CF = np.mean(conv_factors)
CF2 = residuals[-1]/residuals[i-2]

print "Trace-min, problem size ",A.shape[0]," x ",A.shape[0],", ",A.nnz," nonzeros"
print "\tSetup time 		 = ",setup_time
print "\tSolve time 		 = ",solve_time
print "\tSetup complexity 	 = ",SC
print "\tOperator complexity 	 = ",OC
print "\tCycle complexity 	 = ",CC
print "\tAverage con. factor 	 = ",CF
print "\tFinal con. factor 	 = ",CF2


pdb.set_trace()

