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

# Strength of connection 
# ----------------------
#	- symmetric, strong connection if |A[i,j]| >= theta * sqrt( |A[i,i]| * |A[j,j]| )
#		+ theta (0)
#	- classical, strong connection if |A[i,j]| >= theta * max( |A[i,k]| )
#		+ theta (0)
# 	- evolution
#		+ epsilon (4)- drop tolerance, > 1. Larger e -> denser matrix. 
#		+ k (2)- ODE num time steps, step size = 1/rho(DinvA)
#		+ block_flag (F)- True / False for block matrices
#		+ symmetrize_measure (T)- True / False, True --> Atilde = 0.5*(Atilde + Atilde.T)
#		+ proj_type (l2)- Define norm for constrained min prob, l2 or D_A
# strength = ('classical', {'theta': 0.35} )
strength = ('evolution', {'k': 2, 'epsilon': 5.0})

# AMG CF-splitting 
# ----------------
# 	- RS : Original Ruge-Stuben method
#    	+ Good C/F splittings, inherently serial.
# 	- PMIS: Parallel Modified Independent Set
#     	+ Very fast construction, low operator complexity. Convergence
#		  can deteriorate with increasing problem size.
# 	- PMISc: Parallel Modified Independent Set in Color
#     	+ Fast construction with low operator complexity.
#         Better scalability than PMIS on structured meshes.
# 	- CLJP: Clearly-Luby-Jones-Plassmann
#     	+ Parallel method, cost and complexity comparable to RS.
#     	  Convergence can deteriorate with increasing problem
#       size on structured meshes.
# 	- CLJP-c: Clearly-Luby-Jones-Plassmann in Color
#     	+ Parallel method, cost and complexity comparable to RS.
#     	  Better scalability than CLJP on structured meshes.
#	- CR : Compatible relaxation
#		+ 
#		+ 
splitting = 'RS'
# splitting = None

# Aggregation 
# -----------
#	- standard
#	- naive
#		+ Differs from standard - "Each dof is considered. If it has been aggregated
# 		  skip over. Otherwise, put dof and any unaggregated neighbors in an aggregate.
#   	  Results in possibly much higher complexities than standard aggregation." 
#	- lloyd (don't know how this works...)
#		+ ratio (0.03)- fraction of the nodes which will be seeds.
#		+ maxiter (10)- maximum number iterations to perform
#		+ distance (unit)- edge weight of graph G used in Lloyd clustering.
#		  For each C[i,j]!=0,
#	    	~ unit - G[i,j] = 1
#	        ~ abs  - G[i,j] = abs(C[i,j])
#	        ~ inv  - G[i,j] = 1.0/abs(C[i,j])
#	        ~ same - G[i,j] = C[i,j]
#	        ~ sub  - G[i,j] = C[i,j] - min(C)
aggregate = ('standard')
aggregate = None


# Relaxation
# ---------- 
# 	- jacobi
#		+ omega (1.0)- damping parameter.
#		+ iterations (1)- number of iterations to perform.
# 	- gauss_seidel
#		+ iterations (1)- number of iterations to perform.
#		+ sweep (forward)- direction of relaxation sweep.
#			~ forward
#			~ backward
#			~ symmetric
# 	- sor
#		+ omega - damping parameter. If omega = 1.0, SOR <--> Gauss-Seidel.
#		+ iterations (1)- number of iterations to perform.
#		+ sweep (forward)- direction of relaxation sweep.
#			~ forward
#			~ backward
#			~ symmetric
# 	- block_jacobi
#		+ omega (1.0)- damping parameter.
#		+ iterations (1)- number of relaxation iterations.
#		+ blocksize (1)- block size of bsr matrix
#		+ Dinv (None)- Array holding block diagonal inverses of A.
#		  size (numBlocks, blocksize, blocksize)
#	- block_gauss_seidel
#		+ iterations (1)- number of relaxation iterations.
#		+ sweep (forward)- direction of relaxation sweep.
#			~ forward
#			~ backward
#			~ symmetric
#		+ blocksize (1)- block size of bsr matrix
#		+ Dinv (None)- Array holding block diagonal inverses of A.
#		  size (numBlocks, blocksize, blocksize)
#
# Note, Schwarz relaxation, polynomial relaxation, Cimmino relaxation,
# Kaczmarz relaxation, indexed Gauss-Seidel, and one other variant of 
# Gauss-Seidel are also available - see relaxation.py. 
# relaxation = ('jacobi', {'omega': 4.0/3.0, 'iterations': 1} )
relaxation = ('gauss_seidel', {'sweep': 'symmetric', 'iterations': 1} )
# relaxation = ('richardson', {'iterations': 1})

improve_candidates = ('gauss_seidel', {'sweep': 'forward', 'iterations': 4})
# improve_candidates = ('jacobi', {'omega': 4.0/3.0, 'iterations': 4})


# Trace-minimization parameters
# -----------------------------
trace_min={'deg': 1, 'maxiter': 10,
           'tol': 1e-8, 'debug': False,
           'get_tau': 1.0, 'precondition': True}


# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

# Problem parameters and variables
# ---------------------------------

rand_guess 	= True
zero_rhs 	= True
problem_dim = 2
N 			= 1000
epsilon 	= 0.00			# 'Strength' of aniostropy (only for 2d)
theta 		= 3.0*math.pi/16.0	# Angle of anisotropy (only for 2d)

# 1d Poisson 
if problem_dim == 1:
	grid_dims = [N,1]
	A = poisson((N,), format='csr')
# 2d Poisson
elif problem_dim == 2:
	grid_dims = [N,N]
	stencil = diffusion_stencil_2d(epsilon,theta)
	A = stencil_grid(stencil, grid_dims, format='csr')

# # Vectors and additional variables
[d,d,A] = symmetric_rescaling(A)
vec_size = np.prod(grid_dims)

# Zero right hand side or sin(pi x)
if zero_rhs:
	b = np.zeros((vec_size,1))
	# If zero rhs and zero initial guess, throw error
	if not rand_guess:
		print "Zero rhs and zero initial guess converges trivially."
# Note, this vector probably doesn't make sense in 2d... 
else: 
	b = np.sin(math.pi*np.arange(0,vec_size)/(vec_size-1.0))

# Random vs. zero initial guess
if rand_guess:
	x0 = np.random.rand(vec_size,1)
else:
	x0 = np.zeros(vec_size,1)

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
                      improve_candidates=improve_candidates, max_levels=max_levels,
                      max_coarse=max_coarse, keep=keep)
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

