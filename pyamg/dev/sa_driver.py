# from pyamg.util.utils import scale_T, get_Cpt_params
# from pyamg.strength import classical_strength_of_connection, symmetric_strength_of_connection, evolution_strength_of_connection, energy_based_strength_of_connection
# from pyamg.aggregation.aggregate import standard_aggregation
# from pyamg.aggregation.tentative import pseudo_interpolation, fit_candidates, new_ideal_interpolation
# from pyamg.aggregation.smooth import jacobi_prolongation_smoother, richardson_prolongation_smoother, energy_prolongation_smoother
# from pyamg.util.utils import relaxation_as_linear_operator



import pdb, cProfile, pstats

import time
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from pyamg.gallery.diffusion import diffusion_stencil_2d
from pyamg.gallery.stencil import stencil_grid
from pyamg.aggregation.rootnode import rootnode_solver
from pyamg.aggregation.aggregation import smoothed_aggregation_solver
from pyamg.aggregation.pairwise import pairwise_solver
from pyamg.util.utils import symmetric_rescaling
from pyamg.gallery import poisson

from scipy import sparse
from scipy.sparse import csr_matrix

# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

# General multilevel parameters
# -----------------------------
max_levels 		   = 20			# Max levels in hierarchy
max_coarse 		   = 100 		# Max points allowed on coarse grid
tol 			   = 1e-8		# Residual convergence tolerance
is_pdef 		   = True		# Assume matrix positive definite (only for aSA)
keep_levels 	   = False		# Also store SOC, aggregation, and tentative P operators
diagonal_dominance = False		# Avoid coarsening diagonally dominant rows 
coarse_solver = 'pinv'
accel = 'cg'
keep = True

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
strength_connection = ('classical', {'theta': 0.2} )
# strength_connection =  ('evolution', {'k': 2, 'epsilon': 4.0, 'symmetrize_measure':True})	# 'symmetric', 'classical', 'evolution'


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
aggregation = ('standard')
# pairwise = ('notay', {'matchings': 2, 'beta': 0.1, 'improve_candidates': False})
pairwise = ('matching', {'matchings': 3, 'improve_candidates': True, 'get_weights': False})


# Interpolation smooother (Jacobi seems slow...)
# -----------------------
# 	- richardson
#		+ omega (4/3)- weighted Richardson w/ weight omega
#		+ degree (1)- number of Richardson iterations 
# 	- jacobi
#		+ omega (4/3)- weighted Jacobi w/ weight omega
#		+ degree (1)- number of Jacobi iterations 
#		+ filter (F)- True / False, filter smoothing matrix S w/ nonzero 
#		  indices of SOC matrix. Can greatly control complexity? (appears to slow convergence)
#		+ weighting (diagonal)- construction of diagonal preconditioning
#			~ local - local row-wise weight, avoids under-damping (appears to slow convergence)
#			~ diagonal - inverse of diagonal of A
#			~ block - block diagonal inverse for A, USE FOR BLOCK SYSTEMS
# 	- energy
#		+ krylov (cg)- descent method for energy minimization
#			~ cg - use cg for SPD systems. 
#			~ cgnr - use for nonsymmetric or indefinite systems.
#			  Only supports diagonal weighting.
#			~ gmres - use for nonsymmetric or indefinite systems.
#		+ degree (1)- sparsity pattesa for P based on (Atilde^degree T).
#		+ maxiter (4)- number of energy minimization steps to apply to P.
#		+ weighting (local)- construction of diagonal preconditioning
#			~ local - local row-wise weight, avoids under-damping 
#			~ diagonal - inverse of diagonal of A
#			~ block - block diagonal inverse for A, USE FOR BLOCK SYSTEMS
# interp_smooth = ('jacobi', {'omega': 4.0/3.0, 'degree': 1 })
interp_smooth = ('energy', {'degree': 3, 
							'prefilter': {'theta': 0.1}, 
							'postfilter': {'theta': 0.1},
							'maxiter': 5} )
 
# interp_smooth2 = interp_smooth1

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
relaxation = ('jacobi', {'omega': 2.0/3.0, 'iterations': 1} )
# relaxation = ('gauss_seidel', {'sweep': 'symmetric', 'iterations': 1} )
# relaxation = ('richardson', {'iterations': 1})


# Improve near null space candidates
# ----------------------------------
# improve_candidates = [('gauss_seidel', {'sweep': 'forward', 'iterations': 4})]
improve_candidates = [('jacobi', {'omega': 2.0/3.0, 'iterations': 4})]
# improve_candidates = ('richardson', {'omega': 3.0/2.0, 'iterations': 4} )

# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

# Problem parameters and variables
# ---------------------------------

rand_guess 	= True
zero_rhs 	= True
problem_dim = 2
N 			= 750
epsilon 	= 0.00				# 'Strength' of aniostropy (only for 2d)
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
	# b = np.array([np.sin(math.pi*np.arange(0,vec_size)/(vec_size-1.0)),np.ones(vec_size)]).T
	

# Random vs. zero initial guess
if rand_guess:
	x0 = np.random.rand(vec_size,1)
else:
	x0 = np.zeros(vec_size,1)




# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

# Classical smoothed aggregation solver
# --------------------------

# pdb.set_trace()
sa_residuals = []

# Form classical smoothed aggregation multilevel solver object
start = time.clock()
ml_sa = smoothed_aggregation_solver(A, B=None, symmetry='symmetric', strength=strength_connection,
									aggregate=aggregation, smooth=interp_smooth, max_levels=max_levels,
									max_coarse=max_coarse, presmoother=relaxation, postsmoother=relaxation,
						 			improve_candidates=improve_candidates, keep=keep )

sol = ml_sa.solve(b, x0, tol, accel=accel, residuals=sa_residuals)
end = time.clock()

SC = ml_sa.setup_complexity()
CC = ml_sa.cycle_complexity()

sa_time = end-start
sa_conv_factors = np.zeros((len(sa_residuals)-1,1))
for i in range(1,len(sa_residuals)):
	sa_conv_factors[i-1] = sa_residuals[i]/sa_residuals[i-1]

CF = np.mean(sa_conv_factors[1:])
print "SA - ", sa_time, " seconds"
print "\tCF - ",CF
print "\tSetup complexity - ",SC
print "\tCycle complexity - ",CC
print "\tEffectve CF - ", CF**(1.0/CC)



# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

# Classical smoothed aggregation solver
# --------------------------

pw_residuals = []

# Form classical smoothed aggregation multilevel solver object
start = time.clock()
ml_pw = pairwise_solver(A, B=None, strength=strength_connection, symmetry='symmetric', aggregate=pairwise,
						smooth=interp_smooth, max_levels=max_levels, max_coarse=max_coarse,
						presmoother=relaxation, postsmoother=relaxation,
						improve_candidates=improve_candidates, keep=keep )

sol = ml_pw.solve(b, x0, tol, accel=accel, residuals=pw_residuals)
end = time.clock()

SC = ml_pw.setup_complexity()
CC = ml_pw.cycle_complexity()

pw_time = end-start
pw_conv_factors = np.zeros((len(pw_residuals)-1,1))
for i in range(1,len(pw_residuals)):
	pw_conv_factors[i-1] = pw_residuals[i]/pw_residuals[i-1]

CF = np.mean(pw_conv_factors[1:])
print "PW - ", pw_time, " seconds"
print "\tCF - ",CF
print "\tSetup complexity - ",SC
print "\tCycle complexity - ",CC
print "\tEffectve CF - ", CF**(1.0/CC)

