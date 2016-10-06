import pdb

import time
import math
import numpy as np
from scipy import sparse, io
from scipy.sparse import csr_matrix
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D

from pyamg.gallery import poisson
from pyamg.gallery.diffusion import diffusion_stencil_2d
from pyamg.gallery.stencil import stencil_grid
from pyamg.aggregation.aggregation import smoothed_aggregation_solver
from pyamg.aggregation.adaptive import adaptive_sa_solver
from pyamg.aggregation.new_adaptive import asa_solver
from pyamg.aggregation.gtyr_adaptive import gtyr_solver
from pyamg.util.utils import symmetric_rescaling

# from poisson import get_poisson
# from elasticity_bar import get_elasticity_bar


def A_norm(x, A):
    """
    Calculate A-norm of x
    """
    x = np.ravel(x)
    return np.sqrt(np.dot(x.conjugate(), A*x))


# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

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
# strength = ('symmetric', {'theta': 0.0} )
strength = ('classical', {'theta': 0.5} )
# strength = ('evolution', {'epsilon': 4.0, 'k' : 2} )


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


# Interpolation smooother (Jacobi seems slow...)
# -----------------------
# 	- richardson
#		+ omega (4/3)- weighted Richardson w/ weight omega
#		+ degree (1)- number of Richardson iterations 
# 	- jacobi
#		+ omega (4/3)- weighted Jacobi w/ weight omega
#		+ degree (1)- number of Jacobi iterations 
#		+ filter (F)- True / False, filter smoothing matrix S w/ nonzero 
#		  indices of SOC matrix. Can greatly control complexity? 
#		+ weighting (diagonal)- construction of diagonal preconditioning
#			~ local - local row-wise weight, avoids under-damping
#			~ diagonal - inverse of diagonal of A
#			~ block - block diagonal inverse for A, USE FOR BLOCK SYSTEMS
# 	- energy
#		+ krylov (cg)- descent method for energy minimization
#			~ cg - use cg for SPD systems. 
#			~ cgnr - use for nonsymmetric or indefinite systems.
#			  Only supports diagonal weighting.
#			~ gmres - use for nonsymmetric or indefinite systems.
#		+ degree (1)- sparsity pattern for P based on (Atilde^degree T).
#		+ maxiter (4)- number of energy minimization steps to apply to P.
#		+ weighting (local)- construction of diagonal preconditioning
#			~ local - local row-wise weight, avoids under-damping 
#			~ diagonal - inverse of diagonal of A
#			~ block - block diagonal inverse for A, USE FOR BLOCK SYSTEMS
interp_smooth = ('jacobi', {'omega' : 4.0/3.0,
							'degree' : 1,
							'filter' : False,
							'weighting' : 'diagonal'} )
# interp_smooth = ('richardson', {'omega': 3.0/2.0, 'degree': 1} )
# interp_smooth = ('energy', {'krylov' : 'cg',
# 							'degree' : 1,
# 							'maxiter' : 3,
# 							'weighting' : 'local'} )


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
# relaxation = ('jacobi', {'omega': 3.0/3.0, 'iterations': 1} )
relaxation = ('gauss_seidel', {'sweep': 'forward', 'iterations': 1} )
# relaxation = ('richardson', {'iterations': 1})


# Adaptive parameters
# -------------------
candidate_iters		= 5 	# number of smoothings/cycles used at each stage of adaptive process
num_candidates 		= 1		# number of near null space candidated to generate
target_convergence	= 0.25 	# target convergence factor, called epsilon in adaptive solver input
eliminate_local		= (False, {'Ca': 1.0})	# aSA, supposedly not useful I think

# New adaptive parameters
# -----------------------
weak_tol 		   = 1.0	# new aSA 
max_bad_guys	   = 8
max_bullets		   = 3
max_iterations 	   = 5
improvement_iters  = 4		# number of times a target bad guy is improved
num_targets 	   = 1		# number of near null space candidates to generate

# from SA --> WHY WOULD WE DEFINE THIS TO BE DIFFERENT THAN THE RELAXATION SCHEME USED??
improve_candidates = ('gauss_seidel', {'sweep': 'forward', 'iterations': improvement_iters})
# improve_candidates = ('jacobi', {'omega': 3.0/3.0, 'iterations': 4})
# improve_candidates = ('richardson', {'omega': 3.0/2.0, 'iterations': 4} )


# General multilevel parameters
# -----------------------------
max_levels 		   = 20 		# Max levels in hierarchy
max_coarse 		   = 20 		# Max points allowed on coarse grid
tol 			   = 1e-8		# Residual convergence tolerance
is_pdef 		   = True		# Assume matrix positive definite (only for aSA)
keep_levels 	   = False		# Also store SOC, aggregation, and tentative P operators
diagonal_dominance = False		# Avoid coarsening diagonally dominant rows 
coarse_solver = 'pinv'
accel = 'gmres'
cycle = 'V'
keep = False
gtyr = True			# Run GTYR solver 

# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

# Problem parameters and variables
# ---------------------------------

# Poisson
# -------
n0 = 500
eps = 0.001
theta = 3*np.pi / 14.0

# A, b = get_poisson(n=n0, eps=eps, theta=theta, rand=False)
# n = A.shape[0]
# x0 = np.random.rand(n,1)
# b = np.zeros((n,1))

# 2d Poisson pyAMG gallery
grid_dims = [n0,n0]
stencil = diffusion_stencil_2d(eps,theta)
A = stencil_grid(stencil, grid_dims, format='csr')


# A = io.mmread("/Users/ben/Desktop/aSA_dataset/MMMAT/SPD-MTX/bundle1.mtx")		# Easy
# A = io.mmread("/Users/ben/Desktop/aSA_dataset/MMMAT/SPD-MTX/chem97ztz.mtx")	# Easy
# A = io.mmread("/Users/ben/Desktop/aSA_dataset/MMMAT/SPD-MTX/crankseg.mtx")	# Hard and interesting
# A = io.mmread("/Users/ben/Desktop/aSA_dataset/MMMAT/SPD-MTX/crystm03.mtx")	# Easy
# A = io.mmread("/Users/ben/Desktop/aSA_dataset/MMMAT/SPD-MTX/cvxbqp1.mtx")		# Really hard
# A = io.mmread("/Users/ben/Desktop/aSA_dataset/MMMAT/SPD-MTX/obstacle.mtx")	# Easy
# A = io.mmread("/Users/ben/Desktop/aSA_dataset/MMMAT/SPD-MTX/qa8fm.mtx")		# Easy, kind of interesting
# A = io.mmread("/Users/ben/Desktop/aSA_dataset/MMMAT/SPD-MTX/sts4098.mtx")		# Small, medium hard, probably hard to scale (need big tol)
# A = io.mmread("/Users/ben/Desktop/aSA_dataset/MMMAT/SPD-MTX/thread.mtx")		# Really hard, almost dense matrix

# Vectors and additional variables
n = A.shape[0]
b = np.zeros((n,1))
x0 = np.random.rand(n,1)

bad_guy = np.ones((n,1))
# bad_guy = None
# bad_guy = np.array((np.sin(np.linspace(0,1,A.shape[0])*np.pi),)).T

# Elasticity 
# ----------
# nx = 2
# ny = 100
# nz = 5
# A, b, bad_guy = get_elasticity_bar(nx=nx, ny=ny, nz=nz)
# A.eliminate_zeros()
# n = A.shape[0]
# x0 = np.random.rand(n,1)
# # bad_guy = None

# [D, dum, dum] = symmetric_rescaling(A, copy=False)
# bad_guy[:,0] = D * bad_guy[:,0]


# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

# Classical SA solver
# -------------------

sa_residuals = []
start = time.clock()
ml_sa = smoothed_aggregation_solver(A, B=bad_guy, symmetry='symmetric', strength=strength, aggregate=aggregate,
						 			smooth=interp_smooth, max_levels=max_levels, max_coarse=max_coarse,
						 			presmoother=relaxation, postsmoother=relaxation,
						 			improve_candidates=improve_candidates, coarse_solver=coarse_solver,
						 			keep=keep_levels )

sa_sol = ml_sa.solve(b, x0, tol, residuals=sa_residuals, cycle=cycle, accel=accel)

end = time.clock()
sa_time = end-start

# Get complexities
OC = ml_sa.operator_complexity()
CC = ml_sa.cycle_complexity()
SC = ml_sa.setup_complexity()

# Convergence factors 
sa_conv_factors = np.zeros((len(sa_residuals)-1,1))
for i in range(1,len(sa_residuals)-1):
	sa_conv_factors[i] = sa_residuals[i]/sa_residuals[i-1]

CF = np.mean(sa_conv_factors[1:])

print "SA Solver : ", A.shape[0]," DOF, ", A.nnz," nonzeros"
# print "\tSetup time      	- ",sa_setup_time, " seconds"
# print "\tSolve time      	- ", sa_solve_time, " seconds"
print "\tConv. factor    	- ", CF
print "\tSetup complexity 	- ", SC
print "\tOp. complexity  	- ", OC
print "\tCyc. complexity 	- ", CC


# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

# Classical aSA solver
# --------------------

# asa_residuals = []
# start = time.clock()
# [ml_asa, work] = asa_solver(A, B=bad_guy, pdef=is_pdef, num_candidates=num_candidates,
# 									candidate_iters=candidate_iters, improvement_iters=improvement_iters,
# 									epsilon=target_convergence, max_levels=max_levels, max_coarse=max_coarse,
# 									aggregate=aggregate, prepostsmoother=relaxation, smooth=interp_smooth,
# 									strength=strength, coarse_solver=coarse_solver,
# 									eliminate_local=(False, {'Ca': 1.0}), keep=keep_levels)

# asa_sol = ml_asa.solve(b, x0, tol, residuals=asa_residuals, cycle=cycle)

# end = time.clock()
# asa_time = end-start
# asa_conv_factors = np.zeros((len(asa_residuals)-1,1))
# for i in range(1,len(asa_residuals)-1):
# 	asa_conv_factors[i] = asa_residuals[i]/asa_residuals[i-1]

# print "Classical aSA - ", asa_time, " seconds"
# print asa_conv_factors

# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

# New aSA solver
# --------------

new_asa_residuals = []
start = time.clock()
ml_new_asa = asa_solver(A, B=bad_guy,
						strength=strength,
						aggregate=aggregate,
						smooth=interp_smooth,
						presmoother=relaxation,
						postsmoother=relaxation,
						improvement_iters=improvement_iters,
						max_coarse=max_coarse,
						max_levels=max_levels,
						target_convergence=target_convergence,
						max_bullets=max_bullets,
						max_bad_guys=max_bad_guys,
						num_targets=num_targets,
						max_iterations=max_iterations,
						weak_tol=weak_tol,
						diagonal_dominance=diagonal_dominance,
						coarse_solver=coarse_solver,
						cycle=cycle,
						verbose=True,
						keep=keep,
						setup_complexity=True)

new_asa_sol = ml_new_asa.solve(b, x0, tol, residuals=new_asa_residuals, cycle=cycle, accel=accel)
end = time.clock()
new_asa_time = end-start

# Get complexities
OC = ml_new_asa.operator_complexity()
CC = ml_new_asa.cycle_complexity()
SC = ml_new_asa.setup_complexity(verbose=False)

# Convergence factors 
new_asa_conv_factors = np.zeros((len(new_asa_residuals)-1,1))
for i in range(1,len(new_asa_residuals)-1):
	new_asa_conv_factors[i] = new_asa_residuals[i]/new_asa_residuals[i-1]

CF = np.mean(new_asa_conv_factors[1:])

print "aSA Solver : ", A.shape[0]," DOF, ", A.nnz," nonzeros"
# print "\tSetup time      	- ",sa_setup_time, " seconds"
# print "\tSolve time      	- ", sa_solve_time, " seconds"
print "\tConv. factor    	- ", CF
print "\tSetup complexity 	- ", SC
print "\tOp. complexity  	- ", OC
print "\tCyc. complexity 	- ", CC

# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

# New GTYR solver
# --------------
if gtyr:

	gtyr_residuals = []
	start = time.clock()
	ml_gtyr = gtyr_solver(A, B=bad_guy,
							strength=strength,
							aggregate=aggregate,
							smooth=interp_smooth,
							presmoother=relaxation,
							postsmoother=relaxation,
							improvement_iters=improvement_iters,
							max_coarse=max_coarse,
							max_levels=max_levels,
							target_convergence=target_convergence,
							max_bullets=max_bullets,
							max_bad_guys=max_bad_guys,
							num_targets=num_targets,
							max_level_iterations=max_iterations,
							weak_tol=weak_tol,
							diagonal_dominance=diagonal_dominance,
							coarse_solver=coarse_solver,
							cycle=cycle,
							verbose=True,
							keep=keep,
							setup_complexity=True)

	gtyr_sol = ml_gtyr.solve(b, x0, tol, residuals=gtyr_residuals, cycle=cycle, accel=accel)
	end = time.clock()
	gtyr_time = end-start

	# Get complexities
	OC = ml_gtyr.operator_complexity()
	CC = ml_gtyr.cycle_complexity()
	SC = ml_gtyr.setup_complexity(verbose=False)

	# Convergence factors 
	gtyr_conv_factors = np.zeros((len(gtyr_residuals)-1,1))
	for i in range(1,len(gtyr_residuals)-1):
		gtyr_conv_factors[i] = gtyr_residuals[i]/gtyr_residuals[i-1]

	CF = np.mean(gtyr_conv_factors[1:])

	print "GTYR solver : ", A.shape[0]," DOF, ", A.nnz," nonzeros"
	# print "\tSetup time      	- ",sa_setup_time, " seconds"
	# print "\tSolve time      	- ", sa_solve_time, " seconds"
	print "\tConv. factor    	- ", CF
	print "\tSetup complexity 	- ", SC
	print "\tOp. complexity  	- ", OC
	print "\tCyc. complexity 	- ", CC


pdb.set_trace()

