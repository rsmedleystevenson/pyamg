import pdb
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import sparse
from scipy.sparse import csr_matrix, spdiags

from pyamg.gallery.diffusion import diffusion_stencil_2d
from pyamg.gallery.stencil import stencil_grid
from pyamg.aggregation.rootnode import rootnode_solver
from pyamg.aggregation.aggregation import smoothed_aggregation_solver
from pyamg.util.utils import symmetric_rescaling

from pyamg.util.utils import scale_rows, scale_columns

from lumped_transport import lumped_transport
from Jacob_complexity import *

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
symmetry = 'symmetric'
accel = 'gmres'
cycle = 'V'

# Strength of connection 
# ----------------------
#	- symmetric, strong connection if |A[i,j]| >= theta * sqrt( |A[i,i]| * |A[j,j]| )
#		+ theta (0)
#	- classical, strong connection if |A[i,j]| >= theta * max( |A[i,k]| )
#		+ theta (0)
# 	- evolution
#		+ epsilon (4)- drop tolerance, > 1. Larger e -> denser matrix. 
#		+ k (2)- ODE num time steps, step size = 1/rho(DinvA) - should be in powers of 2
#		+ block_flag (F)- True / False for block matrices
#		+ symmetrize_measure (T)- True / False, True --> Atilde = 0.5*(Atilde + Atilde.T)
#		+ proj_type (l2)- Define norm for constrained min prob, l2 or D_A
# strength_connection = ('symmetric', {'theta': 0} )
strength_connection =  ('evolution', {'k': 2, 'epsilon': 4.0, 'symmetrize_measure': True})	# 'symmetric', 'classical', 'evolution'
# strength_connection = None

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
# aggregation = ('lloyd', {'ratio': 0.1, 'distance': 'abs'})
aggregation = ('standard')
# aggregation = ('pairwise', {'matchings': 3, 'algorithm': 'drake_C'})


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
#		+ degree (1)- sparsity pattern for P based on (Atilde^degree T).
#		+ maxiter (4)- number of energy minimization steps to apply to P.
#		+ weighting (local)- construction of diagonal preconditioning
#			~ local - local row-wise weight, avoids under-damping 
#			~ diagonal - inverse of diagonal of A
#			~ block - block diagonal inverse for A, USE FOR BLOCK SYSTEMS
interp_smooth = ('energy', {'krylov': 'gmres', \
                            'degree': 6, \
                            'maxiter': 5, \
                            'weighting': 'local', \
                            'prefilter' : {'theta' : 0.1}, \
                            'postfilter' : {'theta' : 0.0} }) 

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
# relaxation = ('jacobi', {'omega': 2.0/3.0, 'iterations': 1} )
relaxation = ('gauss_seidel', {'sweep': 'forward', 'iterations': 1} )
# relaxation = ('richardson', {'iterations': 1})


# Improve near null space candidates
# ----------------------------------
improve_candidates = [('gauss_seidel', {'sweep': 'forward', 'iterations': 4})]
# improve_candidates = [('jacobi', {'omega': 2.0/3.0, 'iterations': 4})]
# improve_candidates = ('richardson', {'omega': 3.0/2.0, 'iterations': 4} )

# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

# Discontinuous sigma values
class sigtc(object):
    def eval(self, x):
        v = 1e-4
        if (x[0] <= 1.0 and x[1] >= 0.5):
            v = 10
        if (x[0] <= 2.0 and x[1] <= 0.25):
            v = 100
        if (1.0 <= x[0] <= 2.0 and x[1] >= 0.75):
            v = 50
        if (2.5 <= x[0] <= 3.5 and 0.25 <= x[1] <= 0.75):
            v = 1000
        if (4.0 <= x[0] <= 5.0 and 0.0 <= x[1] <= 0.25):
            v = 10
        if (4.0 <= x[0] <= 5.0 and 0.75 <= x[1] <= 1.0):
            v = 100
        return v

    def get_max(self):
        return 1000

sigt = sigtc()

# Problem parameters and variables
# ---------------------------------

rand_guess 	= True
zero_rhs 	= True
theta = 3*np.pi/16.0 		# Must be in (0, pi/2)
num_rows = 200
num_cols = 5*num_rows
rn_residuals = []

L = lumped_transport(num_rows=num_rows, num_cols=num_cols, theta=theta, sigt=sigt, scale=False, size_x=5.0, size_y=1.0)
A = csr_matrix(L.T*L)
# [d,d,A] = symmetric_rescaling(A)  # This is not symmetric... 
vec_size = A.shape[0]

# Zero right hand side or sin(pi x)
if zero_rhs:
	b = np.zeros((vec_size,1))
	# If zero rhs and zero initial guess, throw error
	if not rand_guess:
		print "Zero rhs and zero initial guess converges trivially."

# Random vs. zero initial guess
if rand_guess:
	x0 = np.random.rand(vec_size,1)
else:
	x0 = np.zeros(vec_size,1)

# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #


# Classical root node solver
# --------------------------
maxiter = 400

# Form classical root node multilevel solver object
start = time.clock()
ml_rn = rootnode_solver(A, B=None, symmetry=symmetry, strength=strength_connection,
						aggregate=aggregation, smooth=interp_smooth, max_levels=max_levels,
						max_coarse=max_coarse, presmoother=relaxation, postsmoother=relaxation,
						improve_candidates=improve_candidates, keep=keep_levels )

# pdb.set_trace()

sol = ml_rn.solve(b, x0, tol, residuals=rn_residuals, cycle=cycle, accel=accel, maxiter=maxiter)
end = time.clock()
rn_time = end-start

setup_cost = setup_complexity(sa=ml_rn, strength=strength_connection, smooth=interp_smooth,
							  improve_candidates=improve_candidates, aggregate=aggregation,
							  presmoother=relaxation, postsmoother=relaxation, keep=keep_levels,
							  max_levels=max_levels, max_coarse=max_coarse, coarse_solver=coarse_solver,
							  symmetry=symmetry)

cycle_cost = cycle_complexity(solver=ml_rn, presmoothing=relaxation,
							  postsmoothing=relaxation, cycle=cycle)

rn_conv_factors = np.zeros((len(rn_residuals)-1,1))
for i in range(0,len(rn_residuals)-1):
	rn_conv_factors[i] = rn_residuals[i]/rn_residuals[i-1]

av_fact = np.mean(rn_conv_factors[1::])

# print rn_conv_factors
print "Root node - ", rn_time, " seconds"
print "\tSetup cost - ", setup_cost
print "\tCycle cost - ", cycle_cost
print "\tAverage CF - ", av_fact
print "\tNum DOFS   - ", A.shape[0]
