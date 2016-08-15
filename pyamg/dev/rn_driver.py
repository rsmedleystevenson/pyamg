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
from pyamg.util.utils import symmetric_rescaling
from pyamg.gallery import poisson

from scipy import sparse
from scipy.sparse import csr_matrix
import scipy.io as sio

from recirc import get_recirculating
from biharmonic import get_biharmonic
from poisson import get_poisson
from elasticity_bar import get_elasticity_bar

# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

# General multilevel parameters
# -----------------------------
max_levels 		   = 20			# Max levels in hierarchy
max_coarse 		   = 20 		# Max points allowed on coarse grid
tol 			   = 1e-8		# Residual convergence tolerance
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
strength_connection = ('classical', {'theta': 0.35} )
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
# aggregation = ('standard')
aggregation = None
splitting = 'RS'

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
pre = 0.1
post = 0.1
deg = 2
interp_smooth = ('energy', {'krylov': 'cg', \
							'degree': deg, \
							'maxiter': int(np.ceil(deg*3)), \
							'weighting': 'diagonal', \
							'prefilter' : {'theta' : pre}, \
							'postfilter' : {'theta' : post} })

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
relaxation = ('gauss_seidel_nr', {'sweep': 'symmetric', 'iterations': 1} )
# relaxation = ('richardson', {'iterations': 1})


# Improve near null space candidates
# ----------------------------------
improve_candidates = [('gauss_seidel_nr', {'sweep': 'symmetric', 'iterations': 4})]
# improve_candidates = [('jacobi', {'omega': 4.0/3.0, 'iterations': 4})]
# improve_candidates = ('richardson', {'omega': 3.0/2.0, 'iterations': 4} )

# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

# Problem parameters and variables
# ---------------------------------

# 3d isotropic diffusion on unstructured mesh
# A = sio.loadmat("./unstructured_triangles_3D_ani_diff_ref2.mat")['A']
# b = np.zeros((A.shape[0],1))

# 2d diffusion from pyamg gallery
N = 300
epsilon = 1.0
theta = 3*np.pi / 16.0
stencil = diffusion_stencil_2d(epsilon,theta)
A = stencil_grid(stencil, [N,N], format='csr')
[d,d,A] = symmetric_rescaling(A)
b = np.zeros((A.shape[0],1))
B = None

# Recirculating flow
# eps = 0.005
# theta= 1.0
# n0 = 50
# # A, b = get_poisson(n=n0,eps=eps,theta=theta,rand=True)
# A, b = get_recirculating(n=n0, eps=eps)
# # A, b = get_biharmonic(n0=500)

# # [d,d,A] = symmetric_rescaling(A)
# n = A.shape[0]
# x0 = np.random.rand(n,1)

# nx = 10
# ny = 10
# nz = 500
# A, b, B = get_elasticity_bar(nx=nx, ny=ny, nz=nz)
# [d,d,A] = symmetric_rescaling(A)
# n = A.shape[0]


# Note
#	- Oddly enough, recirculating problem is not stable with GS
#	- Is good with GS_nr, and stable anyways with weighted Jacobi
#	- Need to compare by doing V(2,2) with Jacobi, also need to play with omega


# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

# Classical root node solver
# --------------------------

rn_residuals = []
n = A.shape[0]
x0 = np.random.rand(n,1)


# Form classical root node multilevel solver object
start = time.clock()
ml_rn = rootnode_solver(A, B=B, strength=strength_connection, aggregate=aggregation,
						splitting=splitting, smooth=interp_smooth, max_levels=max_levels,
						max_coarse=max_coarse, presmoother=relaxation, postsmoother=relaxation,
						improve_candidates=improve_candidates, keep=keep, setup_complexity=True )

end = time.clock()
rn_setup_time = end-start

start = time.clock()
sol = ml_rn.solve(b, x0, tol, residuals=rn_residuals, accel=accel)
end = time.clock()
rn_solve_time = end-start

# Get complexities
OC = ml_rn.operator_complexity()
CC = ml_rn.cycle_complexity()
SC = ml_rn.setup_complexity()

# Convergence factors 
rn_conv_factors = np.zeros((len(rn_residuals)-1,1))
for i in range(1,len(rn_residuals)-1):
	rn_conv_factors[i] = rn_residuals[i]/rn_residuals[i-1]

CF = np.mean(rn_conv_factors[1:])

print "Problem : ", n," DOF, ", A.nnz," nonzeros"
print "\tSetup time      - ",rn_setup_time, " seconds"
print "\tSolve time      - ", rn_solve_time, " seconds"
print "\tConv. factor    - ", CF
print "\tSetup complexity - ", SC
print "\tOp. complexity  - ", OC
print "\tCyc. complexity - ", CC

# SC = ml_rn.setup_complexity(verbose=True)

# results = open('./transport_tests.csv', 'a')
# results.write('%i,%i,%i,%ipi/%i,%1.1f,%i,%1.1f,%i,%i,%s,%s,%s,%1.4f,%1.2f,%1.2f,%1.2f,%1.2f\n' %(n,eps,A.nnz,OC,CC,SC,CF


pdb.set_trace()

