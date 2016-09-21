"""Support for aggregation-based AMG"""
from __future__ import absolute_import

__docformat__ = "restructuredtext en"

from warnings import warn
from scipy.sparse import csr_matrix, isspmatrix_csr, isspmatrix_bsr,\
	SparseEfficiencyWarning, kron, kronsum
import numpy as np

from pyamg.multilevel import multilevel_solver
from pyamg.relaxation.smoothing import change_smoothers
from pyamg.aggregation.aggregation import smoothed_aggregation_solver
from pyamg.aggregation.rootnode import rootnode_solver
from pyamg.gallery import poisson


__all__ = ['tensor_product_solver']


def tensor_product_solver(A,
						  B=None,
						  BH=None,
						  solver_handles=[rootnode_solver],
						  solver_args=[{'smooth': ('energy', {'degree': 2, 'maxiter': 10}), 
						  				'improve_candidates' : ('jacobi', {'omega': 0.5}) } ],
						  **kwargs):

	if len(solver_handles) != len(solver_args):
		raise ValueError("Must have same number of solver handles "
						 "and solver arguments.")

	num_mats = len(A)
	for i in range(len(solver_args),num_mats):
		solver_args.append(solver_args[-1])
		solver_handles.append(solver_handles[-1])

	if B is None:
		B = num_mats * [None]

	if BH is None:
		BH = num_mats * [None]

	# Construct solver for each product matrix
	solvers = []
	for i in range(0,num_mats):
		solvers.append( solver_handles[i](A=A[i], **solver_args[i]) )

	# TODO : Need to generalize to any number of product matrices
	num_levels = len(solvers[0].levels)
	levels = []
	levels.append(multilevel_solver.level())
	levels[-1].A = kron(solvers[0].levels[0].A, solvers[1].levels[0].A, format='csr')

	# Assuming Galerkin coarse grid for now
	for i in range(0,num_levels-1):
		levels[-1].P = kron(solvers[0].levels[i].P, solvers[1].levels[i].P, format='csr')
		levels[-1].R = levels[-1].P.T
		levels.append(multilevel_solver.level())
		levels[-1].A = kron(solvers[0].levels[i+1].A, solvers[1].levels[i+1].A, format='csr')

	# TODO : what to do with presmoother and postsmoother
	# TODO : Fix smoother
	tensor_solver = multilevel_solver(levels, **kwargs)
	change_smoothers(tensor_solver, ('jacobi', {'omega': 1.0}), ('jacobi', {'omega': 1.0}))

	return tensor_solver


# ------------------------------------------------------------ #
# ------------------------------------------------------------ #
# - Apparently the tensor product of two 1d poisson problems
# 	is a hard problem for AMG to solve. Not sure why...
# - Tensor product AMG does at least as well (or poorly for that
# 	matter) as full AMG. Now just need test problems...
# - TODO : WHAT TO DO WITH DIFFERENT NUMBER OF LEVELS FOR DIFFERENT
# 		   HIERACHIES??
# ------------------------------------------------------------ #
# ------------------------------------------------------------ #

import pdb 

n0 = 300
N = n0*n0
A1 = poisson(grid=[n0])
A2 = poisson(grid=[n0])
ml = tensor_product_solver(A=[A1,A2])
ml2 = rootnode_solver(A=ml.levels[0].A,
					  presmoother=('jacobi', {'omega': 1.0}),
					  postsmoother=('jacobi', {'omega': 1.0}),
					  smooth=('energy', {'degree': 2, 'maxiter': 10}))
accel = 'cg'
tol = 1e-8
maxiter = 200
b = np.zeros((N,1))
x0 = np.random.rand(N,1)

# ------------------------------------------------------------ #

res = []
ml.solve(b=b, x0=x0, tol=tol, maxiter=maxiter, residuals=res, accel=accel)
CFs = np.zeros((len(res)-1,1))
for i in range(1,len(res)-1):
	CFs[i] = res[i]/res[i-1]

CF = np.mean(CFs)

print "Tensor product AMG"
# print "\tSetup time 		 = ",setup_time
# print "\tSetup complexity 	 = ",SC
# print "\tSolve time 		 = ",solve_time
# print "\tOperator complexity 	 = ",OC
# print "\tCycle complexity 	 = ",CC
print "\tConvergence factor 	 = ",CF

# ------------------------------------------------------------ #

res2 = []
ml2.solve(b=b, x0=x0, tol=tol, maxiter=maxiter, residuals=res2, accel=accel)
CFs2 = np.zeros((len(res2)-1,1))
for i in range(1,len(res2)-1):
	CFs2[i] = res2[i]/res2[i-1]

CF2 = np.mean(CFs2)

print "Root node AMG"
# print "\tSetup time 		 = ",setup_time
# print "\tSetup complexity 	 = ",SC
# print "\tSolve time 		 = ",solve_time
# print "\tOperator complexity 	 = ",OC
# print "\tCycle complexity 	 = ",CC
print "\tConvergence factor 	 = ",CF2

pdb.set_trace()





