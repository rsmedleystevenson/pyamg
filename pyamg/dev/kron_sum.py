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


__all__ = ['kronecker_sum_solver']


def kronecker_sum_solver(A,
						 B=None,
						 BH=None,
						 solver_handles=[rootnode_solver],
						 solver_args=[{}],
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

	import pdb
	pdb.set_trace()

	# Construct solver for each product matrix
	solvers = []
	for i in range(0,num_mats):
		solvers.append( solver_handles[i](A=A[i], **solver_args[i]) )

	# TODO : Need to generalize to any number of product matrices
	num_levels = len(solvers[0].levels)
	levels = []
	levels.append(multilevel_solver.level())
	levels[-1].A = kronsum(solvers[0].levels[0].A, solvers[1].levels[0].A, format='csr')

	# Assuming Galerkin coarse grid for now
	for i in range(0,num_levels-1):
		levels[i].P = kron(solvers[0].levels[i].P, solvers[1].levels[i].P, format='csr')
		levels[i].R = levels[i].P.T
		levels.append(multilevel_solver.level())
		# levels[-1].A = kronsum(solvers[0].levels[i+1].A, solvers[1].levels[i+1].A, format='csr')
		levels[i+1].A = levels[i].R * levels[i].A * levels[i].P

	# TODO : what to do with presmoother and postsmoother
	# TODO : Fix smoother
	kron_solver = multilevel_solver(levels, **kwargs)
	change_smoothers(kron_solver, 'jacobi', 'jacobi')

	return kron_solver


# ------------------------------------------------------------ #
# ------------------------------------------------------------ #
# - Diverges with naive non-Galerkin coarse grid (P^TP = I)
# - Could try collapsing stencil of P'*P, and then factor
# 	out the diagonal? 
# - Does okay with Galerkin coarse grid, but I think this
#	eliminates the obvious benefit of kronecker sum AMG...
#
#
# ------------------------------------------------------------ #
# ------------------------------------------------------------ #

import pdb 

n0 = 500
N = n0*n0
A1 = poisson(grid=[n0])
A2 = poisson(grid=[n0])
ml = kronecker_sum_solver(A=[A1,A2])
ml2 = rootnode_solver(A=ml.levels[0].A,
					  presmoother='jacobi',
					  postsmoother='jacobi')
accel = None
tol = 1e-10
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

print "Kronecker sum AMG"
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





