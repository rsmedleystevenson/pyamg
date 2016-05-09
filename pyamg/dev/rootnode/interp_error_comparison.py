import pdb
import time
import math
import numpy as np
import scipy
import matplotlib.pyplot as plt
from pyamg.gallery.diffusion import diffusion_stencil_2d
from pyamg.gallery.stencil import stencil_grid
from pyamg.aggregation.rootnode import rootnode_solver
from pyamg.aggregation.rootnode_nii import newideal_solver
from pyamg.gallery import poisson
from pyamg.util.utils import symmetric_rescaling, scale_T, get_Cpt_params, relaxation_as_linear_operator
from pyamg.strength import classical_strength_of_connection, symmetric_strength_of_connection, evolution_strength_of_connection, energy_based_strength_of_connection
from pyamg.aggregation.aggregate import standard_aggregation
from pyamg.aggregation.tentative import fit_candidates, new_ideal_interpolation
from pyamg.aggregation.smooth import jacobi_prolongation_smoother, richardson_prolongation_smoother, energy_prolongation_smoother
from pyamg import amg_core
from pyamg.classical import CR
from pyamg.aggregation.tentative import fit_candidates

from scipy.sparse import csr_matrix, bsr_matrix, identity, linalg, identity, vstack

from copy import deepcopy

# ------------------------------------------------------------------------- #
# ------------------------------------------------------------------------- #

SOC 	  = 'evol'
SOC_drop  = 4.0
SA 		  = 1 		# Use SA type coarsening or CR
pow_G 	  = 1
epsilon   = 0.1
theta 	  = 3.0*math.pi/16.0
N 		  = 40
n 	 	  = N*N
grid_dims = [N,N]
stencil   = diffusion_stencil_2d(epsilon,theta)
A 		  = stencil_grid(stencil, grid_dims, format='csr')
[d,d,A]   = symmetric_rescaling(A)
A 		  = csr_matrix(A)
B 		  = np.kron(np.ones((A.shape[0]/1, 1), dtype=A.dtype),np.eye(1))
tol 	  = 1e-12	# Drop tolerance for singular values 

if SA:
	if SOC=='evol': 
		C = evolution_strength_of_connection(A, B, epsilon=SOC_drop, k=2)
	else:
		SOC = 'symm'
		C = symmetric_strength_of_connection(A, theta=SOC_drop)

	AggOp, Cpts = standard_aggregation(C)
else: 
	splitting = CR(A, method='habituated')
	Cpts = [i for i in range(0,n) if splitting[i]==1]

Fpts = [i for i in range(0,n) if i not in Cpts]
num_Fpts = len(Fpts)
num_Cpts = len(Cpts)
num_bad_guys = 1
cf_ratio = float(num_Cpts) / num_Fpts

# Permutation matrix to sort rows
permute = identity(n,format='csr')
permute.indices = np.concatenate((Fpts,Cpts))
permute = permute.T;

# Smooth bad guys
b = np.ones((A.shape[0], 1), dtype=A.dtype)
smooth_fn = ('gauss_seidel', {'sweep': 'symmetric', 'iterations': 4})
B = relaxation_as_linear_operator((smooth_fn), A, b) * B

# Compute eigenvalues / eigenvectors 
[eval_A,evec_A] = linalg.eigsh(A, k=n/4, which='SM')
[norm_A, dum] = linalg.eigsh(A, k=1, which='LM')
norm_A = norm_A[0]
del dum

# B = np.array(evec_A[:,0:2])		# Let bad guy be smoothest eigenvector(s)
# B = np.array(evec_A[:,0:1])		# Let bad guy be smoothest eigenvector(s)
# B = np.array((evec_A[:,0],evec_A[:,0])).T
# B = B / np.mean(B)

# Form operators
Afc    = -A[Fpts,:][:,Cpts]
Acf    = Afc.transpose()
AfcAcf = Afc*Acf
K 	   = identity(num_Fpts,format='csr')
rhsTop = K - A[Fpts,:][:,Fpts]      # rhsTop = G^j
G = deepcopy(rhsTop)
for i in range(1,pow_G):
    K = K + rhsTop             # K = I + G + ... + G^(j-1)
    rhsTop = rhsTop * G        # G = G^j

# Sparsity pattern
sparse_ind = 0
if sparse_ind == 0:
	sparse_string = r'$C^2_{Fpts}$'
	test = C*C
	Y = csr_matrix(test[Fpts,:][:,Fpts], dtype=np.float64)
elif sparse_ind == 1:
	sparse_string = r'$G^jA_{fc}A_{cf}$'
	Y = csr_matrix(rhsTop*AfcAcf)

sparse_nnz = float(len(Y.data)) / np.prod(Y.shape)

# Use constrained minimization to form NII P for given sparsity pattern
weighting  = 1000.0
lqBottomOp = weighting*(B[Cpts,:].T*Acf)
rhsBottom  = weighting*B[Fpts,:].T - lqBottomOp*K

fn = amg_core.new_ideal_interpolation
fn( Y.indptr,
    Y.indices,
    Y.data,
    AfcAcf.indptr,
    AfcAcf.indices,
    AfcAcf.data,
    lqBottomOp.ravel(order='F'),
    rhsTop.indptr,
    rhsTop.indices,
    rhsTop.data,
    rhsBottom.ravel(order='F'),
    num_Fpts,
    num_Cpts,
    num_bad_guys )

P_nii = vstack( (csr_matrix((K+Y)*Afc), identity(num_Cpts)), format='csr' )
P_nii = permute*P_nii
P_nii_dagger = np.linalg.pinv(P_nii.todense(), rcond=tol)
err_nii = identity(n) - P_nii*P_nii_dagger
P_nii_nnz = float(len(P_nii.data)) / np.prod(P_nii.shape)

# Form standard SA prolongation operator
T, B_coarse = fit_candidates(AggOp, B)
P_sa = jacobi_prolongation_smoother(A, T, C, B_coarse)
# P_sa = richardson_prolongation_smoother(A, T, **kwargs)
P_sa_dagger = np.linalg.pinv(P_sa.todense(), rcond=tol)
err_sa = identity(n) - P_sa*P_sa_dagger
P_sa_nnz = float(len(P_sa.data)) / np.prod(P_sa.shape)

# Form RN prolongation operator
T, B_coarse = fit_candidates(AggOp, B[:,0:1])
Cpt_params = (True, get_Cpt_params(A, Cpts, AggOp, T))
T = scale_T(T, Cpt_params[1]['P_I'], Cpt_params[1]['I_F'])
B_coarse = Cpt_params[1]['P_I'].T*B
P_rn = energy_prolongation_smoother(A, T, C, B_coarse, B,
                 Cpt_params=Cpt_params, maxiter=8, degree=2, weighting='local')
P_rn_dagger = np.linalg.pinv(P_rn.todense(), rcond=tol)
err_rn = identity(n) - P_rn*P_rn_dagger
P_rn_nnz = float(len(P_rn.data)) / np.prod(P_rn.shape)

# num_plot = num_Fpts
num_plot = len(eval_A[eval_A<0.25])

# Empty arrays
interp_error_sa = np.zeros(num_plot)
interp_error_rn = np.zeros(num_plot)
interp_error_nii = np.zeros(num_plot)

for i in range(0,num_plot):
	vec_A = evec_A[:,i]
	vec_Ac = evec_A[Cpts,i]
	# Theoretical error in interpolation using P from smoothed aggregation 
	interp_error_sa[i] = np.linalg.norm( np.dot(err_sa, vec_A ) )
	# Theoretical error in interpolation using P from Jacob's root node
	interp_error_rn[i] = np.linalg.norm( np.dot(err_rn, vec_A ) )
	# Theoretical error in interpolation using constrained new ideal P
	interp_error_nii[i] = np.linalg.norm( np.dot(err_nii, vec_A ) )


# Figure and subplot variables
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=False)
# fig.suptitle('N = %i, theta = 3pi / 16, e = %1.1f, SOC = %s(%1.1f), C/F ratio = %1.2f' %(N,epsilon,SOC,SOC_drop,cf_ratio), fontsize=14, fontweight='bold')
fig.suptitle('N = %i, theta = 3pi / 16, e = %1.1f, C/F ratio = %1.2f' %(N,epsilon,cf_ratio), fontsize=14, fontweight='bold')
# fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=False)
# fig.suptitle('N = %i, theta = 3pi / 16, e = %1.1f, SOC = %s('r'$\theta=$'r'%1.1f), C/F ratio = %1.2f' %(N,epsilon,SOC,SOC_drop,cf_ratio), fontsize=14, fontweight='bold')
indices = [i for i in range(0,num_plot)]

# Error in interpolation for SA P
ax1.plot(indices, interp_error_sa, color='red', label='Interp. error - 'r'$min ||v - Pw_c||$')
ssap1 = ax1.plot(indices, eval_A[0:num_plot]/norm_A, color='blue', label='SSAP - 'r'$\lambda_i(A) / ||A||$')
wap1 = ax1.plot(indices, np.sqrt(eval_A[0:num_plot])/norm_A, color='darkgreen', label='WAP - 'r'$\sqrt{\lambda_i(A)} / ||A||$')
ax1.legend(loc='upper left')
ax1.set_title('SA interpolation error\nNonzero ratio of P = %1.2f'%(P_sa_nnz))
ax1.set_ylim((0,1))
ax1.grid(color='k')
plt.setp(ssap1, linewidth=3, linestyle='--')
plt.setp(wap1, linewidth=3, linestyle='--')

# Error in interpolation for RN P
ax2.plot(indices, interp_error_rn, color='red', label='Interp. error - 'r'$min ||v - Pw_c||$')
ssap2 = ax2.plot(indices, eval_A[0:num_plot]/norm_A, color='blue', label='SSAP - 'r'$\lambda_i(A) / ||A||$')
wap2 = ax2.plot(indices, np.sqrt(eval_A[0:num_plot])/norm_A, color='darkgreen', label='WAP - 'r'$\sqrt{\lambda_i(A)} / ||A||$')
ax2.legend(loc='upper left')
ax2.set_title('RN interpolation error\nNonzero ratio of P = %1.2f'%(P_rn_nnz))
ax2.set_ylim((0,1))
ax2.grid(color='k')
plt.setp(ssap2, linewidth=3, linestyle='--')
plt.setp(wap2, linewidth=3, linestyle='--')

# Error in interpolation for NII P
ax3.plot(indices, interp_error_nii, color='red', label='Interp. error - 'r'$min ||v - Pw_c||$')
ssap3 = ax3.plot(indices, eval_A[0:num_plot]/norm_A, color='blue', label='SSAP - 'r'$\lambda_i(A) / ||A||$')
wap3 = ax3.plot(indices, np.sqrt(eval_A[0:num_plot])/norm_A, color='darkgreen', label='WAP - 'r'$\sqrt{\lambda_i(A)} / ||A||$')
ax3.legend(loc='upper left')
# ax3.set_title('New ideal interpolation error\n Sparsity pattern = %s, sparsity nonzero ratio %1.2f, Nonzero ratio P = %1.2f'%(sparse_string,sparse_nnz,P_nii_nnz))
ax3.set_title('New RN interpolation error, G^%d\nNonzero ratio of P = %1.2f'%(pow_G,P_nii_nnz))
ax3.set_ylim((0,1))
ax3.grid(color='k')
plt.setp(ssap3, linewidth=3, linestyle='--')
plt.setp(wap3, linewidth=3, linestyle='--')

fig.set_size_inches(18.5, 10.5, forward=True)
# plt.savefig('test.pdf')

plt.show()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=False)
# fig.suptitle('N = %i, theta = 3pi / 16, e = %1.1f, SOC = %s(%1.1f), C/F ratio = %1.2f' %(N,epsilon,SOC,SOC_drop,cf_ratio), fontsize=14, fontweight='bold')
fig.suptitle('N = %i, theta = 3pi / 16, e = %1.1f, C/F ratio = %1.2f' %(N,epsilon,cf_ratio), fontsize=14, fontweight='bold')
# fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=False)
# fig.suptitle('N = %i, theta = 3pi / 16, e = %1.1f, SOC = %s('r'$\theta=$'r'%1.1f), C/F ratio = %1.2f' %(N,epsilon,SOC,SOC_drop,cf_ratio), fontsize=14, fontweight='bold')
indices = [i for i in range(0,num_plot)]

# Error in interpolation for SA P
ax1.semilogy(indices, interp_error_sa, color='red', label='Interp. error - 'r'$min ||v - Pw_c||$')
ssap1 = ax1.semilogy(indices, eval_A[0:num_plot]/norm_A, color='blue', label='SSAP - 'r'$\lambda_i(A) / ||A||$')
wap1 = ax1.semilogy(indices, np.sqrt(eval_A[0:num_plot])/norm_A, color='darkgreen', label='WAP - 'r'$\sqrt{\lambda_i(A)} / ||A||$')
ax1.legend(loc='lower right')
ax1.set_title('SA interpolation error\nNonzero ratio of P = %1.2f'%(P_sa_nnz))
ax1.set_ylim((0,1))
ax1.grid(color='k')
plt.setp(ssap1, linewidth=3, linestyle='--')
plt.setp(wap1, linewidth=3, linestyle='--')

# Error in interpolation for RN P
ax2.semilogy(indices, interp_error_rn, color='red', label='Interp. error - 'r'$min ||v - Pw_c||$')
ssap2 = ax2.semilogy(indices, eval_A[0:num_plot]/norm_A, color='blue', label='SSAP - 'r'$\lambda_i(A) / ||A||$')
wap2 = ax2.semilogy(indices, np.sqrt(eval_A[0:num_plot])/norm_A, color='darkgreen', label='WAP - 'r'$\sqrt{\lambda_i(A)} / ||A||$')
ax2.legend(loc='lower right')
ax2.set_title('RN interpolation error\nNonzero ratio of P = %1.2f'%(P_rn_nnz))
# ax2.set_ylim((0,2.25))
ax2.set_ylim((0,1))
ax2.grid(color='k')
plt.setp(ssap2, linewidth=3, linestyle='--')
plt.setp(wap2, linewidth=3, linestyle='--')

# Error in interpolation for NII P
ax3.semilogy(indices, interp_error_nii, color='red', label='Interp. error - 'r'$min ||v - Pw_c||$')
ssap3 = ax3.semilogy(indices, eval_A[0:num_plot]/norm_A, color='blue', label='SSAP - 'r'$\lambda_i(A) / ||A||$')
wap3 = ax3.semilogy(indices, np.sqrt(eval_A[0:num_plot])/norm_A, color='darkgreen', label='WAP - 'r'$\sqrt{\lambda_i(A)} / ||A||$')
ax3.legend(loc='lower right')
# ax3.set_title('New ideal interpolation error\n Sparsity pattern = %s, sparsity nonzero ratio %1.2f, Nonzero ratio P = %1.2f'%(sparse_string,sparse_nnz,P_nii_nnz))
ax3.set_title('New RN interpolation error, G^%d\nNonzero ratio of P = %1.2f'%(pow_G,P_nii_nnz))
ax3.set_ylim((0,1))
ax3.grid(color='k')
plt.setp(ssap3, linewidth=3, linestyle='--')
plt.setp(wap3, linewidth=3, linestyle='--')

fig.set_size_inches(18.5, 10.5, forward=True)
# plt.savefig('test.pdf')

plt.show()



pdb.set_trace()



