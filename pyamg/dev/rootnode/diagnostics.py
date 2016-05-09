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

from scipy.sparse import csr_matrix, identity, linalg, identity, vstack

from copy import deepcopy

# ------------------------------------------------------------------------- #
# ------------------------------------------------------------------------- #
# This section computes the analytic, unconstrained prolongation operator 	#
# and then computes the analytical error in the construction of the 		#
# interpolation operator, applied to eigenvalues of A. 
# ------------------------------------------------------------------------- #
# ------------------------------------------------------------------------- #

constrained = 1
SOC 	  = 'evol'
SOC_drop  = 4.0
SA 		  = 1 		# Use SA type coarsening or CR
pow_G 	  = 1
SOC_width = 2
epsilon   = 0.01
theta 	  = 3.0*math.pi/16.0
N 		  = 80
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

# Smooth bad guys
b = np.ones((A.shape[0], 1), dtype=A.dtype)
# b = np.random.random((A.shape[0], 1))
smooth_fn = ('gauss_seidel', {'sweep': 'symmetric', 'iterations': 4})
B = relaxation_as_linear_operator((smooth_fn), A, b) * B

# Test of Acc sparsity pattern 
Acc = A[Cpts,:][:,Cpts]
test = Acc - identity(num_Cpts)
test.data[test.data < 1e-12] = 0.0
test.eliminate_zeros()
if len(test.data) > 0:
	print "Acc is not the identity."

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

# Compute pseudoinverse (AfcAcf)^+
dagger = np.linalg.pinv(AfcAcf.todense(), rcond=tol)

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

# Use constrained minimization to form P in practice for given sparsity pattern
weighting  = 1000.0
lqBottomOp = weighting*(B[Cpts,:].T*Acf)
rhsBottom  = weighting*B[Fpts,:].T - lqBottomOp*K

if constrained:
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
else:
	fn = amg_core.unconstrained_new_ideal
	fn( Y.indptr,
	    Y.indices,
	    Y.data,
	    AfcAcf.indptr,
	    AfcAcf.indices,
	    AfcAcf.data,
	    rhsTop.indptr,
	    rhsTop.indices,
	    rhsTop.data,
	    num_Fpts,
	    num_Cpts)


# Form P found in constrained L2 minimization and error operator, which 
# projects onto ker(P^*)
P_min = vstack( (csr_matrix((K+Y)*Afc), identity(num_Cpts)), format='csr' )
permute = identity(n,format='csr')
permute.indices = np.concatenate((Fpts,Cpts))
permute = permute.T;
P_min = permute*P_min
P_min_dagger = np.linalg.pinv(P_min.todense(), rcond=tol)
err_min = identity(n) - P_min*P_min_dagger
P_nnz = float(len(P_min.data)) / np.prod(P_min.shape)

# Form theoretical P, minimized P, and theoretical error operator in forming Y,
# depending on power of G being used.
if pow_G == 0:
	errOp = identity(num_Fpts) - (dagger*AfcAcf) 	# This is error in forming Y, not error in interpolation
	P = vstack( (dagger*Afc, identity(num_Cpts)), format='csr')
else:
	errOp = rhsTop - rhsTop*(dagger*AfcAcf) 	# This is error in forming Y, not error in interpolation
	P = vstack( ( (K + rhsTop*dagger)*Afc, identity(num_Cpts)), format='csr' )

P = csr_matrix(permute*P)
P_dagger = np.linalg.pinv(P.todense(), rcond=tol)
err = identity(n) - P*P_dagger
analytical_nnz = float(len(P.data)) / np.prod(P.shape)

# Compute eigenvalues / eigenvectors 
[eval_A,evec_A] = linalg.eigsh(A, k=n/4, which='SM')
[norm_A, dum] = linalg.eigsh(A, k=1, which='LM')
norm_A = norm_A[0]

# num_plot = num_Fpts
num_plot = len(eval_A[eval_A<0.25])

# Empty arrays
AfcAcf_norm = np.zeros(num_plot)
interp_error_A = np.zeros(num_plot)
interp_error2_A = np.zeros(num_plot)
bound_error_A = np.zeros(num_plot)
bound_error_AfcAcf = np.zeros(num_plot)

for i in range(0,num_plot):
	vec_A  = evec_A[:,i]
	vec_Af = evec_A[Fpts,i]
	vec_Ac = evec_A[Cpts,i]
	# Upper bound on minimization error for v_f (see proof 2 in notes)
	bound_error_A[i] = np.linalg.norm( np.dot(errOp, vec_Af ) )
	# Theoretical error in interpolation
	interp_error_A[i] = np.linalg.norm( np.dot(err, vec_A ) )
	# Theoretical error in interpolation using constrained minimum P
	interp_error2_A[i] = np.linalg.norm( np.dot(err_min, vec_A ) )


# Figure and subplot variables
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=False)
indices = [i for i in range(0,num_plot)]
# fig.suptitle('N = %i, theta = 3pi / 16, e = %1.2f, SOC = %s(%1.1f), C/F ratio = %1.2f, G^%d' %(N,epsilon,SOC,SOC_drop,cf_ratio,pow_G), fontsize=14, fontweight='bold')
fig.suptitle('N = %i, theta = 3pi / 16, e = %1.2f, C/F ratio = %1.2f, G^%d' %(N,epsilon,cf_ratio,pow_G), fontsize=14, fontweight='bold')

# Error in minimizing to form P
ssap1 = ax1.plot(indices, eval_A[0:num_plot]/norm_A, color='blue', label='SSAP - 'r'$\lambda_i(A) / ||A||$')
wap1 = ax1.plot(indices, np.sqrt(eval_A[0:num_plot])/norm_A, color='darkgreen', label='WAP - 'r'$\sqrt{\lambda_i(A)} / ||A||$')
ax1.plot(indices, bound_error_A, color='red', label='Min. error, v_if')
# ax1.plot(indices, bound_error_AfcAcf, color='black', label='Min. error, eigenvector AfcAcf')
ax1.legend(loc='upper left')
ax1.set_title('Analytical error in minimization')
ax1.set_ylim((0,1))
ax1.grid(color='k')
plt.setp(ssap1, linewidth=3, linestyle='--')
plt.setp(wap1, linewidth=3, linestyle='--')

# Error in interpolation for analytical P
ax2.plot(indices, interp_error_A, color='red', label='Interp. error - 'r'$||Pv_c - v_f||$')
ssap2 = ax2.plot(indices, eval_A[0:num_plot]/norm_A, color='blue', label='SSAP - 'r'$\lambda_i(A) / ||A||$')
wap2 = ax2.plot(indices, np.sqrt(eval_A[0:num_plot])/norm_A, color='darkgreen', label='WAP - 'r'$\sqrt{\lambda_i(A)} / ||A||$')
ax2.legend(loc='upper left')
ax2.set_title('Interpolation error using analytical P\nNonzero ratio of P = %1.2f' %(analytical_nnz))
ax2.set_ylim((0,1))
ax2.grid(color='k')
plt.setp(ssap2, linewidth=3, linestyle='--')
plt.setp(wap2, linewidth=3, linestyle='--')

# Error in interpolation for experimental P
ax3.plot(indices, interp_error2_A, color='red', label='Interp. error - 'r'$||Pv_c - v_f||$')
ssap3 = ax3.plot(indices, eval_A[0:num_plot]/norm_A, color='blue', label='SSAP - 'r'$\lambda_i(A) / ||A||$')
wap3 = ax3.plot(indices, np.sqrt(eval_A[0:num_plot])/norm_A, color='darkgreen', label='WAP - 'r'$\sqrt{\lambda_i(A)} / ||A||$')
ax3.legend(loc='upper left')
if constrained:
	# ax3.set_title('Interpolation error using contrained minimized P\n Sparsity pattern = %s, sparsity nonzero ratio %1.2f, nonzero ratio P = %1.2f'%(sparse_string,sparse_nnz,P_nnz))
	ax3.set_title('Interpolation error using contrained minimized P\n Nonzero ratio of P = %1.2f'%(P_nnz))
else:
	ax3.set_title('Interpolation error using uncontrained minimized P\n Sparsity pattern = %s, sparsity nonzero ratio %1.2f, nonzero ratio P = %1.2f'%(sparse_string,sparse_nnz,P_nnz))

ax3.set_ylim((0,1))
ax3.grid(color='k')
plt.setp(ssap3, linewidth=3, linestyle='--')
plt.setp(wap3, linewidth=3, linestyle='--')

fig.set_size_inches(18.5, 10.5, forward=True)
# plt.savefig('test.pdf')

plt.show()


# Figure and subplot variables
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=False)
indices = [i for i in range(0,num_plot)]
# fig.suptitle('N = %i, theta = 3pi / 16, e = %1.2f, SOC = %s(%1.1f), C/F ratio = %1.2f, G^%d' %(N,epsilon,SOC,SOC_drop,cf_ratio,pow_G), fontsize=14, fontweight='bold')
fig.suptitle('N = %i, theta = 3pi / 16, e = %1.2f, C/F ratio = %1.2f, G^%d' %(N,epsilon,cf_ratio,pow_G), fontsize=14, fontweight='bold')

# Error in minimizing to form P
ssap1 = ax1.semilogy(indices, eval_A[0:num_plot]/norm_A, color='blue', label='SSAP - 'r'$\lambda_i(A) / ||A||$')
wap1 = ax1.semilogy(indices, np.sqrt(eval_A[0:num_plot])/norm_A, color='darkgreen', label='WAP - 'r'$\sqrt{\lambda_i(A)} / ||A||$')
ax1.semilogy(indices, bound_error_A, color='red', label='Min. error, v_if')
# ax1.semilogy(indices, bound_error_AfcAcf, color='black', label='Min. error, eigenvector AfcAcf')
ax1.legend(loc='lower right')
ax1.set_title('Analytical error in minimization')
ax1.set_ylim((0,1))
ax1.grid(color='k')
plt.setp(ssap1, linewidth=3, linestyle='--')
plt.setp(wap1, linewidth=3, linestyle='--')

# Error in interpolation for analytical P
ax2.semilogy(indices, interp_error_A, color='red', label='Interp. error - 'r'$||Pv_c - v_f||$')
ssap2 = ax2.semilogy(indices, eval_A[0:num_plot]/norm_A, color='blue', label='SSAP - 'r'$\lambda_i(A) / ||A||$')
wap2 = ax2.semilogy(indices, np.sqrt(eval_A[0:num_plot])/norm_A, color='darkgreen', label='WAP - 'r'$\sqrt{\lambda_i(A)} / ||A||$')
ax2.legend(loc='lower right')
ax2.set_title('Interpolation error using analytical P\nNonzero ratio of P = %1.2f' %(analytical_nnz))
ax2.set_ylim((0,1))
ax2.grid(color='k')
plt.setp(ssap2, linewidth=3, linestyle='--')
plt.setp(wap2, linewidth=3, linestyle='--')

# Error in interpolation for experimental P
ax3.semilogy(indices, interp_error2_A, color='red', label='Interp. error - 'r'$||Pv_c - v_f||$')
ssap3 = ax3.semilogy(indices, eval_A[0:num_plot]/norm_A, color='blue', label='SSAP - 'r'$\lambda_i(A) / ||A||$')
wap3 = ax3.semilogy(indices, np.sqrt(eval_A[0:num_plot])/norm_A, color='darkgreen', label='WAP - 'r'$\sqrt{\lambda_i(A)} / ||A||$')
ax3.legend(loc='lower right')
if constrained:
	# ax3.set_title('Interpolation error using contrained minimized P\n Sparsity pattern = %s, sparsity nonzero ratio %1.2f, nonzero ratio P = %1.2f'%(sparse_string,sparse_nnz,P_nnz))
	ax3.set_title('Interpolation error using contrained minimized P\n Nonzero ratio of P = %1.2f'%(P_nnz))
else:
	ax3.set_title('Interpolation error using uncontrained minimized P\n  Sparsity pattern = %s, sparsity nonzero ratio %1.2f, nonzero ratio P = %1.2f'%(sparse_string,sparse_nnz.P_nnz))

ax3.set_ylim((0,1))
ax3.grid(color='k')
plt.setp(ssap3, linewidth=3, linestyle='--')
plt.setp(wap3, linewidth=3, linestyle='--')

fig.set_size_inches(18.5, 10.5, forward=True)
# plt.savefig('test.pdf')

plt.show()



pdb.set_trace()






# ------------------------------------------------------------------------- #
# ------------------------------------------------------------------------- #
#	- Don't think Eq. 183 leads to Eq. 184, analytically or numerically
# 	- What basis do we use for the null space? I get different results for 
#	  columns vs. rows of V from SVD. Is this even the right null space? 
# 	- Where does the S^(-1) come in? Do we use eigenvectors from S?

# bound_error_Ainv = np.zeros(num_Fpts)

# Get A-inverse optimal
# [temp,singVals,Vt] = scipy.linalg.svd(AfcAcf.todense()) 
# null = [i for i in range(0,num_Fpts) if singVals[i] < 1e-10]
# Vt = Vt[null,:]
# [U,R] = np.linalg.qr(Afc.todense())
# Rstar_inv = np.linalg.inv(R*R.T)
# # Rstar_inv = np.linalg.inv(R.T)
# B21 = np.dot(Vt, G.todense())
# # Size of this guy (for identity)??
# I_min_B22 = np.identity(B21.shape[0]) - B21*Vt.T
# B21 = B21 * U
# bottom = csr_matrix( np.linalg.inv(I_min_B22) * B21 * Rstar_inv)
# WAfc = scipy.sparse.vstack([csr_matrix(Rstar_inv), bottom], format='csr')
# WAfc = scipy.sparse.hstack([WAfc,np.zeros((num_Fpts,num_Fpts-Rstar_inv.shape[0]))])
# WAfc = WAfc * Afc

# pdb.set_trace()
# FOR...
	# Error in A inverse optimal interpolation
	# interp_error_Ainv[i] = np.linalg.norm(vec_A - WAfc*vec_Ac )
	# print i," - A_vec = ",err,", AfcAcf_vec = ", err2, "eval_A = ",eval_A[i]
	# print i," = ",AfcAcf_norm[i]," eval_A = ",eval_A[i]
