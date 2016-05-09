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
epsilon   = 1
theta 	  = 3.0*math.pi/16.0
N 		  = 50
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
dagger2 = np.linalg.pinv(Acf.todense(), rcond=tol)
dagger[dagger<1e-10] = 0.0
dagger2[dagger2<1e-10] = 0.0

permute = identity(n,format='csr')
permute.indices = np.concatenate((Fpts,Cpts))
permute = permute.T;

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

# New Ben's L2-optimal
Pben = vstack( ( dagger2*Acc, identity(num_Cpts)), format='csr' )
Pben = csr_matrix(permute*Pben)
Pben_dagger  = np.linalg.pinv(Pben.todense(), rcond=tol)
err_ben = identity(n) - Pben*Pben_dagger
analytical_nnz_ben = float(len(Pben.data)) / np.prod(Pben.shape)

# Compute eigenvalues / eigenvectors 
[eval_A,evec_A] = linalg.eigsh(A, k=n/4, which='SM')
[norm_A, dum] = linalg.eigsh(A, k=1, which='LM')
norm_A = norm_A[0]

# num_plot = num_Fpts
num_plot = len(eval_A[eval_A<0.25])

# Empty arrays
interp_error_A = np.zeros(num_plot)
interp_error_ben_A = np.zeros(num_plot)

for i in range(0,num_plot):
	vec_A  = evec_A[:,i]
	vec_Af = evec_A[Fpts,i]
	vec_Ac = evec_A[Cpts,i]
	# Theoretical error in interpolation
	interp_error_A[i] = np.linalg.norm( np.dot(err, vec_A ) )
	# Theoretical error in interpolation using constrained minimum P
	interp_error_ben_A[i] = np.linalg.norm( np.dot(err_ben, vec_A ) )


# Figure and subplot variables
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=False)
indices = [i for i in range(0,num_plot)]
fig.suptitle('N = %i, theta = 3pi / 16, e = %1.2f, SOC = %s(%1.1f), C/F ratio = %1.2f, G^%d' %(N,epsilon,SOC,SOC_drop,cf_ratio,pow_G), fontsize=14, fontweight='bold')

# Error in interpolation for analytical P
ax1.plot(indices, interp_error_A, color='red', label='Interp. error - 'r'$||Pv_c - v_f||$')
ssap1 = ax1.plot(indices, eval_A[0:num_plot]/norm_A, color='blue', label='SSAP - 'r'$\lambda_i(A) / ||A||$')
wap1 = ax1.plot(indices, np.sqrt(eval_A[0:num_plot])/norm_A, color='darkgreen', label='WAP - 'r'$\sqrt{\lambda_i(A)} / ||A||$')
ax1.legend(loc='upper left')
ax1.set_title('Interpolation error using analytical P, nonzero ratio = %1.2f' %(analytical_nnz))
ax1.set_ylim((0,1.2))
ax1.grid(color='k')
plt.setp(ssap1, linewidth=3, linestyle='--')
plt.setp(wap1, linewidth=3, linestyle='--')

# Error in interpolation for experimental P
ax2.plot(indices, interp_error_ben_A, color='red', label='Interp. error - 'r'$||Pv_c - v_f||$')
ssap2 = ax2.plot(indices, eval_A[0:num_plot]/norm_A, color='blue', label='SSAP - 'r'$\lambda_i(A) / ||A||$')
wap2 = ax2.plot(indices, np.sqrt(eval_A[0:num_plot])/norm_A, color='darkgreen', label='WAP - 'r'$\sqrt{\lambda_i(A)} / ||A||$')
ax2.legend(loc='upper left')
ax2.set_title('Interpolation error using Bens analytical P, nonzero ratio = %1.2f' %(analytical_nnz_ben))
ax2.set_ylim((0,1.2))
ax2.grid(color='k')
plt.setp(ssap2, linewidth=3, linestyle='--')
plt.setp(wap2, linewidth=3, linestyle='--')

fig.set_size_inches(18.5, 10.5, forward=True)
# plt.savefig('test.pdf')

plt.show()


pdb.set_trace()







