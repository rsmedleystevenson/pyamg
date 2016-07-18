from pyamg.utils.linalg import norm
from scipy.sparse import csr_matrix


def trace_min_cg(A, B, Sp, Cpts, Fpts, maxit, tol, tau, debug=False):
	""""
	CG to minimize trace functional. Fill this in. 

	"""
	# Function to return trace
	def get_trace(A):
		return np.sum(A.diagonal())

	# Function to efficiently stack two csr matrices. Note,
	# does not check for CSR or matching dimensions. 
	def stack(W, I):
		temp = W.indptr[-1]
		P = csr_matrix( (np.concatenate((W.data, I.data)),
						 np.concatenate((W.indices, I.indices)),
						 np.concatenate((W.indptr, temp+I.indptr[1:])) ),
						shape=[(W.shape[0]+I.shape[0]), W.shape[1]] )
		return P

	# Get sizes and permutation matrix from [F, C] block
	# ordering to natural matrix ordering.
    n = A.shape[0]
	nc = length(Cpts) 
    nf = length(Fpts)
    nb = B.shape[1]
  	permute = eye(n,format='csr')
    permute.indices = np.concatenate((Fpts,Cpts))
    permute = permute.T

	# Zero initial guess for W, empty matrices with appropriate
	# sparsity pattern and sorted indices. 
    Sp.sort_indices()
    correction1 = Sp 		# Reference to Sp (can overwrite, don't need Sp.data)
    correction2 = csr_matrix((np.ones((len(Sp.data),)), Sp.indices, Sp.indptr), shape=Sp.shape)
    W = csr_matrix((np.zeros((len(Sp.data),)), Sp.indices, Sp.indptr), shape=Sp.shape)
    R = csr_matrix((np.zeros((len(Sp.data),)), Sp.indices, Sp.indptr), shape=Sp.shape)
    C0 = csr_matrix((np.zeros((len(Sp.data),)), Sp.indices, Sp.indptr), shape=Sp.shape)

    # Form RHS, C0 = Bf*Bc^T - tau*Afc
    Bc = B[Cpts,:]
	incomplete_mat_mult_dense2sparse(B[Fpts,:].ravel(order='C'),
                                     Bc.ravel(order='C'),
                                     C0.indptr,
                                     C0.indices,
                                     C0.data,
                                     nf,
                                     nb,
                                     nc)

	# C0 -= tau*Afc restricted to sparsity pattern
	# 	TODO : More efficient to hard code this in C
	temp = A[Fpts,:][:,Cpts].multiply(correction2)
	temp.eliminate_zeros()
    C0 -= tau * temp
    C0.sort_indices()

	# Initial residual
    Aff = A[Fpts,:][:,Fpts]
    D = csr_matrix((C0.data, C0.indices, C0.indptr), shape=C0.shape)
    rold = norm(C0.data, sqrt=False)
    it = 0

    # Construct matrices only used for debugging purposes
    if debug:
	    I = eye(n=n, format='csr');
	    Rc = csr_matrix( (np.ones((nc,)),
	    				  np.array(Cpts),
	    				  np.arange(0,nc)), shape=[nc,n])
	    funcval = []

	# Do CG iterations until residual tolerance or maximum
	# number of iterations reached
    while it < maxiter and rold > tol:

    	# Compute and add correction, where 
    	# 		App = tau * Tr(D^TAD) + Tr(DCD^T)
    	# Note, 
	    #		Tr(DCD^T) = Tr(DBcBc^TD^T) = Tr(Bc^TD^TDBc) = ||DBc||_F^2
	    # PyAMG vector 2-norm on a 2d array returns Frobenius
	    temp = D * Bc
	    App = tau*get_trace(D.T*Aff*D) + norm(temp, sqrt=False)
	    alpha = rold / App
	    W.data += alpha * D.data

	    # Form matrix products
	    # correction1 = Aff*W, restricted to sparsity pattern
        pyamg.amg_core.incomplete_mat_mult_bsr(Aff.indptr,
        									   Aff.indices,
                                               Aff.data,
                                               W.indptr,
                                               W.indices,
                                               W.data,
                                               correction1.indptr,
                                               correction1.indices,
                                               correction1.data,
                                               Aff.shape[0],
                                               correction1.shape[1],
                                               1, 1, 1)

	    # correction2 = WC = WBcBc^T, restricted to sparsity pattern
	    temp = W * Bc
		incomplete_mat_mult_dense2sparse(temp.ravel(order='C'),
	                                     Bc.ravel(order='C'),
	                                     correction2.indptr,
	                                     correction2.indices,
	                                     correction2.data,
	                                     nf,
	                                     nb,
	                                     nc)

	    # Get residual, R = C0 - tau*Aff*W - W*C
	    R.data = C0.data - tau*correction1.data - correction2.data
	    rnew = norm(R.data, sqrt=False)

		# Get new search direction, increase iteration count    
	    D.data *= (rnew/rold)
	    D.data += R.data
	    rold = rnew
	    it += 1;

	    # Evaluate functional (just for debugging)
	    if debug:
	    	# Form P
			P = stack(W, eye(nc, format='csr'))
			P = csr_matrix(permute * P)

			# Compute functionals
		    F1P = get_trace(B.T * (I-Rc.T*P.T) * (I-P*Rc) * B)
		    F2P = get_trace(P.T*A*P)
		    FP = F1P/tau + F2P

		    print('Iter ',it,' - |Res| = %3.4f'%sqrt(rold),', F1(P) = %3.4f',F1P, \
			', F2(P) = %3.4f',F2P,', F(P) = %3.4f',FP)
			funcval.append(FP)

	# Form P = [W; I], reorder and return
	P = stack(W, eye(nc, format='csr'))
	P = csr_matrix(permute * P)
	return P


def trace_min(A, B, SOC, Cpts, Fpts=None, T=None,
			  deg=1, maxit=100, tol=1e-8, get_tau='size',
			  diagonal_dominance=False, debug=False):
	""" 
	Trace-minimization of P. Fill this in.

	"""
	# Currently only implemented for CSR matrices
    if not isspmatrix_csr(A):
        A = csr_matrix(A)
        warn("Implicit conversion of A to CSR", SparseEfficiencyWarning)

	# Make sure C-points are an array, get F-points
	Cpts = np.array(Cpts)
	if Fpts is None:
		temp = np.zeros((A.shape[0],))
		temp[Cpts] = 1
		Fpts = np.where(temp == 0)[0]
		del temp

	nf = len(Fpts)
	nc = len(Cpts)

	# Form tau
	if get_tau == 'size':
		tau = B.shape[1] / nc
	else:
		raise ValueError("Unrecognized method to compute weight tau.")

	# Form initial sparsity pattern as identity along C-points
	# if tentative operator is not passed in. 
	if T == None:
		rowptr = np.zeros((n+1,),dtype='intc')
		rowptr[Cpts+1] = 1
		np.cumsum(rowptr, out=rowptr)
		S = csr_matrix((np.ones((num_Cnodes,), dtype='intc'),
		             	np.arange(0,num_Cnodes),
		             	rowptr),
		               dtype='float64')
	else:
		S = csr_matrix(T, dtype='float64')

	# Expand sparsity pattern by multiplying with SOC matrix
	for i in range(0,deg):
		S = SOC * S

	# Check if any rows have empty sparsity pattern --> either
	# boundary nodes or have no strong connections (diagonally
	# dominant). Include diagonally dominant nodes in sparsity
	# pattern if diagonal_dominance=False.
	# if not diagonal_dominance:
	if not diagonal_dominance or False:
		missing = np.where( np.ediff1d(P.indptr) == 0)[0]
		for row in missing:
			# Not connected to any other points, x[row]
			# is fixed.
			if (A.indptr[row+1] - A.indptr[row]) == 1:
				pass
			
			# Diagonally dominant
			# TODO : address this 


	# Need to chop off S to be only fine grid rows, size
	# nf x nc, before passing to trace_min_cg(). 
	S = S[Fpts,:]

	# Pick weghting parameter
    tau = B.shape[1] / len(Cpts);

	# Form P
	P = trace_min_cg(A=A, B=B, Sp=S, Cpts=Cpts, Fpts=Fpts,
					 maxit=maxit, tol=tol, tau=tau, debug=debug)

	# How do we form coarse-grid bad guys? By injection? 
	Bc = B[Cpts,:]

	return P, Bc



