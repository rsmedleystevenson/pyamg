from pyamg.utils.linalg import norm



def trace_min_cg(A, B, Sp, Cpts, Fpts, maxit, tol, debug=False):
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
  	permute = eye(n,format='csr')
    permute.indices = np.concatenate((Fpts,Cpts))
    permute = permute.T;

    # These appear fixed. Should restrict C0 sparsity pattern now. 
    # Also, is it worth saving Xct / Xf if they are not used again? 
    #
    #	TODO: restrict C0 to sparsity
    #		  New function for incomplete mat mul from dense arrays? 
    #		  C is dense... this is not good. 
    #
    tau = X.shape[1] / nc;
    Xf = X[Fpts,:];
    Xct = X[Cpts,:].T;
    C0 = Xf*Xct - tau*A[Fpts,:][:,Cpts];
    C = Xct.T * Xct; 
    
	# Zero initial guess for W, empty matrices with sparsity pattern
	#
	#	TODO : Can we do better? All ones? Something? Note, if
	#		   this changes, need to fix initial residual and D
	#
    W = csr_matrix((np.zeros((len(Sp.data),)), Sp.indices, Sp.indptr), shape=Sp.shape)
    temp1 = csr_matrix((np.zeros((len(Sp.data),)), Sp.indices, Sp.indptr), shape=Sp.shape)
    temp2 = Sp

	# Initial residual
    Aff = A[Fpts,:][:,Fpts]
    D = C0;
    rold = norm(C0.data, sqrt=False);
    it = 0;

    # Construct matrices only used for debugging purposes
    if debug:
	    I = eye(n=n, format='csr');
	    R = csr_matrix(permute * eye(m=nc, n=n))
	    funcval = []

	# Do CG iterations until residual tolerance or maximum
	# number of iterations reached
    while it < maxiter and rold > tol:

    	# Compute and add correction
    	#
    	# 	TODO : This can maybe be optimized more using fancy trace stuff?
	    #
	    App = tau*get_trace(D.T*Aff*D) + get_trace(D*C*D.T);
	    alpha = rold / App;
	    W = W + alpha*D;

	    # Form matrix products
	    # 	temp1 = W*C, restricted to sparsity pattern
        pyamg.amg_core.incomplete_mat_mult_bsr(W.indptr,
        									   W.indices,
                                               W.data,
                                               C.indptr,
                                               C.indices,
                                               C.data,
                                               temp1.indptr,
                                               temp1.indices,
                                               temp1.data,
                                               W.shape[0],
                                               temp1.shape[1],
                                               1, 1, 1)

	    # 	temp2 = Aff*W, restricted to sparsity pattern
        pyamg.amg_core.incomplete_mat_mult_bsr(Aff.indptr,
        									   Aff.indices,
                                               Aff.data,
                                               W.indptr,
                                               W.indices,
                                               W.data,
                                               temp2.indptr,
                                               temp2.indices,
                                               temp2.data,
                                               Aff.shape[0],
                                               temp2.shape[1],
                                               1, 1, 1)

	    # Get residual. Note, these matrix additions can also be
	    # 	optimized by just adding the data, if we make sure the
	    # 	column indices are sorted. 
	    R = C0 - temp1 - tau*temp2;
	    rnew = norm(R.data, sqrt=False);

		# Get new search direction, increase iteration count    
	    D = R + (rnew/rold)*D;
	    rold = rnew;
	    it += 1;

	    # Evaluate functional (just for debugging)
	    if debug:
	    	# Form P
			P = stack(W, eye(nc, format='csr'))
			P = csr_matrix(permute * P)

			# Compute functionals
		    F1P = get_trace(X.T * (I-R.T*P.T) * (I-P*R) * X);
		    F2P = get_trace(P.T*A*P);
		    FP = F1P/tau + F2P;

		    print 'Iter ',it,' - |Res| = %3.4f'%sqrt(rold),', F1(P) = %3.4f',F1P, \
			', F2(P) = %3.4f',F2P,', F(P) = %3.4f',FP
			funcval.append(FP);

	# Form P = [W; I], reorder and return
	P = stack(W, eye(nc, format='csr'))
	P = csr_matrix(permute * P)
	return P


def trace_min(A, B, SOC, Cpts, Fpts=None, T=None,
			  deg=1, maxit=100, tol=1e-8, debug=False):
	""" 
	Trace-minimization of P. Fill this in.

	"""
	# Currently only implemented for CSR matrices
    if not (isspmatrix_csr(A):
        A = csr_matrix(A)
        warn("Implicit conversion of A to CSR", SparseEfficiencyWarning)

	# Make sure C-points are an array, get F-points
	Cpts = np.array(Cpts)
	if Fpts is None:
		temp = np.zeros((A.shape[0],))
		temp[Cpts] = 1
		Fpts = np.where(temp == 0)[0]
		del temp

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

	# Check if any rows have empty sparsity pattern. If so,
	missing = np.where( np.ediff1d(P.indptr) == 0)[0]
	for row in missing:



	# Need to chop off S to be only fine grid rows, size
	# nf x nc, before passing to trace_min_cg(). 
	S = S[Fpts,:]

	# Form P
	P = trace_min_cg(A=A, B=B, Sp=S, Cpts=Cpts, Fpts=Fpts,
					 maxit=maxit, tol=tol, debug=debug)


	# How do we form coarse-grid bad guys? By injection? 
	Bc = B[Cpts,:]

	return P, Bc



