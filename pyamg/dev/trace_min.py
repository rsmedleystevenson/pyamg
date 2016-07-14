from pyamg.utils.linalg import norm



# What is X?
def trace_min_cg(A, Sp, X, Cpts, Fpts, maxit, tol, debug=False):

	# Function to return trace
	def get_trace(A):
		return np.sum(A.diagonal())

	# Get sizes and permutation matrix from [F, C] block
	# ordering to natural matrix ordering.
    n = A.shape[0]
	nc = length(Cpts) 
    nf = length(Fpts)
  	permute = eye(n,format='csr')
    permute.indices = np.concatenate((Fpts,Cpts))
    permute = permute.T;

    # These appear fixed. Should restrict C0 sparsity pattern now. 
    # Also, is it worth saving Xct / Xf as they are not used again? 
    tau = X.shape[1] / nc;
    Xf = X[Fpts,:];
    Xct = X[Cpts,:].T;
    C0 = Xf*Xct - tau*A[Fpts,:][:,Cpts];
    C = Xct.T * Xct; 
    
	# Zero initial guess for W
	#
	#	TODO : Can we do better? All ones? Something? Note, if
	#		   this changes, need to fix initial residual and D
	#
    W = csr_matrix((np.zeros((len(Sp.data),)), Sp.indices, Sp.indptr), shape=Sp.shape)

	# Initial residual
    Aff = A[Fpts,:][:,Fpts]
    D = C0;
    rold = norm(C0.data, sqrt=False);
    it = 0;

    # Construct matrices only used for debugging purposes
    if debug:
	    I = eye(n=n, format='csr');
	    R = csr_matrix(permute * eye(m=nc, n=n))

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
	    temp1 = incomplete_mat_mat_mul(Aff,W)
	    temp2 = incomplete_mat_mat_mul(W,C)

	    # Get residual
	    #
	    #	TODO : Do temp1, temp2, C0, definitely have
	    #		   same sparsity pattern as Sp?
	    #
	    R = C0 - temp2 - tau*temp1;
	    rnew = norm(R.data, sqrt=False);

		# Get new search direction, increase iteration count    
	    D = R + (rnew/rold)*D;
	    rold = rnew;
	    it += 1;

	    # Evaluate functional (just for debugging)
	    if debug:
	    	# Form P
			P = vstack([W, eye(nc, format='csr')], format='csr')
			P = csr_matrix(permute * P)

			# Compute functionals
		    F1P = get_trace(X.T * (I-R.T*P.T) * (I-P*R) * X);
		    F2P = get_trace(P.T*A*P);
		    FP = F1P/tau + F2P;

		    print 'Iter ',it,' - |Res| = %3.4f'%sqrt(rold),', F1(P) = %3.4f',F1P, \
			', F2(P) = %3.4f',F2P,', F(P) = %3.4f',FP
			funcval.append(FP);

	# Form P = [W; I], reorder and return
	#
	# 	TODO : vstack is slow, should be able to do this faster manually
	# 		   by adding identity rows by hand to sparse structure. 
	#
	P = vstack([W, eye(nc, format='csr')], format='csr')
	P = csr_matrix(permute * P)
	return P


def trace_min(A, C, Cpts, maxit=100, tol=1e-8, debug=False):

	# Check for CSR Matrix (because we select Aff)


	# Get F-points
	temp = np.zeros((A.shape[0],))
	temp[Cpts] = 1
	Fpts = np.where(temp == 0)[0]
	del temp

	# Form sparsity pattern -- Note, need to chop off to
	# be nf x nc before passing to trace_min_cg(). 




	# Form P
	P = trace_min_cg(A=A, Sp=Sp, X=X, Cpts=Cpts, Fpts=Fpts,
					 maxit=maxit, tol=tol, debug=debug)










