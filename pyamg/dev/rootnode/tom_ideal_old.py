
# # -------------------------------- CHECK ON / FIX -------------------------------- #
#     # NEED TO IMPLEMENT THIS FOR SUPER NODES AS WELL...
#     # K1 = B.shape[0] / num_pts  # dof per supernode (e.g. 3 for 3d vectors)
#     # K2 = B.shape[1]           # candidates
def new_ideal_interpolation(A, AggOp, Cnodes, B=None, SOC=None, max_power=1, weighting=10.0, tol=1e-10):

    blocksize = None

    if not isspmatrix_csr(AggOp):
        raise TypeError('expected csr_matrix for argument AggOp')

    if B is not None:
        B = np.asarray(B)
        if B.dtype not in ['float32', 'float64', 'complex64', 'complex128']:
            B = np.asarray(B, dtype='float64')

        if len(B.shape) != 2:
            raise ValueError('expected 2d array for argument B')

        if B.shape[0] % AggOp.shape[0] != 0:
            raise ValueError('dimensions of AggOp %s and B %s are \
                              incompatible' % (AggOp.shape, B.shape))

    # BSR matrix - get blocksize and convert to csr matrix to extract submatrices.
    if isspmatrix_bsr(A):
        blocksize = A.blocksize
        A = A.tocsr()
        num_pts = A.shape[0]
        Cpts = [blocksize[0]*i+j for i in Cnodes for j in range(0,blocksize[0])]
        Fpts = [i for i in range(0,num_pts) if i not in Cnodes]
        num_Fpts = len(Fpts)
        num_Cpts = len(Cpts)
        num_bad_guys = B.shape[1]
        if isspmatrix_bsr(B):
            B = B.tocsr()

        if blocksize[0] != blocksize[1]:
            warnings.warn('A has rectangular block size.\n New ideal interpolation is not set up to accomodate this.')

    # CSR matrix
    else:
        num_pts = AggOp.shape[0]
        Cpts = deepcopy(Cnodes)
        Fpts = [i for i in range(0,num_pts) if i not in Cpts]
        num_Fpts = len(Fpts)
        num_Cpts = len(Cpts)
        num_bad_guys = B.shape[1]

    # Get necessary submatrices / form operators for minimization
    Afc = -1.0*A[Fpts,:][:,Cpts]
    Acf = Afc.transpose()
    K = identity(num_Fpts,format='csr')
    rhsTop = K - A[Fpts,:][:,Fpts]      # rhsTop = G^j
    if max_power > 1:
        G = deepcopy(rhsTop)
        for i in range(1,max_power):
            K = K + rhsTop                  # K = I + G + ... + G^(j-1)
            rhsTop = rhsTop * G             # G = G^j

    lqTopOp = Afc*Acf

    # Pre-allocate sparsity pattern for Y
    # Y = csr_matrix( K + lqTopOp!=0, dtype=np.float64)
    # Y = csr_matrix( K + rhsTop!=0, dtype=np.float64)    # This appears to be better... Slightly better convergence, half the size coarse grid...
    # Y = csr_matrix(scale*Y)
    
    # pdb.set_trace()
    temp = SOC*SOC
    Y = csr_matrix(temp[Fpts,:][:,Fpts], dtype=np.float64)


    # test2 = lqTopOp*lqTopOp
    # test2 = rhsTop*lqTopOp      # G(AfcAfc)^+, seems to be best yet?
    # Y = csr_matrix(rhsTop*lqTopOp, dtype=np.float64)

    # Unconstrained new ideal interpolation if no bad guys are provided
    if B is None:
        warnings.warn("No bad guys provided - using unconstrained minimization.")
        fn = amg_core.unconstrained_new_ideal
        fn( Y.indptr,
            Y.indices,
            Y.data,
            lqTopOp.indptr,
            lqTopOp.indices,
            lqTopOp.data,
            rhsTop.indptr,
            rhsTop.indices,
            rhsTop.data,
            num_Fpts,
            num_Cpts )
    # Constrained new ideal interpolation if bad guys are provided
    else:
        lqBottomOp = weighting*(B[Cpts,:].T*Acf)
        rhsBottom = weighting*B[Fpts,:].T - lqBottomOp*K
        fn = amg_core.new_ideal_interpolation
        fn( Y.indptr,
            Y.indices,
            Y.data,
            lqTopOp.indptr,
            lqTopOp.indices,
            lqTopOp.data,
            lqBottomOp.ravel(order='F'),
            rhsTop.indptr,
            rhsTop.indices,
            rhsTop.data,
            rhsBottom.ravel(order='F'),
            num_Fpts,
            num_Cpts,
            num_bad_guys )

    # Form P
    P = vstack( [(K+Y)*Afc, identity(num_Cpts,format='csr')], format='csr')
    # NOTE, FIX THIS
    #     ---> vstack and hstack are very slow, because they 
    #          convert to COO matrix stack, then revert... 
    #     Instead should modify underlying data arrays and just add identity to (K+Y)Afc. 
    # Maybe I can preallocate P including A_{cc} = I before calling fn? 

    # Arrange rows of P in proper order, convert to bsr matrix
    permute = identity(num_pts,format='csr')
    permute.indices = np.concatenate((Fpts,Cpts))
    permute = permute.T;
    if blocksize:
        P = bsr_matrix(permute*P,blocksize=blocksize)
        A.tobsr(blocksize=blocksize)
    else:
        P = bsr_matrix(permute*P,blocksize=(1,1))

    return P


def py_ideal_interpolation(A, AggOp, Cnodes, B=None, SOC=None, tol=1e-10):

    # Parameters needed
    max_power = 1              # power which we take G to
    weighting = 10.0           # constrained leqast squares weighting parameter

    if not isspmatrix_csr(AggOp):
        raise TypeError('expected csr_matrix for argument AggOp')

    if B is not None:
        B = np.asarray(B)
        if B.dtype not in ['float32', 'float64', 'complex64', 'complex128']:
            B = np.asarray(B, dtype='float64')

        if len(B.shape) != 2:
            raise ValueError('expected 2d array for argument B')

        if B.shape[0] % AggOp.shape[0] != 0:
            raise ValueError('dimensions of AggOp %s and B %s are \
                              incompatible' % (AggOp.shape, B.shape))

    # Get number of F-points, C-points and bad guys.
    num_pts = AggOp.shape[0]
    Fpts = [i for i in range(0,num_pts) if i not in Cnodes]
    num_Fpts = len(Fpts)
    num_Cpts = len(Cnodes)
    num_bad_guys = B.shape[1]

    # Get necessary submatrices / form operators for minimization
    Afc = A[Fpts,:][:,Cnodes]
    Acf = Afc.transpose()
    AfcAcf = Afc*Acf
    K = identity(num_Fpts,format='csr')
    G = identity(num_Fpts) - A[Fpts,:][:,Fpts]
    temp = deepcopy(G)
    for i in range(1,max_power):
        K = K + G      # K = I + G + ... + G^(j-1)
        G = G * temp      # G = G^j

    rhsBottom = weighting*(B[Fpts,:].T + B[Cnodes,:].T*Acf*K)

    # bad guy operator for least squares problem, only need compute it once
    if B is not None:
        bad_guy_op = -weighting*(B[Cnodes,:].T*Acf) # The negative belongs to Acf

    # Let sparsity pattern be the nonzeros in G^j and AfcAcf/
    # data = array([[1]*num_Fpts]).repeat(3,axis=0)
    # offsets = array([0,-1,1])
    # scale = dia_matrix( (data,offsets), shape=(num_Fpts,num_Fpts))
    # sparsity = csr_matrix( K + AfcAcf!=0, dtype=np.float64)
    # sparsity = csr_matrix(scale*sparsity)
    # sparsity = csr_matrix( K + AfcAcf!=0, dtype=np.float64)
    sparsity = csr_matrix( K + G!=0, dtype=np.float64)    # This appears to be better... Slightly better convergence, half the size coarse grid...

    pdb.set_trace()

    # solve minimization for every row
    for row in range(0,num_Fpts):

        # get indices for rows / columns of A needed for the sparsity pattern of this row
        col_ind = sorted([i for i in range(0,num_Fpts) if sparsity[row,i]!=0])
        row_ind = [i for j in col_ind for i in range(0,num_Fpts) if AfcAcf[i,j]!=0]
        row_ind = sorted(list(set(row_ind)))
        lqNumCols = len(col_ind)
        lqNumRows = len(row_ind) + num_bad_guys

        if B == None:
            # compute right hand side vectors 
            rhs = G[row_ind,:][:,row]  # G^je_r restricted to row_ind 

            # form least squares operator
            #   --> Looks like we need to multiply Afc*Acf. Should probably do it once before we
            #       we loop over rows and then use the results again. Is this going to be an
            #       expensive part of the algorithm?
            operator = AfcAcf[row_ind,:][:,col_ind]

        elif len(row_ind):
            # compute right hand side vectors 
            rhs = G[row_ind,:][:,row]   # G^je_r restricted to row_ind 
            bottom = rhsBottom[:,row]           #   
            rhs = vstack([rhs,bottom],'csr')

            # form least squares operator
            operator = vstack( [ AfcAcf[row_ind,:][:,col_ind], bad_guy_op[:,col_ind] ], 'csr')

        # In the case of zero row indices, Python gets upset seeking a submatrix
        else: 

            rhs = csr_matrix(rhsBottom[:,row])
            operator = csr_matrix(bad_guy_op[:,col_ind])

        # --------------------------------------------------------------------------------#
        # --------------------------------------------------------------------------------#

        # PRINT OUT NONZERO COLUMNS AND ROWS TO COMPARE WITH C CODE
        print "Row ", row, " - ", lqNumRows, " x ", lqNumCols;
        # sys.stdout.write('\t');
        # for i in range(0,len(row_ind)):
        #     sys.stdout.write('%i ,' %row_ind[i])

        # sys.stdout.write('\n\t');
        # for i in range(0,lqNumCols):
        #     sys.stdout.write('%i ,' %col_ind[i])

        # print "\n"

        # PRINT LQ OPERATOR AND RHS TO COMPARE WITH C CODE
        # for i in range(0,lqNumRows):
        #     sys.stdout.write('\t');
        #     for j in range(0,lqNumCols):
        #         sys.stdout.write('%.3f ,' %operator[i,j])

        #     sys.stdout.write('\n');

        # sys.stdout.write('\n');

        # for i in range(0,lqNumRows):
        #     sys.stdout.write('\t%.3f ,' %rhs[i].todense())

        # sys.stdout.write('\n\n');

        # --------------------------------------------------------------------------------#
        # --------------------------------------------------------------------------------#

        # take SVD of operator, form minimum norm least squares solution as
        # y = A^+(rhs) = V S^\dagger U^T
        [U,S,V] = linalg.svd(operator.todense(),full_matrices=0)
        V = V.T     # linalg.svd returns V^T, not V. Confusing. 

        # sys.stdout.write('\t');
        # for i in range(0,lqNumCols):
        #     sys.stdout.write('%f ,' %S[i])

        # sys.stdout.write('\n\n');        

        # pdb.set_trace()

    # I THINK THIS MAY ORDER SINGULAR VALUES BACKWARDS... DOES THIS MATTER?
        S[S<tol] = 0
        nonzero_diag = S>0
        S[nonzero_diag] = 1.0/S[nonzero_diag]
        S = linalg.diagsvd(S,min(operator.shape),min(operator.shape))
        y = U.T*rhs
        y = np.dot(S,y)
        y = np.dot(V,y)

        # remove numerical noise from interpolation and store in matrix
        y[abs(y)<tol] = 0
        # test = abs(y) / linalg.norm(y,ord=np.inf)
        # y[test <= 1e-1] = 0.0
        sparsity[row,col_ind] = y.T


        # sys.stdout.write('\t');
        # for i in range(0,lqNumCols):
        #     sys.stdout.write('%f ,' %y[i])

        # sys.stdout.write('\n\n');


    # Stack Y with identity matrix on Cnodes to form prolongation operator
    P = vstack([ -1.0*(K+sparsity)*Afc, identity(num_Cpts)], 'csr')
 

    # Arrange rows of P in proper order
    permute = identity(num_pts,format='csr')
    permute.indices = np.concatenate((Fpts,Cnodes))
    permute = permute.T;
    P = bsr_matrix(permute*P)

    if isspmatrix_csr(A):
        P = bsr_matrix(P, blocksize=[1,1])
    else:
        P = bsr_matrix(P, blocksize=A.blocksize)

    return P

