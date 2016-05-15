def pairwise_aggregation(A, B, Bh=None, symmetry='hermitian',
                        algorithm='drake_C', matchings=1,
                        weights=None, improve_candidates=None,
                        strength=None, **kwargs):
    """ Pairwise aggregation of nodes. 

    Parameters
    ----------
    A : csr_matrix or bsr_matrix
        matrix for linear system.
    B : array_like
        Right near-nullspace candidates stored in the columns of an NxK array.
    BH : array_like : default None
        Left near-nullspace candidates stored in the columns of an NxK array.
        BH is only used if symmetry is 'nonsymmetric'.
        The default value B=None is equivalent to BH=B.copy()
    algorithm : string : default 'drake'
        Algorithm to perform pairwise matching. Current options are 
        'drake', 'preis', 'notay', referring to the Drake (2003), 
        Preis (1999), and Notay (2010), respectively. 
    matchings : int : default 1
        Number of pairwise matchings to do. k matchings will lead to 
        a coarsening factor of under 2^k.
    weights : function handle : default None
        Optional function handle to compute weights used in the matching,
        e.g. a strength of connection routine. Additional arguments for
        this routine should be provided in **kwargs. 
    improve_candidates : {tuple, string, list} : default None
        The list elements are relaxation descriptors of the form used for
        presmoother and postsmoother.  A value of None implies no action on B.
    strength : {(int, string) or None} : default None
        If a strength of connection matrix should be returned along with
        aggregation. If None, no SOC matrix returned. To return a SOC matrix,
        pass in a tuple (a,b), for int a > matchings telling how many matchings
        to use to construct a SOC matrix, and string b with data type.
        E.g. strength = (4,'bool'). 

    THINGS TO NOTE
    --------------
        - Not implemented for non-symmetric and/or block systems
            + Need to set up pairwise aggregation to be applicable for 
              nonsymmetric matrices (think it actually is...) 
            + Need to define how a matching is done nodally.
            + Also must consider what targets are used to form coarse grid'
              in nodal approach...
        - Once we are done w/ Python implementations of matching, we can remove 
          the deepcopy of A to W --> Don't need it, waste of time/memory.  

    """

    def unpack_arg(v):
        if isinstance(v, tuple):
            return v[0], v[1]
        else:
            return v, {}

    if not isinstance(matchings, int):
        raise TypeError("Number of matchings must be an integer.")

    if matchings < 1:
        raise ValueError("Number of matchings must be > 0.")

    if (algorithm is not 'drake') and (algorithm is not 'preis') and \
       (algorithm is not 'notay') and (algorithm is not 'drake_C'):
       raise ValueError("Only drake, notay and preis algorithms implemeted.")

    if (symmetry != 'symmetric') and (symmetry != 'hermitian') and \
            (symmetry != 'nonsymmetric'):
        raise ValueError('expected \'symmetric\', \'nonsymmetric\' or\
                         \'hermitian\' for the symmetry parameter ')

    if strength is not None:
        if strength[0] < matchings:
            warn("Expect number of matchings for SOC >= matchings for aggregation.")
            diff = 0
        else:
            diff = strength[0] - matchings  # How many more matchings to do for SOC
    else:
        diff = 0

    # Compute weights if function provided, otherwise let W = A
    if weights is not None:
        W = weights(A, **kwargs)
    else:
        W = deepcopy(A)

    if not isspmatrix_csr(W):
        warn("Requires CSR matrix - trying to convert.", SparseEfficiencyWarning)
        try:
            W = W.tocsr()
        except:
            raise TypeError("Could not convert to csr matrix.")

    n = A.shape[0]
    if (symmetry == 'nonsymmetric') and (Bh == None):
        print "Warning - no left near null-space vector provided for nonsymmetric matrix.\n\
        Copying right near null-space vector."
        Bh = deepcopy(B[0:n,0:1])

    # Dictionary of function names for matching algorithms 
    get_matching = {
        'drake': drake_matching,
        'preis': preis_matching_1999,
        'notay': notay_matching_2010
    }

    # Get initial matching


    # Form sparse P from pairwise aggregation
    rowptr = np.empty(n, dtype='intc')
    colinds = np.empty(n, dtype='intc')
    data = np.empty(n, dtype=float)




    AggOp = csr_matrix( (np.ones((n,), dtype=bool), (row_inds,col_inds)), shape=(n,Nc) )

    # Predefine SOC matrix is only one pairwise pass is done for aggregation
    if (matchings == 1) and (diff > 0):
        AggOp2 = csr_matrix(AggOp, dtype=strength[1])

    # If performing multiple pairwise matchings, form coarse grid operator
    # and repeat process
    if (matchings+diff) > 1:

        # P = csr_matrix( (B[0:n,0], (row_inds,col_inds)), shape=(n,Nc) )

        Bc = np.ones((Nc,1))
        if symmetry == 'hermitian':
            R = P.H
            Ac = R*A*P
        elif symmetry == 'symmetric':
            R = P.T            
            Ac = R*A*P
        elif symmetry == 'nonsymmetric':
            # R = csr_matrix( (Bh[0:n,0], (col_inds,row_inds)), shape=(Nc,n) )
            Ac = R*A*P
            AcH = Ac.H.asformat(Ac.format)
            Bhc = np.ones((Nc,1))

        # Loop over the number of pairwise matchings to be done
        for i in range(1,(matchings+diff)):
            if weights is not None:
                W = weights(Ac, **kwargs)
            else:
                W = Ac
            
            # Improve near nullspace candidates by relaxing on A B = 0
            fn, kwargs = unpack_arg(improve_candidates)
            if fn is not None:
                b = np.zeros((n, 1), dtype=Ac.dtype)
                Bc = relaxation_as_linear_operator((fn, kwargs), Ac, b) * Bc
                if symmetry == "nonsymmetric":
                    Bhc = relaxation_as_linear_operator((fn, kwargs), AcH, b) * Bhc


            # Get matching and form sparse P
            rowptr = np.empty(n, dtype='intc')
            colinds = np.empty(n, dtype='intc')
            data = np.empty(n, dtype=float)



            # Pick C-points and save in list




            # Form coarse grid operator and update aggregation matrix
            if i < (matchings-1):

                # P = csr_matrix( (Bc[0:n,0], (row_inds,col_inds)), shape=(n,Nc) )
                
                if symmetry == 'hermitian':
                    R = P.H
                    Ac = R*Ac*P
                elif symmetry == 'symmetric':
                    R = P.T            
                    Ac = R*Ac*P
                elif symmetry == 'nonsymmetric':
                    
                    # R = csr_matrix( (Bhc[0:n,0], (col_inds,row_inds)), shape=(Nc,n) )

                    Ac = R*Ac*P
                    AcH = Ac.H.asformat(Ac.format)
                    Bhc = np.ones((Nc,1))

                AggOp = csr_matrix(AggOp * P, dtype=bool)
                Bc = np.ones((Nc,1))
            # Construct final aggregation matrix
            elif i == (matchings-1):

                # P = csr_matrix( (np.ones((n,), dtype=bool), (row_inds,col_inds)), shape=(n,Nc) )
                
                AggOp = csr_matrix(AggOp * P, dtype=bool)
                # Construct coarse grids and additional aggregation matrix
                # only if doing more matchings for SOC.
                if diff > 0:
                    if symmetry == 'hermitian':
                        R = P.H
                        Ac = R*Ac*P
                    elif symmetry == 'symmetric':
                        R = P.T            
                        Ac = R*Ac*P
                    elif symmetry == 'nonsymmetric':
                        # R = csr_matrix( (Bhc[0:n,0], (col_inds,row_inds)), shape=(Nc,n) )
                        Ac = R*Ac*P
                        AcH = Ac.H.asformat(Ac.format)
                        Bhc = np.ones((Nc,1))

                    Bc = np.ones((Nc,1))
                    AggOp2 = csr_matrix(AggOp, dtype=strength[1])
            # Pairwise iterations for SOC
            elif i < (matchings+diff-1):

                # P = csr_matrix( (Bc[0:n,0], (row_inds,col_inds)), shape=(n,Nc) )
                
                if symmetry == 'hermitian':
                    R = P.H
                    Ac = R*Ac*P
                elif symmetry == 'symmetric':
                    R = P.T            
                    Ac = R*Ac*P
                elif symmetry == 'nonsymmetric':
                    # R = csr_matrix( (Bhc[0:n,0], (col_inds,row_inds)), shape=(Nc,n) )
                    Ac = R*Ac*P
                    AcH = Ac.H.asformat(Ac.format)
                    Bhc = np.ones((Nc,1))

                AggOp2 = csr_matrix(AggOp2 * P, dtype=bool)
                Bc = np.ones((Nc,1))
            # Final matching for SOC matrix. Construct SOC as AggOp*AggOp^T.
            elif i == (matchings+diff-1):
                # P = csr_matrix( (np.ones((n,), dtype=bool), (row_inds,col_inds)), shape=(n,Nc) )
                AggOp2 = csr_matrix(AggOp2 * P, dtype=bool)
                AggOp2 = csr_matrix(AggOp2*AggOp2.T, dtype=strength[1])

    if strength is None:
        return AggOp, Cpts
    else:
        return AggOp, Cpts, AggOp2

