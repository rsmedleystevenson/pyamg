import amg_core

def pairwise_aggregation(A, B=None, Bh=None, symmetry='hermitian',
                        algorithm='drake', matchings=2,
                        weights=None, improve_candidates=None,
                        get_Cpts=False, **kwargs):
    """ Pairwise aggregation of nodes. 

    Parameters
    ----------
    A : csr_matrix or bsr_matrix
        matrix for linear system.
    B : array_like : default None
        Right near-nullspace candidates stored in the columns of an NxK array.
        If no target vector provided, constant vector is used. In the case of
        multiple targets, k>1, only the first is used to construct coarse grid
        matrices for pairwise aggregations. 
    BH : array_like : default None
        Left near-nullspace candidates stored in the columns of an NxK array.
        BH is only used if symmetry is 'nonsymmetric'.
        The default value B=None is equivalent to BH=B.copy()
    algorithm : string : default 'drake'
        Algorithm to perform pairwise matching. Current options are 
        'drake', 'notay', referring to the Drake (2003), and Notay (2010),
        respectively. For Notay, optional filtering threshold, beta, can be 
        passed in as algorithm=('notay', {'beta' : 0.25}). Default beta=0.25.
    matchings : int : default 2
        Number of pairwise matchings to do. k matchings will lead to 
        a coarsening factor of under 2^k.
    weights : function handle : default None
        Optional function handle to compute weights used in the matching,
        e.g. a strength of connection routine. Additional arguments for
        this routine should be provided in **kwargs. 
    improve_candidates : {tuple, string, list} : default None
        The list elements are relaxation descriptors of the form used for
        presmoother and postsmoother.  A value of None implies no action on B.

            - This is a bit complicated, make sure to explain well how to use it. 


    NOTES
    -----
        - Not implemented for non-symmetric, block systems, or complex. 
            + Need to set up pairwise aggregation to be applicable for 
              nonsymmetric matrices.
            + Need to define how a matching is done nodally.
            + Also must consider what targets are used to form coarse grid
              in nodal approach...
            + Drake should be accessible in complex, but not Notay due to the
              hard minimum. Is there a < operator overloaded for complex?
              Could I overload it perhaps? Probably would do magnitude or something
              though, which is not what we want... 

    REFERENCES
    ----------
    [1] D’Ambra, Pasqua, and Panayot S. Vassilevski. "Adaptive AMG with
    coarsening based on compatible weighted matching." Computing and
    Visualization in Science 16.2 (2013): 59-76.

    [2] Notay, Yvan. "An aggregation-based algebraic multigrid method." 
    Electronic transactions on numerical analysis 37.6 (2010): 123-146.

    [3] Drake, Doratha E., and Stefan Hougardy. "A simple approximation
    algorithm for the weighted matching problem." Information Processing
    Letters 85.4 (2003): 211-213.

    """

    def unpack_arg(v):
        if isinstance(v, tuple):
            return v[0], v[1]
        else:
            return v, {}

    if A.dtype == 'complex':
        raise TypeError("Not currently implemented for complex.")

    if not isinstance(matchings, int):
        raise TypeError("Number of matchings must be an integer.")

    if matchings < 1:
        raise ValueError("Number of matchings must be > 0.")

    if (symmetry != 'symmetric') and (symmetry != 'hermitian') and \
            (symmetry != 'nonsymmetric'):
        raise ValueError('expected \'symmetric\', \'nonsymmetric\' or\
                         \'hermitian\' for the symmetry parameter ')

    n = A.shape[0]
    if (symmetry == 'nonsymmetric'): # and (Bh == None):
        raise ValueError("Not implemented for non-symmetric matrices.")
        # print "Warning - no left near null-space vector provided for nonsymmetric matrix.\n\
        # Copying right near null-space vector."
        # Bh = deepcopy(B[0:n,0:1])

    # Compute weights if function provided, otherwise let W = A
    if weights is not None:
        W = weights(A, **kwargs)
    else:
        W = A

    if not isspmatrix_csr(W):
        warn("Only implemented for CSR matrix - trying to convert.", SparseEfficiencyWarning)
        try:
            W = W.tocsr()
        except:
            raise TypeError("Could not convert to csr matrix.")

    # Get initial targets
    improve_fn, improve_args = unpack_arg(improve_candidates)
    target = None

    # If target vectors provided, take first. Note, no improvement is 
    # done on provided targets in the assumption that they are smooth.
    if B is not None:
        if len(B.shape[1]) == 2:
            target = B[:,0]
        else:
            target = B[:,]
    # If no targets provided, check if a random or constant initial
    # target is specified in improve_candidates. If not, set default 
    # to random and improve target as specified. 
    else:
        if (improve_fn is not None):
            if 'init' not in improve_args:
                improve_args['init'] = 'rand'

            if improve_args['init'] == 'rand':
                target = np.random.rand(n,)
            elif improve_args['init'] == 'ones':
                target = np.ones(n,)
            else:
                raise ValueError("Only initial guesses of 'init' = 'ones'\
                                 and 'init' = 'rand' supported.")
    
            # Improve near nullspace candidates by relaxing on A B = 0
            b = np.zeros((n, 1), dtype=Ac.dtype)
            target = relaxation_as_linear_operator((improve_fn, improve_args), A, b) * target
        
        # If B = None and improve candidates = None, a default target of ones
        # is used on all levels, which is specified by target = None. 
        else: 
            target = None

    # Get matching algorithm 
    beta = 0.25
    choice, alg_args = unpack_arg(algorithm)
    if choice == 'drake': 
        get_pairwise = amg_core.drake_matching
    elif choice == 'notay':
        get_pairwise = amg_core.notay_pairwise
        if 'beta' in alg_args:
            beta = alg_args['beta']
    else:
       raise ValueError("Only drake amd notay pairwise algorithms implemented.")

    # Loop over the number of pairwise matchings to be done
    for i in range(0,matchings-1):

        # Get matching and form sparse P
        rowptr = np.empty(n, dtype='intc')
        colinds = np.empty(n, dtype='intc')
        if target == None:
            get_pairwise(W.indptr, 
                         W.indices,
                         W.data,
                         rowptr,
                         colinds,
                         shape,
                         beta )
            P = csr_matrix( (np.ones(n,), colinds, rowptr), shape=shape )
        else:
            data = np.empty(n, dtype=float)
            get_pairwise(W.indptr, 
                         W.indices,
                         W.data,
                         target,
                         rowptr,
                         colinds,
                         data,
                         shape,
                         beta )
            P = csr_matrix( (data, colinds, rowptr), shape=shape )

        # Form aggregation matrix 
        if i == 0:
            AggOp = csr_matrix( (np.ones(n, dtype=bool), colinds, rowptr), shape=shape )
        else:
            AggOp = csr_matrix(AggOp * P, dtype=bool)

        # Form coarse grid operator
        if symmetry == 'hermitian':
            R = P.H
            Ac = R*Ac*P
        elif symmetry == 'symmetric':
            R = P.T            
            Ac = R*Ac*P

        # Compute optional weights on coarse grid operator
        n = Ac.shape[0]
        if weights is not None:
            W = weights(Ac, **kwargs)
        else:
            W = Ac
                
        # Form new target vector on coarse grid if this is not last iteration.
        # If last iteration, we will not use target - set to None.  
        if (improve_fn is not None) and (i < matchings-2):
            if improve_args['init'] == 'rand':
                target = np.random.rand(n,)
            elif improve_args['init'] == 'ones':
                target = np.ones(n,)

            # Improve near nullspace candidates by relaxing on A B = 0
            b = np.zeros((n, 1), dtype=Ac.dtype)
            target = relaxation_as_linear_operator((improve_fn, improve_args), Ac, b) * target

        else:
            target = None            

    # Get final matching, form aggregation matrix 
    rowptr = np.empty(n, dtype='intc')
    colinds = np.empty(n, dtype='intc')
    shape = np.empty(2, dtype='intc')
    get_pairwise(W.indptr, 
                 W.indices,
                 W.data,
                 rowptr,
                 colinds,
                 shape,
                 beta )

    if matching > 1:
        temp_agg = csr_matrix( (np.ones(n, dtype='int8'), colinds, rowptr), shape=shape )
        AggOp = csr_matrix( AggOp * temp_agg, dtype='int8')
    else:
        AggOp = csr_matrix( (np.ones(n, dtype='int8'), colinds, rowptr), shape=shape )


    # NEED TO IMPLEMENT A WAY TO CHOOSE C-POINTS
    if get_Cpts:
        raise TypeError("Cannot return C-points - not yet implemented.")
    else:
        return AggOp


