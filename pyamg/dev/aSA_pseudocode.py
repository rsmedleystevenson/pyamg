
def global_ritz_process(A, B1, B2=None, weak_tol=15., level=0, verbose=False):

    # Orthonormalize the vectors.
    B = [B1, B2]
    [Q,R] = qr(B)

    # Solve A^2 generalized eigenvalue problem
    QtAAQ = (Q.T*A.T) * (A*Q)
    [E,V] = eig(QtAAQ)

    # Compute Ritz vectors and normalize in energy. Mark vectors
    # that trivially satisfy weak approximation property.
    V = Q*V
    num_candidates = -1
    entire_const = weak_tol / approximate_spectral_radius(A)

    for j in num_eigenvectors:
        V_j /= sqrt(E_j)
        if 1./E[j] <= entire_const:
            num_candidates = j
            break


    return V[:, :num_candidates]


def local_ritz_process(A, AggOp, B, weak_tol=15., level=0, verbose=False):

    # scale the weak tolerence by the radius of A
    tol = weak_tol / approximate_spectral_radius(A)

    # row, col, and val arrays to store entries of T
    max_size = B.shape[1]*AggOp.getnnz()
    row_i = numpy.empty(max_size)
    col_i = numpy.empty(max_size)
    val_i = numpy.empty(max_size)
    cur_col = 0
    index = 0

    # iterate over aggregates
    for i in 0,...,num_aggregates:
        agg = AggOpCsc[:,i] # get the current aggregate
        rows = agg.nonzero()[0] # non zero rows of aggregate
        Ba = B[rows] # restrict B to aggregate

        BatBa = numpy.dot(Ba.transpose(), Ba) # Ba^T*Ba

        [E, V] = numpy.linalg.eigh(BatBa) # Eigen decomp of Ba^T*Ba
        E = E[::-1] # eigenvalues are ascending, we want them descending
        V = numpy.fliplr(V) # flip eigenvectors to match new order of eigenvalues

        num_targets = 0
        # iterate over eigenvectors
        for j in range(V.shape[1]):
            local_const = agg.getnnz() * tol / AggOp.getnnz()
            if E[j] <= local_const: # local candidate trivially satisfies local WAP
                break
            num_targets += 1

        # having at least 1 target greatly improves performance
        num_targets = min(max(1, num_targets), V.shape[1])
        per_agg_count[rows] = num_targets

        basis = numpy.dot(Ba, V[:,0:num_targets]) # new local basis is Ba * V

        # add 0 to num_targets-1 columns of U to T
        for j in range(num_targets):
            basis[:,j] /= numpy.sqrt(E[j])
            for x in range(rows.size):
                row_i[index] = rows[x]
                col_i[index] = cur_col
                val_i[index] = basis[x,j]
                index += 1
            cur_col += 1


    row_i.resize(index)
    col_i.resize(index)
    val_i.resize(index)

    # build csr matrix
    return csr_matrix((val_i, (row_i, col_i)), (B.shape[0], cur_col)), per_agg_count


def asa_solver(A, B=None,
               symmetry='hermitian'
               strength='symmetric',
               aggregate='standard',
               smooth='jacobi',
               presmoother=('block_gauss_seidel',
                            {'sweep': 'symmetric'}),
               postsmoother=('block_gauss_seidel',
                             {'sweep': 'symmetric'}),
               improve_candidates=('block_gauss_seidel',
                                    {'sweep': 'symmetric',
                                     'iterations': 4}),
               max_coarse=20,
               max_levels=20,
               conv_tol=0.5,
               max_targets=100,
               min_targets=0,
               num_targets=1,
               max_level_iterations=10,
               weak_tol=15.,
               local_weak_tol=15.,
               diagonal_dominance=False,
               coarse_solver='pinv2',
               verbose=False,
               keep=True,
               **kwargs):



    # Call recursive adaptive process starting from finest grid, level 0,
    # to construct adaptive hierarchy. 
    return try_solve(level=0)



def try_solve(A, levels,
              level,
              symmetry,
              strength,
              aggregate,
              smooth,
              presmoother,
              postsmoother,
              improve_candidates,
              max_coarse,
              max_levels,
              conv_tol,
              max_targets,
              min_targets,
              num_targets,
              max_level_iterations,
              weak_tol,
              local_weak_tol,
              coarse_solver,
              diagonal_dominance,
              verbose,
              keep,
              hierarchy=None):

    
    # Delete previously constructed lower levels

    # Add new level to hierarchy, matrix A, size n

    # Test if we are at the coarsest level
    if n <= max_coarse or level >= max_levels - 1:
        return

    # Generate initial targets as random vectors relaxed on AB = 0.
    B = rand(n,num_targets)
    B = relax(B)

    # Get SOC matrix C
    C = strength(A)

    # Compute aggregation matrix AggOp
    AggOp = aggregate(C)

    # Loop over adaptive hierarchy until CF is sufficient or
    # we have reached maximum iterations
    level_iter = 0
    conv_factor = 1
    target = None
    while (conv_factor > conv_tol) and (level_iter < max_level_iterations):

        # Add new target. Orthogonalize using global /
        # local Ritz and reconstruct T.  
        B = global_ritz_process(B1=B, B2=target)
        T = local_ritz_process()

        # Smooth tentative prolongator
        P = smooth(T, A)
        R = P.T

        # Construct coarse grid
        Ac = (R * A * P).tocsr()

        # Symmetrically scale diagonal of A
        [dum, Dinv, dum] = symmetric_rescaling(Ac, copy=False)
        P = (P * diags(Dinv, offsets=0)).tocsr()
        R = P.H

        # Recursively call try_solve() with coarse grid operator
        try_solve(level+1, Ac)
        level_iter += 1

        # Test convergence of new hierarchy on random vector
        target = rand(n,1)
        target = solve(b=0, x0=target, cycle=cycle, maxiter=iters)
        conv_factor = residuals[end] / residuals[end-1]

    return

