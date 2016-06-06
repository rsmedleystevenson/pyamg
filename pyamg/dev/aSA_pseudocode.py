def global_ritz_process(A, B1, B2=None, weak_tol=15., level=0, verbose=False):

    # Orthonormalize the vectors.
    B = [B1, B2]
    [Q,R] = qr(B)

    # Solve A^2 generalized eigenvalue problem
    QtAAQ = (Q.T*A.T) * (A*Q)
    [E,V] = eig(QtAAQ)

    # Compute Ritz vectors and normalize in energy. Mark vectors
    # that trivially satisfy weak approximation property.
    entire_const = weak_tol / approximate_spectral_radius(A)
    V = Q*V
    num_candidates = 0
    for j in num_eigenvectors:
        if 1./E[j] <= entire_const:
            break
        else:
            V_j /= sqrt(E_j)
            num_candidates += 1

    # Make sure at least one candidate is kept
    if num_candidates == 0:
        V[:,0] /= np.sqrt(E[0])
        num_candidates = 1

    return V[:, 0:num_candidates]


def local_ritz_process(A, AggOp, B, weak_tol=15., level=0, verbose=False):

    # Scale weak tolerence by spectral radius of A
    #   -> This is expensive, do we need approximate spectral radius?
    tol = weak_tol / approximate_spectral_radius(A)

    # Iterate over aggregates
    for each aggregate:

        Ba = B[aggregate]   # restrict B to aggregate
        BatBa = Ba.T * Ba   # Form Ba^T*Ba
        [E, V] = eig(BatBa) # Eigen decomposition of Ba^T*Ba in descending order

        # Iterate over eigenvectors
        local_const = tol * size(aggregate) / n
        num_targets = 0
        for j in num_eigenvectors:
            if E[j] <= local_const: # local candidate trivially satisfies local WAP
                break
            else:
                V_j /= sqrt(E_j)
                num_targets += 1

        # Keeping at least one target greatly improves convergence
        if num_targets == 0:
            V[:,0] /= np.sqrt(E[0])
            num_targets = 1

        # Define new local basis, Ba * V
        basis = Ba * V[:,0:num_targets] 

        # Add columns 0,...,(num_targets-1) of U to T
        # over current aggregate
        for j in 0,...,(num_targets-1):
            add basis[:,j] as column to T

    return T


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
               local_weak_tol=15.):

    # Call recursive adaptive process starting from finest grid, level 0,
    # to construct adaptive hierarchy. 
    return try_solve(A=A, level=0, args)


def try_solve(A, level, args):

    # Delete previously constructed lower levels
    del hierarchy[level, ..., end]

    # Add new level to hierarchy, matrix A, size n
    hierarchy.append(A)

    # Test if we are at the coarsest level
    if n <= max_coarse or level >= max_levels - 1:
        return

    # Generate initial targets as random vectors relaxed on AB = 0.
    B = rand(n,num_targets)
    B = relax(B)

    # Get SOC matrix, C
    C = strength(A)

    # Compute aggregation matrix, AggOp
    AggOp = aggregate(C)

    # Loop over adaptive hierarchy until CF is sufficient or
    # we have reached maximum iterations
    level_iter = 0
    conv_factor = 1
    target = None
    while (conv_factor > conv_tol) and (level_iter < max_level_iterations):

        # Add new target. Orthogonalize using global / local
        # Ritz and construct tentative prolongator, T.  
        B = global_ritz_process(B1=B, B2=target)
        T = local_ritz_process()

        # Smooth tentative prolongator
        P = smooth(T, A)
        R = P.T

        # Construct coarse grid
        Ac = (R * A * P).tocsr()

        # Symmetrically scale diagonal of Ac, modify R, P accodingly
        sqrt_Dinv, Ac = symmetric_rescaling(Ac)
        P = P * sqrt_Dinv
        R = P.H

        # Save operators to this level of hierarchy
        hierarchy[level].P = P
        hierarchy[level].R = R

        # Recursively call try_solve() with coarse grid operator
        try_solve(Ac, level+1, args)
        level_iter += 1

        # Test convergence of new hierarchy on random vector,
        # the result of which will be used as the next target
        target = rand(n,1)
        target = solve(b=0, x0=target, maxiter=iters)
        conv_factor = residuals[end] / residuals[end-1]

    return

