"""Basic Two-Level Adaptive Smoothed Aggregation"""

__docformat__ = "restructuredtext en"

import pdb

import time
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg as linalg
import scipy.linalg
from scipy.sparse import bsr_matrix, isspmatrix_csr, isspmatrix_bsr, eye, csr_matrix, diags
from scipy.sparse.linalg.isolve.utils import make_system
from scipy.sparse.linalg.interface import LinearOperator
from scipy import stats

from pyamg.multilevel import multilevel_solver, coarse_grid_solver
from pyamg.strength import symmetric_strength_of_connection, ode_strength_of_connection
from pyamg.relaxation.smoothing import change_smoothers
from pyamg.util.linalg import norm, approximate_spectral_radius
from pyamg.util.utils import levelize_strength_or_aggregation, levelize_smooth_or_improve_candidates, \
                            symmetric_rescaling, relaxation_as_linear_operator
from .aggregate import standard_aggregation, naive_aggregation,\
    lloyd_aggregation
from pyamg.aggregation.smooth import jacobi_prolongation_smoother, energy_prolongation_smoother, \
                   richardson_prolongation_smoother
from .tentative import fit_candidates


__all__ = ['A_norm', 'my_rand', 'tl_sa_solver', 'asa_solver']

def tabs(level):
    return '\t'*level


def A_norm(x, A):
    """
    Calculate A-norm of x
    """
    x = np.ravel(x)
    return np.sqrt(scipy.dot(x.conjugate(), A*x))


def my_rand(d1, d2, zero_crossings=True):
    """
    Uniformly random vectors of size (d1,d2)
    If zero_crossings are allowed, then range
    of values is [-1,1].  If not, the range is
    [0,1].
    """

    x = scipy.rand(d1,d2)
    if zero_crossings:
        x = (x - 0.5)*2.0
    # x = np.ones((d1,d2))

    return x


def global_ritz_process(A, B1, B2=None, weak_tol=15., level=0, verbose=False):
    """
    Helper function that compresses two sets of targets B1 and B2 into one set
    of candidates. This is the Ritz procedure.

    Parameters
    ---------
    A : {sparse matrix}
        SPD matrix used to compress the candidates so that the weak
        approximation property is satisfied.
    B1 : {array}
        n x m1 array of m1 potential candidates
    B2 : {array}
        n x m2 array of m2 potential candidates
    weak_tol : {float}
        The constant in the weak approximation property.

    Returns
    -------
    New set of candidates forming an Euclidean orthogonal and energy
    orthonormal subset of span(B1,B2). The candidates that trivially satisfy
    the weak approximation property are deleted.
    """

    # TODO : hstack is slow as shit
    if B2 is not None:
        B = np.hstack((B1, B2.reshape(-1,1)))
    else:
        B = B1

    # Orthonormalize the vectors.
    [Q,R] = scipy.linalg.qr(B, mode='economic')

    # Formulate and solve the eigenpairs problem returning eigenvalues in
    # ascending order.
    # QtAQ = scipy.dot(Q.conjugate().T, A*Q)        # WAP  
    # [E,V] = scipy.linalg.eigh(QtAQ)
    QtAAQ = A*Q
    QtAAQ = scipy.dot(QtAAQ.conjugate().T, QtAAQ)   # WAP_{A^2} = SAP
    [E,V] = scipy.linalg.eigh(QtAAQ)

    # Make sure eigenvectors are real. Eigenvalues must be already real.
    try:
        V = np.real(V)
    except:
        import pdb; pdb.set_trace()

    # Compute Ritz vectors and normalize them in energy. Also, mark vectors
    # that trivially satisfy the weak approximation property.
    V = scipy.dot(Q, V)
    entire_const = weak_tol / approximate_spectral_radius(A)
    if verbose:
        print
        print tabs(level), "WAP const", entire_const

    num_candidates = 0
    for j in range(V.shape[1]):
        if verbose:
            print tabs(level), "Vector 1/e", j, 1./E[j], "ELIMINATED" if 1./E[j] <= entire_const else ""
        
        if 1./E[j] <= entire_const:
            num_candidates = j
            break
        else:
            V[:,j] /= np.sqrt(E[j])
            num_candidates += 1
            # verify energy norm is 1
            # print tabs(level), "&&&&&&&&&&&&&&&&&&&&", scipy.dot(A*V[:,j],V[:,j])

    # Make sure at least one candidate is kept
    if num_candidates == 0:
        V[:,0] /= np.sqrt(E[0])
        num_candidates = 1


    if verbose:
        # print tabs(level), "Finished global ritz process, eliminated", B.shape[1]-num_candidates, "candidates", num_candidates, ". candidates remaining"
        print

    return V[:, 0:num_candidates]


# TODO : this has to be moved to C
def local_ritz_process(A, AggOp, B, weak_tol=15., level=0, verbose=False):
    """
    Helper function that finds the minimal local basis of a set of candidates.

    Parameters
    ----------
    A : {csr_matrix}
        SPD Matrix used to calculate tolerance.
    AggOp : {csr_matrix}
        The aggregation opterator. Used to determine aggregate groups.
    B : {array}
        n x m array of m candidates.
    weak_tol : {float}
        The weak approximation constant divided by the approximate spectral
        radius of the Matrix.

    Returns
    -------
    T : {csr_matrix}
        The tentative prolongator.
    """

    # TODO: keep this?
    # return fit_candidates(AggOp, B)[0], []
    # if B.shape[1] < 2:
    #     return AggOp

    # scale the weak tolerence by the radius of A
    tol = weak_tol / approximate_spectral_radius(A)

    # TODO : Can we avoid reallocation? 
    AggOpCsc = AggOp.tocsc() # we are slicing columns, this makes it much faster

    # row, col, and val arrays to store entries of T
    max_size = B.shape[1]*AggOp.getnnz()
    row_i = np.empty(max_size)
    col_i = np.empty(max_size)
    val_i = np.empty(max_size)
    cur_col = 0
    index = 0

    # store how many basis functions we keep per aggregate
    per_agg_count = np.zeros((B.shape[0], 1))

    # iterate over aggregates
    for i in range(AggOpCsc.shape[1]):

        agg = AggOpCsc[:,i] # get the current aggregate
        rows = agg.nonzero()[0] # non zero rows of aggregate
        Ba = B[rows] # restrict B to aggregate

        BatBa = np.dot(Ba.transpose(), Ba) # Ba^T*Ba

        [E, V] = np.linalg.eigh(BatBa) # Eigen decomp of Ba^T*Ba
        E = E[::-1] # eigenvalues are ascending, we want them descending
        V = np.fliplr(V) # flip eigenvectors to match new order of eigenvalues

        local_const = agg.getnnz() * tol / AggOp.getnnz()
        num_targets = 0

        # iterate over eigenvectors
        for j in range(V.shape[1]):
            if E[j] <= local_const: # local candidate trivially satisfies local WAP
                break
            else:
                V[:,j] /= np.sqrt(E[j])
                num_targets += 1

 
        # Keeping at least one target greatly improves convergence
        if num_targets == 0:
            V[:,0] /= np.sqrt(E[0])
            num_targets = 1

        per_agg_count[rows] = num_targets

        # Define new local basis, Ba * V
        basis = np.dot(Ba, V[:,0:num_targets]) 

        # Add columns 0,...,(num_targets-1) of U to T
        for j in range(num_targets):
            for x in range(rows.size):
                row_i[index] = rows[x]
                col_i[index] = cur_col
                val_i[index] = basis[x,j]
                index += 1
            cur_col += 1

    if verbose:
        print tabs(level), "Eliminated %d local vectors (%.2f%% remaining nonzeros)" % \
                (B.shape[1]*AggOp.shape[1] - cur_col, index*100.0/max_size)
    
    # TODO : don't need this, can construct CSR w/ hanging zeros. 
    #        Will move this function to C anyways.
    row_i.resize(index)
    col_i.resize(index)
    val_i.resize(index)

    # build csr matrix
    return csr_matrix((val_i, (row_i, col_i)), (B.shape[0], cur_col)), per_agg_count


def asa_solver(A, B=None,
               symmetry='hermitian',
               strength='symmetric',
               aggregate='standard',
               smooth='jacobi',
               presmoother=('block_gauss_seidel',
                            {'sweep': 'symmetric'}),
               postsmoother=('block_gauss_seidel',
                             {'sweep': 'symmetric'}),
               improvement_iters=10,
               max_coarse=20,
               max_levels=20,
               target_convergence=0.5,
               max_targets=100,
               min_targets=0,
               num_targets=1,
               max_level_iterations=10,
               weak_tol=15.,
               local_weak_tol=15.,
               diagonal_dominance=False,
               coarse_solver='pinv2',
               cycle='V',
               verbose=False,
               keep=True,
               **kwargs):
    """
    Create a two-level solver using Adaptive Smoothed Aggregation (aSA)

    Parameters
    ----------
    A : {csr_matrix, bsr_matrix}
        Square matrix in CSR or BSR format
    B : {None, n x m dense matrix}
        If a matrix, then this forms the basis for the first m targets.
        Also in this case, the initial setup stage is skipped, because this
        provides the first target(s).  If None, then a random initial guess
        and relaxation are used to inform the initial target.
    max_targets : {integer}
        Maximum number of near-nullspace targets to generate
    min_targets : {integer}
        Minimum number of near-nullspace targets to generate
    num_targets : {integer}
        Number of initial targets to generate
    targets_iters : {integer}
        Number of smoothing passes/multigrid cycles used in the adaptive
        process to obtain targets.
    target_convergence : {float}
        Convergence factor tolerance before the adaptive solver is accepted.
    weak_tol : {float}
        Weak approximation tolerance for dropping targets globally.
    local_weak_tol : {float}
        Weak approximation tolerance for dropping targets locally.
    max_coarse : {integer}
        Maximum number of variables permitted on the coarse grid. 
    max_levels : {integer}
        Maximum number of levels.
    max_level_iterations : {integer}
        Maximum number of aSA iterations per level.
    prepostsmoother : {string or dict}
        Pre- and post-smoother used in the adaptive method
    smooth : ['jacobi', 'richardson', 'energy', None]
        Method used used to smooth the tentative prolongator.  See
        smoothed_aggregation_solver(...) documentation
    strength : ['symmetric', 'classical', 'ode', ('predefined', {'C' : csr_matrix}), None]
        Method used to determine the strength of connection between unknowns
        of the linear system.  See smoothed_aggregation_solver(...) documentation.
    aggregate : ['standard', ('predefined', {'AggOp' : csr_matrix})]
        Method used to aggregate nodes.  See smoothed_aggregation_solver(...)
        documentation.
    coarse_solver : ['splu','lu', ... ]
        Solver used at the coarsest level of the MG hierarchy
    verbose : [True, False]
        If True, print information from each level visited
    keep : [True, False]
        If True, keep all temporary operators in hierarchy.  This should be True.

    Returns
    -------
    Smoothed aggregation solver with adaptively generated targets.

    Floating point value representing the "work" required to generate
    the solver.  This value is the total cost of just relaxation, relative
    to the fine grid.

    Notes
    -----
        Unlike the standard Smoothed Aggregation (SA) method, adaptive SA
        does not require knowledge of near-nullspace target vectors.
        Instead, an adaptive procedure computes one or more targets
        'from scratch'.  This approach is useful when no targets are known
        or the targets have been invalidated due to changes to matrix A.

    Examples
    --------
    >>> from pyamg.gallery import stencil_grid
    >>> from pyamg.aggregation import go_to_room_sa_solver
    >>> import np
    >>> A=stencil_grid([[-1,-1,-1],[-1,8.0,-1],[-1,-1,-1]], (31,31),format='csr')
    >>> sa = tl_sa_solver(A,num_targets=1)
    >>> residuals=[]
    >>> x=sa.solve(b=np.ones((A.shape[0],)),x0=np.ones((A.shape[0],)),residuals=residuals)
    """

    if not (isspmatrix_csr(A) or isspmatrix_bsr(A)):
        try:
            print 'Implicit conversion of A to CSR in tl_sa_solver()...'
            A = csr_matrix(A)
        except:
            raise TypeError('Argument A must have type csr_matrix or bsr_matrix, ' \
                            'or be convertible to csr_matrix.')

    A = A.asfptype()
    if A.shape[0] != A.shape[1]:
        raise ValueError('Expected square matrix!')

    levels = []

    # Levelize the user parameters, so that they become lists describing the
    # desired user option on each level.
    max_levels, max_coarse, strength =\
        levelize_strength_or_aggregation(strength, max_levels, max_coarse)
    max_levels, max_coarse, aggregate =\
        levelize_strength_or_aggregation(aggregate, max_levels, max_coarse)
    smooth = levelize_smooth_or_improve_candidates(smooth, max_levels)



    # TODO : Should maybe levelize adaptive parameters level_iterations, max targets, etc.?
    #   - Assume tolerances fixed on all levels
    #   - Assume max / min / num targets fixed on all levels too, don't see why they change>
    #   - Max_level_iterations may be reasonable to change per level



    # Call recursive adaptive process starting from finest grid, level 0,
    # to construct adaptive hierarchy. 
    hierarchy = multilevel_solver(levels=levels)
    try_solve(A=A, levels=levels, level=0, symmetry=symmetry,
              strength=strength, aggregate=aggregate, smooth=smooth,
              presmoother=presmoother, postsmoother=postsmoother,
              improvement_iters=improvement_iters,
              max_coarse=max_coarse, max_levels=max_levels,
              target_convergence=target_convergence, max_targets=max_targets,
              min_targets=min_targets, num_targets=num_targets, 
              max_level_iterations=max_level_iterations,
              weak_tol=weak_tol, local_weak_tol=local_weak_tol,
              diagonal_dominance=diagonal_dominance,
              coarse_solver=coarse_solver, cycle=cycle,
              verbose=verbose, keep=keep, hierarchy=hierarchy)

    return hierarchy


# Weird bugs 
#   - Richardson P smoothing calls some pow funciton in scipy??




# Important :
#   ==> Levels are passed by reference, but a coarse grid solver
#       is constructed on the first pass, and invalid if we 
#       remove a level. Thus we can pop() off the levels list,
#       and it modifies the hierarchy created from the list. 
def try_solve(A, levels,
              level,
              symmetry,
              strength,
              aggregate,
              smooth,
              presmoother,
              postsmoother,
              improvement_iters,
              max_coarse,
              max_levels,
              target_convergence,
              max_targets,
              min_targets,
              num_targets,
              max_level_iterations,
              weak_tol,
              local_weak_tol,
              diagonal_dominance,
              coarse_solver,
              cycle,
              verbose,
              keep,
              hierarchy):

    """
    Needs some love.
    """

    def unpack_arg(v):
        if isinstance(v, tuple):
            return v[0], v[1]
        else:
            return v, {}

    # If coarserer hierarchies have already been defined, remove
    # because they will be reconstructed
    while len(levels) > level:
        levels.pop()

    # Add new level to hierarchy, define reference 'current' to current level
    # Set n and rhs for testing convergence on homogeneous problem
    levels.append(multilevel_solver.level())
    current = levels[level]
    current.A = A
    n = A.shape[0]

    # Test if we are at the coarsest level. If no hierarchy has been
    # constructed in adaptive process yet, construct hierarchy. If 
    # a hierarchy exists, update coarse grid solver. 
    if n <= max_coarse or level >= max_levels - 1:
        hierarchy.coarse_solver = coarse_grid_solver(coarse_solver)
        change_smoothers(hierarchy, presmoother, postsmoother)
        return

    # initialize history
    current.history = {}
    current.history['B'] = []
    current.history['conv'] = []
    current.history['agg'] = []

    # Generate initial targets as random vectors relaxed on AB = 0.
    current.B = my_rand(n,num_targets, zero_crossings=False)
    fn, kwargs = unpack_arg(presmoother)
    kwargs['iterations'] = improvement_iters
    if fn is None:
        raise ValueError("Must improve candidates for aSA.")

    b = np.zeros((A.shape[0], 1), dtype=A.dtype)
    for i in range(0,num_targets):
        current.B[:,i:(i+1)] = relaxation_as_linear_operator( \
                                (fn, kwargs), \
                                current.A, b) * current.B[:,i:(i+1)]

    # Compute the strength-of-connection matrix C, where larger
    # C[i,j] denote stronger couplings between i and j.
    fn, kwargs = unpack_arg(strength[level])
    if fn == 'symmetric':
        C = symmetric_strength_of_connection(current.A, **kwargs)
    elif fn == 'classical':
        C = classical_strength_of_connection(current.A, **kwargs)
    elif fn == 'distance':
        C = distance_strength_of_connection(current.A, **kwargs)
    elif (fn == 'ode') or (fn == 'evolution'):
        if 'B' in kwargs:
            C = evolution_strength_of_connection(current.A, **kwargs)
        else:
            C = evolution_strength_of_connection(current.A, current.B, **kwargs)
    elif fn == 'energy_based':
        C = energy_based_strength_of_connection(current.A, **kwargs)
    elif fn == 'predefined':
        C = kwargs['C'].tocsr()
    elif fn == 'algebraic_distance':
        C = algebraic_distance(current.A, **kwargs)
    elif fn == 'affinity':
        C = affinity_distance(current.A, **kwargs)
    elif fn is None:
        C = currrent.A.tocsr()
    else:
        raise ValueError('unrecognized strength of connection method: %s' %
                         str(fn))

     # Avoid coarsening diagonally dominant rows
    flag, kwargs = unpack_arg(diagonal_dominance)
    if flag:
        C = eliminate_diag_dom_nodes(current.A, C, **kwargs)

    # Compute the aggregation matrix AggOp (i.e., the nodal coarsening of A).
    # AggOp is a boolean matrix, where the sparsity pattern for the k-th column
    # denotes the fine-grid nodes agglomerated into k-th coarse-grid node.
    fn, kwargs = unpack_arg(aggregate[level])
    if fn == 'standard':
        current.AggOp = standard_aggregation(C, **kwargs)[0]
    elif fn == 'naive':
        current.AggOp = naive_aggregation(C, **kwargs)[0]
    elif fn == 'lloyd':
        current.AggOp = lloyd_aggregation(C, **kwargs)[0]
    elif fn == 'predefined':
        current.AggOp = kwargs['AggOp'].tocsr()
    else:
        raise ValueError('unrecognized aggregation method %s' % str(fn))

    # Loop over adaptive hierarchy until CF is sufficient or we have 
    # reached maximum iterations
    level_iter = 0
    conv_factor = 1
    target = None
    while (conv_factor > target_convergence) and (level_iter < max_level_iterations):

        # Add new target. Orthogonalize using global / local Ritz and reconstruct T.  
        current.B = global_ritz_process(A=current.A, B1=current.B, B2=target, \
                                        weak_tol=weak_tol, level=level, \
                                        verbose=verbose)
        current.T, per_agg = local_ritz_process(A=current.A, AggOp=current.AggOp, \
                                                B=current.B, weak_tol=local_weak_tol, \
                                                level=level, verbose=verbose)
        current.history['B'].append(current.B)
        current.history['agg'].append(per_agg)

        # Smooth tentative prolongator
        fn, kwargs = unpack_arg(smooth[level])
        if fn == 'jacobi':
            current.P = jacobi_prolongation_smoother(current.A, current.T, C, \
                                                     current.B, **kwargs)
        elif fn == 'richardson':
            current.P = richardson_prolongation_smoother(current.A, current.T \
                                                         **kwargs)
        elif fn == 'energy':
            current.P = energy_prolongation_smoother(current.A, current.T, C, \
                                                     current.B, None, (False, {}), \
                                                     **kwargs)
        elif fn is None:
            current.P = T
        else:
            raise ValueError('unrecognized prolongation smoother method %s' %
                             str(fn))
        
        current.R = current.P.H

        # Construct coarse grid
        Ac = (current.R * current.A * current.P).tocsr()

        # Symmetrically scale diagonal of Ac, modify R,P accodingly
        #
        #   - TODO : Make sure this is doing the right thing
        #     TODO : Do we need to do this? 
        #
        [dum, Dinv, dum] = symmetric_rescaling(Ac, copy=False)
        current.P = (current.P * diags(Dinv, offsets=0)).tocsr()
        current.R = current.P.H

        # Recursively call try_solve() with coarse grid operator
        try_solve(A=Ac, levels=levels, level=(level+1), symmetry=symmetry,
                  strength=strength, aggregate=aggregate, smooth=smooth,
                  presmoother=presmoother, postsmoother=postsmoother,
                  improvement_iters=improvement_iters,
                  max_coarse=max_coarse, max_levels=max_levels,
                  target_convergence=target_convergence, max_targets=max_targets,
                  min_targets=min_targets, num_targets=num_targets, 
                  max_level_iterations=max_level_iterations,
                  weak_tol=weak_tol, local_weak_tol=local_weak_tol,
                  diagonal_dominance=diagonal_dominance,
                  coarse_solver=coarse_solver, cycle=cycle,
                  verbose=verbose, keep=keep, hierarchy=hierarchy)

        level_iter += 1

        # Test convergence of new hierarchy
        #
        #   TODO: be smarter about approximating the convergence factor?
        #         Tristan thinks some kind of linear fit to CFs could
        #         provide a better estimate of CF.
        #
        residuals = []
        target = my_rand(n, 1, current.A.dtype)
        target = hierarchy.solve(b, x0=target, cycle=cycle, \
                                 maxiter=improvement_iters, \
                                 tol=1e-16, residuals=residuals,
                                 init_level=level, accel=None)
        conv_factor = residuals[-1] / residuals[-2]
        current.history['conv'].append(conv_factor)

        if verbose:
            print tabs(level), "Iter = ",level_iter,", num targets = ",\
                  num_targets,", CF = ", conv_factor

    return

