"""Basic Two-Level Adaptive Smoothed Aggregation"""

__docformat__ = "restructuredtext en"

import pdb

# TODO : Be consistent with scipy importing. Remove unnecessary imports. 

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
from pyamg.strength import symmetric_strength_of_connection,\
                        classical_strength_of_connection, evolution_strength_of_connection
from pyamg.relaxation.smoothing import change_smoothers
from pyamg.util.linalg import norm, approximate_spectral_radius
from pyamg.util.utils import levelize_strength_or_aggregation, levelize_smooth_or_improve_candidates, \
                            symmetric_rescaling, relaxation_as_linear_operator, mat_mat_complexity
from .aggregate import standard_aggregation, naive_aggregation,\
    lloyd_aggregation
from pyamg.aggregation.smooth import jacobi_prolongation_smoother, energy_prolongation_smoother, \
                   richardson_prolongation_smoother
from .tentative import fit_candidates


__all__ = ['A_norm', 'my_rand', 'tl_sa_solver', 'asa_solver']


def unpack_arg(v, cost=True):
    if isinstance(v, tuple):
        if cost:
            (v[1])['cost'] = [0.0]
            return v[0], v[1]
        else:
            return v[0], v[1]
    else:
        if cost:
            return v, {'cost' : [0.0]}
        else:
            return v, {}


def tabs(level):
    return '  '*level


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


def global_ritz_process(A, B1, B2, sap_tol, level, max_bad_guys,
                        verbose=False, cost=[0]):
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
    level
        This is only for debugging purposes --> TODO : Remove
    max_bad_guys : int 
        Maximum number of global bad guys to keep

    Returns
    -------
    New set of candidates forming an Euclidean orthogonal and energy
    orthonormal subset of span(B1,B2). The candidates that trivially satisfy
    the weak approximation property are deleted.
    """

    # TODO : hstack is very slow
    if B2 is not None:
        B = np.hstack((B1, B2.reshape(-1,1)))
    else:
        B = B1

    # Orthonormalize the vectors. Cost taken from Golub/Van Loan ~ 2mn^2
    [Q,R] = scipy.linalg.qr(B, mode='economic')
    cost[0] += 2.0 * B.shape[0] * B.shape[1]**2 / float(A.nnz)

    # Formulate and solve the eigenpairs problem returning eigenvalues in
    # ascending order. Eigenvalue cost ~ 9n^3 for symmetric QR, Golub/Van Loan
    QtAAQ = A*Q
    QtAAQ = scipy.dot(QtAAQ.conjugate().T, QtAAQ)   # WAP_{A^2} = SAP
    [E,V] = scipy.linalg.eigh(QtAAQ)
    cost[0] += Q.shape[1] + float(Q.shape[0] * Q.shape[1]**2) / A.nnz + \
                float(9.0 * QtAAQ.shape[0]**3) / A.nnz

    # Make sure eigenvectors are real. Eigenvalues must be already real.
    V = np.real(V)

    # Compute Ritz vectors and normalize them in energy. 
    V = scipy.dot(Q, V)
    cost[0] += float(V.shape[0] * V.shape[1]**2) / A.nnz

    # Select candidates that don't trivially satisfy the WAP(A^2).
    num_candidates = 0
    for j in range(V.shape[1]):
        if 1./E[j] <= sap_tol:
            num_candidates = j
            break
        else:
            V[:,j] /= np.sqrt(E[j])
            num_candidates += 1

    # Make sure at least one candidate is kept
    if num_candidates == 0:
        V[:,0] /= np.sqrt(E[0])
        num_candidates = 1

    cost[0] += float(V.shape[0] * num_candidates) / A.nnz

    print tabs(level), "Glob cand - ", num_candidates, ", max norm = ", np.dot(V[:,0].T,V[:,0]) # targets = ","%.2f"%av_num, \

    # Only allow for for max_bad_guys to be kept
    num_candidates = np.min((num_candidates,max_bad_guys))
    return V[:, 0:num_candidates]


# TODO : this has to be moved to C
def local_ritz_process(A, AggOp, B, sap_tol, level, max_bullets,
                       verbose=False, cost=[0]):
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
        radius of A, both squared for the WAP(A^2).

    Returns
    -------
    T : {csr_matrix}
        The tentative prolongator.
    """

    # TODO: keep this?
    # return fit_candidates(AggOp, B)[0], []
    # if B.shape[1] < 2:
    #     return AggOp

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

    list_targets = []
    agg_sizes = []

    # iterate over aggregates
    for i in range(AggOpCsc.shape[1]):

        # Get current aggregate and restrict bad guys to it
        agg = AggOpCsc[:,i]
        rows = agg.nonzero()[0]
        Ba = B[rows]

        # Get eigenvalue decomposition of Ba^T*Ba, sprt eigenvalues
        # and vectors in descending order.
        BatBa = np.dot(Ba.transpose(), Ba)
        [E, V] = np.linalg.eigh(BatBa)
        E = E[::-1]
        V = np.fliplr(V)
        cost[0] +=  float(Ba.shape[1]*Ba.shape[0]**2 + 9*V.shape[0]**3) / A.nnz

        # Form constant for local WAP(A^2)
        local_const = sap_tol * float(agg.nnz) / AggOp.nnz
        num_targets = 0

        # iterate over eigenvectors
        for j in range(V.shape[1]):
            if E[j] <= local_const: # local candidate trivially satisfies local WAP
                break
            else:
                V[:,j] /= np.sqrt(E[j])
                num_targets += 1
 
        # Make sure at least one candidate is kept
        if num_targets == 0:
            V[:,0] /= np.sqrt(E[0])
            num_targets = 1

        per_agg_count[rows] = num_targets
        num_targets = np.min((num_targets,max_bullets))
        cost[0] += float(V.shape[0] * num_targets) / A.nnz

        # Define new local basis, Ba * V
        basis = np.dot(Ba, V[:,0:num_targets]) 
        cost[0] += float(Ba.shape[0] * Ba.shape[1] * num_targets) / A.nnz

        # Diagnostics data
        list_targets.append(num_targets)
        agg_sizes.append(len(rows))

        # Add columns 0,...,(num_targets-1) of U to T
        for j in range(num_targets):
            for x in range(rows.size):
                row_i[index] = rows[x]
                col_i[index] = cur_col
                val_i[index] = basis[x,j]
                index += 1
            cur_col += 1

    if verbose:
        av_size = np.mean(agg_sizes)
        av_num = np.mean(list_targets)

        print tabs(level), "Av. agg size = ","%.2f"%av_size,", Av. # targets = ","%.2f"%av_num, \
                ", Max = ",np.max(list_targets),", Min = ",np.min(list_targets)
        # print tabs(level), "Eliminated %d local vectors (%.2f%% remaining nonzeros)" % \
        #         (B.shape[1]*AggOp.shape[1] - cur_col, index*100.0/max_size)
    
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
               max_bullets=100,
               max_bad_guys=0,
               num_targets=1,
               max_level_iterations=10,
               weak_tol=15.,
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
    max_bullets : {integer}
        Maximum number of near-nullspace targets to generate
        TODO : THIS IS NOT USED
    max_bad_guys : {integer}
        Minimum number of near-nullspace targets to generate
        TODO : THIS IS NOT USED
    num_targets : {integer}
        Number of initial targets to generate
    improvement_iters : {integer}
        Number of smoothing passes/multigrid cycles used in the adaptive
        process to obtain targets.
    target_convergence : {float}
        Convergence factor tolerance before the adaptive solver is accepted.
    weak_tol : {float}
        Weak approximation tolerance for dropping targets globally.
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

    if ('setup_complexity' in kwargs):
        if kwargs['setup_complexity'] == True:
            mat_mat_complexity.__detailed__ = True
        del kwargs['setup_complexity']

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
    #   - max_level_iterations may be reasonable to change per level

    # Dictionary for complexity tracking on each level
    complexity = []
    for i in range(0,max_levels):
        complexity.append( {'RAP': 0.0,
                            'aggregation': 0.0,
                            'strength': 0.0,
                            'candidates': 0.0,
                            'test_solve': 0.0,
                            'global_ritz': 0.0,
                            'local_ritz': 0.0,
                            'smooth_P': 0.0} )

    # Call recursive adaptive process starting from finest grid, level 0,
    # to construct adaptive hierarchy. 
    hierarchy = multilevel_solver(levels=levels)
    try_solve(A=A, levels=levels, level=0, symmetry=symmetry,
              strength=strength, aggregate=aggregate, smooth=smooth,
              presmoother=presmoother, postsmoother=postsmoother,
              improvement_iters=improvement_iters,
              max_coarse=max_coarse, max_levels=max_levels,
              target_convergence=target_convergence, max_bullets=max_bullets,
              max_bad_guys=max_bad_guys, num_targets=num_targets, 
              max_level_iterations=max_level_iterations,
              weak_tol=weak_tol, diagonal_dominance=diagonal_dominance,
              coarse_solver=coarse_solver, cycle=cycle,
              verbose=verbose, keep=keep, hierarchy=hierarchy,
              complexity=complexity, B=B)

    # Add complexity dictionary to levels in hierarchy 
    nlvls = len(hierarchy.levels)
    for i in range(0,nlvls):

        # Scale complexity on level i to be WUs w.r.t. the operator
        # on level i for consistency with other methods. Rescaled to
        # total WUs w.r.t. the fine grid in multilevel.py.
        temp = float(levels[0].A.nnz) / levels[i].A.nnz 
        for method in complexity[i]:
            complexity[i][method] *= temp

        hierarchy.levels[i].complexity = complexity[i]

    # TODO : This needs to be added to SC somewhere
    extra_work = 0.0
    for i in range(nlvls,max_levels):
        for method, cost in complexity[i].iteritems():
            extra_work += cost

    print("Extra setup cost on unaccounted for levels = ", extra_work)

    return hierarchy


# Weird bugs / Notes
#   - Filtering does not decrease complexity, omproves CF slightly?
#   - ^Same with local weighting
#   - No filter, local weighting seems best for TAD / SAD
#   - Don't think we need to symmetrically scale diagonal each iteration.
#     Lot of WUs, initial tests didn't suggest it improved convergene or
#     complexity enough to be worth the effort.
#   - Seems to like energy smoothing. Low degree, e.g. 1 still increases
#     complexity, but improves CF even more. 
#   - For TAD, aSA seems to like energy smoothing w/ degree 1.
#     E.g. for theta=3pi/16, n=1250, a modest increase in OC from 3 to 3.65
#     with energy smoothing bumps CF from ~0.55 --> 0.35. Increasing 
#     Jacobi degree to 2 was totally intractable moving OC to 6.9,
#     with CF still ~0.55
# ==> Approximate spectral radii should be for A^2!
#     Should also only do once, and do less work to get it (i.e. less exact, row sum?).

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
              max_bullets,
              max_bad_guys,
              num_targets,
              max_level_iterations,
              weak_tol,
              diagonal_dominance,
              coarse_solver,
              cycle,
              verbose,
              keep,
              hierarchy,
              complexity,
              B = None):

    """
    Needs some love.

    TODO : add SC for relaxation targets
           Check on SC for small eigenvalue problems -- really 9n^3??

           ----> BAD GUYS DO NOT APPEAR TO BE A-ORTHOGONAL COMING OUT OF GRITZ???

    """

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

    # Leading constant in approximation properties
    #   TODO : Approximate this in a cheaper way?
    sap_tol = (weak_tol / approximate_spectral_radius(A) )**2

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

    # Leading constant for complexity w.r.t. the finest grid
    chi = float(A.nnz) / levels[0].A.nnz

    # Generate initial targets as random vectors relaxed on AB = 0.
    if B == None:
        current.B = my_rand(n,num_targets, zero_crossings=False)
    else:
        current.B = B

    # TODO : This needs some work, turns out changing kwargs changes
    # internal argument. Currently reset to original value after improving...
    fn, kwargs = unpack_arg(presmoother, cost=False)
    try:
        temp = kwargs['iterations']
    except:
        temp = 1
    kwargs['iterations'] = improvement_iters
    if fn is None:
        raise ValueError("Must improve candidates for aSA.")

    b = np.zeros((A.shape[0], 1), dtype=A.dtype)
    current.B = relaxation_as_linear_operator( \
                            (fn, kwargs), \
                            current.A, b) * current.B

    kwargs['iterations'] = temp
    complexity[level]['candidates'] += improvement_iters * B.shape[1]

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
        C = current.A.tocsr()
    else:
        raise ValueError('unrecognized strength of connection method: %s' %
                         str(fn))

    complexity[level]['strength'] += kwargs['cost'][0] * chi

     # Avoid coarsening diagonally dominant rows
    flag, kwargs = unpack_arg(diagonal_dominance)
    if flag:
        C = eliminate_diag_dom_nodes(current.A, C, **kwargs)
        complexity[level]['strength'] += kwargs['cost'][0] * chi

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

    complexity[level]['aggregation'] += kwargs['cost'][0] * (float(C.nnz)/levels[0].A.nnz)

    # Loop over adaptive hierarchy until CF is sufficient or we have 
    # reached maximum iterations. Maximum iterations is checked inside
    # loop to prevent running test iterations that will not be used. 
    level_iter = 0
    conv_factor = 1
    target = None
    while (conv_factor > target_convergence):

        # Add new target. Orthogonalize using global / local Ritz and reconstruct T. 
        temp_cost = [0] 
        current.B = global_ritz_process(A=current.A, B1=current.B, B2=target, \
                                        sap_tol=sap_tol, level=level, \
                                        max_bad_guys=max_bad_guys, \
                                        verbose=verbose, cost=temp_cost)
        complexity[level]['global_ritz'] += temp_cost[0] * chi
        
        if level == 0:
            pdb.set_trace()

        temp_cost[0] = 0 
        current.T, per_agg = local_ritz_process(A=current.A, AggOp=current.AggOp, \
                                                B=current.B, sap_tol=sap_tol, \
                                                max_bullets=max_bullets, level=level, \
                                                verbose=verbose, cost=temp_cost)
        complexity[level]['local_ritz'] += temp_cost[0] * chi
        
        # Restrict bad guy using tentative prolongator
        Bc = current.T.T * current.B
        current.history['B'].append(current.B)
        current.history['agg'].append(per_agg)
        complexity[level]['smooth_P'] += float(current.T.nnz) / levels[0].A.nnz

        # Smooth tentative prolongator
        fn, kwargs = unpack_arg(smooth[level])
        if fn == 'jacobi':
            current.P = jacobi_prolongation_smoother(current.A, current.T, C, \
                                                     Bc, **kwargs)
        elif fn == 'richardson':
            current.P = richardson_prolongation_smoother(current.A, current.T, \
                                                         **kwargs)
        elif fn == 'energy':
            current.P = energy_prolongation_smoother(current.A, current.T, C, \
                                                     Bc, None, (False, {}), \
                                                     **kwargs)
        elif fn is None:
            current.P = T
        else:
            raise ValueError('unrecognized prolongation smoother method %s' %
                             str(fn))
        
        current.R = current.P.H
        complexity[level]['smooth_P'] += kwargs['cost'][0] * chi        

        # Form coarse grid operator, get complexity
        complexity[level]['RAP'] += mat_mat_complexity(current.R, current.A) / float(levels[0].A.nnz)
        RA = current.R * current.A
        complexity[level]['RAP'] += mat_mat_complexity(RA, current.P) / float(levels[0].A.nnz)
        Ac = csr_matrix(RA * current.P)      # Galerkin operator, Ac = RAP

        # Symmetrically scale diagonal of Ac, modify R,P accodingly
        #     TODO : Do we need to do this? 
        #             Add cost
        [D, Dinv, dum] = symmetric_rescaling(Ac, copy=False)
        current.P = (current.P * diags(Dinv, offsets=0)).tocsr()
        current.R = current.P.H
        for i in range(0,Bc.shape[1]):
            Bc[:,i] = D * Bc[:,i]

        # Recursively call try_solve() with coarse grid operator
        try_solve(A=Ac, levels=levels, level=(level+1), symmetry=symmetry,
                  strength=strength, aggregate=aggregate, smooth=smooth,
                  presmoother=presmoother, postsmoother=postsmoother,
                  improvement_iters=improvement_iters,
                  max_coarse=max_coarse, max_levels=max_levels,
                  target_convergence=target_convergence, max_bullets=max_bullets,
                  max_bad_guys=max_bad_guys, num_targets=num_targets, 
                  max_level_iterations=max_level_iterations,
                  weak_tol=weak_tol, diagonal_dominance=diagonal_dominance,
                  coarse_solver=coarse_solver, cycle=cycle,
                  verbose=verbose, keep=keep, hierarchy=hierarchy,
                  complexity=complexity, B=Bc)

        # Break if this was last adaptive iteration to prevent computing
        # test iterations that won't be used to improve solver
        level_iter += 1
        if level_iter == max_level_iterations:
            break

        # Test convergence of new hierarchy
        residuals = []
        target = my_rand(n, 1, current.A.dtype)
        target = hierarchy.solve(b, x0=target, cycle=cycle, \
                                 maxiter=improvement_iters, \
                                 tol=1e-16, residuals=residuals,
                                 init_level=level, accel=None)
        conv_factor = residuals[-1] / residuals[-2]
        current.history['conv'].append(conv_factor)
        temp_CC = hierarchy.cycle_complexity(cycle=cycle, init_level=level,
                                             recompute=True)

        # Count WUs to run test iterations - note, the CC is in terms of
        # the fine grid operator, so we do not scale by chi = Ai.nnz / A0.nnz
        complexity[level]['test_solve'] += temp_CC * improvement_iters

        if verbose:
            print tabs(level), "Iter = ",level_iter,", CF = ", conv_factor

    return

