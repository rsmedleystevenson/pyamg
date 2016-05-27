"""Basic Two-Level Adaptive Smoothed Aggregation"""

__docformat__ = "restructuredtext en"

import pdb

import time
import numpy
import scipy
import scipy.sparse
import scipy.sparse.linalg as linalg
import scipy.linalg
from scipy.sparse import bsr_matrix, isspmatrix_csr, isspmatrix_bsr, eye, csr_matrix, diags
from scipy.sparse.linalg.isolve.utils import make_system
from scipy.sparse.linalg.interface import LinearOperator
from scipy import stats

from pyamg.multilevel import multilevel_solver, coarse_grid_solver
from pyamg.util.linalg import residual_norm
from pyamg.relaxation.relaxation import gauss_seidel, gauss_seidel_nr, gauss_seidel_ne, \
                                        gauss_seidel_indexed, jacobi, polynomial
from pyamg.strength import symmetric_strength_of_connection, ode_strength_of_connection
from pyamg.relaxation.smoothing import change_smoothers
from pyamg.util.linalg import norm, approximate_spectral_radius
from pyamg.util.utils import levelize_strength_or_aggregation, levelize_smooth_or_improve_candidates, \
                            symmetric_rescaling, relaxation_as_linear_operator
from pyamg.aggregation.aggregation import smoothed_aggregation_solver
from pyamg.aggregation.aggregate import standard_aggregation
from pyamg.aggregation.smooth import jacobi_prolongation_smoother, energy_prolongation_smoother, \
                   richardson_prolongation_smoother
from pyamg.aggregation.tentative import fit_candidates


__all__ = ['A_norm', 'my_rand', 'tl_sa_solver', 'asa_solver']

def tabs(level):
    return '\t'*level


def A_norm(x, A):
    """
    Calculate A-norm of x
    """
    x = numpy.ravel(x)
    return numpy.sqrt(scipy.dot(x.conjugate(), A*x))

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
    # x = numpy.ones((d1,d2))

    return x

def find_slope(y):
    """
    Assume that y are the y-values for a line, find its slope, assuming equal
    spacing on the x-axis
    """
    y = numpy.ravel(y)
    npts = y.shape[0]
    x = numpy.arange(1, npts + 1)
    A = numpy.zeros((npts, 2))
    A[:,0] = 1.0
    A[:,1] = x
    return scipy.linalg.lstsq(A,y)[0][1]


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

    if B2 is not None:
        B = numpy.hstack((B1, B2.reshape(-1,1)))
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
        V = numpy.real(V)
    except:
        import pdb; pdb.set_trace()

    # Compute Ritz vectors and normalize them in energy. Also, mark vectors
    # that trivially satisfy the weak approximation property.
    V = scipy.dot(Q, V)
    num_candidates = -1
    entire_const = weak_tol / approximate_spectral_radius(A)
    if verbose:
        print
        print tabs(level), "WAP const", entire_const
    for j in range(V.shape[1]):
        V[:,j] /= numpy.sqrt(E[j])
        # verify energy norm is 1
        # print tabs(level), "&&&&&&&&&&&&&&&&&&&&", scipy.dot(A*V[:,j],V[:,j])
        if verbose:
            print tabs(level), "Vector 1/e", j, 1./E[j], "ELIMINATED" if 1./E[j] <= entire_const else ""
        if 1./E[j] <= entire_const:
            num_candidates = j
            break

    if num_candidates == 0:
        num_candidates = 1
    if num_candidates == -1:
        num_candidates = V.shape[1]

    if verbose:
        # print tabs(level), "Finished global ritz process, eliminated", B.shape[1]-num_candidates, "candidates", num_candidates, ". candidates remaining"
        print

    return V[:, :num_candidates]

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

    AggOpCsc = AggOp.tocsc() # we are slicing columns, this makes it much faster

    # row, col, and val arrays to store entries of T
    max_size = B.shape[1]*AggOp.getnnz()
    row_i = numpy.empty(max_size)
    col_i = numpy.empty(max_size)
    val_i = numpy.empty(max_size)
    cur_col = 0
    index = 0

    # store how many basis functions we keep per aggregate
    per_agg_count = numpy.zeros((B.shape[0], 1))

    # iterate over aggregates
    for i in range(AggOpCsc.shape[1]):
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

    if verbose:
        print tabs(level), "Eliminated %d local vectors (%.2f%% remaining nonzeros)" % \
                (B.shape[1]*AggOp.shape[1] - cur_col, index*100.0/max_size)
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
               prepostsmoother='richardson',
               conv_tol=0.5,
               max_coarse=20,
               max_levels=20,
               max_targets=100,
               min_targets=0,
               num_targets=1,
               targets_iters=15,
               max_level_iterations=10,
               weak_tol=15.,
               local_weak_tol=15.,
               coarse_solver='pinv2',
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
    conv_tol : {float}
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
    >>> import numpy
    >>> A=stencil_grid([[-1,-1,-1],[-1,8.0,-1],[-1,-1,-1]], (31,31),format='csr')
    >>> sa = tl_sa_solver(A,num_targets=1)
    >>> residuals=[]
    >>> x=sa.solve(b=numpy.ones((A.shape[0],)),x0=numpy.ones((A.shape[0],)),residuals=residuals)
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

    # I think this is where I should levelize all of the parameter 
    # input, e.g. strength, smoothing, etc. 
    # ==> Ideally make into levelized list of function handles
    #     with arguments??



    # Call recursive adaptive process starting from finest grid, level 0,
    # to construct adaptive hierarchy. 
    try_solve(A=A, levels=levels, level=0, symmetry=symmetry,
              strength=strength, aggregate=aggregate, smooth=smooth,
              prepostsmoother=prepostsmoother, conv_tol=conv_tol,
              max_coarse=max_coarse, max_levels=max_levels,
              max_targets=max_targets, min_targets=min_targets,
              num_targets=num_targets, targets_iters=targets_iters,
              max_level_iterations=max_level_iterations, weak_tol=weak_tol,
              local_weak_tol=local_weak_tol, coarse_solver=coarse_solver,
              verbose=verbose, keep=keep)


    return [multilevel_solver(levels, coarse_solver=coarse_solver), work]

def get_targets(A, num_targets, targets_iters, prepostsmoother):
    """
    Compute an initial target.

    Parameters
    ----------
    See tl_sa_solver(...) documentation.

    Returns
    -------
    The initial target and an estimate of the asymptotic convergence factor.
    """

# fn, kwargs = unpack_arg(improve_candidates[len(levels)-1])
# if fn is not None:
#     b = np.zeros((A.shape[0], 1), dtype=A.dtype)
#     B = relaxation_as_linear_operator((fn, kwargs), A, b) * B

# TODO --> This is slow. Preallocate np.array() of size for num_targets,
# and fill in loop.

    targets = None
    factors = []
    for i in range(num_targets):
        x = my_rand(A.shape[0], 1, A.dtype)
        relax = relaxation_as_linear_operator(A, prepostsmoother)
        conv_factor = A_norm(X[:, -1], A) / \
                      A_norm(X[:, -2], A)
        x = X[:, -1].reshape(-1, 1)
        del X
        if targets is None:
            targets = x
        else:
            targets = numpy.hstack((targets, x))
        factors.append(conv_factor)

    return targets, sum(factors)/len(factors)

def rayleigh(A, x):
    """
    Compute the Rayleigh quotient of a matrix and a vector.

    Parameters
    ----------
    A: {csr_matrix}
    B: {vector}

    Returns
    -------
    The Rayleigh quotient
    """
    # TODO: actual matrix norm?
    return scipy.dot((A*x).conjugate(), x) / \
           (csr_matrix.sum(abs(A.getrow(1000)))*scipy.dot(x.conjugate(), x))


def test_level_conv(levels, level, cycle, iters, coarse_solver, prepostsmoother):  
    """
    Approximate the convergence factor of a multilevel solver starting at a given level

    """

    size = levels[level].A.shape[0]
    dtype = levels[level].A.dtype
    x = my_rand(size, 1, dtype)
    b = numpy.zeros_like(x)

    lvls = levels[level:]
    ml = multilevel_solver(lvls, coarse_solver=coarse_solver)
    change_smoothers(ml, prepostsmoother, prepostsmoother)

    residuals = []
    # TODO: what should tolerence be?
    x = ml.solve(b, x0=x, cycle=cycle, maxiter=iters, tol=1e-16, residuals=residuals)
    # TODO: be smarter about approximating the convergence factor
    return x, residuals[-1] / residuals[-2]


# TODO: this needs to be refactored into a much cleaner recursive code
def try_solve(A, levels,
              level,
              symmetry,
              strength,
              aggregate,
              smooth,
              prepostsmoother,
              conv_tol,
              max_coarse,
              max_levels,
              max_targets,
              min_targets,
              num_targets,
              targets_iters,
              max_level_iterations,
              weak_tol,
              local_weak_tol,
              coarse_solver,
              verbose,
              keep):

    """
    Needs some love.
    """

    cycle = "V"

    # If coarserer hierarchies have already been defined, remove
    # because they will be reconstructed
    #
    #   - Eventually may want to keep one multilevel solver
    #     object and add / remove hierarchies instead of keeping
    #     a list and reconstructing the hierarchy when necessary...
    #
    while len(levels) > level:
        levels.pop()

    # Add new level to hierarchy, define reference 'current' to current level
    levels.append(multilevel_solver.level())
    current = levels[level]
    current.A = A

    # Test if we are at the coarsest level
    #
    #   - What do we do if we are? How does this play out in the recusive stuff?
    #     ==> First time this happens, maybe construct/return multilevel object?
    #
    if current.A.shape[0] <= max_coarse or level >= max_levels - 1:
        return

    # initialize history
    current.history = {}
    current.history['B'] = []
    current.history['conv'] = []
    current.history['agg'] = []

    # Generate initial targets
    #
    #   - Need to adress the possibility that relaxation is sufficient.
    #     ==> What do we do then? Set relaxation as coarse solver?
    #
    current.B, relax_CF = get_targets(current.A, num_targets, \
                               targets_iters, prepostsmoother )

    # Get SOC matrix and aggregation
    #
    #   - Should probably pass a function handle to avoid 20 if-statements picking
    #     the SOC and aggregation routine every recursive call. Looks bad.
    #
    C = symmetric_strength_of_connection(current.A)
    current.AggOp = standard_aggregation(C)[0]

    level_iter = 0
    conv_factor = 1
    target = None
    while conv_factor > conv_tol and level_iter < max_level_iterations:

        # Add new target. Orthogonalize using global / local Ritz and reconstruct T.  
        current.B = global_ritz_process(A=current.A, B1=current.B, B2=target,
                                        weak_tol=weak_tol, level=level,
                                        verbose=verbose)
        current.T, per_agg = local_ritz_process(A=current.A, AggOp=current.AggOp,
                                                B=current.B, weak_tol=local_weak_tol,
                                                level=level, verbose=verbose)
        current.history['B'].append(current.B)
        current.history['agg'].append(per_agg)

        # Smooth tentative prolongator
        #
        #   - Need to use arbitrary smoother here.
        #     ==> again with possible pass in function handle to try_solve()?
        #
        current.P = richardson_prolongation_smoother(current.A, current.T, omega=1)
        current.R = current.P.H

        # Construct coarse grid
        Ac = (current.R * current.A * current.P).tocsr()

        # Symmetrically scale diagonal of A
        #
        #   - Make sure this is doing the right thing
        #
        [dum, Dinv, dum] = symmetric_rescaling(Ac, copy=False)
        current.P = (current.P * diags(Dinv, offsets=0)).tocsr()
        current.R = current.P.H

        # Recursively call try_solve() with coarse grid operator
        try_solve(A=Ac, levels=levels, level=(level+1), symmetry=symmetry,
                  strength=strength, aggregate=aggregate, smooth=smooth,
                  prepostsmoother=prepostsmoother, conv_tol=conv_tol,
                  max_coarse=max_coarse, max_levels=max_levels,
                  max_targets=max_targets, min_targets=min_targets,
                  num_targets=num_targets, targets_iters=targets_iters,
                  max_level_iterations=max_level_iterations, weak_tol=weak_tol,
                  local_weak_tol=local_weak_tol, coarse_solver=coarse_solver,
                  verbose=verbose, keep=keep)
        level_iter += 1

        # Test convergence of new hierarchy
        target, conv_factor = test_level_conv(levels, level, cycle, targets_iters, \
                                        coarse_solver, prepostsmoother)
        current.history['conv'].append(conv_factor)

        if verbose:
            print tabs(level), "Iter = ",level_iter,", num targets = ",num_targets,", CF = ", conv_factor

    return

