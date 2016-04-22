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
from pyamg.relaxation.smoothing import change_smoothers, rho_block_D_inv_A, rho_D_inv_A
from pyamg.util.linalg import norm, approximate_spectral_radius
from pyamg.util.utils import levelize_strength_or_aggregation, levelize_smooth_or_improve_candidates, \
                            symmetric_rescaling
from pyamg.aggregation.aggregation import smoothed_aggregation_solver
from pyamg.aggregation.aggregate import standard_aggregation
from pyamg.aggregation.smooth import jacobi_prolongation_smoother, energy_prolongation_smoother, \
                   richardson_prolongation_smoother
from pyamg.aggregation.tentative import fit_candidates


__all__ = ['A_norm', 'my_rand', 'tl_sa_solver']

def tabs(level):
    return '\t'*level

def unpack_arg(v):
    """Helper function for local methods"""
    if isinstance(v,tuple):
        return v[0],v[1]
    else:
        return v,{}

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

def relaxation_as_linear_operator(A, smoother):
    """
    Input: relaxation descriptor in smoother and a sparse matrix A

    Output: a linear operator that when applied to a vector x
            carries out the relaxation method described by smoother.
            In general, this operator will act like (I - M^{-1} A) x.
    """

    # TODO: Currently there is some imitation, in the case of Richardson and
    #       Jacobi, of the parameters that the multilevel solver routines
    #       take. This is to make sure that the initial target is pinned
    #       appropriately. This way (duplicating initialization) is very ugly
    #       and error-prone. Needs to be done right as soon as possible!

    fn, kwargs = unpack_arg(smoother)
    if fn == 'gauss_seidel':
        def matvec(x):
            xcopy = x.copy()
            gauss_seidel(A, xcopy, numpy.zeros_like(xcopy), iterations=2, sweep='symmetric')
            return xcopy
    elif fn == 'gauss_seidel_ne':
        def matvec(x):
            xcopy = x.copy()
            gauss_seidel_ne(A, xcopy, numpy.zeros_like(xcopy), iterations=2, sweep='symmetric')
            return xcopy
    elif fn == 'gauss_seidel_nr':
        def matvec(x):
            xcopy = x.copy()
            gauss_seidel_nr(A, xcopy, numpy.zeros_like(xcopy), iterations=2, sweep='symmetric')
            return xcopy
    elif fn == 'jacobi':
        withrho = True
        omega = 1.0
        if 'withrho' in kwargs:
            withrho = kwargs['withrho']
        if 'omega' in kwargs:
            omega = kwargs['omega']
        if withrho:
            omega /= rho_D_inv_A(A)
        def matvec(x):
            xcopy = x.copy()
            jacobi(A, xcopy, numpy.zeros_like(xcopy), iterations=2, omega=omega)
            return xcopy
    elif fn == 'richardson':
        omega = 1.0
        if 'omega' in kwargs:
            omega = kwargs['omega']
        omega /= approximate_spectral_radius(A)
        def matvec(x):
            xcopy=x.copy()
            polynomial(A, xcopy, numpy.zeros_like(xcopy), iterations=2, coefficients=[omega])
            return xcopy
    else:
        raise TypeError('Unrecognized smoother')

    return LinearOperator(A.shape, matvec, dtype=A.dtype)

def solver_as_homogeneous_lin_op(solver):
    """ Return a linear operator based on solver that executes
        one iteration of solving A x = 0 """
    A = solver.levels[0].A

    def solver_matvec(vec):
        vec = solver.solve(numpy.zeros_like(vec), x0=vec,
                           tol=float(numpy.finfo(numpy.float).tiny), maxiter=1)
        return vec

    return LinearOperator(A.shape, solver_matvec, dtype=A.dtype)

def relaxation_vectors(relax, x, num_iters, normalize=False):
    """
    Run relaxation num_iters times with initial guess x.

    Parameters
    ---------
    relax : {linear operator}
        Carries out relaxation for some  A x = 0
    x : {array}
        n x 1 with 1 initial guess
    num_iters : {int}
        Number of relaxation steps (i.e., vectors) to generate

    Returns
    -------
    Set of num_iters+1 vectors, [x, relax x, relax^2 x, ...]
    """
    nvecs = x.shape[1]
    if normalize:
        x /= norm(x, 'inf')

    for i in range(num_iters):
        x = numpy.hstack((x, relax*x[:,-nvecs:]))
        if normalize:
            x[:,i+1] /= norm(x[:,i+1], 'inf')

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

def asa_solver(A, initial_targets=None, max_targets=100, min_targets=0,
        num_initial_targets=1, targets_iters=15, conv_tol=0.5, weak_tol=15.,
        local_weak_tol=15., max_coarse=1000, coarse_size=1000, max_levels=20,
        max_level_iterations=10, prepostsmoother='richardson', smooth='jacobi',
        strength='symmetric', aggregate='standard', coarse_solver='pinv2',
        verbose=False, keep=True):
    """
    Create a two-level solver using Adaptive Smoothed Aggregation (aSA)

    Parameters
    ----------
    A : {csr_matrix, bsr_matrix}
        Square matrix in CSR or BSR format
    initial_targets : {None, n x m dense matrix}
        If a matrix, then this forms the basis for the first m targets.
        Also in this case, the initial setup stage is skipped, because this
        provides the first target(s).  If None, then a random initial guess
        and relaxation are used to inform the initial target.
    max_targets : {integer}
        Maximum number of near-nullspace targets to generate
    min_targets : {integer}
        Minimum number of near-nullspace targets to generate
    num_initial_targets : {integer}
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
    coarse_size : {integer}
        Size below which a direct solver should be used.
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

    # Initializations
    work = numpy.zeros((1,))

    try_solve(A, levels, 0, max_targets, min_targets, num_initial_targets, targets_iters, conv_tol, weak_tol, local_weak_tol, max_coarse, coarse_size, smooth, max_levels, max_level_iterations, coarse_solver, work, verbose, prepostsmoother)

    return [multilevel_solver(levels, coarse_solver=coarse_solver), work]

def tl_initial_target(A, num_targets, targets_iters, prepostsmoother, work):
    """
    Compute an initial target.

    Parameters
    ----------
    See tl_sa_solver(...) documentation.

    Returns
    -------
    The initial target and an estimate of the asymptotic convergence factor.
    """

    ts = None
    factors = []
    for i in range(num_targets):
        x = my_rand(A.shape[0], 1, A.dtype)
        relax = relaxation_as_linear_operator(A, prepostsmoother)
        work[:] += A.nnz*targets_iters*2
        X = relaxation_vectors(relax, x, targets_iters, normalize=False)
        conv_factor = A_norm(X[:, -1], A) / \
                      A_norm(X[:, -2], A)
        x = X[:, -1].reshape(-1, 1)
        del X
        if ts is None:
            ts = x
        else:
            ts = numpy.hstack((ts, x))
        factors.append(conv_factor)

    return ts, sum(factors)/len(factors)

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

# Approximate the convergence factor of a multilevel solver starting at a given level
def test_level_conv(levels, level, cycle, iters, coarse_solver, prepostsmoother):
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

# Adds a new target to the existing targets. Returns new targets, tentative prolongator
def add_target(A, AggOp, B, t, weak_tol, local_weak_tol, level, verbose):
    B_new = global_ritz_process(A, B, t, weak_tol, level, verbose)
    T_new, per_agg = local_ritz_process(A, AggOp, B_new, local_weak_tol, level, verbose)
    return B_new, T_new, per_agg

# TODO: this needs to be refactored into a much cleaner recursive code
def try_solve(A, levels, level, max_targets, min_targets, num_initial_targets,
        targets_iters, conv_tol, weak_tol, local_weak_tol, max_coarse,
        coarse_size, smooth, max_levels, max_level_iterations, coarse_solver,
        work, verbose, prepostsmoother):
    # try:
    #     smallest = linalg.eigsh(levels[level].A, k=1, which='SM')[0]
    #     print smallest
    #     largest = linalg.eigsh(levels[level].A, k=1, which='LM')[0]
    #     print largest
    #     print "------ Condition number",largest/smallest,"level", level
    # except:
    #     pass
    cycle = "V"

    # initialize current level
    if level >= len(levels):
        levels.append(multilevel_solver.level())
    else:
        levels[level] = multilevel_solver.level()
        # TODO: remove levels below us?
        while len(levels) > level+1:
            levels.pop()
    current = levels[level]
    current.A = A

    # Test if we are at the coarsest level
    if current.A.shape[0] <= coarse_size or level >= max_levels - 1:
        # coarse solver should solve exactly i.e. conv factor = 0
        return
    else:
        # initialize history
        current.history = {}
        current.history['B'] = []
        current.history['conv'] = []
        current.history['agg'] = []

        # generate initial targets
        current.B, _ = tl_initial_target(current.A, num_initial_targets, \
                                               targets_iters, prepostsmoother, work)

        # find T and AggOp
        # TODO: this should use specified strength of connection, aggregation, and smoother
        C = symmetric_strength_of_connection(current.A)
        AggOp = standard_aggregation(C)[0]
        current.B, current.T, per_agg = \
                add_target(current.A, AggOp, current.B, None, \
                           weak_tol, local_weak_tol, level, verbose)
        current.AggOp = AggOp
        current.history['B'].append(current.B)
        current.history['agg'].append(per_agg)

        # TODO: what happens if smoothing is good enough?
        count = 0
        factor = 1
        while factor > conv_tol and count < max_level_iterations:
            if count >= max_level_iterations:
                if verbose:
                    print tabs(level), "Too many aSA iterations on level", level, "Stopping"
                return

            if current.B.shape[1] >= max_targets:
                if verbose:
                    print tabs(level), "Too many targets on level", level, "stopping"
                return

            # smooth tentative prolongator
            # TODO: use specified smoother
            current.P = richardson_prolongation_smoother(current.A, current.T, omega=1)
            current.R = current.P.H

            # create coarse A
            Ac = (current.R * current.A * current.P).tocsr()

            # Symmetrically scale out diagonal of A
            # TODO: move out of loop?
            [dum, Dinv, dum] = symmetric_rescaling(Ac, copy=False)
            current.P = (current.P * diags(Dinv,offsets=0)).tocsr()
            current.R = current.P.H

            # recurse
            try_solve(Ac, levels, level+1, max_targets, min_targets, \
                    num_initial_targets, targets_iters, conv_tol, weak_tol, \
                    local_weak_tol, max_coarse, coarse_size, smooth, \
                    max_levels, max_level_iterations, coarse_solver, work, \
                    verbose, prepostsmoother)
            # see if new coarse level lets us converge fast enough
            t, factor = test_level_conv(levels, level, cycle, targets_iters, \
                    coarse_solver, prepostsmoother)
            if verbose:
                print
                print tabs(level), "Convergence factor:", factor

            current.history['conv'].append(factor)

            if factor < conv_tol:
                # coarse grid is good, we are done
                return

            if verbose:
                print tabs(level), "Convergence tolerance not reached, adding additional target"
            # coarse grid is slow, add another target
            current.B, current.T, per_agg = \
                    add_target(current.A, current.AggOp, \
                               current.B, t, weak_tol, local_weak_tol, level, verbose)
            count += 1
            current.history['B'].append(current.B)
            current.history['agg'].append(per_agg)
            # try coarse solve again

# Approximate the convergence factor of using this solver
def test_solver(solver):
    relax = solver_as_homogeneous_lin_op(solver)
    x = my_rand(solver.levels[0].A.shape[0], 1, solver.levels[0].A.dtype)
    V = relaxation_vectors(relax, x, targets_iters, normalize=False)
    # TODO: better estimate of convergence?
    conv_factor = A_norm(V[:, -1], solver.levels[0].A) / \
                  A_norm(V[:, -2], solver.levels[0].A)
    return conv_factor
