"""Adaptive Smoothed Aggregation"""

__docformat__ = "restructuredtext en"

import pdb

# TODO : Be consistent with scipy importing. Remove unnecessary imports. 

import copy
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg as linalg
import scipy.linalg
from scipy.sparse import bsr_matrix, isspmatrix_csr, isspmatrix_bsr, eye, csr_matrix, diags

from pyamg.multilevel import multilevel_solver, coarse_grid_solver
from pyamg.strength import symmetric_strength_of_connection,\
                        classical_strength_of_connection, evolution_strength_of_connection
from pyamg.relaxation.smoothing import change_smoothers
from pyamg.util.linalg import norm, approximate_spectral_radius
from pyamg.util.utils import levelize_strength_or_aggregation, levelize_smooth_or_improve_candidates, \
                            symmetric_rescaling, relaxation_as_linear_operator, mat_mat_complexity, \
                            blocksize
from .aggregate import standard_aggregation, naive_aggregation,\
    lloyd_aggregation
from pyamg.aggregation.smooth import jacobi_prolongation_smoother, energy_prolongation_smoother, \
                   richardson_prolongation_smoother
from .tentative import fit_candidates
from pyamg.aggregation.aggregation import smoothed_aggregation_solver


__all__ = ['A_norm', 'my_rand', 'asa_solver']


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

        # Get eigenvalue decomposition of Ba^T*Ba, sort eigenvalues
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
               max_bullets=5,
               max_bad_guys=10,
               num_targets=1,
               max_iterations=10,
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
    max_iterations : {integer}
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
    >>> residuals=[]
    >>> x=sa.solve(b=np.ones((A.shape[0],)),x0=np.ones((A.shape[0],)),residuals=residuals)
    """

    if ('setup_complexity' in kwargs):
        if kwargs['setup_complexity'] == True:
            mat_mat_complexity.__detailed__ = True
        del kwargs['setup_complexity']

    if not (isspmatrix_csr(A) or isspmatrix_bsr(A)):
        try:
            print 'Implicit conversion of A to CSR.'
            A = csr_matrix(A)
        except:
            raise TypeError('Argument A must have type csr_matrix or bsr_matrix, ' \
                            'or be convertible to csr_matrix.')

    A = A.asfptype()
    if A.shape[0] != A.shape[1]:
        raise ValueError('Expected square matrix!')

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

    # Right near nullspace candidates use constant for each variable as default
    if B is None:
        B = np.kron(np.ones((int(A.shape[0]/blocksize(A)), 1), dtype=A.dtype),
                    np.eye(blocksize(A)))
    else:
        B = np.asarray(B, dtype=A.dtype)
        if len(B.shape) == 1:
            B = B.reshape(-1, 1)
        if B.shape[0] != A.shape[0]:
            raise ValueError('The near null-space modes B have incorrect \
                              dimensions for matrix A')
        if B.shape[1] < blocksize(A):
            warn('Having less target vectors, B.shape[1], than \
                  blocksize of A can degrade convergence factors.')

    # Make improve candidates dictionary based on presmoother
    # and improvement iters.
    improve_candidates = copy.deepcopy(presmoother)
    if type(improve_candidates) is list:
        improve_candidates = improve_candidates[0]
        if len(improve_candidates) == 1:
            improve_candidates.append({'iterations': improvement_iters})
        else:
            improve_candidates[1]['iterations'] = improvement_iters
        improve_candidates = tuple(improve_candidates)
    elif type(improve_candidates) is tuple:
        improve_candidates = list(improve_candidates)
        if len(improve_candidates) == 1:
            improve_candidates.append({'iterations': improvement_iters})
        else:
            improve_candidates[1]['iterations'] = improvement_iters
        improve_candidates = tuple(improve_candidates)
    elif type(improve_candidates) is str:
        improve_candidates = (improve_candidates, {'iterations': improvement_iters})
    else:
        raise ValueError("Presmoother must be string, tuple or list.")

    # Build SA solver based on initial bad guys
    #   TODO - Keep T in SA w/o keeping C and AggOp
    ml = smoothed_aggregation_solver(A, B=B, BH=None, symmetry=symmetry, strength=strength,
                                     aggregate=aggregate, smooth=smooth, presmoother=presmoother,
                                     postsmoother=postsmoother, improve_candidates=improve_candidates,
                                     max_levels = max_levels, max_coarse = max_coarse,
                                     diagonal_dominance=diagonal_dominance,
                                     keep=True, **kwargs)

    # Loop over adaptive hierarchy until CF is sufficient or we have reached
    # maximum iterations. Note, iterations and convergence checked inside
    # loop to prevent running test iterations that will not be used, and for
    # simple / readable code (no large block of code before loop). 
    level_iter = 0
    conv_factor = 1
    sap_tol = (weak_tol / approximate_spectral_radius(A) )**2
    while True:

        # Generate random new target
        target = my_rand(A.shape[0], 1, A.dtype)

        # Improve target in energy by relaxing on A B = 0
        temp_cost = [0.0]
        fn, sm_args = unpack_arg(improve_candidates, cost=False)
        if fn is not None:
            b = np.zeros((A.shape[0], 1), dtype=A.dtype)
            B = relaxation_as_linear_operator((fn, sm_args), A, b, temp_cost) * B
            if A.symmetry == "nonsymmetric":
                BH = relaxation_as_linear_operator((fn, sm_args), AH, b, temp_cost) * BH

        complexity[0]['candidates'] = temp_cost[0] * B.shape[1]

        # Test solver on new target
        residuals = []
        ml.solve(b, x0=target, cycle=cycle, maxiter=improvement_iters, \
                 tol=1e-16, residuals=residuals, accel=None)
        temp_CC = ml.cycle_complexity()
        complexity[0]['test_solve'] += temp_CC * improvement_iters

        # TODO - Need to estimate CF better
        conv_factor = residuals[-1] / residuals[-2]
        print "Iteration ",level_iter,", CF = ",conv_factor,", ",B.shape[1]," targets."
        level_iter += 1

        # Check if good convergence achieved or maximum iterations done
        if (level_iter > max_iterations) or (conv_factor < target_convergence):
            break

        # Interpolate targets in hierarchy up from coarsest grid,
        # form set of bad guys
        for i in range(len(ml.levels)-2,-1,-1):
            # B2 = ml.levels[i].P * ml.levels[i+1].B
            B2 = ml.levels[i].T * ml.levels[i+1].B

        # TODO account for complexity here

        # Store bad guys as one vector
        # -- It seems that stacking the new guys and the old guys
        #    is wrong, i.e. degrades convrergence. Unclear if replacing
        #    B with B2 can offer suffcient improvement to be worthwhile...
        if False:
            B = np.hstack((B, B2))
        elif True:
            B = B2

        # Add new target. Orthogonalize using global Ritz and reconstruct T. 
        #   TODO - worth keeping stuff that trivially satisfies?
        temp_cost = [0] 
        B = global_ritz_process(A=A, B1=B, B2=target, sap_tol=sap_tol, level=0, \
                                max_bad_guys=max_bad_guys, verbose=verbose, \
                                cost=temp_cost)
        complexity[0]['global_ritz'] += temp_cost[0]

        # TODO - put in Local Ritz! Should not be skipping this.


        # Build new hierarchy
        ml = smoothed_aggregation_solver(A, B=B[:,0:min(B.shape[1],max_bullets)], BH=None, symmetry=symmetry, strength=strength,
                                         aggregate=aggregate, smooth=smooth, presmoother=presmoother,
                                         postsmoother=postsmoother, improve_candidates=improve_candidates,
                                         max_levels = max_levels, max_coarse = max_coarse,
                                         diagonal_dominance=diagonal_dominance,
                                         keep=True, **kwargs)

        # TODO - Store complexity from building new hierarchy


    return ml


