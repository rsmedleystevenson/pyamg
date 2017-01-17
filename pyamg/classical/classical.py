"""Classical AMG (Ruge-Stuben AMG)"""
from __future__ import absolute_import

__docformat__ = "restructuredtext en"

from warnings import warn
from scipy.sparse import csr_matrix, isspmatrix_csr, SparseEfficiencyWarning, block_diag
import numpy as np

from pyamg.multilevel import multilevel_solver
from pyamg.relaxation.smoothing import change_smoothers
from pyamg.strength import classical_strength_of_connection, \
    symmetric_strength_of_connection, evolution_strength_of_connection,\
    distance_strength_of_connection, energy_based_strength_of_connection,\
    algebraic_distance, affinity_distance
from pyamg.util.utils import mat_mat_complexity, unpack_arg, extract_diagonal_blocks
from .interpolate import direct_interpolation, standard_interpolation,\
     trivial_interpolation, approximate_ideal_restriction
from .split import *
from .cr import CR

__all__ = ['ruge_stuben_solver']

def ruge_stuben_solver(A,
                       strength=('classical', {'theta': 0.25 ,'do_amalgamation': False}),
                       CF='RS',
                       influence=None,
                       interp='standard',
                       restrict=None,
                       presmoother=('gauss_seidel', {'sweep': 'symmetric'}),
                       postsmoother=('gauss_seidel', {'sweep': 'symmetric'}),
                       max_levels=20, max_coarse=20, keep=False,
                       block_starts=None, **kwargs):
    """Create a multilevel solver using Classical AMG (Ruge-Stuben AMG)

    Parameters
    ----------
    A : csr_matrix
        Square matrix in CSR format
    strength : ['symmetric', 'classical', 'evolution', 'distance',
                'algebraic_distance','affinity', 'energy_based', None]
        Method used to determine the strength of connection between unknowns
        of the linear system.  Method-specific parameters may be passed in
        using a tuple, e.g. strength=('symmetric',{'theta' : 0.25 }). If
        strength=None, all nonzero entries of the matrix are considered strong.
    CF : {string} : default 'RS'
        Method used for coarse grid selection (C/F splitting)
        Supported methods are RS, PMIS, PMISc, CLJP, CLJPc, and CR.
    influence: {np array size of num dofs} : default is None
        If set, this adds influence to the lambda values of points for RS coarsening
        This makes points with high influence values more likely to become C points
    interp : {string} : default 'standard'
        Use direct or standard interpolation.
    restrict : {string} : default None
        Optional flag to use R != P^T. only option is 'air'.
    presmoother : {string or dict}
        Method used for presmoothing at each level.  Method-specific parameters
        may be passed in using a tuple, e.g.
        presmoother=('gauss_seidel',{'sweep':'symmetric}), the default.
    postsmoother : {string or dict}
        Postsmoothing method with the same usage as presmoother
    max_levels: {integer} : default 10
        Maximum number of levels to be used in the multilevel solver.
    max_coarse: {integer} : default 500
        Maximum number of variables permitted on the coarse grid.
    keep: {bool} : default False
        Flag to indicate keeping extra operators in the hierarchy for
        diagnostics.  For example, if True, then strength of connection (C) and
        tentative prolongation (T) are kept.
    block_starts: list of integers
        If non-trivial, list of starting row indices of blocks of A if A represents a system
        (used for unknown-based approach for systems).

    Returns
    -------
    ml : multilevel_solver
        Multigrid hierarchy of matrices and prolongation operators

    Other Parameters
    ----------------
    cycle_type : ['V','W','F']
        Structrure of multigrid cycle
    coarse_solver : ['splu', 'lu', 'cholesky, 'pinv', 'gauss_seidel', ... ]
        Solver used at the coarsest level of the MG hierarchy.
            Optionally, may be a tuple (fn, args), where fn is a string such as
        ['splu', 'lu', ...] or a callable function, and args is a dictionary of
        arguments to be passed to fn.
    setup_complexity : bool
        For a detailed, more accurate setup complexity, pass in 
        'setup_complexity' = True. This will slow down performance, but
        increase accuracy of complexity count. 

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg import ruge_stuben_solver
    >>> A = poisson((10,),format='csr')
    >>> ml = ruge_stuben_solver(A,max_coarse=3)

    Notes
    -----

    Standard interpolation is generally considered more robust than
    direct, but direct is the currently the default until our new 
    implementation of standard has been more rigorously tested.

    "coarse_solver" is an optional argument and is the solver used at the
    coarsest grid.  The default is a pseudo-inverse.  Most simply,
    coarse_solver can be one of ['splu', 'lu', 'cholesky, 'pinv',
    'gauss_seidel', ... ].  Additionally, coarse_solver may be a tuple
    (fn, args), where fn is a string such as ['splu', 'lu', ...] or a callable
    function, and args is a dictionary of arguments to be passed to fn.


    References
    ----------
    .. [1] Trottenberg, U., Oosterlee, C. W., and Schuller, A.,
       "Multigrid" San Diego: Academic Press, 2001.  Appendix A

    See Also
    --------
    aggregation.smoothed_aggregation_solver, multilevel_solver,
    aggregation.rootnode_solver

    """

    if ('setup_complexity' in kwargs):
        if kwargs['setup_complexity'] == True:
            mat_mat_complexity.__detailed__ = True
        del kwargs['setup_complexity']

    # convert A to csr
    if not ( isspmatrix_csr(A) ):
        try:
            A = csr_matrix(A)
            warn("Implicit conversion of A to CSR",
                 SparseEfficiencyWarning)
        except:
            raise TypeError('Argument A must have type csr_matrix, \
                             or be convertible to csr_matrix')
    # preprocess A
    A = A.asfptype()
    if A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')

    levels = [multilevel_solver.level()]
    levels[-1].A = A
    levels[-1].block_starts = block_starts
    levels[-1].influence = influence

    while len(levels) < max_levels and levels[-1].A.shape[0] > max_coarse:
        extend_hierarchy(levels, strength, CF, interp, restrict, keep)

    ml = multilevel_solver(levels, **kwargs)
    change_smoothers(ml, presmoother, postsmoother)
    return ml


# internal function
def extend_hierarchy(levels, strength, CF, interp, restrict, keep):
    """ helper function for local methods """

    A = levels[-1].A
    block_starts = levels[-1].block_starts
    influence = levels[-1].influence

    # If this is a system, apply the unknown approach by coarsening and
    # generating interpolation based on each diagonal block of A
    if (block_starts):
        A_diag = extract_diagonal_blocks(A, block_starts)
    else:
        A_diag = [A]

    # Zero initial complexities for strength, splitting and interpolation
    levels[-1].complexity['CF'] = 0.0
    levels[-1].complexity['strength'] = 0.0
    levels[-1].complexity['interpolate'] = 0.0

    # Empty arrays to store operators for each diagonal block of A
    C_diag = []
    P_diag = []
    R_diag = []
    splitting = []
    next_lvl_block_starts = [0]
    block_cnt = 0

    # Form interpolation for each diagonal block in A (nodal AMG interpolation)
    for mat in A_diag:

        # Compute the strength-of-connection matrix C, where larger
        # C[i,j] denote stronger couplings between i and j.
        fn, kwargs = unpack_arg(strength)
        if fn == 'symmetric':
            C_diag.append( symmetric_strength_of_connection(mat, **kwargs) )
        elif fn == 'classical':
            C_diag.append( classical_strength_of_connection(mat, **kwargs) )
        elif fn == 'distance':
            C_diag.append( distance_strength_of_connection(mat, **kwargs) )
        elif (fn == 'ode') or (fn == 'evolution'):
            C_diag.append( evolution_strength_of_connection(mat, **kwargs) )
        elif fn == 'energy_based':
            C_diag.append( energy_based_strength_of_connection(mat, **kwargs) )
        elif fn == 'algebraic_distance':
            C_diag.append( algebraic_distance(mat, **kwargs) )
        elif fn == 'affinity':
            C_diag.append( affinity_distance(mat, **kwargs) )
        elif fn is None:
            C_diag.append( mat )
        else:
            raise ValueError('unrecognized strength of connection method: %s' %
                             str(fn))
        levels[-1].complexity['strength'] += kwargs['cost'][0] * mat.nnz / float(A.nnz)

        # Generate the C/F splitting
        fn, kwargs = unpack_arg(CF)
        if fn == 'RS':
            splitting.append( RS(C_diag[-1], influence, **kwargs) )
        elif fn == 'PMIS':
            splitting.append( PMIS(C_diag[-1], **kwargs) )
        elif fn == 'PMISc':
            splitting.append( PMISc(C_diag[-1], **kwargs) )
        elif fn == 'CLJP':
            splitting.append( CLJP(C_diag[-1], **kwargs) )
        elif fn == 'CLJPc':
            splitting.append( CLJPc(C_diag[-1], **kwargs) )
        elif fn == 'CR':
            splitting.append( CR(C_diag[-1], **kwargs) )
        elif fn == 'weighted_matching':
            sp, soc = weighted_matching(C_diag[-1], **kwargs)
            splitting.append(sp)
            if soc is not None:
                C_diag[-1] = soc
        else:
            raise ValueError('unknown C/F splitting method (%s)' % CF)
        levels[-1].complexity['CF'] += kwargs['cost'][0] * C_diag[-1].nnz / float(A.nnz)

        # Generate the interpolation matrix that maps from the coarse-grid to the
        # fine-grid
        fn, kwargs = unpack_arg(interp)
        if fn == 'standard':
            P_diag.append( standard_interpolation(mat, C_diag[-1], splitting[-1], **kwargs) )
        elif fn == 'direct':
            P_diag.append( direct_interpolation(mat, C_diag[-1], splitting[-1], **kwargs) )
        elif fn == 'trivial':
            P_diag.append( trivial_interpolation(mat, splitting[-1], **kwargs) )
        else:
            raise ValueError('unknown interpolation method (%s)' % interp)
        levels[-1].complexity['interpolate'] += kwargs['cost'][0] * mat.nnz / float(A.nnz)

        next_lvl_block_starts.append( next_lvl_block_starts[-1] + P_diag[-1].shape[1])
        block_cnt = block_cnt + 1

        # Build restriction operator
        fn, kwargs = unpack_arg(restrict)
        if fn is None:
            R_diag.append(P_diag[-1].T.tocsr())
        elif fn == 'air':
            R_diag.append( approximate_ideal_restriction(mat, C_diag[-1], splitting[-1], **kwargs) )
        else:
            raise ValueError('unknown restriction method (%s)' % restrict)


    # Build P to be block diagonal and R = P^T.
    P = block_diag(P_diag, format='csr')
    R = block_diag(R_diag, format='csr')



    Fpts = np.array(np.where(splitting[0] == 0)[0], dtype='int32')
    Aff =  A[Fpts,:][:,Fpts]
    import pdb
    pdb.set_trace()


    # Store relevant information for this level
    splitting = np.concatenate(splitting)
    if keep:
        C = block_diag(C_diag)
        levels[-1].C = C                  # strength of connection matrix

    levels[-1].P = P                  # prolongation operator
    levels[-1].R = R                  # restriction operator
    levels[-1].splitting = splitting  # C/F splitting

    # Form coarse grid operator, get complexity
    levels[-1].complexity['RAP'] = mat_mat_complexity(R,A) / float(A.nnz)
    RA = R * A
    levels[-1].complexity['RAP'] += mat_mat_complexity(RA,P) / float(A.nnz)
    A = RA * P      # Galerkin operator, Ac = RAP

    levels.append(multilevel_solver.level())
    levels[-1].A = A

    # Store influence and block starts on next level
    if (influence != None):
        I = (R==1).astype('intc')
        levels[-1].influence = I*influence
    else:
        levels[-1].influence = None
    if (block_starts):
        levels[-1].block_starts = next_lvl_block_starts
    else:
        levels[-1].block_starts = None

