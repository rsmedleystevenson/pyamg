"""Classical AMG (Ruge-Stuben AMG)"""
from __future__ import absolute_import

__docformat__ = "restructuredtext en"

from warnings import warn
from scipy.sparse import csr_matrix, isspmatrix_csr, isspmatrix_bsr, \
    SparseEfficiencyWarning
from pyamg.multilevel import multilevel_solver
from pyamg.relaxation.smoothing import change_smoothers
from pyamg.strength import classical_strength_of_connection, \
    symmetric_strength_of_connection, evolution_strength_of_connection,\
    distance_strength_of_connection, energy_based_strength_of_connection,\
    algebraic_distance, affinity_distance
from pyamg.util.utils import mat_mat_complexity, unpack_arg

from .interpolate import direct_interpolation, standard_interpolation, \
    distance_two_interpolation, one_point_interpolation, \
    injection_interpolation
from pyamg.classical.split import *
from .cr import CR

__all__ = ['ruge_stuben_solver']


def ruge_stuben_solver(A,
                       strength=('classical', {'theta': 0.25}),
                       CF='RS',
                       interpolation='direct',
                       restriction='galerkin',
                       presmoother=('gauss_seidel', {'sweep': 'symmetric'}),
                       postsmoother=('gauss_seidel', {'sweep': 'symmetric'}),
                       max_levels=10, max_coarse=10, keep=False, **kwargs):
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
    interpolation : {string} : default 'direct'
        Method for interpolation. Options include 'direct', 'standard', 'injection',
        'one_point', and 'distance_two'.
    restriction : {string or dict} : default 'galerkin'
        'Galerkin' means set R := P^T for a Galerkin coarse-grid operator. Can also specify
        an interpolation method as above, to build the restriciton operator based on A^T. 
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

    # Convert A to csr
    if not (isspmatrix_csr(A) or isspmatrix_bsr(A)):
        try:
            A = csr_matrix(A)
            warn("Implicit conversion of A to CSR", SparseEfficiencyWarning)
        except:
            raise TypeError('Argument A must have type csr_matrix, bsr_matrix, \
                             or be convertible to csr_matrix')
    
    # if isspmatrix_bsr(A):
    #     warn("Classical AMG is often more effective on CSR matrices.")

    # preprocess A
    A = A.asfptype()
    if A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')

    levels = [multilevel_solver.level()]
    levels[-1].A = A

    while len(levels) < max_levels and levels[-1].A.shape[0] > max_coarse:
        extend_hierarchy(levels, strength, CF, interpolation, restriction, keep)

    ml = multilevel_solver(levels, **kwargs)
    change_smoothers(ml, presmoother, postsmoother)
    return ml


# internal function
def extend_hierarchy(levels, strength, CF, interpolation, restriction, keep):
    """ helper function for local methods """

    A = levels[-1].A

    # Compute the strength-of-connection matrix C, where larger
    # C[i,j] denote stronger couplings between i and j.
    fn, kwargs = unpack_arg(strength)
    if fn == 'symmetric':
        C = symmetric_strength_of_connection(A, **kwargs)
    elif fn == 'classical':
        C = classical_strength_of_connection(A, **kwargs)
    elif fn == 'distance':
        C = distance_strength_of_connection(A, **kwargs)
    elif (fn == 'ode') or (fn == 'evolution'):
        C = evolution_strength_of_connection(A, **kwargs)
    elif fn == 'energy_based':
        C = energy_based_strength_of_connection(A, **kwargs)
    elif fn == 'algebraic_distance':
        C = algebraic_distance(A, **kwargs)
    elif fn == 'affinity':
        C = affinity_distance(A, **kwargs)
    elif fn is None:
        C = A
    else:
        raise ValueError('unrecognized strength of connection method: %s' %
                         str(fn))

    levels[-1].complexity['strength'] = kwargs['cost'][0]

    # Generate the C/F splitting
    fn, kwargs = unpack_arg(CF)
    if fn == 'RS':
        splitting = split.RS(C, **kwargs)
    elif fn == 'PMIS':
        splitting = split.PMIS(C, **kwargs)
    elif fn == 'PMISc':
        splitting = split.PMISc(C, **kwargs)
    elif fn == 'CLJP':
        splitting = split.CLJP(C, **kwargs)
    elif fn == 'CLJPc':
        splitting = split.CLJPc(C, **kwargs)
    elif fn == 'CR':
        splitting = CR(C, **kwargs)
    elif fn == 'weighted_matching':
        splitting, soc = weighted_matching(C, **kwargs)
        if soc is not None:
            C = soc
    else:
        raise ValueError('unknown C/F splitting method (%s)' % CF)

    levels[-1].complexity['CF'] = kwargs['cost'][0]

    # Generate the interpolation matrix that maps from the coarse-grid to the
    # fine-grid
    fn, kwargs = unpack_arg(interpolation)
    if fn == 'standard':
        P = standard_interpolation(A, C, splitting, **kwargs)
    elif fn == 'distance_two':
        P = distance_two_interpolation(A, C, splitting, **kwargs)
    elif fn == 'direct':
        P = direct_interpolation(A, C, splitting, **kwargs)
    elif fn == 'one_point':
        P = one_point_interpolation(A, C, splitting, **kwargs)
    elif fn == 'injection':
        P = injection_interpolation(A, splitting, **kwargs)
    else:
        raise ValueError('unknown interpolation method (%s)' % interpolation)
    levels[-1].complexity['interpolate'] = kwargs['cost'][0]

    # Generate the restriction matrix that maps from the fine-grid to the
    # coarse-grid. Must make sure transpose matrices remain in CSR or BSR
    fn, kwargs = unpack_arg(restriction)
    if isspmatrix_csr(A):
        if restriction == 'galerkin':
            R = P.T.tocsr()
        elif fn == 'standard':
            temp_A = A.T.tocsr()
            temp_C = C.T.tocsr()
            R = standard_interpolation(temp_A, temp_C, splitting, **kwargs)
            R = R.T.tocsr()
        elif fn == 'distance_two':
            temp_A = A.T.tocsr()
            temp_C = C.T.tocsr()
            R = distance_two_interpolation(temp_A, temp_C, splitting, **kwargs)
            R = R.T.tocsr()
        elif fn == 'direct':
            temp_A = A.T.tocsr()
            temp_C = C.T.tocsr()
            R = direct_interpolation(temp_A, temp_C, splitting, **kwargs)
            R = R.T.tocsr()
        elif fn == 'one_point':         # Don't need A^T here
            temp_C = C.T.tocsr()
            R = one_point_interpolation(A, temp_C, splitting, **kwargs)
            R = R.T.tocsr()
        elif fn == 'injection':         # Don't need A^T or C^T here
            R = injection_interpolation(A, splitting, **kwargs)
            R = R.T.tocsr()
        else:
            raise ValueError('unknown interpolation method (%s)' % interpolation)
    else: 
        if restriction == 'galerkin':
            R = P.T.tobsr()
        elif fn == 'standard':
            temp_A = A.T.tobsr()
            temp_C = C.T.tocsr()
            R = standard_interpolation(temp_A, temp_C, splitting, **kwargs)
            R = R.T.tobsr()
        elif fn == 'distance_two':
            temp_A = A.T.tobsr()
            temp_C = C.T.tocsr()
            R = distance_two_interpolation(temp_A, temp_C, splitting, **kwargs)
            R = R.T.tobsr()
        elif fn == 'direct':
            temp_A = A.T.tobsr()
            temp_C = C.T.tocsr()
            R = direct_interpolation(temp_A, temp_C, splitting, **kwargs)
            R = R.T.tobsr()
        elif fn == 'one_point':         # Don't need A^T here
            temp_C = C.T.tocsr()
            R = one_point_interpolation(A, temp_C, splitting, **kwargs)
            R = R.T.tobsr()
        elif fn == 'injection':         # Don't need A^T or C^T here
            R = injection_interpolation(A, splitting, **kwargs)
            R = R.T.tobsr()
        else:
            raise ValueError('unknown interpolation method (%s)' % interpolation)
    
    levels[-1].complexity['restriction'] = kwargs['cost'][0]

    # Store relevant information for this level
    if keep:
        levels[-1].C = C                  # strength of connection matrix

    levels[-1].P = P                  # prolongation operator
    levels[-1].R = R                  # restriction operator
    levels[-1].splitting = splitting  # C/F splitting

    # Form coarse grid operator, get complexity
    levels[-1].complexity['RAP'] = mat_mat_complexity(R,A) / float(A.nnz)
    RA = R * A
    levels[-1].complexity['RAP'] += mat_mat_complexity(RA,P) / float(A.nnz)
    A = RA * P      # Galerkin operator, Ac = RAP

    # Make sure coarse-grid operator is in correct sparse format
    if (isspmatrix_csr(P) and (not isspmatrix_csr(A))):
        A = A.tocsr()
    elif (isspmatrix_bsr(P) and (not isspmatrix_bsr(A))):
        A = A.tobsr()

    # Form next level through Galerkin product
    levels.append(multilevel_solver.level())
    levels[-1].A = A
