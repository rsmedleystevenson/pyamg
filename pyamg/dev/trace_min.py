"""Support for aggregation-based AMG"""

__docformat__ = "restructuredtext en"

import pdb

import numpy as np
from warnings import warn
from scipy.sparse import csr_matrix, bsr_matrix, isspmatrix_csr, \
    isspmatrix_bsr, identity, SparseEfficiencyWarning, diags

from pyamg.multilevel import multilevel_solver
from pyamg.relaxation.smoothing import change_smoothers
from pyamg.util.utils import relaxation_as_linear_operator,\
    symmetric_rescaling, eliminate_diag_dom_nodes, blocksize, \
    levelize_strength_or_aggregation, mat_mat_complexity, \
    levelize_smooth_or_improve_candidates
from pyamg.strength import classical_strength_of_connection,\
    symmetric_strength_of_connection, evolution_strength_of_connection,\
    energy_based_strength_of_connection, distance_strength_of_connection,\
    algebraic_distance
from aggregate import standard_aggregation, naive_aggregation, \
    lloyd_aggregation

from pyamg.classical.split import RS, PMIS, PMISc, MIS, CLJP, CLJPc
from pyamg.classical.cr import CR
from tentative import ben_ideal_interpolation


__all__ = ['ben_ideal_solver']


def trace_min_solver(A, B=None, BH=None,
                    symmetry='hermitian',
                    strength='symmetric',
                    aggregate=None,
                    splitting='RS',
                    presmoother=('block_gauss_seidel',
                                 {'sweep': 'symmetric'}),
                    postsmoother=('block_gauss_seidel',
                                  {'sweep': 'symmetric'}),
                    improve_candidates=('block_gauss_seidel',
                                        {'sweep': 'symmetric',
                                         'iterations': 4}),
                    trace_min={'deg': 1, 'max_iter': 100,
                               'tol': 1e-8, 'debug': False}
                    max_levels = 10, max_coarse = 10,
                    diagonal_dominance=False,
                    keep=False, **kwargs):
    """




    Parameters
    ----------
    A : {csr_matrix, bsr_matrix}
        Sparse NxN matrix in CSR or BSR format
    B : {None, array_like}
        Right near-nullspace candidates stored in the columns of an NxK array.
        K must be >= the blocksize of A (see reference [2]). The default value
        B=None is equivalent to choosing the constant over each block-variable,
        B=np.kron(np.ones((A.shape[0]/blocksize(A), 1)), np.eye(blocksize(A)))
    BH : {None, array_like}
        Left near-nullspace candidates stored in the columns of an NxK array.
        BH is only used if symmetry is 'nonsymmetric'.  K must be >= the
        blocksize of A (see reference [2]). The default value B=None is
        equivalent to choosing the constant over each block-variable,
        B=np.kron(np.ones((A.shape[0]/blocksize(A), 1)), np.eye(blocksize(A)))
    symmetry : {string}
        'symmetric' refers to both real and complex symmetric
        'hermitian' refers to both complex Hermitian and real Hermitian
        'nonsymmetric' i.e. nonsymmetric in a hermitian sense
        Note that for the strictly real case, symmetric and hermitian are
        the same
        Note that this flag does not denote definiteness of the operator.
    strength : {list} : default
        ['symmetric', 'classical', 'evolution', ('predefined',
                                                 {'C' : csr_matrix}), None]
        Method used to determine the strength of connection between unknowns of
        the linear system.  Method-specific parameters may be passed in using a
        tuple, e.g. strength=('symmetric',{'theta' : 0.25 }). If strength=None,
        all nonzero entries of the matrix are considered strong.
        See notes below for varying this parameter on a per level basis.  Also,
        see notes below for using a predefined strength matrix on each level.
    aggregate : {list} : default ['standard', 'lloyd', 'naive', 'pairwise',
                ('predefined', {'AggOp' : csr_matrix})]
        Method used to aggregate nodes.  See notes below for varying this
        parameter on a per level basis.  Also, see notes below for using a
        predefined aggregation on each level. Method-specific parameters may be
        passed in using a tuple, e.g. aggregate=('pairwise',{'num_matchings': 2 })
    splitting : {list} : 


    presmoother : {tuple, string, list} : default ('block_gauss_seidel',
                                                   {'sweep':'symmetric'})
        Defines the presmoother for the multilevel cycling.  The default block
        Gauss-Seidel option defaults to point-wise Gauss-Seidel, if the matrix
        is CSR or is a BSR matrix with blocksize of 1.  See notes below for
        varying this parameter on a per level basis.
    postsmoother : {tuple, string, list}
        Same as presmoother, except defines the postsmoother.
    improve_candidates : {tuple, string, list} : default
                         [('block_gauss_seidel',
                          {'sweep': 'symmetric', 'iterations': 4}), None]
        The ith entry defines the method used to improve the candidates B on
        level i.  If the list is shorter than max_levels, then the last entry
        will define the method for all levels lower.  If tuple or string, then
        this single relaxation descriptor defines improve_candidates on all
        levels.
        The list elements are relaxation descriptors of the form used for
        presmoother and postsmoother.  A value of None implies no action on B.
    max_levels : {integer} : default 10
        Maximum number of levels to be used in the multilevel solver.
    max_coarse : {integer} : default 500
        Maximum number of variables permitted on the coarse grid.
    diagonal_dominance : {bool, tuple} : default False
        If True (or the first tuple entry is True), then avoid coarsening
        diagonally dominant rows.  The second tuple entry requires a
        dictionary, where the key value 'theta' is used to tune the diagonal
        dominance threshold.
    keep : {bool} : default False
        Flag to indicate keeping extra operators in the hierarchy for
        diagnostics.  For example, if True, then strength of connection (C),
        tentative prolongation (T), aggregation (AggOp), and arrays
        storing the C-points (Cpts) and F-points (Fpts) are kept at
        each level.

    Other Parameters
    ----------------
    cycle_type : ['V','W','F']
        Structrure of multigrid cycle
    coarse_solver : ['splu', 'lu', 'cholesky, 'pinv', 'gauss_seidel', ... ]
        Solver used at the coarsest level of the MG hierarchy.
            Optionally, may be a tuple (fn, args), where fn is a string such as
        ['splu', 'lu', ...] or a callable function, and args is a dictionary of
        arguments to be passed to fn.

    Returns
    -------
    ml : multilevel_solver
        Multigrid hierarchy of matrices and prolongation operators

    See Also
    --------
    multilevel_solver, aggregation.smoothed_aggregation_solver,
    aggregation.rootnode_solver, classical.ruge_stuben_solver

    Notes
    -----

    Examples
    --------

    References
    ----------

    """

    if ('setup_complexity' in kwargs):
        if kwargs['setup_complexity'] == True:
            mat_mat_complexity.__detailed__ = True
        del kwargs['setup_complexity']

    if not (isspmatrix_csr(A) or isspmatrix_bsr(A)):
        try:
            A = csr_matrix(A)
            warn("Implicit conversion of A to CSR",
                 SparseEfficiencyWarning)
        except:
            raise TypeError('Argument A must have type csr_matrix, \
                             bsr_matrix, or be convertible to csr_matrix')

    A = A.asfptype()

    if (symmetry != 'symmetric') and (symmetry != 'hermitian') and \
            (symmetry != 'nonsymmetric'):
        raise ValueError('expected \'symmetric\', \'nonsymmetric\' \
                          or \'hermitian\' for the symmetry parameter ')
    A.symmetry = symmetry

    if A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')
    # Right near nullspace candidates use constant for each variable as default
    if B is None:
        B = np.kron(np.ones((A.shape[0]/blocksize(A), 1), dtype=A.dtype),
                    np.eye(blocksize(A)))
    else:
        B = np.asarray(B, dtype=A.dtype)
        if len(B.shape) == 1:
            B = B.reshape(-1, 1)
        if B.shape[0] != A.shape[0]:
            raise ValueError('The near null-space modes B have incorrect \
                              dimensions for matrix A')
        if B.shape[1] < blocksize(A):
            raise ValueError('B.shape[1] must be >= the blocksize of A')

    # Left near nullspace candidates
    if A.symmetry == 'nonsymmetric':
        if BH is None:
            BH = B.copy()
        else:
            BH = np.asarray(BH, dtype=A.dtype)
            if len(BH.shape) == 1:
                BH = BH.reshape(-1, 1)
            if BH.shape[1] != B.shape[1]:
                raise ValueError('The number of left and right near \
                                  null-space modes B and BH, must be equal')
            if BH.shape[0] != A.shape[0]:
                raise ValueError('The near null-space modes BH have \
                                  incorrect dimensions for matrix A')

    # Levelize the user parameters, so that they become lists describing the
    # desired user option on each level.
    max_levels, max_coarse, strength =\
        levelize_strength_or_aggregation(strength, max_levels, max_coarse)
    max_levels, max_coarse, aggregate =\
        levelize_strength_or_aggregation(aggregate, max_levels, max_coarse)
    improve_candidates =\
        levelize_smooth_or_improve_candidates(improve_candidates, max_levels)

    # Construct multilevel structure
    levels = []
    levels.append(multilevel_solver.level())
    levels[-1].A = A          # matrix

    # Append near nullspace candidates
    levels[-1].B = B          # right candidates
    if A.symmetry == 'nonsymmetric':
        levels[-1].BH = BH    # left candidates

    while len(levels) < max_levels and \
            levels[-1].A.shape[0]/blocksize(levels[-1].A) > max_coarse:
        extend_hierarchy(levels, strength, aggregate, splitting,
                    improve_candidates, diagonal_dominance, keep)

    ml = multilevel_solver(levels, **kwargs)
    change_smoothers(ml, presmoother, postsmoother)
    return ml


def extend_hierarchy(levels, strength, aggregate, splitting,
                     improve_candidates, diagonal_dominance, keep):
    """Service routine to implement the strength of connection, aggregation,
    tentative prolongation construction, and prolongation smoothing.  Called by
    smoothed_aggregation_solver.
    """

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

    A = levels[-1].A
    B = levels[-1].B
    if A.symmetry == "nonsymmetric":
        AH = A.H.asformat(A.format)
        BH = levels[-1].BH

    # Improve near nullspace candidates by relaxing on A B = 0
    temp_cost = [0.0]
    fn, kwargs = unpack_arg(improve_candidates[len(levels)-1], cost=False)
    if fn is not None:
        b = np.zeros((A.shape[0], 1), dtype=A.dtype)
        B = relaxation_as_linear_operator((fn, kwargs), A, b, temp_cost) * B
        levels[-1].B = B
        if A.symmetry == "nonsymmetric":
            BH = relaxation_as_linear_operator((fn, kwargs), AH, b, temp_cost) * BH
            levels[-1].BH = BH

    levels[-1].complexity['candidates'] = temp_cost[0] * B.shape[1]

    # Compute the strength-of-connection matrix C, where larger
    # C[i, j] denote stronger couplings between i and j.
    fn, kwargs = unpack_arg(strength[len(levels)-1])
    if fn == 'symmetric':
        C = symmetric_strength_of_connection(A, **kwargs)
    elif fn == 'classical':
        C = classical_strength_of_connection(A, **kwargs)
    elif fn == 'distance':
        C = distance_strength_of_connection(A, **kwargs)
    elif (fn == 'ode') or (fn == 'evolution'):
        if 'B' in kwargs:
            C = evolution_strength_of_connection(A, **kwargs)
        else:
            C = evolution_strength_of_connection(A, B, **kwargs)
    elif fn == 'energy_based':
        C = energy_based_strength_of_connection(A, **kwargs)
    elif fn == 'predefined':
        C = kwargs['C'].tocsr()
    elif fn == 'algebraic_distance':
        C = algebraic_distance(A, **kwargs)
    elif fn is None:
        C = A.tocsr()
    else:
        raise ValueError('unrecognized strength of connection method: %s' %
                         str(fn))

    levels[-1].complexity['strength'] = kwargs['cost'][0]

    # Avoid coarsening diagonally dominant rows
    flag, kwargs = unpack_arg(diagonal_dominance)
    if flag:
        C = eliminate_diag_dom_nodes(A, C, **kwargs)
        levels[-1].complexity['diag_dom'] = kwargs['cost'][0]

    # Compute the aggregation matrix AggOp (i.e., the nodal coarsening of A).
    # AggOp is a boolean matrix, where the sparsity pattern for the k-th column
    # denotes the fine-grid nodes agglomerated into k-th coarse-grid node.
    fn, kwargs = unpack_arg(aggregate[len(levels)-1])
    if fn == 'standard':
        AggOp, Cnodes = standard_aggregation(C, **kwargs)
    elif fn == 'naive':
        AggOp, Cnodes = naive_aggregation(C, **kwargs)
    elif fn == 'lloyd':
        AggOp, Cnodes = lloyd_aggregation(C, **kwargs)
    elif fn == 'pairwise':
        AggOp, Cnodes = pairwise_aggregation(A, B, **kwargs)
    elif fn == 'predefined':
        AggOp = kwargs['AggOp'].tocsr()
        Cnodes = kwargs['Cnodes']
    elif fn == None:
        AggOp = None
    else:
        raise ValueError('unrecognized aggregation method %s' % str(fn))

    levels[-1].complexity['aggregation'] = kwargs['cost'][0] * (float(C.nnz)/A.nnz)

    # Check for CF-splitting to generate C-points. Must use either a
    # CF-splitting or aggregation routine. If both provided, Aggregation
    # routine used to generate sparsity and C-points taken from CF-splitting.
    # 
    # TODO : levelize splitting in here and in classical.py
    #
    # fn, kwargs = unpack_arg(splitting[len(levels)-1])
    fn, kwargs = unpack_arg(splitting)
    if fn == 'RS':
        splitting = RS(C, **kwargs)
    elif fn == 'PMIS':
        splitting = PMIS(C, **kwargs)
    elif fn == 'PMISc':
        splitting = PMISc(C, **kwargs)
    elif fn == 'CLJP':
        splitting = CLJP(C, **kwargs)
    elif fn == 'CLJPc':
        splitting = CLJPc(C, **kwargs)
    elif fn == 'CR':
        splitting = CR(C, **kwargs)
    elif fn == None and AggOp == None:
        raise ValueError('Must provide either aggregation routine ' \
                         'or CF splitting routine.')
    elif fn == None:
        # Ensure C-points are sorted
        splitting = np.zeros((A.shape[0],), dtype='intc')
        splitting[Cnodes] = 1
    else:
        raise ValueError('unknown C/F splitting method (%s)' % splitting)
    
    Cnodes = np.array(np.where(splitting == 1)[0], dtype='intc')
    Fnodes = np.array(np.where(splitting == 0)[0], dtype='intc')
    levels[-1].complexity['CF'] = kwargs['cost'][0]

    # Compute prolongation operator.
    P, Bc = trace_min(A=A, B=B, SOC=C, Cpts=Cnodes, 
                      Fpts=Fnodes, T=AggOp, **trace_min)

    # Compute the restriction matrix R, which interpolates from the fine-grid
    # to the coarse-grid.  If A is nonsymmetric, then R must be constructed
    # based on A.H.  Otherwise R = P.H or P.T.
    symmetry = A.symmetry
    if symmetry == 'hermitian':
        R = P.H
    elif symmetry == 'symmetric':
        R = P.T
    elif symmetry == 'nonsymmetric':
        raise TypeError('Trace-min not implemented for non-symmetric matrices.')

    # Form coarse grid operator, get complexity
    levels[-1].complexity['RAP'] = mat_mat_complexity(R,A) / float(A.nnz)
    RA = R * A
    levels[-1].complexity['RAP'] += mat_mat_complexity(RA,P) / float(A.nnz)
    A = csr_matrix(RA * P)      # Galerkin operator, Ac = RAP
    A.symmetry = symmetry

    if keep:
        levels[-1].C = C                        # strength of connection matrix
        levels[-1].AggOp = AggOp                # aggregation operator

    levels[-1].P = P                            # smoothed prolongator
    levels[-1].R = R                            # restriction operator
    levels[-1].Cpts = Cnodes                    # Cpts (i.e., rootnodes)

    # Add new level to hierarchy
    levels.append(multilevel_solver.level())
    A.symmetry = symmetry
    levels[-1].A = A
    levels[-1].B = Bc                   # right near nullspace candidates

    if A.symmetry == "nonsymmetric":
        levels[-1].BH = BHc                     # left near nullspace candidates
