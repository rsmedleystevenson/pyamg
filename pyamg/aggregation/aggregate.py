"""Aggregation methods"""

__docformat__ = "restructuredtext en"

import pdb 

from warnings import warn
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, isspmatrix_csr, isspmatrix_csc,\
                        SparseEfficiencyWarning
from pyamg.util.utils import relaxation_as_linear_operator
from pyamg import amg_core
from pyamg.graph import lloyd_cluster
from matching import *
from copy import deepcopy

__all__ = ['standard_aggregation', 'naive_aggregation', 'lloyd_aggregation', 'pairwise_aggregation']


def standard_aggregation(C):
    """Compute the sparsity pattern of the tentative prolongator

    Parameters
    ----------
    C : csr_matrix
        strength of connection matrix

    Returns
    -------
    AggOp : csr_matrix
        aggregation operator which determines the sparsity pattern
        of the tentative prolongator
    Cpts : array
        array of Cpts, i.e., Cpts[i] = root node of aggregate i

    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> from pyamg.gallery import poisson
    >>> from pyamg.aggregation.aggregate import standard_aggregation
    >>> A = poisson((4,), format='csr')   # 1D mesh with 4 vertices
    >>> A.todense()
    matrix([[ 2., -1.,  0.,  0.],
            [-1.,  2., -1.,  0.],
            [ 0., -1.,  2., -1.],
            [ 0.,  0., -1.,  2.]])
    >>> standard_aggregation(A)[0].todense() # two aggregates
    matrix([[1, 0],
            [1, 0],
            [0, 1],
            [0, 1]], dtype=int8)
    >>> A = csr_matrix([[1,0,0],[0,1,1],[0,1,1]])
    >>> A.todense()                      # first vertex is isolated
    matrix([[1, 0, 0],
            [0, 1, 1],
            [0, 1, 1]])
    >>> standard_aggregation(A)[0].todense() # one aggregate
    matrix([[0],
            [1],
            [1]], dtype=int8)

    See Also
    --------
    amg_core.standard_aggregation

    """

    if not isspmatrix_csr(C):
        raise TypeError('expected csr_matrix')

    if C.shape[0] != C.shape[1]:
        raise ValueError('expected square matrix')

    index_type = C.indptr.dtype
    num_rows = C.shape[0]

    Tj = np.empty(num_rows, dtype=index_type)  # stores the aggregate #s
    Cpts = np.empty(num_rows, dtype=index_type)  # stores the Cpts

    fn = amg_core.standard_aggregation

    num_aggregates = fn(num_rows, C.indptr, C.indices, Tj, Cpts)
    Cpts = Cpts[:num_aggregates]

    if num_aggregates == 0:
        # return all zero matrix and no Cpts
        return csr_matrix((num_rows, 1), dtype='int8'),\
            np.array([], dtype=index_type)
    else:

        shape = (num_rows, num_aggregates)
        if Tj.min() == -1:
            # some nodes not aggregated
            mask = Tj != -1
            row = np.arange(num_rows, dtype=index_type)[mask]
            col = Tj[mask]
            data = np.ones(len(col), dtype='int8')
            return coo_matrix((data, (row, col)), shape=shape).tocsr(), Cpts
        else:
            # all nodes aggregated
            Tp = np.arange(num_rows+1, dtype=index_type)
            Tx = np.ones(len(Tj), dtype='int8')
            return csr_matrix((Tx, Tj, Tp), shape=shape), Cpts


def naive_aggregation(C):
    """Compute the sparsity pattern of the tentative prolongator

    Parameters
    ----------
    C : csr_matrix
        strength of connection matrix

    Returns
    -------
    AggOp : csr_matrix
        aggregation operator which determines the sparsity pattern
        of the tentative prolongator
    Cpts : array
        array of Cpts, i.e., Cpts[i] = root node of aggregate i

    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> from pyamg.gallery import poisson
    >>> from pyamg.aggregation.aggregate import naive_aggregation
    >>> A = poisson((4,), format='csr')   # 1D mesh with 4 vertices
    >>> A.todense()
    matrix([[ 2., -1.,  0.,  0.],
            [-1.,  2., -1.,  0.],
            [ 0., -1.,  2., -1.],
            [ 0.,  0., -1.,  2.]])
    >>> naive_aggregation(A)[0].todense() # two aggregates
    matrix([[1, 0],
            [1, 0],
            [0, 1],
            [0, 1]], dtype=int8)
    >>> A = csr_matrix([[1,0,0],[0,1,1],[0,1,1]])
    >>> A.todense()                      # first vertex is isolated
    matrix([[1, 0, 0],
            [0, 1, 1],
            [0, 1, 1]])
    >>> naive_aggregation(A)[0].todense() # two aggregates
    matrix([[1, 0],
            [0, 1],
            [0, 1]], dtype=int8)

    See Also
    --------
    amg_core.naive_aggregation

    Notes
    -----
    Differs from standard aggregation.  Each dof is considered.  If it has been
    aggregated, skip over.  Otherwise, put dof and any unaggregated neighbors
    in an aggregate.  Results in possibly much higher complexities than
    standard aggregation.
    """

    if not isspmatrix_csr(C):
        raise TypeError('expected csr_matrix')

    if C.shape[0] != C.shape[1]:
        raise ValueError('expected square matrix')

    index_type = C.indptr.dtype
    num_rows = C.shape[0]

    Tj = np.empty(num_rows, dtype=index_type)  # stores the aggregate #s
    Cpts = np.empty(num_rows, dtype=index_type)  # stores the Cpts

    fn = amg_core.naive_aggregation

    num_aggregates = fn(num_rows, C.indptr, C.indices, Tj, Cpts)
    Cpts = Cpts[:num_aggregates]
    Tj = Tj - 1

    if num_aggregates == 0:
        # all zero matrix
        return csr_matrix((num_rows, 1), dtype='int8'), Cpts
    else:
        shape = (num_rows, num_aggregates)
        # all nodes aggregated
        Tp = np.arange(num_rows+1, dtype=index_type)
        Tx = np.ones(len(Tj), dtype='int8')
        return csr_matrix((Tx, Tj, Tp), shape=shape), Cpts


def lloyd_aggregation(C, ratio=0.03, distance='unit', maxiter=10):
    """Aggregated nodes using Lloyd Clustering

    Parameters
    ----------
    C : csr_matrix
        strength of connection matrix
    ratio : scalar
        Fraction of the nodes which will be seeds.
    distance : ['unit','abs','inv',None]
        Distance assigned to each edge of the graph G used in Lloyd clustering

        For each nonzero value C[i,j]:

        =======  ===========================
        'unit'   G[i,j] = 1
        'abs'    G[i,j] = abs(C[i,j])
        'inv'    G[i,j] = 1.0/abs(C[i,j])
        'same'   G[i,j] = C[i,j]
        'sub'    G[i,j] = C[i,j] - min(C)
        =======  ===========================

    maxiter : int
        Maximum number of iterations to perform

    Returns
    -------
    AggOp : csr_matrix
        aggregation operator which determines the sparsity pattern
        of the tentative prolongator
    seeds : array
        array of Cpts, i.e., Cpts[i] = root node of aggregate i

    See Also
    --------
    amg_core.standard_aggregation

    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> from pyamg.gallery import poisson
    >>> from pyamg.aggregation.aggregate import lloyd_aggregation
    >>> A = poisson((4,), format='csr')   # 1D mesh with 4 vertices
    >>> A.todense()
    matrix([[ 2., -1.,  0.,  0.],
            [-1.,  2., -1.,  0.],
            [ 0., -1.,  2., -1.],
            [ 0.,  0., -1.,  2.]])
    >>> lloyd_aggregation(A)[0].todense() # one aggregate
    matrix([[1],
            [1],
            [1],
            [1]], dtype=int8)
    >>> # more seeding for two aggregates
    >>> Agg = lloyd_aggregation(A,ratio=0.5)[0].todense()
    """

    if ratio <= 0 or ratio > 1:
        raise ValueError('ratio must be > 0.0 and <= 1.0')

    if not (isspmatrix_csr(C) or isspmatrix_csc(C)):
        raise TypeError('expected csr_matrix or csc_matrix')

    if distance == 'unit':
        data = np.ones_like(C.data).astype(float)
    elif distance == 'abs':
        data = abs(C.data)
    elif distance == 'inv':
        data = 1.0/abs(C.data)
    elif distance is 'same':
        data = C.data
    elif distance is 'min':
        data = C.data - C.data.min()
    else:
        raise ValueError('unrecognized value distance=%s' % distance)

    if C.dtype == complex:
        data = np.real(data)

    assert(data.min() >= 0)

    G = C.__class__((data, C.indices, C.indptr), shape=C.shape)

    num_seeds = int(min(max(ratio * G.shape[0], 1), G.shape[0]))

    distances, clusters, seeds = lloyd_cluster(G, num_seeds, maxiter=maxiter)

    row = (clusters >= 0).nonzero()[0]
    col = clusters[row]
    data = np.ones(len(row), dtype='int8')
    AggOp = coo_matrix((data, (row, col)),
                       shape=(G.shape[0], num_seeds)).tocsr()
    return AggOp, seeds



def pairwise_aggregation(A, B, Bh=None, symmetry='hermitian',
                        algorithm='drake_C', matchings=1,
                        weights=None, improve_candidates=None, **kwargs):
    """ Pairwise aggregation of nodes. 

    Parameters
    ----------
    A : csr_matrix or bsr_matrix
        matrix for linear system.
    B : array_like
        Right near-nullspace candidates stored in the columns of an NxK array.
    BH : array_like : default None
        Left near-nullspace candidates stored in the columns of an NxK array.
        BH is only used if symmetry is 'nonsymmetric'.
        The default value B=None is equivalent to BH=B.copy()
    algorithm : string : default 'drake'
        Algorithm to perform pairwise matching. Current options are 
        'drake', 'preis', 'notay', referring to the Drake (2003), 
        Preis (1999), and Notay (2010), respectively. 
    matchings : int : default 1
        Number of pairwise matchings to do. k matchings will lead to 
        a coarsening factor of under 2^k.
    weights : function handle : default None
        Optional function handle to compute weights used in the matching,
        e.g. a strength of connection routine. Additional arguments for
        this routine should be provided in **kwargs. 
    improve_candidates : {tuple, string, list} : default None
        The list elements are relaxation descriptors of the form used for
        presmoother and postsmoother.  A value of None implies no action on B.

    THINGS TO NOTE
    --------------
        - Not implemented for non-symmetric and/or block systems
            + Need to set up pairwise aggregation to be applicable for 
              nonsymmetric matrices (think it actually is...) 
            + Need to define how a matching is done nodally.
            + Also must consider what targets are used to form coarse grid'
              in nodal approach...
        - Once we are done w/ Python implementations of matching, we can remove 
          the deepcopy of A to W --> Don't need it, waste of time/memory.  

    """

    def unpack_arg(v):
        if isinstance(v, tuple):
            return v[0], v[1]
        else:
            return v, {}

    if not isinstance(matchings, int):
        raise TypeError("Number of matchings must be an integer.")

    if matchings < 1:
        raise ValueError("Number of matchings must be > 0.")

    if (algorithm is not 'drake') and (algorithm is not 'preis') and \
       (algorithm is not 'notay') and (algorithm is not 'drake_C'):
       raise ValueError("Only drake, notay and preis algorithms implemeted.")

    if (symmetry != 'symmetric') and (symmetry != 'hermitian') and \
            (symmetry != 'nonsymmetric'):
        raise ValueError('expected \'symmetric\', \'nonsymmetric\' or\
                         \'hermitian\' for the symmetry parameter ')

    # Compute weights if function provided, otherwise let W = A
    if weights is not None:
        W = weights(A, **kwargs)
    else:
        W = deepcopy(A)

    if not isspmatrix_csr(W):
        warn("Requires CSR matrix - trying to convert.", SparseEfficiencyWarning)
        try:
            W = W.tocsr()
        except:
            raise TypeError("Could not convert to csr matrix.")

    n = A.shape[0]
    if (symmetry == 'nonsymmetric') and (Bh == None):
        print "Warning - no left near null-space vector provided for nonsymmetric matrix.\n\
        Copying right near null-space vector."
        Bh = deepcopy(B[0:n,0:1])

    # Dictionary of function names for matching algorithms 
    get_matching = {
        'drake': drake_matching_2003,
        'drake_C': drake_C,
        'preis': preis_matching_1999,
        'notay': notay_matching_2010
    }

    # Get initial matching
    [M,S] = get_matching[algorithm](W, order='backward', **kwargs)
    num_pairs = M.shape[0]
    num_sing = S.shape[0]
    Nc = num_pairs+num_sing
    # Pick C-points and save in list
    Cpts = np.zeros((Nc,),dtype=int)
    Cpts[0:num_pairs] = M[:,0]
    Cpts[num_pairs:Nc] = S

    # Form sparse P from pairwise aggregation
    row_inds = np.empty(n)
    row_inds[0:(2*num_pairs)] = M.flatten()
    row_inds[(2*num_pairs):n] = S
    col_inds = np.empty(n)
    col_inds[0:(2*num_pairs)] = ( np.array( ( np.arange(0,num_pairs),np.arange(0,num_pairs) ) ).T).flatten()
    col_inds[(2*num_pairs):n] = np.arange(num_pairs,Nc)
    AggOp = csr_matrix( (np.ones((n,), dtype=bool), (row_inds,col_inds)), shape=(n,Nc) )

    # If performing one matching, return P and list of C-points
    if matchings == 1:
        return AggOp, Cpts
    # If performing multiple pairwise matchings, form coarse grid operator
    # and repeat process
    else:
        P = csr_matrix( (B[0:n,0], (row_inds,col_inds)), shape=(n,Nc) )
        Bc = np.ones((Nc,1))
        if symmetry == 'hermitian':
            R = P.H
            Ac = R*A*P
        elif symmetry == 'symmetric':
            R = P.T            
            Ac = R*A*P
        elif symmetry == 'nonsymmetric':
            R = csr_matrix( (Bh[0:n,0], (col_inds,row_inds)), shape=(Nc,n) )
            Ac = R*A*P
            AcH = Ac.H.asformat(Ac.format)
            Bhc = np.ones((Nc,1))

        # Loop over the number of pairwise matchings to be done
        for i in range(1,matchings):
            if weights is not None:
                W = weights(Ac, **kwargs)
            else:
                W = Ac
            # Get matching
            [M,S] = get_matching[algorithm](W, order='forward', **kwargs)
            n = Ac.shape[0]
            num_pairs = M.shape[0]
            num_sing = S.shape[0]
            Nc = num_pairs+num_sing
            # Pick C-points and save in list
            temp = np.zeros((Nc,),dtype=int)
            temp[0:num_pairs] = M[:,0]
            temp[num_pairs:Nc] = S
            Cpts = Cpts[temp]

            # Improve near nullspace candidates by relaxing on A B = 0
            fn, kwargs = unpack_arg(improve_candidates)
            if fn is not None:
                b = np.zeros((n, 1), dtype=Ac.dtype)
                Bc = relaxation_as_linear_operator((fn, kwargs), Ac, b) * Bc
                if symmetry == "nonsymmetric":
                    Bhc = relaxation_as_linear_operator((fn, kwargs), AcH, b) * Bhc

            # Form sparse P from pairwise aggregation
            row_inds = np.empty(n)
            row_inds[0:(2*num_pairs)] = M.flatten()
            row_inds[(2*num_pairs):n] = S
            col_inds = np.empty(n)
            col_inds[0:(2*num_pairs)] = ( np.array( ( np.arange(0,num_pairs),np.arange(0,num_pairs) ) ).T).flatten()
            col_inds[(2*num_pairs):n] = np.arange(num_pairs,Nc)

            # Form coarse grid operator and update aggregation matrix
            if i<(matchings-1):
                P = csr_matrix( (Bc[0:n,0], (row_inds,col_inds)), shape=(n,Nc) )
                if symmetry == 'hermitian':
                    R = P.H
                    Ac = R*Ac*P
                elif symmetry == 'symmetric':
                    R = P.T            
                    Ac = R*Ac*P
                elif symmetry == 'nonsymmetric':
                    R = csr_matrix( (Bhc[0:n,0], (col_inds,row_inds)), shape=(Nc,n) )
                    Ac = R*Ac*P
                    AcH = Ac.H.asformat(Ac.format)
                    Bhc = np.ones((Nc,1))

                AggOp = csr_matrix(AggOp * P, dtype=bool)
                Bc = np.ones((Nc,1))
            else:
                P = csr_matrix( (np.ones((n,), dtype=bool), (row_inds,col_inds)), shape=(n,Nc) )
                AggOp = csr_matrix(AggOp * P, dtype=bool)

        return AggOp, Cpts




