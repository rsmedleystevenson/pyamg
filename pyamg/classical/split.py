"""Functions to compute C/F splittings for use in Classical AMG

Overview
--------

A C/F splitting is a partitioning of the nodes in the graph of as connection
matrix (denoted S for strength) into sets of C (coarse) and F (fine) nodes.
The C-nodes are promoted to the coarser grid while the F-nodes are retained
on the finer grid.  Ideally, the C-nodes, which represent the coarse-level
unknowns, should be far fewer in number than the F-nodes.  Furthermore,
algebraically smooth error must be well-approximated by the coarse level
degrees of freedom.


Representation
--------------

C/F splitting is represented by an array with ones for all the C-nodes
and zeros for the F-nodes.


C/F Splitting Methods
---------------------

RS : Original Ruge-Stuben method
    - Produces good C/F splittings but is inherently serial.
    - May produce AMG hierarchies with relatively high operator complexities.
    - See References [1,4]

PMIS: Parallel Modified Independent Set
    - Very fast construction with low operator complexity.
    - Convergence can deteriorate with increasing problem
      size on structured meshes.
    - Uses method similar to Luby's Maximal Independent Set algorithm.
    - See References [1,3]

PMISc: Parallel Modified Independent Set in Color
    - Fast construction with low operator complexity.
    - Better scalability than PMIS on structured meshes.
    - Augments random weights with a (graph) vertex coloring
    - See References [1]

CLJP: Clearly-Luby-Jones-Plassmann
    - Parallel method with cost and complexity comparable to Ruge-Stuben.
    - Convergence can deteriorate with increasing problem
      size on structured meshes.
    - See References [1,2]

CLJP-c: Clearly-Luby-Jones-Plassmann in Color
    - Parallel method with cost and complexity comparable to Ruge-Stuben.
    - Better scalability than CLJP on structured meshes.
    - See References [1]


Summary
-------

In general, methods that use a graph coloring perform better on structured
meshes [1].  Unstructured meshes do not appear to benefit substantially
from coloring.

    ========  ========  ========  ==========
     method   parallel  in color     cost
    ========  ========  ========  ==========
       RS        no        no      moderate
      PMIS      yes        no      very low
      PMISc     yes       yes        low
      CLJP      yes        no      moderate
      CLJPc     yes       yes      moderate
    ========  ========  ========  ==========


References
----------

..  [1] David M. Alber and Luke N. Olson
    "Parallel coarse-grid selection"
    Numerical Linear Algebra with Applications 2007; 14:611-643.

..  [2] Cleary AJ, Falgout RD, Henson VE, Jones JE.
    "Coarse-grid selection for parallel algebraic multigrid"
    Proceedings of the 5th International Symposium on Solving Irregularly
    Structured Problems in Parallel. Springer: Berlin, 1998; 104-115.

..  [3] Hans De Sterck, Ulrike M Yang, and Jeffrey J Heys
    "Reducing complexity in parallel algebraic multigrid preconditioners"
    SIAM Journal on Matrix Analysis and Applications 2006; 27:1019-1039.

..  [4] Ruge JW, Stuben K.
    "Algebraic multigrid (AMG)"
    In Multigrid Methods, McCormick SF (ed.),
    Frontiers in Applied Mathematics, vol. 3.
    SIAM: Philadelphia, PA, 1987; 73-130.


"""

import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix, isspmatrix_csr

from pyamg.graph import vertex_coloring
from pyamg import amg_core
from pyamg.util.utils import remove_diagonal
from pyamg.strength import classical_strength_of_connection

__all__ = ['RS', 'PMIS', 'PMISc', 'CLJP', 'CLJPc', 'MIS', 'weighted_matching']
__docformat__ = "restructuredtext en"


def RS(S, cost=[0]):
    """Compute a C/F splitting using Ruge-Stuben coarsening

    Parameters
    ----------
    S : csr_matrix
        Strength of connection matrix indicating the strength between nodes i
        and j (S_ij)
    influence : TODO -- what is this?

    Returns
    -------
    splitting : ndarray
        Array of length of S of ones (coarse) and zeros (fine)

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.classical import RS
    >>> S = poisson((7,), format='csr') # 1D mesh with 7 vertices
    >>> splitting = RS(S)

    See Also
    --------
    amg_core.rs_cf_splitting

    References
    ----------
    .. [1] Ruge JW, Stuben K.  "Algebraic multigrid (AMG)"
       In Multigrid Methods, McCormick SF (ed.),
       Frontiers in Applied Mathematics, vol. 3.
       SIAM: Philadelphia, PA, 1987; 73-130.

    """
    if not isspmatrix_csr(S):
        raise TypeError('expected csr_matrix')
    S = remove_diagonal(S)

    T = S.T.tocsr()  # transpose S for efficient column access
    splitting = np.empty(S.shape[0], dtype='intc')
    influence = np.zeros((S.shape[0],), dtype='intc')

    amg_core.rs_cf_splitting(S.shape[0],
                             S.indptr, S.indices,
                             T.indptr, T.indices,
                             influence,
                             splitting)
    amg_core.rs_cf_splitting_pass2(S.shape[0], S.indptr,
                                   S.indices, splitting)

    return splitting


def PMIS(S, cost=[0]):
    """C/F splitting using the Parallel Modified Independent Set method

    Parameters
    ----------
    S : csr_matrix
        Strength of connection matrix indicating the strength between nodes i
        and j (S_ij)

    Returns
    -------
    splitting : ndarray
        Array of length of S of ones (coarse) and zeros (fine)

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.classical import PMIS
    >>> S = poisson((7,), format='csr') # 1D mesh with 7 vertices
    >>> splitting = PMIS(S)

    See Also
    --------
    MIS

    References
    ----------
    .. [1] Hans De Sterck, Ulrike M Yang, and Jeffrey J Heys
       "Reducing complexity in parallel algebraic multigrid preconditioners"
       SIAM Journal on Matrix Analysis and Applications 2006; 27:1019-1039.

    """
    S = remove_diagonal(S)
    weights, G, S, T = preprocess(S)
    return MIS(G, weights)


def PMISc(S, method='JP', cost=[0]):
    """C/F splitting using Parallel Modified Independent Set (in color)

    PMIS-c, or PMIS in color, improves PMIS by perturbing the initial
    random weights with weights determined by a vertex coloring.

    Parameters
    ----------
    S : csr_matrix
        Strength of connection matrix indicating the strength between nodes i
        and j (S_ij)
    method : string
        Algorithm used to compute the initial vertex coloring:
            * 'MIS' - Maximal Independent Set
            * 'JP'  - Jones-Plassmann (parallel)
            * 'LDF' - Largest-Degree-First (parallel)

    Returns
    -------
    splitting : array
        Array of length of S of ones (coarse) and zeros (fine)

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.classical import PMISc
    >>> S = poisson((7,), format='csr') # 1D mesh with 7 vertices
    >>> splitting = PMISc(S)

    See Also
    --------
    MIS

    References
    ----------
    .. [1] David M. Alber and Luke N. Olson
       "Parallel coarse-grid selection"
       Numerical Linear Algebra with Applications 2007; 14:611-643.

    """
    S = remove_diagonal(S)
    weights, G, S, T = preprocess(S, coloring_method=method)
    return MIS(G, weights)


def CLJP(S, color=False, cost=[0]):
    """Compute a C/F splitting using the parallel CLJP algorithm

    Parameters
    ----------
    S : csr_matrix
        Strength of connection matrix indicating the strength between nodes i
        and j (S_ij)
    color : bool
        use the CLJP coloring approach

    Returns
    -------
    splitting : array
        Array of length of S of ones (coarse) and zeros (fine)

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.classical.split import CLJP
    >>> S = poisson((7,), format='csr') # 1D mesh with 7 vertices
    >>> splitting = CLJP(S)

    See Also
    --------
    MIS, PMIS, CLJPc

    References
    ----------
    .. [1] David M. Alber and Luke N. Olson
       "Parallel coarse-grid selection"
       Numerical Linear Algebra with Applications 2007; 14:611-643.

    """
    if not isspmatrix_csr(S):
        raise TypeError('expected csr_matrix')
    S = remove_diagonal(S)

    colorid = 0
    if color:
        colorid = 1

    T = S.T.tocsr()  # transpose S for efficient column access
    splitting = np.empty(S.shape[0], dtype='intc')

    amg_core.cljp_naive_splitting(S.shape[0],
                                  S.indptr, S.indices,
                                  T.indptr, T.indices,
                                  splitting,
                                  colorid)

    return splitting


def CLJPc(S, cost=[0]):
    """Compute a C/F splitting using the parallel CLJP-c algorithm

    CLJP-c, or CLJP in color, improves CLJP by perturbing the initial
    random weights with weights determined by a vertex coloring.

    Parameters
    ----------
    S : csr_matrix
        Strength of connection matrix indicating the strength between nodes i
        and j (S_ij)

    Returns
    -------
    splitting : array
        Array of length of S of ones (coarse) and zeros (fine)

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.classical.split import CLJPc
    >>> S = poisson((7,), format='csr') # 1D mesh with 7 vertices
    >>> splitting = CLJPc(S)

    See Also
    --------
    MIS, PMIS, CLJP

    References
    ----------
    .. [1] David M. Alber and Luke N. Olson
       "Parallel coarse-grid selection"
       Numerical Linear Algebra with Applications 2007; 14:611-643.

    """
    S = remove_diagonal(S)
    return CLJP(S, color=True)


def MIS(G, weights, maxiter=None, cost=[0]):
    """Compute a maximal independent set of a graph in parallel

    Parameters
    ----------
    G : csr_matrix
        Matrix graph, G[i,j] != 0 indicates an edge
    weights : ndarray
        Array of weights for each vertex in the graph G
    maxiter : int
        Maximum number of iterations (default: None)

    Returns
    -------
    mis : array
        Array of length of G of zeros/ones indicating the independent set

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.classical import MIS
    >>> import numpy as np
    >>> G = poisson((7,), format='csr') # 1D mesh with 7 vertices
    >>> w = np.ones((G.shape[0],1)).ravel()
    >>> mis = MIS(G,w)

    See Also
    --------
    fn = amg_core.maximal_independent_set_parallel

    """

    if not isspmatrix_csr(G):
        raise TypeError('expected csr_matrix')
    G = remove_diagonal(G)

    mis = np.empty(G.shape[0], dtype='intc')
    mis[:] = -1

    fn = amg_core.maximal_independent_set_parallel

    if maxiter is None:
        fn(G.shape[0], G.indptr, G.indices, -1, 1, 0, mis, weights, -1)
    else:
        if maxiter < 0:
            raise ValueError('maxiter must be >= 0')

        fn(G.shape[0], G.indptr, G.indices, -1, 1, 0, mis, weights, maxiter)

    return mis


# internal function
def preprocess(S, coloring_method=None):
    """Common preprocess for splitting functions

    Parameters
    ----------
    S : csr_matrix
        Strength of connection matrix
    method : {string}
        Algorithm used to compute the vertex coloring:
            * 'MIS' - Maximal Independent Set
            * 'JP'  - Jones-Plassmann (parallel)
            * 'LDF' - Largest-Degree-First (parallel)

    Returns
    -------
    weights: ndarray
        Weights from a graph coloring of G
    S : csr_matrix
        Strength matrix with ones
    T : csr_matrix
        transpose of S
    G : csr_matrix
        union of S and T

    Notes
    -----
    Performs the following operations:
        - Checks input strength of connection matrix S
        - Replaces S.data with ones
        - Creates T = S.T in CSR format
        - Creates G = S union T in CSR format
        - Creates random weights
        - Augments weights with graph coloring (if use_color == True)

    """

    if not isspmatrix_csr(S):
        raise TypeError('expected csr_matrix')

    if S.shape[0] != S.shape[1]:
        raise ValueError('expected square matrix, shape=%s' % (S.shape,))

    N = S.shape[0]
    S = csr_matrix((np.ones(S.nnz, dtype='int8'), S.indices, S.indptr),
                   shape=(N, N))
    T = S.T.tocsr()  # transpose S for efficient column access

    G = S + T  # form graph (must be symmetric)
    G.data[:] = 1

    weights = np.ravel(T.sum(axis=1))  # initial weights
    # weights -= T.diagonal()          # discount self loops

    if coloring_method is None:
        weights = weights + sp.rand(len(weights))
    else:
        coloring = vertex_coloring(G, coloring_method)
        num_colors = coloring.max() + 1
        weights = weights + (sp.rand(len(weights)) + coloring)/num_colors

    return (weights, G, S, T)



# TODO : This cannot coarsen well on lower levels


def weighted_matching(A, B=None, theta=0.5, use_weights=True, get_SOC=False, cost=[0.0], **kwargs):
    """ Pairwise aggregation of nodes using Drake approximate
        1/2-matching algorithm.

    Parameters
    ----------
    A : csr_matrix or bsr_matrix
        matrix for linear system.
    B : array_like : default None
        Right near-nullspace candidates stored in the columns of an NxK array.
        If no target vector provided, constant vector is used. In the case of
        multiple targets, k>1, only the first is used to construct coarse grid
        matrices for pairwise aggregations. 
    use_weights : {bool} : default True
        Optional function handle to compute weights used in the matching,
        e.g. a strength of connection routine. Additional arguments for
        this routine should be provided in **kwargs. 
    get_SOC : {bool} : default False
        TODO
    theta : float
        connections deemed "strong" if |a_ij| > theta*|a_ii|

    NOTES
    -----
        - Not implemented for block systems or complex. 
            + Need to define how a matching is done nodally.
            + Also must consider what targets are used to form coarse grid
              in nodal approach...
            + Drake should be accessible in complex, but not Notay due to the
              hard minimum. Is there a < operator overloaded for complex?
              Could I overload it perhaps? Probably would do magnitude or something
              though, which is not what we want... 

    REFERENCES
    ----------
    [1] D'Ambra, Pasqua, and Panayot S. Vassilevski. "Adaptive AMG with
    coarsening based on compatible weighted matching." Computing and
    Visualization in Science 16.2 (2013): 59-76.

    [2] Drake, Doratha E., and Stefan Hougardy. "A simple approximation
    algorithm for the weighted matching problem." Information Processing
    Letters 85.4 (2003): 211-213.

    """

    def unpack_arg(v):
        if isinstance(v, tuple):
            return v[0], v[1]
        else:
            return v, {}

    if A.dtype == 'complex':
        raise TypeError("Not currently implemented for complex.")

    if (A.getformat() != 'csr'):
        try:
            A = A.tocsr()
        except:
            raise TypeError("Must pass in CSR matrix, or sparse matrix "
                            "which can be converted to CSR.")

    n = A.shape[0]

    # If target vectors provided, take first.
    if B is not None:
        if len(B.shape) == 2:
            target = B[:,0]
        else:
            target = B[:,]
    else:
        target = None

    # Compute weights if function provided, otherwise let W = A
    if use_weights:
        weights = np.empty((A.nnz,),dtype=A.dtype)
        temp_cost = np.ones((1,), dtype=A.dtype)
        if target is None:
            amg_core.compute_weights(A.indptr, A.indices, A.data,
                                     weights, temp_cost)
        else:
            amg_core.compute_weights(A.indptr, A.indices, A.data,
                                     weights, target, temp_cost)

        cost[0] += temp_cost[0] / float(A.nnz)
    else:
        weights = A.data

    # Get CF splitting
    temp_cost = np.ones((1,), dtype=A.dtype)
    splitting = np.empty(n, dtype='int32')
    amg_core.drake_CF_matching(A.indptr, A.indices, weights, splitting, theta, temp_cost )
    cost[0] += temp_cost[0] / float(A.nnz)

    if get_SOC:
        temp = csr_matrix((weights, A.indices, A.indptr), copy=True)
        C = classical_strength_of_connection(temp, theta, cost=cost)
        return splitting, C
    else:
        return splitting, None





