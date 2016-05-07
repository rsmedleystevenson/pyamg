import pdb
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, isspmatrix_csr, isspmatrix_bsr
from pyamg import amg_core


__all__ = ['preis_matching_1999', 'drake_matching_2003','notay_matching_2010','drake_C']


def preis_try_match(a0, b0, w_ab, W, M, M_ind, G, aggregated, recurse = False):

    # Removed edges from nodes a and b. Stored as list of lists,
    # where each inner list looks like [c,w], for adjacent node 
    # c and edge weight w.
    C_a = []
    C_b = []

    # Move edge [a,b] and [b0,a] out of graph
    if not recurse:    
        b_ind = G.rows[a0].index(b0)
        a_ind = G.rows[b0].index(a0)
        C_a.append([b0,G.data[a0][b_ind]])
        del G.rows[a0][b_ind]
        del G.data[a0][b_ind]
        C_b.append([a0,G.data[b0][a_ind]])
        del G.rows[b0][a_ind]
        del G.data[b0][a_ind]

    # While nodes a,b, are unaggregated and a or b has at leat one unchecked edge
    while ( (aggregated[a0,1] == 0) and (aggregated[b0,1] == 0) ) and ( (len(G.rows[a0]) > 0) or (len(G.rows[b0]) > 0) ):
        # If a has not been aggregated and has an unchecked edge, [a,c],
        # compare [a,c] with [a,b]
        if (aggregated[a0,1] == 0) and (len(G.rows[a0]) > 0):
            # Move edge [a,c] and [c,a] out of graph as 'checked'
            c0 = G.rows[a0][0]
            w_ac = np.abs(G[a0,c0])
            C_a.append([c0,G.data[a0][0]])
            a_ind = G.rows[c0].index(a0)
            del G.rows[a0][0]
            del G.data[a0][0]
            del G.rows[c0][a_ind]
            del G.data[c0][a_ind]
            # If G(a,c) > G(a,b) recurse with edge [a,c]
            if (w_ac > w_ab):
                [W, M_ind] = preis_try_match(a0, c0, w_ac, W, M, M_ind, G, aggregated, recurse=True)

        # If a has not been aggregated and has an unchecked edge, [a,c],
        # compare [a,c] with [a,b]
        if (aggregated[b0,1] == 0) and (len(G.rows[b0]) > 0):
            # Move edge [b0,d] and [d,b] out of graph as 'checked'
            d0 = G.rows[b0][0]
            w_bd = np.abs(G[b0,d0])
            C_b.append([d0,G.data[b0][0]])
            b_ind = G.rows[d0].index(b0)
            del G.rows[b0][0]
            del G.data[b0][0]
            del G.rows[d0][b_ind]
            del G.data[d0][b_ind]
            # If G(b,d) > G(a,b) recurse with edge [b0,d]
            if (w_bd > w_ab):
                [W, M_ind] = preis_try_match(b0, d0, w_bd, W, M, M_ind, G, aggregated, recurse=True)

    # If both nodes are unaggregated, add edge [a,b] to matching
    if ((aggregated[a0,1] == 0) and (aggregated[b0,1] == 0)):
        M[M_ind,:] = [a0,b0]
        M_ind += 1
        W += w_ab
        aggregated[a0,1] = 1
        aggregated[b0,1] = 1
    # If a is aggregated, but b is not, move b-edges back to graph
    elif (aggregated[b0,1] == 0):
        for edge in C_b:
            d0 = edge[0]
            # Only move edge back if adjacent node is also unaggregated
            if (aggregated[d0,1] == 0):
                G.rows[b0].append(d0)
                G.data[b0].append(edge[1])
                G.rows[d0].append(b0)
                G.data[d0].append(edge[1])
    # If b is aggregated, but a is not, move a-edges back to graph
    elif (aggregated[a0,1] == 0):
        for edge in C_a:
            c0 = edge[0]
            # Only move edge back if adjacent node is also unaggregated
            if (aggregated[c0,1] == 0):
                G.rows[a0].append(c0)
                G.data[a0].append(edge[1])
                G.rows[c0].append(a0)
                G.data[c0].append(edge[1])

    return [W, M_ind]


def preis_matching_1999(G, order='forward', **kwargs):

    # Get off-diagonal in linked list form
    n = G.shape[0]
    V = np.arange(0,n)
    if not isspmatrix_csr(G):
        try:
            G = G.tocsr()
        except:
            raise TypeError("Couldn't convert to csr.")
    
    G.setdiag(np.zeros((n,1)),k=0)
    G.eliminate_zeros()
    G = G.tolil()

    # List of nodes to mark when aggregated
    aggregated = np.zeros((n,2))
    aggregated[:,0] = V

    M = np.empty([n/2,2], dtype=int)
    num_pairs = 0
    W = 0.0
    empty = False

    # Order of nodes on which we grow path
    if order=='forward':
        bottom_loop = 0
        top_loop = n
        step = 1   
    elif order=='backward':
        bottom_loop = n-1
        top_loop = -1 
        step = -1

    # Loop until all nodes have been aggregated, or have no unaggregated neighbors
    while not empty:
        empty = True
        # Loop through starting nodes
        for i in range(bottom_loop,top_loop,step):
            a0 = i
            # Check if node has any unaggregated neighbors
            if (len(G.rows[a0]) > 0):
                empty = False
                b0 = G.rows[a0][0]
                w0 = G[a0,b0]
                [W, num_pairs] = preis_try_match(a0, b0, w0, W, M, num_pairs, G, aggregated)

    # print 'Preis - W = ',W,', aggregated = ',np.sum(aggregated[:,1]),' / ',n
    S = np.where(aggregated[:,1]==0)[0]       # Get singletons (not aggregated nodes)
    return [ M[0:num_pairs,:], S ]


def drake_C(G, order='forward', **kwargs):

    # Get off-diagonal of A in linked list form
    n = int(G.shape[0])
    V = np.arange(0,n)
    if not isspmatrix_csr(G):
        try:
            G = G.tocsr()
        except:
            raise TypeError("Couldn't convert to csr.")

    G.setdiag(np.zeros((n,1)),k=0)
    G.eliminate_zeros()

    # Nodes aggregated in matching 1
    agg1 = np.zeros((n,), dtype=np.int32)
    M1 = np.zeros([(n+2),], dtype=np.int32)

    # Nodes aggregated in matching 2
    agg2 = np.zeros((n,), dtype=np.int32)
    M2 = np.zeros(((n+2),), dtype=np.int32)

    # Singleton nodes -- assume sqrt(n) is enough to store singletons
    S = np.zeros((int(np.sqrt(n)),), dtype=np.int32)

    if order=='forward':
        ord0 = 0
    else:
        ord0 = 1

    match = amg_core.drake_matching
    match( G.indptr, 
           G.indices,
           G.data,
           n,
           ord0,
           agg1,
           M1,
           agg2,
           M2,
           S )

    if M1[1] >= M2[1]:
        num_pairs = M1[0]
        upper_ind_M = 2*(num_pairs+1) 
        M1 = M1[2:upper_ind_M].reshape((num_pairs,2), order='C')
        upper_ind_S = S[0]+1
        S = S[1:upper_ind_S]
        return [M1, S]
    else:
        num_pairs = M2[0]
        upper_ind_M = 2*(num_pairs+1) 
        M2 = M2[2:upper_ind_M].reshape((num_pairs,2), order='C')
        upper_ind_S = S[0]+1
        S = S[1:upper_ind_S]
        return [M2, S]


def drake_matching_2003(G, order='forward', **kwargs):

    # Get off-diagonal of A in linked list form
    n = G.shape[0]
    V = np.arange(0,n)
    if not isspmatrix_csr(G):
        try:
            G = G.tocsr()
        except:
            raise TypeError("Couldn't convert to csr.")

    G.setdiag(np.zeros((n,1)),k=0)
    G.eliminate_zeros()
    G = G.tolil()

    # Nodes aggregated in matching 1
    M1 = np.empty([n/2,2], dtype=int)
    ind1 = 0
    W1 = 0.0
    aggregated1 = np.zeros((n,2))
    aggregated1[:,0] = V

    # Nodes aggregated in matching 2
    M2 = np.empty([n/2,2], dtype=int)
    ind2 = 0
    W2 = 0.0
    aggregated2 = np.zeros((n,2))
    aggregated2[:,0] = V

    # Order of nodes on which we grow path
    if order=='forward':
        bottom_loop = 0
        top_loop = n
        step = 1   
    elif order=='backward':
        bottom_loop = n-1
        top_loop = -1 
        step = -1

    # Path-growing algorithm 
    for i in range(bottom_loop,top_loop,step):
        x = i
        while True:
            if len(G.rows[x]) == 0:
                break

            # Add edge to matching M1
            ind = np.argmax( np.abs( G.data[x] ) )	# find largest edge
            W1 += np.abs(G.data[x][ind])			# add weight to matching 1
            y = G.rows[x][ind]     					# get index of neighbor node   	  
            M1[ind1,:] = [x,y]           			# add edge to matching
            ind1 += 1
            aggregated1[x,1] = 1        			# mark node as aggregated
            aggregated1[y,1] = 1
            # Remove all edges connected to source node
            for neighbor in G.rows[x]:
                del_ind = G.rows[neighbor].index(x)
                del G.rows[neighbor][del_ind]
                del G.data[neighbor][del_ind]

            G.rows[x] = []			
            G.data[x] = []

            if len(G.rows[y]) == 0:
                break

            # Add edge to matching M2
            ind = np.argmax( np.abs( G.data[y] ) )	# find largest edge
            W2 += np.abs(G.data[y][ind])			# add weight to matching 1
            x = G.rows[y][ind]     					# get index of neighbor node  
            M2[ind2,:] = [x,y]                      # add edge to matching
            ind2 += 1
            aggregated2[x,1] = 1        			# mark node as aggregated
            aggregated2[y,1] = 1
            # Remove all edges connected to source node
            for neighbor in G.rows[y]:
                del_ind = G.rows[neighbor].index(y)
                del G.rows[neighbor][del_ind]
                del G.data[neighbor][del_ind]

            G.rows[y] = []
            G.data[y] = []

    # print 'Drake - W1 = ',W1,', aggregated = ',np.sum(aggregated1[:,1]),' / ',n
    # print 'Drake - W2 = ',W2,', aggregated = ',np.sum(aggregated2[:,1]),' / ',n
    if W1 > W2: 
        S = np.where(aggregated1[:,1]==0)[0]       # Get singletons (not aggregated nodes)
    	return [ M1[0:ind1,:], S ]
    else:
        S = np.where(aggregated2[:,1]==0)[0]       # Get singletons (not aggregated nodes)
    	return [ M2[0:ind2,:], S ]


# Notay's algorithm 2.1 from 2010 paper, "An aggregation-based algebraic multigrid method."
# A few things to note - 
#   - This uses a maximum and absolute value for strong connections. In the paper Notay
#     uses a hard minimum, but when I implemented that it produced square aggregates always.
#   - The algorithm in the paper is not linear. In this case, we approximate the 'always 
#	  start with minimal |m_i|' by starting with minimal m_i in the neighborhood of 
#	  the most recently formed aggregate. 
def notay_matching_2010(G, eps=0, order=0, **kwargs):

    # Get strongly coupled, off-diagonal of A in linked list form 
    n = G.shape[0]
    V = np.arange(0,n)
    if not isspmatrix_csr(G):
        try:
            G = G.tocsr()
        except:
            raise TypeError("Couldn't convert to csr.")

    G.setdiag(np.zeros((n,1)),k=0)
    
    # Prune matrix connections based on epsilon
    for row in range(0,n):
        lower_ind = G.indptr[row]
        upper_ind = G.indptr[row+1]
        temp_min = eps*np.max( np.abs( G.data[lower_ind:upper_ind] ) )
        # Don't know why we use negative connections... Doesn't seem general
        for i in range(lower_ind,upper_ind):
            if np.abs(G.data[i]) < temp_min:
                G.data[i] = 0.0

    G.eliminate_zeros()
    G = G.tolil()

    # Aggregated pairs
    M = np.empty([n/2+1,2], dtype=int)
    S = np.empty([n/2,], dtype=int)
    num_conns = map(len, G.rows)
    local_conns = num_conns
    num_aggregated = 0
    max_conns = np.max(num_conns)
    agg_mark = 2*max_conns
    M_ind = 0
    S_ind = 0
    W = 0

    # Loop until all nodes have been aggregated
    while num_aggregated < n:
    	# Get node with minimum connections from neighborhood of previous aggregate.
    	# If all neighbor nodes have been aggregated, search all nodes for minimal |m_i|
        try:
        	node1 = np.argmin(local_conns)
	        if num_conns[node1] > max_conns:
		        node1 = np.argmin(num_conns)
        except:
        	node1 = np.argmin(num_conns)

        # If node is strongly connected to another node, form aggregate pair
        if len(G.rows[node1]) > 0:
            conn_ind = np.argmax( np.abs( G.data[node1] ) )
            node2 = G.rows[node1][conn_ind]     # Again with the negative connections... 
            # If node has already been aggregated, it's number of connections will
            # have been marked >> max_conns. Delete index, continue loop.
            if num_conns[node2] > max_conns:
                del G.rows[node1][conn_ind]
                del G.data[node1][conn_ind] 
                continue

            # Add pair to list of aggregated pairs, save list of neighboring nodes
            neighbors = G.rows[node1]+G.rows[node2]
            M[M_ind,:] = [node1,node2]
            W += np.abs( G.data[node1][conn_ind] )
            num_aggregated += 2
            M_ind += 1

            # Remove all edges connected to strongly connected node and update number of connections
            num_conns[node2] = agg_mark
            for i in range(0,len(G.rows[node2])):
                neighbor = G.rows[node2][i]
                num_conns[neighbor] -= 1
                # After pruning matrix with eps, graph no longer symmmetric
                try:
                    del_ind = G.rows[neighbor].index(node2)
                except ValueError:
                    continue
                del G.rows[neighbor][del_ind]
                del G.data[neighbor][del_ind]

            G.rows[node2] = []
            G.data[node2] = []
        # If node is not strongly connected, form singleton
        else:
            S[S_ind] = node1
            num_aggregated += 1
            S_ind += 1

        # Remove all edges connected to source node and update number of connections
        num_conns[node1] = agg_mark
        for i in range(0,len(G.rows[node1])):
            neighbor = G.rows[node1][i]
            num_conns[neighbor] -= 1
            # After pruning matrix with eps, graph no longer symmmetric
            try:
                del_ind = G.rows[neighbor].index(node1)
            except ValueError:
                continue
            del G.rows[neighbor][del_ind]
            del G.data[neighbor][del_ind]

        G.rows[node1] = []
        G.data[node1] = []
        # Get number of connections for all neighbors of this aggregate
        local_conns = [num_conns[ind] for ind in neighbors]

    # print 'Notay - W = ',W,', aggregated = ',2*M_ind,' / ',n
    return [ M[0:M_ind,:], S[0:S_ind] ]


# def pettie_augment():


# 2/3-e approximation to matching problem. Supposed to converge much faster than
# Drake's initial 2/3-e approximation, so for now will only code up the pettie one.  
# Pettie proposed two algorithms, one random and one deterministic.
#   - Expected complexity should then be about 4/3*1.79m to get a 1/2-approximation.
#     How does this compare to Preis / Drake 2003?
#   - Deterministic algorithm takes ~27m to guarantee 1/2-approxiamtion - slow!
# numsteps = (1/3)n*ln(1/eps)
# def pettie_matching_2004rand(A, eps=1.0/6.0, M=[]):

#     for i in range(0, num_steps):


