#ifndef PAIRWISE_H
#define PAIRWISE_H

#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>
#include <assert.h>
#include <cmath>
#include "linalg.h"


#define F_NODE 0
#define C_NODE 1
#define U_NODE 2
#define PRE_F_NODE 3


/* Function determining which of two elements in the A.data structure is
 * larger. Allows for easy changing between, e.g., absolute value, hard   
 * minimum, etc.   
 *                    
 * Input:
 * ------
 * ind0 : const {int}
 *      Index for element in A.data
 * ind1 : const {int}
 *      Index for element in A.data
 * data : const {float array}  
 *      Data elements for A in sparse format
 *
 * Returns:
 * --------  
 * bool on if data[ind0] > data[ind1] for the measure of choice. For now this is
 * an absolute maximum.
 */
template<class I, class T>
bool is_larger(const I &ind0,
               const I &ind1,
               const T A_data[],
               const T theta = 1.0)
{
    if (std::abs(A_data[ind0]) >= theta*std::abs(A_data[ind1]) ) {
        return true;
    }
    else {
        return false;
    }
}


/* Function that finds the maximum edge connected to a given node and adds
 * this pair to the matching, for a 'maximum' defined by is_larger(). If a 
 * pair is found, each node is marked as aggregated, and the new node index
 * returned. If there are no unaggregated connections to the input node, -1
 * is returned. Input node is marked as F-point and its pair as C-point.
 *
 * Input:
 * ------
 * A_rowptr : const {int array}
 *      Row pointer for sparse array in CSR format.
 * A_colinds : const {int array}
 *      Column indices for sparse array in CSR format.
 * A_data : const {float array}
 *      Data values for sparse array in CSR format.
 * M : {int array}
 *      Approximate matching stored in 1d array, with indices for a pair
 *      as consecutive elements. If new pair is formed, it is added to M[].
 * W : {float}
 *      Additive weight of current matching. If new pair is formed,
 *      corresponding weight is added to W.
 * row : const {int}
 *      Index of base node to form pair for matching. 
 *
 * Returns:
 * --------  
 * Integer index of node connected to input node as a pair in matching. If
 * no unaggregated nodes are connected to input node, -1 is returned. 
 */
template<class I, class T>
I pick_Cpt(const I A_rowptr[],
           const I A_colinds[],
           const T A_data[],
           std::vector<I> &M,
           T &W,
           const I &row,
           T cost[] )
{
    I data_ind0 = A_rowptr[row];
    I data_ind1 = A_rowptr[row+1];
    I new_node = -1;
    I new_ind = data_ind0;

    // Find maximum edge attached to node 'row'
    for (I i=data_ind0; i<data_ind1; i++) {
        I temp_node = A_colinds[i];
        // Check for self-loops and make sure node has not been aggregated 
        if ( (temp_node != row) && (M[temp_node] == -1) ) {
            if (is_larger(i, new_ind, A_data)) {
                new_node = temp_node;
                new_ind = i;
            }
            cost[0] += 1.0;
        }
        cost[0] += 1.0;
    }

    // Add edge to matching and weight to total edge weight. Mark initial
    // node, i.e. this row in matrix, as F-point, and pair as C-point. 
    if (new_node != -1) {
        W += std::abs(A_data[new_ind]);
        M[row] = F_NODE;
        M[new_node] = C_NODE;
        cost[0] += 1.0;
    }

    // Return node index in new edge
    return new_node;
}


/* Function to approximate a graph matching, and use the matching for 
 * a pairwise aggregation of matrix A. This version only constructs the
 * row pointer and column indices for a CSR tentative prolongator. 
 * Matching done via Drake's 2003 1/2-matching algorithm.  
 *
 * Input:
 * ------
 * A_rowptr : const {int array}
 *      Row pointer for sparse array in CSR format.
 * A_colinds : const {int array}
 *      Column indices for sparse array in CSR format.
 * A_data : const {float array}
 *      Data values for sparse array in CSR format.
 * Agg_rowptr : {int array}
 *      Empty length(n+1) row pointer for sparse array in CSR format.
 * Agg_colinds : {int array}
 *      Empty length(n) column indices for sparse array in CSR format.
 * Agg_shape : {int array, size 2} 
 *      Shape array for sparse matrix constructed in function.
 * n : const {int} 
 *      Problem size
 *
 * Returns
 * -------
 * Nothing, splitting list modified in place.
 *
 */
template<class I, class T>
void drake_CF_matching(const I A_rowptr[], const int A_rowptr_size,
                       const I A_colinds[], const int A_colinds_size,
                       const T A_data[], const int A_data_size,
                             I splitting[], const int splitting_size,
                       const T theta, 
                             T cost[], const int cost_size )
{
    I n = A_rowptr_size-1;
        
    // Plan - store M1, M2 as all -a to start, when nodes are aggregated, 
    // say x and y, set M1[x] = y and M1[y] = x. 
    std::vector<I> M1(n,-1);     
    std::vector<I> M2(n,-1);

    // Empty initial weights.
    T W1 = 0;
    T W2 = 0;

    // Form two matchings, M1, M2, starting from last node in DOFs. 
    for (I row=(n-1); row>=0; row--) {
        I x = row;
        while (true) {           
            // Check that x is not already marked.
            if (M1[x] != -1) {
                break;
            }
            // Get new edge in matching, M1. Break loop if node x has no
            // edges to unmarked nodes (function returns -1).
            I y = pick_Cpt(A_rowptr, A_colinds, A_data, M1, W1, x, cost);
            if (y == -1) {
                break;
            }

            // Get new edge in matching, M1. Break loop if node x has no
            // edges to unmarked nodes (function returns -1).
            if (M2[y] != -1) {
                break;
            }
            x = pick_Cpt(A_rowptr, A_colinds, A_data, M2, W2, y, cost);
            if (x == -1) {
                break;
            }
        }
    }

    int *M = NULL; 
    if (std::abs(W1) >= std::abs(W2)) {
        M = &M1[0];
    }
    else {
        M = &M2[0];
    }

    // If node has not been marked, determine if C-point or F-point as follows:
    // Initialize as F-point, if it has any strong connections to F-points,
    // mark as C-point. 
    for (I i=0; i<n; i++) {
        if (M[i] == -1) {
            M[i] = F_NODE;
            // Get diagonal element index
            I diag_ind = -1;
            for (I j=A_rowptr[i]; j<A_rowptr[i+1]; j++) {
                if (A_colinds[j] == i) {
                    diag_ind = j;
                }
            }
            if (diag_ind == -1) {
                M[i] = C_NODE;
            } else {
                // Check if strongly connected to F-points
                for (I j=A_rowptr[i]; j<A_rowptr[i+1]; j++) {
                    if ((diag_ind != j) && is_larger(j,diag_ind,A_data,theta) && 
                        (M[A_colinds[j]] == F_NODE)) {
                        M[i] = C_NODE;
                        break;
                    }
		}
            }
        }
        splitting[i] = M[i];
    }
    M = NULL;
}


/* Function to compute weights of a graph for an approximate matching. 
 * Weights are chosen to construct as well-conditioned of a fine grid
 * as possible based on a given smooth vector:
 *
 *      W_{ij} = 1 - 2a_{ij}b_ib_j / (a_{ii}w_i^2 + a_{jj}w_j^2)
 *
 * Notes: 
 * ------
 *      - weight matrix has same row-pointer and column indices as
 *		  A, so only the weight data is stored in an array.
 *		- If B is not provided, it is assumed to be the constant
 *		  vector.
 *
 * Input:
 * ------
 * A_rowptr : const {int array}
 *      Row pointer for sparse array in CSR format.
 * A_colinds : const {int array}
 *      Column indices for sparse array in CSR format.
 * A_data : const {float array}
 *      Data values for sparse array in CSR format.
 * weights : {float array}
 *      Empty length(n) data array for computed weights
 * B : {float array}, optional
 *      Target algebraically smooth vector to compute weights with.
 *
 * Returns
 * -------
 * Nothing, weights modified in place.
 * 
 */
template<class I, class T>
void compute_weights(const I A_rowptr[], const int A_rowptr_size,
                     const I A_colinds[], const int A_colinds_size,
                     const T A_data[], const int A_data_size,
                      	   T weights[], const int weights_size,
                     const T B[], const int B_size,
                           T cost[], const int cost_size)
{
	I n = A_rowptr_size-1;
	std::vector<T> diag(n);
    T temp_cost = 0.0;

	// Get diagonal elements of matrix
	for (I i=0; i<n; i++) {
		for (I ind=A_rowptr[i]; ind<A_rowptr[i+1]; ind++) {
			if(i == A_colinds[ind]) {
				diag[i] = A_data[ind];
			}
		}
	}

	// Compute matrix weights,
	// 		w{ij} = 1 - 2a_{ij}B_iB_j / (a_{ii}B_i^2 + a_{jj}B_j^2)
	for (I i=0; i<n; i++) {
		for (I ind=A_rowptr[i]; ind<A_rowptr[i+1]; ind++) {
			I j=A_colinds[ind];
			weights[ind] = 1.0 - (2*A_data[ind]*B[i]*B[j]) / (diag[i]*B[i]*B[i] + diag[j]*B[j]*B[j]);
            temp_cost += 3.0;
		}
	}
    temp_cost += n; // Can precompute a_{ii}B_i^2
    cost[0] += temp_cost;   
}

template<class I, class T>
void compute_weights(const I A_rowptr[], const int A_rowptr_size,
                     const I A_colinds[], const int A_colinds_size,
                     const T A_data[], const int A_data_size,
                      	   T weights[], const int weights_size,
                           T cost[], const int cost_size)
{
	I n = A_rowptr_size-1;
	std::vector<T> diag(n);
    T temp_cost = 0.0;

	// Get diagonal elements of matrix
	for (I i=0; i<n; i++) {
		for (I ind=A_rowptr[i]; ind<A_rowptr[i+1]; ind++) {
			if(i == A_colinds[ind]) {
				diag[i] = A_data[ind];
			}
		}
	}

	// Compute matrix weights. B is assumed constant, and
	//		w{ij} = 1 - 2a_{ij} / (a_{ii} + a_{jj})
	for (I i=0; i<n; i++) {
		for (I ind=A_rowptr[i]; ind<A_rowptr[i+1]; ind++) {
			I j=A_colinds[ind];
			weights[ind] = 1.0 - 2*A_data[ind] / (diag[i] + diag[j]);
            temp_cost += 2.0;
		}
	}	
    cost[0] += temp_cost;	
}

#endif
