#include <limits>
#include <math.h>



// Must have beta < 1
template<class I, class T>
void notay_pairwise_common(const I A_rowptr[],
			   		const I A_colinds[],
			   		const T A_data[],
			   		I Agg_rowptr[],
			   		I Agg_colinds[],
			   		T Agg_data[],
			   		I Agg_shape[],
			   		const T &beta,
			   		const I &n,
			   		T B[] = NULL )
{
	I n = A_rowptr_size - 1;

	// Construct sparse, filtered matrix if beta != 0
	if (beta != 0) {
		std::vector<I> rowptr(n+1,0);
		std::vector<I> colinds;
		std::vector<T> data;
		for (I i=0; i<n; i++) {
			// Filtering threshold, -beta * max_{a_ij < 0} |a_ij|.
			T row_thresh = 0;
			for (I j=A_rowptr[i]; j<A_rowptr[i+1]; j++) {
				if (A_data[j] < row_thresh) {
					row_thresh = A_data[j];
				}
			}
			row_thresh *= beta;

			// Construct sparse row of filtered matrix
			I row_size = 0;
			for (I j=A_rowptr[i]; j<A_rowptr[i+1]; j++) {
				if (A_data[j] < row_thresh) {
					colinds.push_back(A_colinds[j]);
					data.push_back(A_data[j]);
					row_size += 1;
				}
			}
			rowptr[i+1] = rowptr[i]+row_size;
		}
	}
	// Otherwise, just use A for pairwise aggregation
	else {
		const I *rowptr = &A_rowptr[0];
		const I *colinds = &A_colinds[0];
		const I *data = &A_data[0];
	}

	// Construct vector, m, to track if each node has been aggregated (-1),
	// and its number of unaggregated neighbors otherwise. Save node with
	// minimum number of neighbors as starting node. 
	std::vector<I> m(n);
	I start_ind = -1;
	I min_neighbor = std::numeric_limits<int>::max();
	for (I i=0; i<n; i++) {
		m[i] = rowptr[i+1] - rowptr[i];
		if (m[i] < min_neighbor) {
			min_neighbor = m[i];
			start_ind = i;
		}
	}

	// Loop until all nodes have been aggregated 
	I Nc = 0;
	I num_aggregated = 0;
	Agg_rowptr[0] = 0;
	while (num_aggregated < n) {

		// Find unaggregated neighbor with strongest (negative) connection
		I neighbor = -1;
		T min_val = 0;
		for (I j=rowptr[start_ind]; j<rowptr[start_ind+1]; j++) {
			// Check for self loop, make sure node is unaggregated 
			if ( (start_ind != colinds[j] ) && (m[colinds[j]] >= 0) )
				// Find hard minimum weight of neighbor nodes 
				if (data[j] < min_val) {
					neighbor = colinds[j];
					min_val = data[j];
				}
			}
		}

		// Form new aggregate as vector of length 1 or 2 and mark
		// nodes as aggregated. 
		std::vector<I> new_agg;
		new_agg.push_back(start_ind);
		m[start_ind] = -1;
		if (neighbor >= 0) {
			new_agg.push_back(neighbor);
			m[neighbor] = -1;
		}

		// Find new starting node
		start_ind = -1;
		min_neighbor = std::numeric_limits<int>::max();
		// For each node in aggregate
		for (auto it=new_agg.begin(); it!=new_agg.end(); it++) {

			// For each node strongly connected to current node
			for (I j=rowptr[*it]; j<rowptr[(*it)+1]; j++) {
				I &neighborhood = m[colinds[j]];
			
				// Check if node has not been aggregated
				if (neighborhood >= 0) {
					// Decrease neighborhood size by one
					neighborhood -= 1;

					// Find neighboring node with smallest neighborhood 
					if (neighborhood < min_neighbor) {
						min_neighbor = neighborhood;
						start_ind = j;
					}
				}
			}
		}

		// If no start node was found, find unaggregated node
		// with least connections out of all nodes.
		if (start_ind == -1) {
			for (I i=0; i<n; i++) {
				if ( (m[i] >= 0) && (m[i] < min_neighbor) ) {
					min_neighbor = m[i];
					start_ind = i;
				}
			}
		}

		// If target B provided, get norm restricted to this aggregate
		T agg_norm = 0;
		if (B != NULL && new_agg.size() > 1) {
			for (auto it=new_agg.begin(); it!=new_agg.end(); it++) {
				agg_norm += B[*it] * B[*it];
			}
			agg_norm = sqrt(agg_norm);
		}

		// Update sparse structure for aggregation matrix with new aggregate.
		for (auto it=new_agg.begin(); it!=new_agg.end(); it++) {
			// Set all nodes in this aggregate to column Nc
			Agg_colinds[*it] = Nc;
			// Set corresponding data value, normalize if B provided.
			// If B is not provided, set constant vector. 
			if (B == NULL || new_agg.size() == 1) {
				Agg_data[*it] = 1.0;
			}
			else {
				Agg_data[*it] = B[*it] / agg_norm;
			}
			// Increase row pointer by one for each node aggregated
			Agg_rowptr[num_aggregated+1] = num_aggregated+1; 
			// Increase count of aggregated nodes
			num_aggregated += 1;
		}

		// Increase coarse grid count
		Nc += 1;
	}

	// Save shape of aggregation matrix
	Agg_shape[0] = n;
	Agg_shape[1] = Nc;
}


template<class I, class T>
void notay_pairwise(const I A_rowptr[], const int A_rowptr_size,
                    const I A_colinds[], const int A_colinds_size,
                    const T A_data[], const int A_data_size,
                    const T B[], const int B_size,
                    I Agg_rowptr[], const int Agg_rowptr_size,
                    I Agg_colinds[], const int Agg_colinds_size,
                    T Agg_data[], const int Agg_data_size,
                    I Agg_shape[], const int Agg_shape_size,
                    const T beta)
{
    I n = A_rowptr_size-1;
    notay_pairwise_common(A_rowptr, A_colinds, A_data,
    					  Agg_rowptr, Agg_colinds, Agg_data,
    					  Agg_shape, beta, n, B);
}

template<class I, class T>
void notay_pairwise(const I A_rowptr[], const int A_rowptr_size,
                    const I A_colinds[], const int A_colinds_size,
                    const T A_data[], const int A_data_size,
                    I Agg_rowptr[], const int Agg_rowptr_size,
                    I Agg_colinds[], const int Agg_colinds_size,
                    T Agg_data[], const int Agg_data_size,
                    I Agg_shape[], const int Agg_shape_size,
                    const T beta)
{
    I n = A_rowptr_size-1;
    notay_pairwise_common(A_rowptr, A_colinds, A_data,
    					  Agg_rowptr, Agg_colinds, Agg_data,
    					  Agg_shape, beta, n);
}


