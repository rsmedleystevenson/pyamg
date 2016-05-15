
/* Function determining which of two elements in the A.data structure is
 * larger. Allows for easy changing between, e.g., absolute value, hard   
 * minimum, etc.                       
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
bool is_larger(const I &ind0, const I &ind1, const T A_data[])
{
    if (std::abs(A_data[ind0]) >= std::abs(A_data[ind1]) ) {
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
 * is returned. 
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
I add_edge(const I A_rowptr[], const I A_colinds[], const T A_data[],
           std::vector<I> &M, T &W, I &ind, const I &row)
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
        }
    }

    // Add edge to matching and weight to total edge weight.
    // Note, matching is indexed (+2) because M[0] tracks the number
    // of pairs added to the matching, and M[1] the total weight. 
    // Mark each node in pair as aggregated. 
    if (new_node != -1) {
        W += A_data[new_ind];
        M[row] = new_node;
        M[new_node] = row;
    }

    // Return node index in new edge
    return new_node;
}


template<class I, class T>
void drake_matching(const I A_rowptr[], const int A_rowptr_size,
                    const I A_colinds[], const int A_colinds_size,
                    const T A_data[], const int A_data_size,
                    I Agg_rowptr[], const int Agg_rowptr_size,
                    I Agg_colinds[], const int Agg_colinds_size,
                    I Agg_shape[], const int Agg_shape )
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
            // Get new edge in matching, M1. Break loop if node x has no
            // edges to unaggregated nodes.
            if (M1[x] != -1) {
                break;
            }    
            I y = add_edge(A_rowptr, A_colinds, A_data, M1, W1, x);
            if (y == -1) {
                break;
            }

            // Get new edge in matching, M2. Break loop if node y has no
            // edges to unaggregated nodes.
            if (M2[y] != -1) {
                break;
            }
            x = add_edge(A_rowptr, A_colinds, A_data, M2, W2, y);
            if (x == -1) {
                break;
            }
        }
    }

    // std::cout << "W1 = " << W1 << ", W2 = " << W2 << std::endl;

    int *M = NULL; 
    if (std::abs(W1) >= std::abs(W2)) {
        M = &M1[0];
    }
    else {
        M = &M2[0];
    }


    // Form sparse structure of aggregation matrix 
    // THIS SECTION DIFFERS FOR AGG_OP VS. BAD GUY
    I Nc = 0;
    T max_single = 0.0;
    Agg_rowptr[0] = 0;
    std::vector<I> singletons;
    for (I i=0; i<n; i++) {

        // Set row pointer value for next row
        Agg_rowptr[i+1] = i+1;

        // Node has not been aggregated --> singleton
        if (M[i] == -1) {

            // Add singleton to sparse structure
            Agg_colinds[i] = Nc;
            Agg_data[i] = B[i];

            // Find largest singleton to normalize all singletons by
            if (abs(B[i] > max_single) {
                max_single = abs(B[i]);
            }
            // Save index to normalize later, mark node as stored (-2),
            // increase coarse grid count
            singletons.push_back(i);
            M[i] = -2;
            Nc += 1;
        }
        // Node has been aggregated, mark pair in aggregation matrix
        else if (M[i] > -1) {

            // Reference to each node in pair for ease of notation
            const I &p1 = i;
            const I &p2 = M[i];

            // Set rows p1, p2 to have column Nc
            Agg_colinds[p1] = Nc;
            Agg_colinds[p2] = Nc;

            // Normalize bad guy over aggregate, store in data vector
            T norm_b = sqrt( B[p1]*B[p1] + B[p2]*B[p2] );
            Agg_data[p1] = B[p1] / norm_b;
            Agg_data[p2] = B[p2] / norm_b;

            // Mark both nodes as stored (-2), and increase coarse grid count
            M[p1] = -2;
            M[p2] = -2;
            Nc += 1;
        }
        else { 
            continue;
        }
    }

    // Normalize singleton data value, s_k <-- s_k / max_k |s_k|.
    if (max_single > 0) {
        for (auto it=singletons.begin(); it!=singletons.end(); it++) {
            Agg_data[*it] /= max_single;
        }
    }

    // Save shape of aggregation matrix
    Agg_shape[0] = n;
    Agg_shape[1] = Nc;
}



// DIFFERENCE BETWEEN NORMALIZING SINGLETONS AND SETTING TO ONE?

// SHOULD FORM P HERE TOO IF DATA PROVIDED? 
// THEN I CAN NORMALIZE OVER EACH AGGREGATE.
// DEFINITELY NEED THIS TO BE AN OPTION.

// Somehow need to pick a set of C-points as well for Notay and Drake... 

