#ifndef RUGE_STUBEN_H
#define RUGE_STUBEN_H

#include <iostream>
#include <vector>
#include <iterator>
#include <cassert>
#include <limits>
#include <algorithm>

#include "linalg.h"
#include "graph.h"


/*
 *  Compute a strength of connection matrix using the classical strength
 *  of connection measure by Ruge and Stuben. Both the input and output
 *  matrices are stored in CSR format.  An off-diagonal nonzero entry
 *  A[i,j] is considered strong if:
 *
 *     | A[i,j] | >= theta * max( -A[i,k] )   where k != i      (_abs)
 *      -A[i,j]   >= theta * max( -A[i,k] )   where k != i      (_min)
 *
 * Otherwise, the connection is weak.
 *
 *  Parameters
 *      num_rows   - number of rows in A
 *      theta      - stength of connection tolerance
 *      Ap[]       - CSR row pointer
 *      Aj[]       - CSR index array
 *      Ax[]       - CSR data array
 *      Sp[]       - (output) CSR row pointer
 *      Sj[]       - (output) CSR index array
 *      Sx[]       - (output) CSR data array
 *
 *
 *  Returns:
 *      Nothing, S will be stored in Sp, Sj, Sx
 *
 *  Notes:
 *      Storage for S must be preallocated.  Since S will consist of a subset
 *      of A's nonzero values, a conservative bound is to allocate the same
 *      storage for S as is used by A.
 *
 */
template<class I, class T, class F>
void classical_strength_of_connection_abs(const I n_row,
                                          const F theta,
                                          const I Ap[], const int Ap_size,
                                          const I Aj[], const int Aj_size,
                                          const T Ax[], const int Ax_size,
                                                I Sp[], const int Sp_size,
                                                I Sj[], const int Sj_size,
                                                T Sx[], const int Sx_size)
{
    I nnz = 0;
    Sp[0] = 0;

    // Compute SOC for each row
    for(I i = 0; i < n_row; i++) {
        F max_offdiagonal = std::numeric_limits<F>::min();

        const I row_start = Ap[i];
        const I row_end   = Ap[i+1];

        // Find maximum off-diagonal in this row
        for(I jj = row_start; jj < row_end; jj++){
            if(Aj[jj] != i){
                max_offdiagonal = std::max(max_offdiagonal,mynorm(Ax[jj]));
            }
        }

        // Set threshold for strong connections
        F threshold = theta*max_offdiagonal;
        for(I jj = row_start; jj < row_end; jj++){
            F norm_jj = mynorm(Ax[jj]);

            // Add entry if it exceeds the threshold
            if(norm_jj >= threshold){
                if(Aj[jj] != i){
                    Sj[nnz] = Aj[jj];
                    Sx[nnz] = Ax[jj];
                    nnz++;
                }
            }

            // Always add the diagonal
            if(Aj[jj] == i){
                Sj[nnz] = Aj[jj];
                Sx[nnz] = Ax[jj];
                nnz++;
            }
        }

        Sp[i+1] = nnz;
    }
}

template<class I, class T>
void classical_strength_of_connection_min(const I n_row,
                                          const T theta,
                                          const I Ap[], const int Ap_size,
                                          const I Aj[], const int Aj_size,
                                          const T Ax[], const int Ax_size,
                                                I Sp[], const int Sp_size,
                                                I Sj[], const int Sj_size,
                                                T Sx[], const int Sx_size)
{
    I nnz = 0;
    Sp[0] = 0;

    for(I i = 0; i < n_row; i++){
        T max_offdiagonal = 0.0;

        const I row_start = Ap[i];
        const I row_end   = Ap[i+1];

        // Find maximum off-diagonal negative entry in this row
        for(I jj = row_start; jj < row_end; jj++){
            if(Aj[jj] != i){
                max_offdiagonal = std::max(max_offdiagonal, -Ax[jj]);
            }
        }

        // Set threshold for strong connections
        T threshold = theta*max_offdiagonal;
        for(I jj = row_start; jj < row_end; jj++){
            T norm_jj = -Ax[jj];

            // Add entry if -A_ij exceeds the threshold
            if(norm_jj >= threshold){
                if(Aj[jj] != i){
                    Sj[nnz] = Aj[jj];
                    Sx[nnz] = Ax[jj];
                    nnz++;
                }
            }

            // Always add the diagonal
            if(Aj[jj] == i){
                Sj[nnz] = Aj[jj];
                Sx[nnz] = Ax[jj];
                nnz++;
            }
        }

        Sp[i+1] = nnz;
    }
}


/*
 *  Compute the maximum in magnitude row value for a CSR matrix
 *
 *  Parameters
 *      num_rows   - number of rows in A
 *      Ap[]       - CSR row pointer
 *      Aj[]       - CSR index array
 *      Ax[]       - CSR data array
 *       x[]       - num_rows array
 *
 *  Returns:
 *      Nothing, x[i] will hold row i's maximum magnitude entry
 *
 */
template<class I, class T, class F>
void maximum_row_value(const I n_row,
                              T x[], const int  x_size,
                       const I Ap[], const int Ap_size,
                       const I Aj[], const int Aj_size,
                       const T Ax[], const int Ax_size)
{

    for(I i = 0; i < n_row; i++){
        F max_entry = std::numeric_limits<F>::min();

        const I row_start = Ap[i];
        const I row_end   = Ap[i+1];

        // Find this row's max entry
        for(I jj = row_start; jj < row_end; jj++){
            max_entry = std::max(max_entry, mynorm(Ax[jj]) );
        }

        x[i] = max_entry;
    }
}


#define F_NODE 0
#define C_NODE 1
#define U_NODE 2
#define PRE_F_NODE 3


/*
 * Compute a C/F (coarse-fine( splitting using the classical coarse grid
 * selection method of Ruge and Stuben.  The strength of connection matrix S,
 * and its transpose T, are stored in CSR format.  Upon return, the  splitting
 * array will consist of zeros and ones, where C-nodes (coarse nodes) are
 * marked with the value 1 and F-nodes (fine nodes) with the value 0.
 *
 * Parameters:
 *   n_nodes   - number of rows in A
 *   Sp[]      - CSR pointer array
 *   Sj[]      - CSR index array
 *   Tp[]      - CSR pointer array
 *   Tj[]      - CSR index array
 *   influence - array that influences splitting (values stored here are added to lambda for each point)
 *   splitting - array to store the C/F splitting
 *
 * Notes:
 *   The splitting array must be preallocated
 *
 */
template<class I>
void rs_cf_splitting(const I n_nodes,
                     const I Sp[], const int Sp_size,
                     const I Sj[], const int Sj_size,
                     const I Tp[], const int Tp_size,
                     const I Tj[], const int Tj_size,
                     const I influence[], const int influence_size,
                           I splitting[], const int splitting_size)
{
    std::vector<I> lambda(n_nodes,0);

    // Compute initial lambda based on C^T
    I lambda_max = 0;
    for(I i = 0; i < n_nodes; i++) {
        lambda[i] = Tp[i+1] - Tp[i] + influence[i];
        if (lambda[i] > lambda_max) {
            lambda_max = lambda[i];
        }
    }

    // For each value of lambda, create an interval of nodes with that value
    //      interval_ptr - the first index of the interval
    //      interval_count - the number of indices in that interval
    //      index_to_node - the node located at a given index
    //      node_to_index - the index of a given node
    lambda_max = lambda_max*2;
    if (n_nodes+1 > lambda_max) {
        lambda_max = n_nodes+1;
    }

    std::vector<I> interval_ptr(lambda_max,0);
    std::vector<I> interval_count(lambda_max,0);
    std::vector<I> index_to_node(n_nodes);
    std::vector<I> node_to_index(n_nodes);

    for(I i = 0; i < n_nodes; i++) {
        interval_count[lambda[i]]++;
    }
    for(I i = 0, cumsum = 0; i < lambda_max; i++) {
        interval_ptr[i] = cumsum;
        cumsum += interval_count[i];
        interval_count[i] = 0;
    }
    for(I i = 0; i < n_nodes; i++) {
        I lambda_i = lambda[i];

        I index    = interval_ptr[lambda_i] + interval_count[lambda_i];
        index_to_node[index] = i;
        node_to_index[i]     = index;
        interval_count[lambda_i]++;
    }

    std::fill(splitting, splitting + n_nodes, U_NODE);


    // All nodes with no neighbors become F nodes
    for (I i = 0; i < n_nodes; i++) {
        if (lambda[i] == 0 || (lambda[i] == 1 && Tj[Tp[i]] == i))
            splitting[i] = F_NODE;
    }

    // Add elements to C and F, in descending order of lambda
    for (I top_index=(n_nodes - 1); top_index>-1; top_index--) {
        
        I i        = index_to_node[top_index];
        I lambda_i = lambda[i];

        // Remove i from its interval
        interval_count[lambda_i]--;

        // ----------------- Sorting every iteration = O(n^2) complexity ----------------- //
        // Search over this interval to make sure we process nodes in descending node order
        // I max_node = i;
        // I max_index = top_index;
        // for (I j = interval_ptr[lambda_i]; j < interval_ptr[lambda_i] + interval_count[lambda_i]; j++) {
        //     if (index_to_node[j] > max_node) {
        //         max_node = index_to_node[j];
        //         max_index = j;
        //     }
        // }
        // node_to_index[index_to_node[top_index]] = max_index;
        // node_to_index[index_to_node[max_index]] = top_index;
        // std::swap(index_to_node[top_index], index_to_node[max_index]);
        // i = index_to_node[top_index];

        // If maximum lambda = 0, break out of loop
        if (lambda[i] <= 0) {
            break;
        }

        // If node is unmarked, set maximum node as C-node and modify
        // lambda values in neighborhood
        if ( splitting[i] == U_NODE) {
            splitting[i] = C_NODE;

            // For each j in S^T_i /\ U, mark j as tentative F-point
            for(I jj = Tp[i]; jj < Tp[i+1]; jj++) {
                I j = Tj[jj];
                if(splitting[j] == U_NODE) {
                    splitting[j] = PRE_F_NODE;
                }
            }

            // For each j in S^T_i /\ U marked as tentative F-point, modify lamdba
            // values for neighborhood of j
            for(I jj = Tp[i]; jj < Tp[i+1]; jj++)
            {
                I j = Tj[jj];
                if(splitting[j] == PRE_F_NODE)
                {
                    splitting[j] = F_NODE;
                    
                    // For each k in S_j /\ U, modify lambda value, lambda_k += 1
                    for(I kk = Sp[j]; kk < Sp[j+1]; kk++){
                        I k = Sj[kk];

                        if(splitting[k] == U_NODE){

                            // Move k to the end of its current interval
                            if(lambda[k] >= n_nodes - 1) {
                                continue;
                            }

                            I lambda_k = lambda[k];
                            I old_pos  = node_to_index[k];
                            I new_pos  = interval_ptr[lambda_k] + interval_count[lambda_k] - 1;

                            node_to_index[index_to_node[old_pos]] = new_pos;
                            node_to_index[index_to_node[new_pos]] = old_pos;
                            std::swap(index_to_node[old_pos], index_to_node[new_pos]);

                            // Update intervals
                            interval_count[lambda_k]   -= 1;
                            interval_count[lambda_k+1] += 1; //invalid write!
                            interval_ptr[lambda_k+1]    = new_pos;

                            // Increment lambda_k
                            lambda[k]++;
                        }
                    }
                }
            }

            // For each j in S_i /\ U, set lambda_j -= 1
            for (I jj = Sp[i]; jj < Sp[i+1]; jj++) {
                I j = Sj[jj];
                // Decrement lambda for node j
                if (splitting[j] == U_NODE) {
                    if (lambda[j] == 0) {
                        continue;
                    }

                    // Move j to the beginning of its current interval
                    I lambda_j = lambda[j];
                    I old_pos  = node_to_index[j];
                    I new_pos  = interval_ptr[lambda_j];

                    node_to_index[index_to_node[old_pos]] = new_pos;
                    node_to_index[index_to_node[new_pos]] = old_pos;
                    std::swap(index_to_node[old_pos],index_to_node[new_pos]);

                    // Update intervals
                    interval_count[lambda_j]   -= 1;
                    interval_count[lambda_j-1] += 1;
                    interval_ptr[lambda_j]     += 1;
                    interval_ptr[lambda_j-1]    = interval_ptr[lambda_j] - interval_count[lambda_j-1];

                    // Decrement lambda_j
                    lambda[j]--;
                }
            }
        }
    }

    // Set any unmarked nodes as F-points
    for (I i=0; i<n_nodes; i++) {
        if (splitting[i] == U_NODE) {
            splitting[i] = F_NODE;
        }
    }
}


template<class I>
void rs_cf_splitting_pass2(const I n_nodes,
                           const I Sp[], const int Sp_size,
                           const I Sj[], const int Sj_size,
                                 I splitting[], const int splitting_size)
{

    // For each F-point
    for (I row=0; row<n_nodes; row++) {
        if (splitting[row] == F_NODE) {

            // Tentative C-point count
            I Cpt0 = -1;

            // For each j in S_row /\ F, test dependence of j on S_row /\ C
            for (I jj=Sp[row]; jj<Sp[row+1]; jj++) {
                I j = Sj[jj];

                if (splitting[j] == F_NODE) {

                    // Test dependence, i.e. check that S_j /\ S_row /\ C is
                    // nonempty. This is simply checking that nodes j and row
                    // have a common strong C-point connection.
                    bool dependence = false;
                    for (I ii=Sp[row]; ii<Sp[row+1]; ii++) {
                        I row_ind = Sj[ii];
                        if (splitting[row_ind] == C_NODE) {
                            for (I kk=Sp[j]; kk<Sp[j+1]; kk++) {
                                if (Sj[kk] == row_ind) {
                                    dependence = true;
                                }
                            }
                        }
                        if (dependence) {
                            break;
                        }
                    }

                    // Node j passed dependence test
                    if (dependence) {
                        continue;   
                    }
                    // Node j did not pass dependence test
                    else {
                        // If no tentative C-point, mark j as tentative C-point
                        if (Cpt0 < 0) {
                            Cpt0 = j;
                            splitting[j] = C_NODE;
                        }
                        // If there is a tentative C-point already, put it back in
                        // set of F-points and mark j as tentative C-point.
                        else {
                            splitting[Cpt0] = F_NODE;
                            Cpt0 = j;
                            splitting[j] = C_NODE;
                        }
                    }
                }
            }
        }
    }
}


/*
 *  Compute a CLJP splitting
 *
 *  Parameters
 *      n          - number of rows in A (number of vertices)
 *      Sp[]       - CSR row pointer (strength matrix)
 *      Sj[]       - CSR index array
 *      Tp[]       - CSR row pointer (transpose of the strength matrix)
 *      Tj[]       - CSR index array
 *      splitting  - array to store the C/F splitting
 *      colorflag  - flag to indicate coloring
 *
 *  Notes:
 *      The splitting array must be preallocated.
 *      CLJP naive since it requires the transpose.
 */

template<class I>
void cljp_naive_splitting(const I n,
                          const I Sp[], const int Sp_size,
                          const I Sj[], const int Sj_size,
                          const I Tp[], const int Tp_size,
                          const I Tj[], const int Tj_size,
                                I splitting[], const int splitting_size,
                          const I colorflag)
{
  // initialize sizes
  int ncolors;
  I unassigned = n;
  I nD;
  int nnz = Sp[n];

  // initialize vectors
  // complexity = 5n
  // storage = 4n
  std::vector<int> edgemark(nnz,1);
  std::vector<int> coloring(n);
  std::vector<double> weight(n);
  std::vector<I> D(n,0);      // marked nodes  in the ind set
  std::vector<I> Dlist(n,0);      // marked nodes  in the ind set
  std::fill(splitting, splitting + n, U_NODE);
  int * c_dep_cache = new int[n];
  std::fill_n(c_dep_cache, n, -1);

  // INITIALIZE WEIGHTS
  // complexity = O(n^2)?!? for coloring
  // or
  // complexity = n for random
  if(colorflag==1){ // with coloring
    //vertex_coloring_jones_plassmann(n, Sp, Sj, &coloring[0],&weight[0]);
    //vertex_coloring_IDO(n, Sp, Sj, &coloring[0]);
    vertex_coloring_mis(n, Sp, Sp_size, Sj, Sj_size, &coloring[0], n);
    ncolors = *std::max_element(coloring.begin(), coloring.end()) + 1;
    for(I i=0; i < n; i++){
      weight[i] = double(coloring[i])/double(ncolors);
    }
  }
  else {
    srand(2448422);
    for(I i=0; i < n; i++){
      weight[i] = double(rand())/RAND_MAX;
    }
  }

  for(I i=0; i < n; i++){
    for(I jj = Sp[i]; jj < Sp[i+1]; jj++){
      I j = Sj[jj];
      if(i != j) {
        weight[j]++;
      }
    }
  }
  // end INITIALIZE WEIGHTS

  // SELECTION LOOP
  I pass = 0;
  while(unassigned > 0){
    pass++;

    // SELECT INDEPENDENT SET
    // find i such that w_i > w_j for all i in union(S_i,S_i^T)
    nD = 0;
    for(I i=0; i<n; i++){
      if(splitting[i]==U_NODE){
        D[i] = 1;
        // check row (S_i^T)
        for(I jj = Sp[i]; jj < Sp[i+1]; jj++){
          I j = Sj[jj];
          if(splitting[j]==U_NODE && weight[j]>weight[i]){
            D[i] = 0;
            break;
          }
        }
        // check col (S_i)
        if(D[i] == 1) {
          for(I jj = Tp[i]; jj < Tp[i+1]; jj++){
            I j = Tj[jj];
            if(splitting[j]==U_NODE && weight[j]>weight[i]){
              D[i] = 0;
              break;
            }
          }
        }
        if(D[i] == 1) {
          Dlist[nD] = i;
          unassigned--;
          nD++;
        }
      }
      else{
        D[i]=0;
      }
    } // end for
    for(I i = 0; i < nD; i++) {
      splitting[Dlist[i]] = C_NODE;
    }
    // end SELECT INDEPENDENT SET

    // UPDATE WEIGHTS
    // P5
    // nbrs that influence C points are not good C points
    for(I iD=0; iD < nD; iD++){
      I c = Dlist[iD];
      for(I jj = Sp[c]; jj < Sp[c+1]; jj++){
        I j = Sj[jj];
        // c <---j
        if(splitting[j]==U_NODE && edgemark[jj] != 0){
          edgemark[jj] = 0;  // "remove" edge
          weight[j]--;
          if(weight[j]<1){
            splitting[j] = F_NODE;
            unassigned--;
          }
        }
      }
    } // end P5

    // P6
    // If k and j both depend on c, a C point, and j influces k, then j is less
    // valuable as a C point.
    for(I iD=0; iD < nD; iD++){
      I c = Dlist[iD];
      for(I jj = Tp[c]; jj < Tp[c+1]; jj++){
        I j = Tj[jj];
        if(splitting[j]==U_NODE)                 // j <---c
          c_dep_cache[j] = c;
      }

      for(I jj = Tp[c]; jj < Tp[c+1]; jj++) {
        I j = Tj[jj];
        for(I kk = Sp[j]; kk < Sp[j+1]; kk++) {
          I k = Sj[kk];
          if(splitting[k] == U_NODE && edgemark[kk] != 0) { // j <---k
            // does c ---> k ?
            if(c_dep_cache[k] == c) {
              edgemark[kk] = 0; // remove edge
              weight[k]--;
              if(weight[k] < 1) {
                splitting[k] = F_NODE;
                unassigned--;
                //kk = Tp[j+1]; // to break second loop
              }
            }
          }
        }
      }
    } // end P6
  }
  // end SELECTION LOOP

  for(I i = 0; i < Sp[n]; i++){
    if(edgemark[i] == 0){
      edgemark[i] = -1;
    }
  }
  for(I i = 0; i < n; i++){
    if(splitting[i] == U_NODE){
      splitting[i] = F_NODE;
    }
  }
  delete[] c_dep_cache;
}


/*
 *   Produce the Ruge-Stuben prolongator using "Direct Interpolation"
 *
 *
 *   The first pass uses the strength of connection matrix 'S'
 *   and C/F splitting to compute the row pointer for the prolongator.
 *
 *   The second pass fills in the nonzero entries of the prolongator
 *
 *   Reference:
 *      Page 479 of "Multigrid"
 *
 */
template<class I>
void rs_direct_interpolation_pass1(const I n_nodes,
                                   const I Sp[], const int Sp_size,
                                   const I Sj[], const int Sj_size,
                                   const I splitting[], const int splitting_size,
                                         I Bp[], const int Bp_size)
{
    I nnz = 0;
    Bp[0] = 0;
    for(I i = 0; i < n_nodes; i++){
        if( splitting[i] == C_NODE ){
            nnz++;
        }
        else {
            for(I jj = Sp[i]; jj < Sp[i+1]; jj++){
                if ( (splitting[Sj[jj]] == C_NODE) && (Sj[jj] != i) )
                    nnz++;
            }
        }
        Bp[i+1] = nnz;
    }
}


template<class I, class T>
void rs_direct_interpolation_pass2(const I n_nodes,
                                   const I Ap[], const int Ap_size,
                                   const I Aj[], const int Aj_size,
                                   const T Ax[], const int Ax_size,
                                   const I Sp[], const int Sp_size,
                                   const I Sj[], const int Sj_size,
                                   const T Sx[], const int Sx_size,
                                   const I splitting[], const int splitting_size,
                                   const I Bp[], const int Bp_size,
                                         I Bj[], const int Bj_size,
                                         T Bx[], const int Bx_size)
{

    for(I i = 0; i < n_nodes; i++){
        if(splitting[i] == C_NODE){
            Bj[Bp[i]] = i;
            Bx[Bp[i]] = 1;
        }
        else {
            T sum_strong_pos = 0, sum_strong_neg = 0;
            for(I jj = Sp[i]; jj < Sp[i+1]; jj++){
                if ( (splitting[Sj[jj]] == C_NODE) && (Sj[jj] != i) ){
                    if (Sx[jj] < 0)
                        sum_strong_neg += Sx[jj];
                    else
                        sum_strong_pos += Sx[jj];
                }
            }

            T sum_all_pos = 0, sum_all_neg = 0;
            T diag = 0;
            for(I jj = Ap[i]; jj < Ap[i+1]; jj++){
                if (Aj[jj] == i){
                    diag += Ax[jj];
                } else {
                    if (Ax[jj] < 0)
                        sum_all_neg += Ax[jj];
                    else
                        sum_all_pos += Ax[jj];
                }
            }

            T alpha = sum_all_neg / sum_strong_neg;
            T beta  = sum_all_pos / sum_strong_pos;

            if (sum_strong_pos == 0){
                diag += sum_all_pos;
                beta = 0;
            }

            T neg_coeff = -alpha/diag;
            T pos_coeff = -beta/diag;

            I nnz = Bp[i];
            for(I jj = Sp[i]; jj < Sp[i+1]; jj++){
                if ( (splitting[Sj[jj]] == C_NODE) && (Sj[jj] != i) ){
                    Bj[nnz] = Sj[jj];
                    if (Sx[jj] < 0)
                        Bx[nnz] = neg_coeff * Sx[jj];
                    else
                        Bx[nnz] = pos_coeff * Sx[jj];
                    nnz++;
                }
            }
        }
    }


    std::vector<I> map(n_nodes);
    for(I i = 0, sum = 0; i < n_nodes; i++){
        map[i]  = sum;
        sum    += splitting[i];
    }
    for(I i = 0; i < Bp[n_nodes]; i++){
        Bj[i] = map[Bj[i]];
    }
}


/*
 *   Produce the Ruge-Stuben prolongator using standard interpolation
 *
 *
 *   The first pass uses the strength of connection matrix 'S'
 *   and C/F splitting to compute the row pointer for the prolongator.
 *
 *   The second pass fills in the nonzero entries of the prolongator
 *
 *   Reference:
 *      Page 144 "A Multigrid Tutorial"
 *
 */
template<class I>
void rs_standard_interpolation_pass1(const I n_nodes,
                                   const I Sp[], const int Sp_size,
                                   const I Sj[], const int Sj_size,
                                   const I splitting[], const int splitting_size,
                                         I Bp[], const int Bp_size)
{
    I nnz = 0;
    Bp[0] = 0;
    for(I i = 0; i < n_nodes; i++){
        if( splitting[i] == C_NODE ){
            nnz++;
        } else {
            for(I jj = Sp[i]; jj < Sp[i+1]; jj++){
                if ( (splitting[Sj[jj]] == C_NODE) && (Sj[jj] != i) )
                    nnz++;
            }
        }
        Bp[i+1] = nnz;
    }
}

template<class I, class T>
void rs_standard_interpolation_pass2(const I n_nodes,
                                   const I Ap[], const int Ap_size,
                                   const I Aj[], const int Aj_size,
                                   const T Ax[], const int Ax_size,
                                   const I Sp[], const int Sp_size,
                                   const I Sj[], const int Sj_size,
                                   const T Sx[], const int Sx_size,
                                   const I splitting[], const int splitting_size,
                                   const I Bp[], const int Bp_size,
                                         I Bj[], const int Bj_size,
                                         T Bx[], const int Bx_size)
{

    for(I i = 0; i < n_nodes; i++) {
        // If node i is a C-point, then set interpolation as injection
        if(splitting[i] == C_NODE) {
            Bj[Bp[i]] = i;
            Bx[Bp[i]] = 1;
        } 
        // Otherwise, use RS standard interpolation formula
        else {

            // Calculate denominator
            T denominator = 0;

            // Start by summing entire row of A
            for(I mm = Ap[i]; mm < Ap[i+1]; mm++) {
                denominator += Ax[mm];
            }

            // Then subtract off the strong connections so that you are left with 
            // denominator = a_ii + sum_{m in weak connections} a_im
            for(I mm = Sp[i]; mm < Sp[i+1]; mm++) {
                if ( Sj[mm] != i ) denominator -= Sx[mm]; // making sure to leave the diagonal entry in there
            }

            // Set entries in P (interpolation weights w_ij from strongly connected C-points)
            I nnz = Bp[i];
            for(I jj = Sp[i]; jj < Sp[i+1]; jj++) {
                if ( (splitting[Sj[jj]] == C_NODE) && (Sj[jj] != i) ) {
                    // Set temporary value for Bj to be mapped to appropriate coarse-grid
                    // column index later and get column index j
                    Bj[nnz] = Sj[jj];
                    I j = Sj[jj];

                    // Initialize numerator as a_ij
                    T numerator = Sx[jj];
                    // printf("  numerator initialized to %e\n", numerator);
                    // Sum over strongly connected fine points
                    for(I kk = Sp[i]; kk < Sp[i+1]; kk++) {
                        if ( (splitting[Sj[kk]] == F_NODE) && (Sj[kk] != i) ) {
                            // Get column index k
                            I k = Sj[kk];

                            // Get a_kj (have to search over k'th row in A for connection a_kj)
                            T a_kj = 0;
                            for(I search_ind = Ap[k]; search_ind < Ap[k+1]; search_ind++) {
                                if ( Aj[search_ind] == j ){
                                    a_kj = Ax[search_ind];
                                }
                            }

                            // If a_kj == 0, then we don't need to do any more work, otherwise
                            // proceed to account for node k's contribution
                            if (a_kj != 0) {
                                // Calculate sum for inner denominator (loop over strongly connected C-points)
                                T inner_denominator = 0;
                                I inner_denom_added_to = 0;
                                for(I ll = Sp[i]; ll < Sp[i+1]; ll++) {
                                    if ( (splitting[Sj[ll]] == C_NODE) && (Sj[ll] != i) ) {
                                        // Get column index l
                                        I l = Sj[ll];
                                        // Add connection a_kl if present in matrix (search over kth row in A for connection)
                                        for(I search_ind = Ap[k]; search_ind < Ap[k+1]; search_ind++) {
                                            // Note: we check here to make sure a_kj and a_kl are same sign
                                            if ( Aj[search_ind] == l && a_kj*Ax[search_ind] > 0) {
                                                inner_denom_added_to = 1;
                                                inner_denominator += Ax[search_ind];
                                            }
                                        }
                                    }
                                }

                                // Add a_ik*a_kj/inner_denominator to the numerator 
                                if (inner_denominator == 0 && !inner_denom_added_to) {
                                    printf("Inner denominator was zero: there was a strongly " \
                                        "connected fine point with no connections to points in C_i\n");
                                }
                                if (inner_denominator == 0 && inner_denom_added_to) {
                                    printf("Inner denominator was zero due to cancellations!\n");
                                }
                                numerator += Sx[kk]*a_kj/inner_denominator;
                            }
                        }
                    }
                    // Set w_ij = -numerator/denominator
                    if (denominator == 0) {
                        printf("Outer denominator was zero: diagonal plus sum of weak connections was zero\n");
                    }
                    Bx[nnz] = -numerator/denominator;
                    nnz++;
                }
            }
        }
    }

    std::vector<I> map(n_nodes);
    for(I i = 0, sum = 0; i < n_nodes; i++) {
        map[i]  = sum;
        sum    += splitting[i];
    }
    for(I i = 0; i < Bp[n_nodes]; i++) {
        Bj[i] = map[Bj[i]];
    }
}


/* Helper function for compatible relaxation to perform steps 3.1d - 3.1f
 * in Falgout / Brannick (2010).  
 *
 * Input:
 * ------
 * A_rowptr : const {int array}
 *      Row pointer for sparse matrix in CSR format.
 * A_colinds : const {int array}
 *      Column indices for sparse matrix in CSR format.
 * B : const {float array}
 *      Target near null space vector for computing candidate set measure. 
 * e : {float array}
 *      Relaxed vector for computing candidate set measure.
 * indices : {int array}
 *      Array of indices, where indices[0] = the number of F indices, nf,
 *      followed by F indices in elements 1:nf, and C indices in (nf+1):n.
 * splitting : {int array}
 *      Integer array with current C/F splitting of nodes, 0 = C-point,
 *      1 = F-point. 
 * gamma : {float array}
 *      Preallocated vector to store candidate set measure.  
 * thetacs : const {float}
 *      Threshold for coarse grid candidates from set measure. 
 *
 * Returns:
 * --------  
 * Nothing, updated C/F-splitting and corresponding indices modified in place. 
 */
template<class I, class T>
void cr_helper(const I A_rowptr[], const int A_rowptr_size,
               const I A_colinds[], const int A_colinds_size, 
               const T B[], const int B_size,
               T e[], const int e_size,
               I indices[], const int indices_size,
               I splitting[], const int splitting_size,
               T gamma[], const int gamma_size,
               const T thetacs, 
               T cost[], const int cost_size )
{
    const T &Annz = A_colinds_size;
    const I &n = splitting_size;
    I &num_Fpts = indices[0];

    // Steps 3.1d, 3.1e in Falgout / Brannick (2010)
    // Divide each element in e by corresponding index in initial target vector.
    // Get inf norm of new e.
    T inf_norm = 0;
    for (I i=1; i<(num_Fpts+1); i++) {
        I pt = indices[i];
        e[pt] = std::abs(e[pt] / B[pt]);
        if (e[pt] > inf_norm) {
            inf_norm = e[pt];
        }   
    }
    cost[0] += num_Fpts / Annz;

    // Compute candidate set measure, pick coarse grid candidates.
    std::vector<I> Uindex;
    for (I i=1; i<(num_Fpts+1); i++) {
        I pt = indices[i];
        gamma[pt] = e[pt] / inf_norm; 
        if (gamma[pt] > thetacs) {
            Uindex.push_back(pt);
        }
    }
    I set_size = Uindex.size();
    cost[0] += num_Fpts / Annz;

    // Step 3.1f in Falgout / Brannick (2010)
    // Find weights: omega_i = |N_i\C| + gamma_i
    std::vector<T> omega(n,0);
    for (I i=0; i<set_size; i++) {
        I pt = Uindex[i];
        I num_neighbors = 0;
        I A_ind0 = A_rowptr[pt];
        I A_ind1 = A_rowptr[pt+1];
        for (I j=A_ind0; j<A_ind1; j++) {
            I neighbor = A_colinds[j];
            if (splitting[neighbor] == 0) {
                num_neighbors += 1;
            }
        }
        omega[pt] = num_neighbors + gamma[pt];
    }

    // Form maximum independent set
    while (true) {
        // 1. Add point i in U with maximal weight to C 
        T max_weight = 0;
        I new_pt = -1;
        for (I i=0; i<set_size; i++) {
            I pt = Uindex[i];
            if (omega[pt] > max_weight) {
                max_weight = omega[pt];
                new_pt = pt;
            }
        }
        // If all points have zero weight (index set is empty) break loop
        if (new_pt < 0) {
            break;
        }
        splitting[new_pt] = 1;
        gamma[new_pt] = 0;

        // 2. Remove from candidate set all nodes connected to 
        // new C-point by marking weight zero.
        std::vector<I> neighbors;
        I A_ind0 = A_rowptr[new_pt];
        I A_ind1 = A_rowptr[new_pt+1];
        for (I i=A_ind0; i<A_ind1; i++) {
            I temp = A_colinds[i];
            neighbors.push_back(temp);
            omega[temp] = 0;
        }

        // 3. For each node removed in step 2, set the weight for 
        // each of its neighbors still in the candidate set +1.
        I num_neighbors = neighbors.size();
        for (I i=0; i<num_neighbors; i++) {
            I pt = neighbors[i];
            I A_ind0 = A_rowptr[pt];
            I A_ind1 = A_rowptr[pt+1];
            for (I j=A_ind0; j<A_ind1; j++) {
                I temp = A_colinds[j];
                if (omega[temp] != 0) {
                    omega[temp] += 1;                   
                }
            }
        }
    }

    // Reorder indices array, with the first element giving the number
    // of F indices, nf, followed by F indices in elements 1:nf, and 
    // C indices in (nf+1):n. Note, C indices sorted largest to smallest.
    num_Fpts = 0;
    I next_Find = 1;
    I next_Cind = n;
    for (I i=0; i<n; i++) {
        if (splitting[i] == 0) {
            indices[next_Find] = i;
            next_Find += 1;
            num_Fpts += 1;
        }
        else {
            indices[next_Cind] = i;
            next_Cind -= 1;
        }
    }
}


/* Interpolate C-points by injection, and each F-point by injection from
 * its strongest connected C-point. 
 * 
 * Parameters
 * ----------
 *      rowptr : const array<int> 
 *          Pre-determined row-pointer for P in CSR format
 *      colinds : array<int>
 *          Empty array for column indices for P in CSR format
 *      C_rowptr : const array<int>
 *          Row pointer for SOC matrix, C
 *      C_colinds : const array<int>
 *          Column indices for SOC matrix, C
 *      C_data : const array<float>
 *          Data array for SOC matrix, C
 *      splitting : const array<int>
 *          Boolean array with 1 denoting C-points and 0 F-points
 *
 * Returns
 * -------
 * Nothing, colinds[] modified in place.
 *
 */
template<class I, class T>
void injection_interpolation(const I rowptr[], const int rowptr_size,
                                   I colinds[], const int colinds_size,
                             const I C_rowptr[], const int C_rowptr_size,
                             const I C_colinds[], const int C_colinds_size,
                             const T C_data[], const int C_data_size,
                             const I splitting[], const int splitting_size )
{
    I n = rowptr_size-1;

    // Get enumeration of C-points, where if i is the jth C-point,
    // then pointInd[i] = j.
    std::vector<I> pointInd(n);
    pointInd[0] = 0;
    for (I i=1; i<n; i++) {
        pointInd[i] = pointInd[i-1] + splitting[i-1];
    }

    // Build interpolation operator as CSR matrix
    I next = 0;
    for (I row=0; row<n; row++) {

        // Set C-point as identity
        if (splitting[row] == C_NODE) {
            colinds[next] = pointInd[row];
            next += 1;
        }
        // For F-points, find strongest connection to C-point
        // and interpolate directly from C-point. 
        else {

            T max = 0.0;
            I ind = 0;
            for (I i=C_rowptr[row]; i<C_rowptr[row+1]; i++) {
                if (splitting[C_colinds[i]] == C_NODE) {
                    if (std::abs(C_data[i]) > max) {
                        max = C_data[i];
                        ind = C_colinds[i];
                    }
                }
            }
            colinds[next] = pointInd[ind];
            next += 1;
        }
    }
}


/* Sorting function for approx_ideal_restriction_pass1, doesn't like being
 * templated I, T for some compilers. Needs to be outside scope of function.
 */
bool sort_2nd(const std::pair<int,double> &left,const std::pair<int,double> &right)
{
       return left.second < right.second;
}
// bool sort_2nd(const std::pair<int,float> &left,const std::pair<int,float> &right)
// {
//        return left.second < right.second;
// }


/* Build row_pointer for approximate ideal restriction in CSR form.
 * 
 * Parameters
 * ----------
 *      rowptr : array<int> 
 *          Empty row-pointer for R
 *      C_rowptr : const array<int>
 *          Row pointer for SOC matrix, C
 *      C_colinds : const array<int>
 *          Column indices for SOC matrix, C
 *      C_data : array<float>
 *          Data array for SOC matrix, C
 *      Cpts : array<int>
 *          List of global C-point indices
 *      splitting : const array<int>
 *          Boolean array with 1 denoting C-points and 0 F-points
 *      max_row : int
 *          Maximum size of sparsity pattern for one row of R
 *
 * Returns
 * -------
 * Nothing, rowptr[] modified in place. If a given C-point has more strong
 * F-connections than max_row, the smallest are set to zero in C_data.
 *
 */
template<class I, class T>
void approx_ideal_restriction_pass1(      I rowptr[], const int rowptr_size,
                                    const I C_rowptr[], const int C_rowptr_size,
                                    const I C_colinds[], const int C_colinds_size,
                                          T C_data[], const int C_data_size,
                                    const I Cpts[], const int Cpts_size,
                                    const I splitting[], const int splitting_size,
                                    const I max_row = std::numeric_limits<I>::max() )
{
    I nnz = 0;
    rowptr[0] = 0;

    // Deterimine number of nonzeros in each row of R.
    for (I row=0; row<Cpts_size; row++) {
        I cpoint = Cpts[row];

        // Determine number of strongly connected F-points in sparsity for R.
        // Store strength values and indices.
        std::vector<std::pair<I,T> > neighborhood;
        for (I i=C_rowptr[cpoint]; i<C_rowptr[cpoint+1]; i++) {
            if ( (splitting[C_colinds[i]] == F_NODE) && (std::abs(C_data[i]) > 1e-16) ) {
                neighborhood.push_back(std::make_pair(i, C_data[i]));
            }
        }

        // If neighborhood is larger than the maximum nonzeros per row, sort
        // F-point neighborhood by strength, set SOC matrix equal to zero in
        // smallest elements (i.e. no longer strongly connected).
        I size = neighborhood.size();
        if (size > max_row) {
            std::sort(neighborhood.begin(), neighborhood.end(), sort_2nd);
            for (I i=max_row; i<size; i++) {
                C_data[neighborhood[i].first] = 0;
            }
        }

        // Set row-pointer for this row of R (including identity on C-points).
        nnz += (1 + std::min(size, max_row));
        rowptr[row+1] = nnz; 
    }
}


/* Build column indices and data array for approximate ideal restriction
 * in CSR format.
 * 
 * Parameters
 * ----------
 *      rowptr : const array<int> 
 *          Pre-determined row-pointer for R in CSR format
 *      colinds : array<int>
 *          Empty array for column indices for R in CSR format
 *      data : array<float>
 *          Empty array for data for R in CSR format
 *      A_rowptr : const array<int>
 *          Row pointer for matrix A
 *      A_colinds : const array<int>
 *          Column indices for matrix A
 *      A_data : const array<float>
 *          Data array for matrix A
 *      C_rowptr : const array<int>
 *          Row pointer for SOC matrix, C
 *      C_colinds : const array<int>
 *          Column indices for SOC matrix, C
 *      C_data : const array<float>
 *          Data array for SOC matrix, C
*      Cpts : array<int>
 *          List of global C-point indices
 *      splitting : const array<int>
 *          Boolean array with 1 denoting C-points and 0 F-points
 *
 * Returns
 * -------
 * Nothing, colinds[] and data[] modified in place.
 *
 */
// Note, pass in A, C as CSR matrices (not A^T or C^T)
template<class I, class T>
void approx_ideal_restriction_pass2(const I rowptr[], const int rowptr_size,
                                          I colinds[], const int colinds_size,
                                          T data[], const int data_size,
                                    const I A_rowptr[], const int A_rowptr_size,
                                    const I A_colinds[], const int A_colinds_size,
                                    const T A_data[], const int A_data_size,
                                    const I C_rowptr[], const int C_rowptr_size,
                                    const I C_colinds[], const int C_colinds_size,
                                    const T C_data[], const int C_data_size,
                                    const I Cpts[], const int Cpts_size,
                                    const I splitting[], const int splitting_size,
                                    const I use_gmres = 0,
                                    const I maxiter = 10,
                                    const I precondition = 1 )
{
    I is_col_major = true;

    // Build column indices and data for each row of R.
    for (I row=0; row<Cpts_size; row++) {

        I cpoint = Cpts[row];
        I ind = rowptr[row];

        // Set column indices for R as strongly connected F-points.
        for (I i=C_rowptr[cpoint]; i<C_rowptr[cpoint+1]; i++) {
            if ( (splitting[C_colinds[i]] == F_NODE) && (std::abs(C_data[i]) > 1e-16) ) {
                colinds[ind] = C_colinds[i];
                ind +=1 ;
            }
        }

        if (ind != (rowptr[row+1]-1)) {
            std::cout << "Error: Row pointer does not agree with neighborhood size.\n\t"
                         "ind = " << ind << ", rowptr[row] = " << rowptr[row] <<
                         ", rowptr[row+1] = " << rowptr[row+1] << "\n";

        }

        // Build local linear system as the submatrix A restricted to the neighborhood,
        // Nf, of strongly connected F-points to the current C-point, that is A0 =
        // A[Nf, Nf]^T, stored in column major form. Since A in row-major = A^T in
        // column-major, A (CSR) is iterated through and A[Nf,Nf] stored in row-major.
        I size_N = ind - rowptr[row];
        std::vector<T> A0(size_N*size_N);
        I temp_A = 0;
        for (I j=rowptr[row]; j<ind; j++) { 
            I this_ind = colinds[j];
            for (I i=rowptr[row]; i<ind; i++) {
                // Search for indice in row of A
                I found_ind = 0;
                for (I k=A_rowptr[this_ind]; k<A_rowptr[this_ind+1]; k++) {
                    if (colinds[i] == A_colinds[k]) {
                        A0[temp_A] = A_data[k];
                        found_ind = 1;
                        temp_A += 1;
                        break;
                    }
                }
                // If indice not found, set element to zero
                if (found_ind == 0) {
                    A0[temp_A] = 0.0;
                    temp_A += 1;
                }
            }
        }

        // Build local right hand side given by b_j = -A_{cpt,N_j}, where N_j
        // is the jth indice in the neighborhood of strongly connected F-points
        // to the current C-point. 
        I temp_b = 0;
        std::vector<T> b0(size_N, 0);
        for (I i=rowptr[row]; i<ind; i++) {
            // Search for indice in row of A. If indice not found, b0 has been
            // intitialized to zero.
            for (I k=A_rowptr[cpoint]; k<A_rowptr[cpoint+1]; k++) {
                if (colinds[i] == A_colinds[k]) {
                    b0[temp_b] = -A_data[k];
                    break;
                }
            }
            temp_b += 1;
        }

        // Solve linear system (least squares solves exactly when full rank)
        // s.t. (RA)_ij = 0 for (i,j) within the sparsity pattern of R. Store
        // solution in data vector for R.
        if (use_gmres) {
            dense_GMRES(&A0[0], &b0[0], &data[rowptr[row]], size_N, is_col_major, maxiter, precondition);
        }
        else {
            least_squares(&A0[0], &b0[0], &data[rowptr[row]], size_N, size_N, is_col_major);
        }

        // Add identity for C-point in this row
        colinds[ind] = cpoint;
        data[ind] = 1.0;
    }
}




// Make sure data[] is passed in initialized to zero

template<class I, class T>
void block_approx_ideal_restriction_pass2(const I rowptr[], const int rowptr_size,
                                                I colinds[], const int colinds_size,
                                                T data[], const int data_size,
                                          const I A_rowptr[], const int A_rowptr_size,
                                          const I A_colinds[], const int A_colinds_size,
                                          const T A_data[], const int A_data_size,
                                          const I C_rowptr[], const int C_rowptr_size,
                                          const I C_colinds[], const int C_colinds_size,
                                          const T C_data[], const int C_data_size,
                                          const I Cpts[], const int Cpts_size,
                                          const I splitting[], const int splitting_size,
                                          const I blocksize )
{
    I is_col_major = true;

    // Build column indices and data for each row of R.
    for (I row=0; row<Cpts_size; row++) {

        I cpoint = Cpts[row];
        I ind = rowptr[row];

        // Set column indices for R as strongly connected F-points.
        for (I i=C_rowptr[cpoint]; i<C_rowptr[cpoint+1]; i++) {
            if ( (splitting[C_colinds[i]] == F_NODE) && (std::abs(C_data[i]) > 1e-16) ) {
                colinds[ind] = C_colinds[i];
                ind += 1 ;
            }
        }

        if (ind != (rowptr[row+1]-1)) {
            std::cout << "Error: Row pointer does not agree with neighborhood size.\n";
        }

        // Build local linear system as the submatrix A^T restricted to the neighborhood,
        // Nf, of strongly connected F-points to the current C-point, that is A0 =
        // A[Nf, Nf]^T, stored in column major form. Since A in row-major = A^T in
        // column-major, A (CSR) is iterated through and A[Nf,Nf] stored in row-major.
        //      - Initialize A0 to zero
        I size_N = ind - rowptr[row];
        I num_DOFs = size_N * blocksize;
        std::vector<T> A0(num_DOFs*num_DOFs, 0.0);
        I this_block_row = 0;

        // Add each block in strongly connected neighborhood to dense linear system.
        // For each column indice in sparsity pattern for this row of R:
        for (I j=rowptr[row]; j<ind; j++) { 
            I this_ind = colinds[j];
            I this_block_col = 0;

            // For this row of A, add blocks to A0 for each entry in sparsity pattern
            for (I i=rowptr[row]; i<ind; i++) {

                // Block row/column indices to normal row/column indices
                I this_row = this_block_row*blocksize;
                I this_col = this_block_col*blocksize;

                // Search for indice in row of A
                for (I k=A_rowptr[this_ind]; k<A_rowptr[this_ind+1]; k++) {

                    // Add block of A to dense array. If indice not found, elements
                    // in A0 have already been initialized to zero.
                    if (colinds[i] == A_colinds[k]) {
                        I block_data_ind = k * blocksize * blocksize;

                        // For each row in block:
                        for (I block_row=0; block_row<blocksize; block_row++) {
                            I row_maj_ind = (this_row + block_row) * num_DOFs + this_col;

                            // For each column in block:
                            for (I block_col=0; block_col<blocksize; block_col++) {

                                // Blocks of A stored in row-major in A_data
                                I A_data_ind = block_data_ind + block_row * blocksize + block_col;
                                A0[row_maj_ind + block_col] = A_data[A_data_ind];
                                if ((row_maj_ind + block_col) > num_DOFs*num_DOFs) {
                                    std::cout << "Warning: Accessing out of bounds index building A0.\n";
                                }
                            }
                        }
                        break;
                    }
                }
                // Increase block column count
                this_block_col += 1;
            }
            // Incresae block row count
            this_block_row += 1;
        }

        // Take QR of local matrix for linear solves, R stored in A0
        std::vector<T> Q = QR(&A0[0],num_DOFs,num_DOFs,is_col_major);

        // Build local right hand side given by blocks b_j = -A_{cpt,N_j}, where N_j
        // is the jth indice in the neighborhood of strongly connected F-points
        // to the current C-point, and c-point the global C-point index corresponding
        // to the current row of R. RHS for each row in block, stored in b0 at indices
        //      b0[0], b0[1*num_DOFs], ..., b0[ (blocksize-1)*num_DOFs ]
        // Mapping between this ordering, say row_ind, and bsr ordering given by
        //      for each block_ind:
        //          for each row in block:    
        //              for each col in block:
        //                  row_ind = num_DOFs*row + block_ind*blocksize + col
        //                  bsr_ind = block_ind*blocksize^2 + row*blocksize + col
        std::vector<T> b0(num_DOFs * blocksize, 0);
        for (I block_ind=0; block_ind<size_N; block_ind++) {
            I temp_ind = rowptr[row] + block_ind;

            // Search for indice in row of A, store data in b0. If not found,
            // b0 has been initialized to zero.
            for (I k=A_rowptr[cpoint]; k<A_rowptr[cpoint+1]; k++) {
                if (colinds[temp_ind] == A_colinds[k]) {
                    for (I this_row=0; this_row<blocksize; this_row++) {
                        for (I this_col=0; this_col<blocksize; this_col++) {
                            I row_ind = num_DOFs*this_row + block_ind*blocksize + this_col;
                            I bsr_ind = k*blocksize*blocksize + this_row*blocksize + this_col;
                            b0[row_ind] = -A_data[bsr_ind];
                        }
                    }
                    break;
                }
            }
        }

        // Solve local linear system for each row in block
        std::vector<T> rhs(num_DOFs);
        for (I this_row=0; this_row<blocksize; this_row++) {
            I b_ind0 = num_DOFs * this_row;

            // Multiply right hand side, rhs := Q^T*b
            for (I i=0; i<num_DOFs; i++) {
                rhs[i] = 0.0;
                for (I k=0; k<num_DOFs; k++) {
                    rhs[i] += b0[b_ind0 + k] * Q[col_major(k,i,num_DOFs)];
                }
            }

            // Solve upper triangular system from QR, store solution in b0
            upper_tri_solve(&A0[0], &rhs[0], &b0[b_ind0], num_DOFs, num_DOFs, is_col_major);
        }
            
        // Add solution for each block row to data array. See section on RHS for
        // mapping between bsr data array and row-major array solution stored in
        for (I block_ind=0; block_ind<size_N; block_ind++) {
            for (I this_row=0; this_row<blocksize; this_row++) {
                for (I this_col=0; this_col<blocksize; this_col++) {
                    I bsr_ind = rowptr[row]*blocksize*blocksize + block_ind*blocksize*blocksize + 
                                this_row*blocksize + this_col;
                    I row_ind = num_DOFs*this_row + block_ind*blocksize + this_col;
                    if (std::abs(b0[row_ind]) > 1e-16) {
                        data[bsr_ind] = b0[row_ind];                    
                    }
                    else {
                        data[bsr_ind] = 0.0;                   
                    }
                }
            }
        }

        // Add identity for C-point in this block row (assume data[] initialized to 0)
        colinds[ind] = cpoint;
        I identity_ind = ind*blocksize*blocksize;
        for (I this_row=0; this_row<blocksize; this_row++) {
            data[identity_ind + (blocksize+1)*this_row] = 1.0;
        }
    }
}





// template<class I>
// void rs_cf_splitting(const I C_rowptr[], const int C_rowptr_size,
//                      const I C_colinds[], const int C_colinds_size,
//                      const I Ct_rowptr[], const int Ct_rowptr_size,
//                      const I Ct_colinds[], const int Ct_colinds_size,
//                            I splitting[], const int splitting_size)
// {
//     I n_nodes = splitting_size;
//     std::vector<I> lambda(n_nodes,0);

//     // Compute initial lambda
//     I lambda_max = 0;
//     for(I i = 0; i < n_nodes; i++) {
//         lambda[i] = Tp[i+1] - Tp[i] + influence[i];
//         if (lambda[i] > lambda_max) {
//             lambda_max = lambda[i];
//         }
//     }










#endif
