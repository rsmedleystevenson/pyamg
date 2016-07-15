/*
 * Calculate A*B = S, but only at the pre-existing sparsity
 * pattern of S, i.e. do an exact, but incomplete mat-mat mult.
 * A is a dense m x d array stored in row-major, and B a dense 
 * d x n array stored in column-major, where generally d << n,m.
 * S is an m x n CSR matrix with a predefined sparsity pattern.
 *
 * Parameters
 * ----------
 * A  : {float|complex array}
 *      Dense m x d array stored in row-major
 * B  : {float|complex array}
 *      Dense d x n array stored in col-major
 * S_rowptr : {int array}
 *      CSR row pointer array
 * S_colinds : {int array}
 *      CSR col index array
 * S_data : {float|complex array}
 *      CSR value array
 * m  : {int}
 *      Number of rows in S / rows in A
 * d  : {int}
 *      Number of columns in A / rows in B
 * n  : {int}
 *      Number of columns in S / columns in B
 *
 * Returns
 * -------
 * Sx is modified in-place to reflect S(i,j) = <A_{i,:}, B_{:,j}>
 * but only for those entries already present in the sparsity pattern
 * of S.
 *
 * Notes
 * -----
 *
 */
template<class I, class T, class F>
void incomplete_mat_mult_dense2sparse(const T A[], const int Ap_size,
                                      const T B[], const int Bp_size,
                                      const I S_rowptr[], const int S_rowptr_size,
                                      const I S_colinds[], const int S_colinds_size,
                                            T S_data[], const int S_data_size,
                                      const I m,
                                      const I d,
                                      const I n)
{
    // Loop over each row in S
    for (I i=0; i<n; i++) {

        // Loop over each nonzero column in sparsity
        // pattern for this row
        for (I k=S_rowptr[i]; k<S_rowptr[i+1]; k++) {

            I j=S_colinds[k];
            S_data[k] = 0;

            // Form S_{ij} = A[i,:] * B[:,j]
            for (I l=0; l<d; l++) {
                S_data[k] += A[i*d + l] * B[j*d + l];
            }
        }
    }
}
