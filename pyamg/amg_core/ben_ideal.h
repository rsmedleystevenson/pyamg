#ifndef BEN_IDEAL_H
#define BEN_IDEAL_H

#include <iostream>
#include <cmath>
#include <vector>
#include <set>
#include "linalg.h"


/* 2d array index A[row,col] to row-major index. */
inline int row_major(const int &row, const int &col, const int &num_cols) 
{
    return row*num_cols + col;
}


/* 2d array index A[row,col] to column-major index. */
inline int col_major(const int &row, const int &col, const int &num_rows) 
{
    return col*num_rows + row;
}


/* Given an array of C-points, generate a vector separating
 * C and F-points. F-points are enumerated using 1-indexing
 * and negative numbers, while C-points are enumerated using
 * 1-indexing and positive numbers, in a vector splitting. 
 *
 * Parameters 
 * ----------
 *         Cpts : int array
 *             Array of Cpts
 *        numCpts : &int
 *            Length of Cpt array
 *        n : &int 
 *            Total number of points
 *
 * Returns
 * -------
 *         splitting - vector
 *            Vector of size n, indicating whether each point is a
 *            C or F-point, with corresponding index.
 *
 */
std::vector<int> get_ind_split(const int Cpts[],
                               const int & numCpts,
                               const int &n)
{
    std::vector<int> ind_split(n,0);
    for (int i=0; i<numCpts; i++) {
        ind_split[Cpts[i]] = 1;
    }
    int find = 1;
    int cind = 1;
    for (int i=0; i<n; i++) {
        if (ind_split[i] == 0) {
            ind_split[i] = -find;
            find += 1;
        }
        else {
            ind_split[i] = cind;
            cind += 1;
        }
    }
    return ind_split;
}


/* Generate column pointer for extracting a CSC submatrix from a
 * CSR matrix. Returns the maximum number of nonzeros in any column,
 * and teh col_ptr is modified in place.
 *
 * Parameters
 * ----------
 *
 *
 *
 *
 * Returns
 * -------
 *
 *
 */
int get_col_ptr(const int A_rowptr[],
                const int A_colinds[],
                const int &n,
                const int is_col_ind[],
                const int is_row_ind[],
                int colptr[], 
                const int &num_cols,
                const int &row_scale = 1,
                const int &col_scale = 1 )
{

    // Count instances of each col-ind submatrix
    for (int i=0; i<n; i++) {
        
        // Continue to next iteration if this row is not a row-ind
        if ( (row_scale*is_row_ind[i]) <= 0 ) {
            continue;
        }

        // Find all instances of col-inds in this row. Increase
        // column pointer to count total instances.
        //     - Note, is_col_ind[] is one-indexed, not zero.
        for (int k=A_rowptr[i]; k<A_rowptr[i+1]; k++) {
            int ind = col_scale * is_col_ind[A_colinds[k]];
            if ( ind > 0) {
                colptr[ind] += 1;
            }
        }
    }

    // Cumulative sum column pointer to correspond with data entries
    int max_nnz = 0;
    for (int i=1; i<=(num_cols); i++) {
        if (colptr[i] > max_nnz) {
            max_nnz = colptr[i];
        }
        colptr[i] += colptr[i-1];
    }
    return max_nnz;
}


/* Generate row_indices and data for CSC submatrix with col_ptr
 * determined in get_col_ptr(). Arrays are modified in place.
 *
 * Parameters 
 * ----------
 *
 *
 * Returns 
 * -------
 *
 *
 */
template<class I, class T>
void get_csc_submatrix(const I A_rowptr[],
                       const I A_colinds[],
                       const T A_data[],
                       const I &n,
                       const I is_col_ind[],
                       const I is_row_ind[],
                       I colptr[], 
                       I rowinds[], 
                       T data[],
                       const I &num_cols,
                       const I &row_scale = 1,
                       const I &col_scale = 1 )
{
    // Fill in rowinds and data for sparse submatrix
    for (I i=0; i<n; i++) {
        
        // Continue to next iteration if this row is not a row-ind
        if ( (row_scale*is_row_ind[i]) <= 0 ) {
            continue;
        }

        // Find all instances of col-inds in this row. Save row-ind
        // and data value in sparse structure. Increase column pointer
        // to mark where next data index is. Will reset after.
        //     - Note, is_col_ind[] is one-indexed, not zero.
        for (I k=A_rowptr[i]; k<A_rowptr[i+1]; k++) {
            I ind = col_scale * is_col_ind[A_colinds[k]];
            if (ind > 0) {
                I data_ind = colptr[ind-1];
                rowinds[data_ind] = std::abs(is_row_ind[i])-1;
                data[data_ind] = A_data[k];
                colptr[ind-1] += 1;
            }
        }
    }

    // Reset colptr for submatrix
    I prev = 0;
    for (I i=0; i<num_cols; i++) {
        I temp = colptr[i];
        colptr[i] = prev;
        prev = temp;
    }    
}


/* QR-decomposition using Householer transformations on dense
 * 2d array stored in either column- or row-major form. 
 * 
 * Parameters
 * ----------
 *         A : double array
 *            2d matrix A stored in 1d column- or row-major.
 *         m : &int
 *            Number of rows in A
 *        n : &int
 *            Number of columns in A
 *        is_col_major : bool
 *            True if A is stored in column-major, false
 *            if A is stored in row-major.
 *
 * Returns
 * -------
 *         Q : vector<double>
 *            Matrix Q stored in same format as A.
 *        R : in-place
 *            R is stored over A in place, in same format.
 *
 */
template<class I, class T>
std::vector<T> QR(T A[],
                       const I &m,
                       const I &n,
                       const I is_col_major)
{
    // Funciton pointer for row or column major matrices
    I (*get_ind)(const I&, const I&, const I&);
    const I *C;
    if (is_col_major) {
        get_ind = &col_major;
        C = &m;
    }
    else {
        get_ind = &row_major;
        C = &n;
    }

    // Initialize Q to identity
    std::vector<T> Q(m*m,0);
    for (I i=0; i<m; i++) {
        Q[get_ind(i,i,m)] = 1;
    }

    // Loop over columns of A using Householder reflections
    for (I j=0; j<n; j++) {

        // Break loop for short fat matrices
        if (m <= j) {
            break;
        }

        // Get norm of next column of A to be reflected. Choose sign
        // opposite that of A_jj to avoid catastrophic cancellation.
        T normx = 0;
        for (I i=j; i<m; i++) {
            T temp = A[get_ind(i,j,*C)];
            normx += temp*temp;
        }
        normx = std::sqrt(normx);
        normx *= -1*signof(A[get_ind(j,j,*C)]);

        // Form vector v for Householder matrix H = I - tau*vv^T
        // where v = R(j:end,j) / scale, v[0] = 1.
        T scale = A[get_ind(j,j,*C)] - normx;
        T tau = -scale / normx;
        std::vector<T> v(m-j,0);
        v[0] = 1;
        for (I i=1; i<(m-j); i++) {
            v[i] = A[get_ind(j+i,j,*C)] / scale;    
        }

        // Modify R in place, R := H*R, looping over columns then rows
        for (I k=j; k<n; k++) {

            // Compute the kth element of v^T * R
            T vtR_k = 0;
            for (I i=0; i<(m-j); i++) {
                vtR_k += v[i] * A[get_ind(j+i,k,*C)];
            }

            // Correction for each row of kth column, given by 
            // R_ik -= tau * v_i * (vtR_k)_k
            for (I i=0; i<(m-j); i++) {
                A[get_ind(j+i,k,*C)] -= tau * v[i] * vtR_k;
            }
        }

        // Modify Q in place, Q = Q*H
        for (I i=0; i<m; i++) {

            // Compute the ith element of Q * v
            T Qv_i = 0;
            for (I k=0; k<(m-j); k++) {
                Qv_i += v[k] * Q[get_ind(i,k+j,m)];
            }

            // Correction for each column of ith row, given by
            // Q_ik -= tau * Qv_i * v_k
            for (I k=0; k<(m-j); k++) { 
                Q[get_ind(i,k+j,m)] -= tau * v[k] * Qv_i;
            }
        }
    }

    return Q;
}


/* Backward substitution solve on upper-triangular linear system,
 * Rx = rhs, where R is stored in column- or row-major form. 
 * 
 * Parameters
 * ----------
 *        R : double array, length m*n
 *            Upper-triangular array stored in column- or row-major.
 *        rhs : double array, length m
 *            Right hand side of linear system
 *        x : double array, length n
 *            Preallocated array for solution
 *        m : &int
 *            Number of rows in R
 *        n : &int
 *            Number of columns in R
 *        is_col_major : bool
 *            True if R is stored in column-major, false
 *            if R is stored in row-major.
 *
 * Returns
 * -------
 *        Nothing, solution is stored in x[].
 *
 * Notes
 * -----
 * R need not be square, the system will be solved over the
 * rank r upper-triangular block. If remaining entries in
 * solution are unused, they will be set to zero.
 *
 */        
template<class I, class T>
void upper_tri_solve(const T R[],
                     const T rhs[],
                     T x[],
                     const I &m,
                     const I &n,
                     const I is_col_major)
{
    // Funciton pointer for row or column major matrices
    I (*get_ind)(const I&, const I&, const I&);
    const I *C;
    if (is_col_major) {
        get_ind = &col_major;
        C = &m;
    }
    else {
        get_ind = &row_major;
        C = &n;
    }

    // Backwards substitution
    I rank = std::min(m,n);
    for (I i=(rank-1); i>=0; i--) {
        T temp = rhs[i];
        for (I j=(i+1); j<rank; j++) {
            temp -= R[get_ind(i,j,*C)]*x[j];
        }
        if (std::abs(R[get_ind(i,i,*C)]) < 1e-12) {
            std::cout << "Warning: Upper triangular matrix near singular.\n"
                         "Dividing by ~ 0.\n";
        }
        x[i] = temp / R[get_ind(i,i,*C)];
    }

    // If rank < size of rhs, set free elements in x to zero
    for (I i=m; i<n; i++) {
        x[i] = 0;
    }
}


/* Forward substitution solve on lower-triangular linear system,
 * Lx = rhs, where L is stored in column- or row-major form. 
 * 
 * Parameters
 * ----------
 *        L : double array, length m*n
 *            Lower-triangular array stored in column- or row-major.
 *        rhs : double array, length m
 *            Right hand side of linear system
 *        x : double array, length n
 *            Preallocated array for solution
 *        m : &int
 *            Number of rows in L
 *        n : &int
 *            Number of columns in L
 *        is_col_major : bool
 *            True if L is stored in column-major, false
 *            if L is stored in row-major.
 *
 * Returns
 * -------
 *        Nothing, solution is stored in x[].
 *
 * Notes
 * -----
 * L need not be square, the system will be solved over the
 * rank r lower-triangular block. If remaining entries in
 * solution are unused, they will be set to zero.
 *
 */
template<class I, class T>
void lower_tri_solve(const T L[],
                     const T rhs[],
                     T x[],
                     const I &m,
                     const I &n,
                     const I is_col_major)
{
    // Funciton pointer for row or column major matrices
    I (*get_ind)(const I&, const I&, const I&);
    const I *C;
    if (is_col_major) {
        get_ind = &col_major;
        C = &m;
    }
    else {
        get_ind = &row_major;
        C = &n;
    }

    // Backwards substitution
    I rank = std::min(m,n);
    for (I i=0; i<rank; i++) {
        T temp = rhs[i];
        for (I j=0; j<i; j++) {
            temp -= L[get_ind(i,j,*C)]*x[j];
        }
        if (std::abs(L[get_ind(i,i,*C)]) < 1e-12) {
            std::cout << "Warning: Lower triangular matrix near singular.\n"
                         "Dividing by ~ 0.\n";
        }
        x[i] = temp / L[get_ind(i,i,*C)];
    }

    // If rank < size of rhs, set free elements in x to zero
    for (I i=m; i<n; i++) {
        x[i] = 0;
    }
}


/* Method to solve the linear least squares problem.
 *
 * Parameters
 * ----------
 *         A : double array, length m*n
 *            2d array stored in column- or row-major.
 *        b : double array, length m
 *            Right hand side of unconstrained problem.
 *        x : double array, length n
 *            Container for solution
 *         m : &int
 *            Number of rows in A
 *        n : &int
 *            Number of columns in A
 *        is_col_major : bool
 *            True if A is stored in column-major, false
 *            if A is stored in row-major.
 *
 * Returns
 * -------
 *         x : vector<double>
 *            Solution to constrained least sqaures problem.
 *
 * Notes
 * -----
 * If system is under determined, free entries are set to zero. 
 *
 */
template<class I, class T>
void least_squares(T A[],
                   T b[],
                   T x[],
                   const I &m,
                   const I &n,
                   const I is_col_major=0)
{
    // Funciton pointer for row or column major matrices
    I (*get_ind)(const I&, const I&, const I&);
    if (is_col_major) {
        get_ind = &col_major;
    }
    else {
        get_ind = &row_major;
    }

    // std::cout << "\t\t\tSet funtion handle" << std::endl;

    // Take QR of A
    std::vector<T> Q = QR(A,m,n,is_col_major);

    // std::cout << "\t\t\tTook QR" << std::endl;

    // Multiply right hand side, b:= Q^T*b. Have to make new vetor, rhs.
    std::vector<T> rhs(m,0);
    for (I i=0; i<m; i++) {
        for (I k=0; k<m; k++) {
            rhs[i] += b[k] * Q[get_ind(k,i,m)];
        }
    }

    // std::cout << "\t\t\tMultiply rhs" << std::endl;

    // Solve upper triangular system, store solution in x.
    upper_tri_solve(A,&rhs[0],x,m,n,is_col_major);
}


/* Method to solve an arbitrary constrained least squares.
 * Can be used one of two ways, with any shape and rank
 * operator A.
 *
 *     1. Suppose we want to solve
 *        x = argmin || xA - b || s.t. xC = d                 (1)
 *     Let C = QR, and make the change of variable z := xQ.
 *     Then (1) is equivalent to
 *        z = argmin || zQ^TA - b ||   s.t. zR = d
 *          = argmin || A^TQz^T - b || s.t. R^Tz^T = d         (2)
 *     This is solved by passing in A in *row major* and
 *     C in *column major.*
 *        - A is nxm
 *        - C is mxs
 *
 *     2. Suppose we want to solve
 *        x = argmin || Ax - b || s.t. Cx = d                 (3)
 *     Let C^T = QR, and make the change of variable z := Q^Tx.
 *     Then (1) is equivalent to
 *        z = argmin || AQx - b ||   s.t. R^Tz = d             (4)
 *     This is solved by passing in A in *column major* and
 *     C in *row major.* 
 *        - A is mxn
 *        - C is sxn
 *
 * Parameters
 * ----------
 *         A : vector<double>
 *            2d array stored in column- or row-major.
 *        b : vector<double>
 *            Right hand side of unconstrained problem.
 *        C : vector<double>
 *            Constraint operator, stored in opposite form as
 *            A. E.g. A in column-major means C in row-major.
 *        d : vector<double> 
 *            Right hand side of contraint equation.
 *        m : &int
 *            Number of columns if A is in row-major, number
 *            of rows if A is in column-major.
 *        n : &int
 *            Number of rows if A is in row-major, number
 *            of columns if A is in column-major.
 *        s : &int
 *            Number of constraints.
 *
 * Returns
 * -------
 *         x : vector<double>
 *            Solution to constrained least sqaures problem.
 *
 */
template<class I, class T>
std::vector<T> constrained_least_squares(std::vector<T> &A,
                                              std::vector<T> &b,
                                              std::vector<T> &C,
                                              std::vector<T> &d,
                                              const I &m,
                                              const I &n,
                                              const I &s)
{

    // Perform QR on matrix C. R is written over C in column major,
    // and Q is returned in column major. 
    std::vector<T> Qc = QR(&C[0],n,s,1);

    // Form matrix product S = A^T * Q. For A passed in as row
    // major, perform S = A * Q, assuming that A is in column major
    // (A^T row major = A column major).
    std::vector<T> temp_vec(n,0);
    for (I i=0; i<m; i++) {
        for (I j=0; j<n; j++) {
            T val = 0.0;
            for (I k=0; k<n; k++) {
                val += A[col_major(i,k,m)] * Qc[col_major(k,j,n)];
            }
            temp_vec[j] = val;
        }
        for (I j=0; j<n; j++) {
            A[col_major(i,j,m)] = temp_vec[j];
        }
    }

    // Change R to R^T. Probably dirtier, cheaper ways to use R^T
    // in place, don't think it's worth it.
    for (I i=1; i<s; i++) {
        for (I j=0; j<i; j++) {
            C[col_major(i,j,n)] = C[col_major(j,i,n)];
        }
    }

    // Satisfy constraIs R^Tz = d, move to rhs of LS
    lower_tri_solve(&C[0],&d[0],&temp_vec[0],n,s,1);
    for (I i=0; i<m; i++) {
        T val = 0.0;
        for (I j=0; j<s; j++) {
            val += A[col_major(i,j,m)] * temp_vec[j];
        }
        b[i] -= val;
    }

    // Call LS on reduced system
    I temp_ind = n-s;
    least_squares(&A[col_major(0,s,m)], &b[0], &temp_vec[s], m, temp_ind, 1);

    // Form x = Q*z
    std::vector<T> x(n,0);
    for (I i=0; i<n; i++) {
        x[i] = 0;
        for (I k=0; k<n; k++) {
            x[i] += Qc[col_major(i,k,n)] * temp_vec[k];
        }
    }

    return x;
}


/* Form interpolation operator using ben ideal interpolation. 
 * 
 * Parameters
 * ----------
 *        A_rowptr : int array
 *            Row pointer for A stored in CSR format.
 *        A_colinds : int array
 *            Column indices for A stored in CSR format.
 *        A_data : double array
 *            Data for A stored in CSR format.
 *        S_rowptr : int array
 *            Row pointer for sparsity pattern stored in CSR format.
 *        S_colinds : int array
 *            Column indices for sparsity pattern stored in CSR format.
 *        P_rowptr : int array
 *            Empty row pointer for interpolation operator stored in
 *            CSR format.
 *        B : double array
 *            Target bad guy vectors to be included in range of
 *            interpolation.
 *        Cpts : int array
 *            List of designated Cpts.
 *        n : int
 *            Degrees of freedom in A.
 *        num_bad_guys : int
 *            Number of target bad guys to include in range of
 *            interpolation.
 *
 * Returns
 * -------
 *         - An STD pair of vector<int> and vector<double>, where the
 *          vector<int> contains column indices for P in a CSR format,
 *          and the vector<double> corresponding data. In Python, this
 *           comes out as a length two tuple of tuples, where the inner
 *           tuples are the column indices and data, respectively. 
 *         - The row pointer for P is modified in place.
 *
 * Notes
 * -----
 * It is important that A has sorted indices before calling this
 * function, and the list of Cpts is sorted. 
 *
 */
//     - TODO : test middle section with new Acf submatrix
//        --> Use get_sub_mat testing function in sparse.cpp
//
template<class I, class T>
// std::pair<std::vector<int>, std::vector<T> > 
void ben_ideal_interpolation(const I A_rowptr[], const I A_rowptr_size,
                             const I A_colinds[], const I A_colinds_size,
                             const T A_data[], const I A_data_size,
                             const I S_rowptr[], const I S_rowptr_size,
                             const I S_colinds[], const I S_colinds_size,
                             I P_rowptr[], const I P_rowptr_size,
                             I P_colinds[], const I P_colinds_size,
                             T P_data[], const I P_data_size,
                             const T B[], const I B_size,
                             const I Cpts[], const I Cpts_size,
                             const I n,
                             const I num_bad_guys )
{
    // Get splitting of points in one vector, Cpts enumerated in positive,
    // one-indexed ordering and Fpts in negative. 
    //         E.g., [-1,1,2,-2,-3] <-- Fpts = [0,3,4], Cpts = [1,2]
    std::vector<I> splitting = get_ind_split(Cpts,Cpts_size,n);

    // Get sparse CSC column pointer for submatrix Acc. Final two arguments
    // positive to select (positive indexed) C-points for rows and columns.
    std::vector<I> Acc_colptr(Cpts_size+1,0);
    get_col_ptr(A_rowptr, A_colinds, n, &splitting[0],
                &splitting[0], &Acc_colptr[0],
                Cpts_size, 1, 1);

    // Allocate row-ind and data arrays for sparse submatrix 
    I nnz = Acc_colptr[Cpts_size];
    std::vector<I> Acc_rowinds(nnz,0);
    std::vector<T> Acc_data(nnz,0);

    // Fill in sparse structure for Acc. 
    get_csc_submatrix(A_rowptr, A_colinds, A_data, n, &splitting[0],
                      &splitting[0], &Acc_colptr[0], &Acc_rowinds[0],
                      &Acc_data[0], Cpts_size, 1, 1);

    // Form constraint vector, \hat{B}_c = A_{cc}B_c, in column major
    std::vector<T> constraint(num_bad_guys*Cpts_size, 0);
    for (I j=0; j<Cpts_size; j++) {
        for (I k=Acc_colptr[j]; k<Acc_colptr[j+1]; k++) {
            for (I i=0; i<num_bad_guys; i++) {
                constraint[col_major(Acc_rowinds[k],i,Cpts_size)] += 
                                    Acc_data[k] * B[col_major(Cpts[j],i,n)];
            }
        }
    }

    // Get sparse CSR submatrix Acf. First estimate number of nonzeros
    // in Acf and preallocate arrays. 
    I Acf_nnz = 0;
    for (I i=0; i<Cpts_size; i++) {
        I temp = Cpts[i];
        Acf_nnz += A_rowptr[temp+1] - A_rowptr[temp];
    }
    Acf_nnz *= (n - Cpts_size) / n;

    std::vector<I> Acf_rowptr(Cpts_size+1,0);
    std::vector<I> Acf_colinds;
    Acf_colinds.reserve(Acf_nnz);
    std::vector<T> Acf_data;
    Acf_data.reserve(Acf_nnz);

    // Loop over the row for each C-point
    for (I i=0; i<Cpts_size; i++) {
        I temp = Cpts[i];
        I nnz = 0;
        // Check if each col_ind is an F-point (splitting < 0)
        for (I k=A_rowptr[temp]; k<A_rowptr[temp+1]; k++) {
            I col = A_colinds[k];
            // If an F-point, store data and F-column index. Note,
            // F-index is negative and 1-indexed in splitting. 
            if (splitting[col] < 0) {
                Acf_colinds.push_back(abs(splitting[col]) - 1);
                Acf_data.push_back(A_data[k]);
                nnz += 1;
            }
        }
        Acf_rowptr[i+1] = Acf_rowptr[i] + nnz;
    }

    // Get maximum number of rows selected in minimization submatrix
    // (equivalent to max columns per row in sparsity pattern for W).
    I max_rows = 0;
    for (I i=1; i<S_rowptr_size; i++) {
        I temp = S_rowptr[i]-S_rowptr[i-1];
        if (max_rows < temp) {
            max_rows = temp;
        } 
    }

    // Get maximum number of nonzero columns per row in Acf submatrix.
    I max_cols = 0;
    for (I i=0; i<Cpts_size; i++) {
        I temp = Acf_rowptr[i+1] - Acf_rowptr[i];
        if (max_cols < temp) {
            max_cols = temp;
        }
    }

    // Preallocate storage for submatrix used in minimization process
    // Generally much larger than necessary, but may be needed in certain
    // cases. 
    I max_size = max_rows * (max_cols * max_rows); 
    std::vector<T> sub_matrix(max_size, 0);

    // Allocate pair of vectors to store P.col_inds and P.data. Use
    // size of sparsity pattern as estimate for number of nonzeros. 
    // Pair returned by the function through SWIG.
    // std::pair<std::vector<int>, std::vector<T> > P_vecs;
    // std::get<0>(P_vecs).reserve(S_colinds_size);
    // std::get<1>(P_vecs).reserve(S_colinds_size);

    /* ------------------ */

    // Form P row-by-row
    P_rowptr[0] = 0;
    I numCpts = 0;
    I data_ind = 0;
    for (I row_P=0; row_P<n; row_P++) {

        // Check if row is a C-point (>0 in splitting vector).
        // If so, add identity to P. Recall, enumeration of C-points
        // in splitting is one-indexed. 
        if (splitting[row_P] > 0) {
            // std::get<0>(P_vecs).push_back(splitting[row_P]-1);
            // std::get<1>(P_vecs).push_back(1.0);
            P_rowptr[row_P+1] = P_rowptr[row_P] + 1;
            P_colinds[data_ind] = splitting[row_P]-1;
            P_data[data_ind] = 1.0;
            data_ind += 1;
            numCpts +=1 ;
        }

        // If row is an F-point, form row of \hat{W} through constrained
        // minimization and multiply by A_{cc} to get row of P. 
        else {

            // Find row indices for all nonzero elements in submatrix of Acf.
            std::set<I> col_inds;

            // Get number of columns in sparsity pattern (rows in submatrix),
            // create pointer to indices
            const I *row_inds = &S_colinds[S_rowptr[row_P]];
            I submat_m = S_rowptr[row_P+1] - S_rowptr[row_P];
            
            // Get all nonzero col indices of any row in sparsity pattern
            for (I j=0; j<submat_m; j++) {
                I temp_row = row_inds[j];
                for (I i=Acf_rowptr[temp_row]; i<Acf_rowptr[temp_row+1]; i++) {
                    col_inds.insert(Acf_colinds[i]);
                }
            }
            I submat_n = col_inds.size();

            // Fill in row major data array for submatrix
            I submat_ind = 0;

            if (submat_m == 0) {
                P_rowptr[row_P+1] = P_rowptr[row_P];
                std::cout << "Warning - empty sparsity pattern for row " <<
                            row_P << " of P will result in zero-row.\n";
                continue;
            }

            // Loop over each row in submatrix
            for (I i=0; i<submat_m; i++) {
                I temp_row = row_inds[i];
                I temp_ind = Acf_rowptr[temp_row];

                // Loop over all column indices
                for (auto it=col_inds.begin(); it!=col_inds.end(); ++it) {
                    
                    // Initialize matrix entry to zero
                    sub_matrix[submat_ind] = 0.0;

                    // Check if each row, col pair is in Acf submatrix. Note, both
                    // sets of indices are ordered and Acf cols a subset of col_inds!
                    for (I k=temp_ind; k<Acf_rowptr[temp_row+1]; k++) {
                        if ( (*it) < Acf_colinds[k] ) {
                            break;
                        }
                        else if ( (*it) == Acf_colinds[k] ) {
                            sub_matrix[submat_ind] = Acf_data[k];
                            temp_ind = k+1;
                            break;
                        }
                    }
                    submat_ind += 1;
                }
            }

            // Make right hand side basis vector for this row of W, which is
            // the current F-point.
            I f_row = row_P - numCpts;
            std::vector<T> sub_rhs(submat_n,0);
            {
                I l=0;
                for (auto it=col_inds.begin(); it!=col_inds.end(); it++, l++) {
                    if ( (*it) == f_row ) {
                        sub_rhs[l] = 1.0;
                    }
                }
            }

            // Restrict constraint vector to sparsity pattern
            std::vector<T> sub_constraint;
            sub_constraint.reserve(submat_m * num_bad_guys);
            for (I k=0; k<num_bad_guys; k++) {
                for (I i=0; i<submat_m; i++) {
                    I temp_row = row_inds[i];
                    sub_constraint.push_back( constraint[col_major(temp_row,k,numCpts)] );
                }
            }

            // Get rhs of constraint - this is just the (f_row)th row of B_f,
            // which is the (row_P)th row of B.
            std::vector<T> constraint_rhs(num_bad_guys,0);
            for (I k=0; k<num_bad_guys; k++) {
                constraint_rhs[k] = B[col_major(row_P,k,n)];
            }

            // Solve constrained least sqaures, store solution in w_l. 
            std::vector<T> w_l = constrained_least_squares(sub_matrix,
                                                           sub_rhs,
                                                           sub_constraint,
                                                           constraint_rhs,
                                                           submat_n,
                                                           submat_m,
                                                           num_bad_guys);
            
            // Loop over each jth column of Acc, taking inner product
            //         (w_l)_j = \hat{w}_l * (Acc)_j
            // to form w_l := \hat{w}_l*Acc.
            I row_length = 0;
            for (I j=0; j<Cpts_size; j++) {
                T temp_prod = 0;
                I temp_v0 = 0;
                // Loop over nonzero indices for this column of Acc and vector w_l.
                // Note, both have ordered, unique indices.
                for (I k=Acc_colptr[j]; k<Acc_colptr[j+1]; k++) {
                    for (I i=temp_v0; i<submat_m; i++) {
                        // Can break here because indices are sorted increasing
                        if ( row_inds[i] > Acc_rowinds[k] ) {
                            break;
                        }
                        // If nonzero, add to dot product 
                        else if (row_inds[i] == Acc_rowinds[k]) {
                            temp_prod += w_l[i] * Acc_data[k];
                            temp_v0 += 1;
                            break;
                        }
                        else {
                            temp_v0 += 1;
                        }
                    }
                }
                // If dot product of column of Acc and vector \hat{w}_l is nonzero,
                // add to sparse structure of P.
                if (std::abs(temp_prod) > 1e-12) {
                    // std::get<0>(P_vecs).push_back(j);
                    // std::get<1>(P_vecs).push_back(temp_prod);
                    P_colinds[data_ind] = j;
                    P_data[data_ind] = temp_prod;
                    row_length += 1;
                    data_ind += 1;
                }
            }

            // Set row pointer for next row in P
            P_rowptr[row_P+1] = P_rowptr[row_P] + row_length;
            row_inds = NULL;
        }

        if (data_ind > P_data_size) {
            std::cout << "Warning - more nonzeros in P than allocated - breaking early.\n";
            break;
        }
    }

    // Check that all C-points were added to P. 
    if (numCpts != Cpts_size) {
        std::cout << "Warning - C-points missed in constructing P.\n";
    }

    // return P_vecs;
}

#endif
