#include <iostream>
#include <vector>
#include <set>


/* I think this is already in PyAMG somewhere? */
inline int signof(int a) { return (a<0 ? -1 : 1); }

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
 * C anf F-points. F-points are enumerated using 1-indexing
 * and negative numbers, while C-points are enumerated using
 * 1-indexing and positive numbers, in a vector splitting. 
 *
 * Parameters 
 * ----------
 * 		Cpts : int array
 * 			Array of Cpts
 *		numCpts : &int
 *			Length of Cpt array
 *		n : &int 
 *			Total number of points
 *
 * Returns
 * -------
 * 		splitting - vector
 *			Vector of size n, indicating whether each point is a
 *			C or F-point, with corresponding index. C-points are
 *			sorted in place in array.
 *
 */
std::vector<int> get_ind_split(int Cpts[], const int & numCpts, const int &n)
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
			Cpts[cind-1] = i;
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
		// 	- Note, is_col_ind[] is one-indexed, not zero.
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
void get_csc_submatrix(const int A_rowptr[],
					   const int A_colinds[],
					   const double A_data[],
					   const int &n,
					   const int is_col_ind[],
					   const int is_row_ind[],
					   int colptr[], 
					   int rowinds[], 
					   double data[],
					   const int &num_cols,
	   				   const int &row_scale = 1,
					   const int &col_scale = 1 )
{
	// Fill in rowinds and data for sparse submatrix
	for (int i=0; i<n; i++) {
		
		// Continue to next iteration if this row is not a row-ind
		if ( (row_scale*is_row_ind[i]) <= 0 ) {
			continue;
		}

		// Find all instances of col-inds in this row. Save row-ind
		// and data value in sparse structure. Increase column pointer
		// to mark where next data index is. Will reset after.
		// 	- Note, is_col_ind[] is one-indexed, not zero.
		for (int k=A_rowptr[i]; k<A_rowptr[i+1]; k++) {
			int ind = col_scale * is_col_ind[A_colinds[k]];
			if (ind > 0) {
				int data_ind = colptr[ind-1];
				rowinds[data_ind] = std::abs(is_row_ind[i])-1;
				data[data_ind] = A_data[k];
				colptr[ind-1] += 1;
			}
		}
	}

	// Reset colptr for submatrix
	int prev = 0;
	for (int i=0; i<num_cols; i++) {
		int temp = colptr[i];
		colptr[i] = prev;
		prev = temp;
	}	
}


/* QR-decomposition using Householer transformations on dense
 * 2d array stored in either column- or row-major form. 
 * 
 * Parameters
 * ----------
 * 		A : double array
 *			2d matrix A stored in 1d column- or row-major.
 * 		m : &int
 *			Number of rows in A
 *		n : &int
 *			Number of columns in A
 *		is_col_major : bool
 *			True if A is stored in column-major, false
 *			if A is stored in row-major.
 *
 * Returns
 * -------
 * 		Q : vector<double>
 *			Matrix Q stored in same format as A.
 *		R : in-place
 *			R is stored over A in place, in same format.
 *
 */
std::vector<double> QR(double A[],
					   const int &m,
					   const int &n,
					   const bool is_col_major)
{
	// Funciton pointer for row or column major matrices
	int (*get_ind)(const int&, const int&, const int&);
	const int *C;
	if (is_col_major) {
		get_ind = &col_major;
		C = &m;
	}
	else {
		get_ind = &row_major;
		C = &n;
	}

	// Initialize Q to identity
	std::vector<double> Q(m*m,0);
	for (int i=0; i<m; i++) {
		Q[get_ind(i,i,m)] = 1;
	}

	// Loop over columns of A using Householder reflections
	for (int j=0; j<n; j++) {

		// Break loop for short fat matrices
		if (m <= j) {
			break;
		}

		// Get norm of next column of A to be reflected. Choose sign
		// opposite that of A_jj to avoid catastrophic cancellation.
		double normx = 0;
		for (int i=j; i<m; i++) {
			double temp = A[get_ind(i,j,*C)];
			normx += temp*temp;
		}
		normx = std::sqrt(normx);
		normx *= -1*signof(A[get_ind(j,j,*C)]);

		// Form vector v for Householder matrix H = I - tau*vv^T
		// where v = R(j:end,j) / scale, v[0] = 1.
		double scale = A[get_ind(j,j,*C)] - normx;
		double tau = -scale / normx;
		std::vector<double> v(m-j,0);
		v[0] = 1;
		for (int i=1; i<(m-j); i++) {
			v[i] = A[get_ind(j+i,j,*C)] / scale;	
		}

		// Modify R in place, R := H*R, looping over columns then rows
		for (int k=j; k<n; k++) {

			// Compute the kth element of v^T * R
			double vtR_k = 0;
			for (int i=0; i<(m-j); i++) {
				vtR_k += v[i] * A[get_ind(j+i,k,*C)];
			}

			// Correction for each row of kth column, given by 
			// R_ik -= tau * v_i * (vtR_k)_k
			for (int i=0; i<(m-j); i++) {
				A[get_ind(j+i,k,*C)] -= tau * v[i] * vtR_k;
			}
		}

		// Modify Q in place, Q = Q*H
		for (int i=0; i<m; i++) {

			// Compute the ith element of Q * v
			double Qv_i = 0;
			for (int k=0; k<(m-j); k++) {
				Qv_i += v[k] * Q[get_ind(i,k+j,m)];
			}

			// Correction for each column of ith row, given by
			// Q_ik -= tau * Qv_i * v_k
			for (int k=0; k<(m-j); k++) { 
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
 *		R : double array, length m*n
 *			Upper-triangular array stored in column- or row-major.
 *		rhs : double array, length m
 *			Right hand side of linear system
 *		x : double array, length n
 *			Preallocated array for solution
 *		m : &int
 *			Number of rows in R
 *		n : &int
 *			Number of columns in R
 *		is_col_major : bool
 *			True if R is stored in column-major, false
 *			if R is stored in row-major.
 *
 * Returns
 * -------
 *		Nothing, solution is stored in x[].
 *
 * Notes
 * -----
 * R need not be square, the system will be solved over the
 * rank r upper-triangular block. If remaining entries in
 * solution are unused, they will be set to zero.
 *
 */		
void upper_tri_solve(const double R[],
					 const double rhs[],
					 double x[],
					 const int &m,
					 const int &n,
					 const bool is_col_major)
{
	// Funciton pointer for row or column major matrices
	int (*get_ind)(const int&, const int&, const int&);
	const int *C;
	if (is_col_major) {
		get_ind = &col_major;
		C = &m;
	}
	else {
		get_ind = &row_major;
		C = &n;
	}

	// Backwards substitution
	int rank = std::min(m,n);
	for (int i=(rank-1); i>=0; i--) {
		double temp = rhs[i];
		for (int j=(i+1); j<rank; j++) {
			temp -= R[get_ind(i,j,*C)]*x[j];
		}
		if (std::abs(R[get_ind(i,i,*C)]) < 1e-12) {
			std::cout << "Warning: Upper triangular matrix near singular.\n"
						 "Dividing by ~ 0.\n";
		}
		x[i] = temp / R[get_ind(i,i,*C)];
	}

	// If rank < size of rhs, set free elements in x to zero
	for (int i=m; i<n; i++) {
		x[i] = 0;
	}
}


/* Forward substitution solve on lower-triangular linear system,
 * Lx = rhs, where L is stored in column- or row-major form. 
 * 
 * Parameters
 * ----------
 *		L : double array, length m*n
 *			Lower-triangular array stored in column- or row-major.
 *		rhs : double array, length m
 *			Right hand side of linear system
 *		x : double array, length n
 *			Preallocated array for solution
 *		m : &int
 *			Number of rows in L
 *		n : &int
 *			Number of columns in L
 *		is_col_major : bool
 *			True if L is stored in column-major, false
 *			if L is stored in row-major.
 *
 * Returns
 * -------
 *		Nothing, solution is stored in x[].
 *
 * Notes
 * -----
 * L need not be square, the system will be solved over the
 * rank r lower-triangular block. If remaining entries in
 * solution are unused, they will be set to zero.
 *
 */
void lower_tri_solve(const double L[],
					 const double rhs[],
					 double x[],
					 const int &m,
					 const int &n,
					 const bool is_col_major)
{
	// Funciton pointer for row or column major matrices
	int (*get_ind)(const int&, const int&, const int&);
	const int *C;
	if (is_col_major) {
		get_ind = &col_major;
		C = &m;
	}
	else {
		get_ind = &row_major;
		C = &n;
	}

	// Backwards substitution
	int rank = std::min(m,n);
	for (int i=0; i<rank; i++) {
		double temp = rhs[i];
		for (int j=0; j<i; j++) {
			temp -= L[get_ind(i,j,*C)]*x[j];
		}
		if (std::abs(L[get_ind(i,i,*C)]) < 1e-12) {
			std::cout << "Warning: Lower triangular matrix near singular.\n"
						 "Dividing by ~ 0.\n";
		}
		x[i] = temp / L[get_ind(i,i,*C)];
	}

	// If rank < size of rhs, set free elements in x to zero
	for (int i=m; i<n; i++) {
		x[i] = 0;
	}
}


/* Method to solve the linear least squares problem.
 *
 * Parameters
 * ----------
 * 		A : double array, length m*n
 *			2d array stored in column- or row-major.
 *		b : double array, length m
 *			Right hand side of unconstrained problem.
 *		x : double array, length n
 *			Container for solution
 * 		m : &int
 *			Number of rows in A
 *		n : &int
 *			Number of columns in A
 *		is_col_major : bool
 *			True if A is stored in column-major, false
 *			if A is stored in row-major.
 *
 * Returns
 * -------
 * 		x : vector<double>
 *			Solution to constrained least sqaures problem.
 *
 * Notes
 * -----
 * If system is under determined, free entries are set to zero. 
 *
 */
void least_squares(double A[],
				   double b[],
				   double x[],
				   const int &m,
				   const int &n,
				   const bool is_col_major=0)
{
	// Funciton pointer for row or column major matrices
	int (*get_ind)(const int&, const int&, const int&);
	if (is_col_major) {
		get_ind = &col_major;
	}
	else {
		get_ind = &row_major;
	}

	// Take QR of A
	std::vector<double> Q = QR(A,m,n,is_col_major);

	// Multiply right hand side, b:= Q^T*b. Have to make new vetor, rhs.
	std::vector<double> rhs(m,0);
	for (int i=0; i<m; i++) {
		for (int k=0; k<m; k++) {
			rhs[i] += b[k] * Q[get_ind(k,i,m)];
		}
	}

	// Solve upper triangular system, store solution in x.
	upper_tri_solve(A,&rhs[0],x,m,n,is_col_major);
}


/* Method to solve an arbitrary constrained least squares.
 * Can be used one of two ways, with any shape and rank
 * operator A.
 *
 * 	1. Suppose we want to solve
 *		x = argmin || xA - b || s.t. xC = d 				(1)
 * 	Let C = QR, and make the change of variable z := xQ.
 * 	Then (1) is equivalent to
 *		z = argmin || zQ^TA - b ||   s.t. zR = d
 *		  = argmin || A^TQz^T - b || s.t. R^Tz^T = d 		(2)
 * 	This is solved by passing in A in *row major* and
 * 	C in *column major.*
 *		- A is nxm
 *		- C is mxs
 *
 * 	2. Suppose we want to solve
 *		x = argmin || Ax - b || s.t. Cx = d 				(3)
 * 	Let C^T = QR, and make the change of variable z := Q^Tx.
 * 	Then (1) is equivalent to
 *		z = argmin || AQx - b ||   s.t. R^Tz = d 			(4)
 * 	This is solved by passing in A in *column major* and
 * 	C in *row major.* 
 *		- A is mxn
 *		- C is sxn
 *
 * Parameters
 * ----------
 * 		A : vector<double>
 *			2d array stored in column- or row-major.
 *		b : vector<double>
 *			Right hand side of unconstrained problem.
 *		C : vector<double>
 *			Constraint operator, stored in opposite form as
 *			A. E.g. A in column-major means C in row-major.
 *		d : vector<double> 
 *			Right hand side of contraint equation.
 *		m : &int
 *			Number of columns if A is in row-major, number
 *			of rows if A is in column-major.
 *		n : &int
 *			Number of rows if A is in row-major, number
 *			of columns if A is in column-major.
 *		s : &int
 *			Number of constraints.
 *
 * Returns
 * -------
 * 		x : vector<double>
 *			Solution to constrained least sqaures problem.
 *
 */
std::vector<double> constrained_least_squares(std::vector<double> &A,
											  std::vector<double> &b,
											  std::vector<double> &C,
											  std::vector<double> &d,
											  const int &m,
											  const int &n,
											  const int &s)
{

	// Perform QR on matrix C. R is written over C in column major,
	// and Q is returned in column major. 
	std::vector<double> Qc = QR(&C[0],n,s,1);

	print_mat(&C[0], n, s, 1);

	// Form matrix product S = A^T * Q. For A passed in as row
	// major, perform S = A * Q, assuming that A is in column major
	// (A^T row major = A column major).
	std::vector<double> temp_vec(n,0);
	for (int i=0; i<m; i++) {
		for (int j=0; j<n; j++) {
			double val = 0.0;
			for (int k=0; k<n; k++) {
				val += A[col_major(i,k,m)] * Qc[col_major(k,j,n)];
			}
			temp_vec[j] = val;
		}
		for (int j=0; j<n; j++) {
			A[col_major(i,j,m)] = temp_vec[j];
		}
	}

	// Change R to R^T. Probably dirtier, cheaper ways to use R^T
	// in place, don't think it's worth it.
	for (int i=1; i<s; i++) {
		for (int j=0; j<i; j++) {
			C[col_major(i,j,n)] = C[col_major(j,i,n)];
		}
	}

	// Satisfy constraints R^Tz = d, move to rhs of LS
	lower_tri_solve(&C[0],&d[0],&temp_vec[0],n,s,1);
	for (int i=0; i<m; i++) {
		double val = 0.0;
		for (int j=0; j<s; j++) {
			val += A[col_major(i,j,m)] * temp_vec[j];
		}
		b[i] -= val;
	}

	// Call LS on reduced system
	int temp_ind = n-s;
	least_squares(&A[col_major(0,s,m)], &b[0], &temp_vec[s], m, temp_ind, 1);

	// Form x = Q*z
	std::vector<double> x(n,0);
	for (int i=0; i<n; i++) {
		x[i] = 0;
		for (int k=0; k<n; k++) {
			x[i] += Qc[col_major(i,k,n)] * temp_vec[k];
		}
	}

	return x;
}


// Need some kind of sparsity pattern? --> Use multiple pairwise aggregations for sparsity...
// 		Then stencil stretches in right direction
//	- Important that A has sorted indices when passed in
// 	- TODO : test middle section with new Acf submatrix
//		--> Use get_sub_mat testing function in sparse.cpp
// 	- TODO : How is P returned?? I have this great construction of it... 

template<class I, class T, class F>
void ben_ideal_interpolation(const I A_rowptr[], const int A_rowptr_size,
		                     const I A_colinds[], const int A_colinds_size,
		                     const T A_data[], const int A_data_size,
		                     I S_rowptr[], const int S_rowptr_size,
		                     I S_colinds[], const int S_colinds_size,
		                     const T B[], const int B_size,
                             I Cpts[], const int Cpts_size,
		                     const I n,
                             const I num_bad_guys )
{

	/* ------ tested ----- */
	// Get splitting of points in one vector, Cpts enumerated in positive,
	// one-indexed ordering and Fpts in negative. 
	// 		E.g., [-1,1,2,-2,-3] <-- Fpts = [0,3,4], Cpts = [1,2]
	// List of Cpts is sorted in the process in place.
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
					  &Acc_data[0], Cpts_size, -1, -1);

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
	int Acf_nnz = 0;
	for (int i=0; i<Cpts_size; i++) {
		int temp = Cpts[i];
		Acf_nnz += A_rowptr[temp+1] - A_rowptr[temp];
	}
	Acf_nnz *= (n - Cpts_size) / n;

	std::vector<int> Acf_rowptr(Cpts_size+1,0);
	std::vector<int> Acf_colinds;
	Acf_colinds.reserve(Acf_nnz);
	std::vector<double> Acf_data;
	Acf_data.reserve(Acf_nnz);

	// Loop over the row for each C-point
	for (int i=0; i<Cpts_size; i++) {
		int temp = Cpts[i];
		int nnz = 0;
		// Check if each col_ind is an F-point (splitting < 0)
		for (int k=A_rowptr[temp]; k<A_rowptr[temp+1]; k++) {
			int col = A_colinds[k];
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
	int max_rows = 0;
	for (int i=1; i<S_rowptr_size; i++) {
		int temp = S_rowptr[i]-S_rowptr[i-1]
		if (max_rows < temp) {
			max_rows = temp;
		} 
	}

	// Get maximum number of nonzero columns per row in Acf submatrix.
	int max_cols = 0;
	for (int i=0; i<Cpts_size; i++) {
		int temp = Acf_rowptr[i+1] - Acf_rowptr[i];
		if (max_cols < temp) {
			max_cols = temp;
		}
	}

	// Preallocate storage for submatrix used in minimization process
	// Generally much larger than necessary, but may be needed in certain
	// cases. 
	int max_size = max_rows * (max_rows * max_rows); 
	vector<T> sub_matrix(max_size, 0);

	/* ------------------ */


	// Form P row-by-row
	I data_ind = 0; 
	I numCpts = 0;
	for (I row_P=0; row_P<n; row_P++) {

		// Check if row is a C-point (>0 in splitting vector).
		// If so, add identity to P. Recall, enumeration of C-points
		// in splitting is one-indexed. 
		if (splitting[row_P] > 0) {
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
	        std::set<int> col_inds;

	        // Get number of columns in sparsity pattern (rows in submatrix),
	        // create pointer to indices
			const int *row_inds = &S_colinds[S_rowptr[row_P]];
			int submat_m = S_rowptr[row_P+1] - S_rowptr[row_P];

			// Get all nonzero col indices of any row in sparsity pattern
			for (int j=0; j<submat_m; j++) {
				int temp_row = row_inds[j];
				for (int i=Acf_rowptr[temp_row]; i<Acf_rowptr[temp_row+1]; i++) {
					col_inds.insert(Acf_colinds[i]);
				}
			}
			int submat_n = col_inds.size();

			// Fill in row major data array for submatrix
			int submat_ind = 0;

			// Loop over each row in submatrix
			for (int i=0; i<submat_m; i++) {
				int temp_row = row_inds[i];
				int temp_ind = Acf_rowptr[temp_row];

				// Loop over all column indices
	            for (auto it=col_inds.begin(); it!=col_inds.end(); ++it) {
	            	
	            	// Initialize matrix entry to zero
					sub_matrix[submat_ind] = 0.0;

	            	// Check if each row, col pair is in Acf submatrix. Note, both
	            	// sets of indices are ordered and Acf cols a subset of col_inds!
	            	for (int k=temp_ind; k<Acf_rowptr[temp_row+1]; k++) {
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
			int f_row = row_P - numCpts;
			std::vector<double> sub_rhs(submat_n,0);
			{
				int l=0;
				for (auto it=rows.begin(); it!=rows.end(); it++, l++) {
					if ( (*it) == f_row ) {
						sub_rhs[l] = 1.0;
					}
				}
			}

			// Restrict constraint vector to sparsity pattern
			vector<double> sub_constraint;
			sub_constraint.reserve(submat_m * num_bad_guys);
			for (int k=0; k<num_bad_guys; k++) {
				for (int i=0; i<submat_m; i++) {
					int temp_row = row_inds[i];
					sub_constraint.push_back( constraint[col_major(temp_row,k,numCpts)] );
				}
			}

			// Get rhs of constraint - this is just the (f_row)th row of B_f,
			// which is the (row_P)th row of B.
			std::vector<double> constraint_rhs(num_bad_guys,0);
			for (int k=0; k<num_bad_guys; k++) {
				constraint_rhs[k] = B[col_major(row_P,k,n)];
			}

			// Solve constrained least sqaures, store solution in w_l. 
			std::vector<double> w_l = constrained_least_squares(sub_matrix,
																sub_rhs,
																sub_constraint,
																constraint_rhs,
																submat_m,
																submat_n,
																num_bad_guys);

			/* ---------- tested ---------- */
			// Loop over each jth column of Acc, taking inner product
			// 		(w_l)_j = \hat{w}_l * (Acc)_j
			// to form w_l := \hat{w}_l*Acc.
			int row_length = 0;
			for (int j=0; j<Cpts_size; j++) {
				double temp_prod = 0;
				int temp_v0 = 0;
				// Loop over nonzero indices for this column of Acc and vector w_l.
				// Note, both have ordered, unique indices.
				for (int k=Acc_colptr[j]; k<Acc_colptr[j+1]; k++) {
					for (int i=temp_v0; i<submat_m; i++) {
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
					P_colinds[data_ind] = j;
					P_data[data_ind] = temp_prod;
					data_ind += 1;
					row_length += 1;
				}
			}

			// Set row pointer for next row in P
			P_rowptr[row_P+1] = P_rowptr[row_P] + row_length;
			colinds = NULL;
		}
	}

	// Check that all C-points were added to P. 
	if (numCpts != Cpts_size) {
		std::cout << "Warning - C-points missed in constructing P.\n";
	}

}

