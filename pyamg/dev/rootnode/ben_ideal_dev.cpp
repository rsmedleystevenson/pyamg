#include <vector>




// Need some kind of sparsity pattern? --> Use multiple pairwise aggregations for sparsity...
// 		Then stencil stretches in right direction
//	- TODO : Make sure C-points are sorted
//		- Probably don't need to, can use splitting array to determine each point!
//		- Might need it in line 46, temp2 = ...

template<class I, class T, class F>
void ben_ideal_interpolation(const I A_rowptr[], const int A_rowptr_size,
		                     const I A_colinds[], const int A_colinds_size,
		                     const T A_data[], const int A_data_size,
		                     I S_rowptr[], const int S_rowptr_size,
		                     I S_colinds[], const int S_colinds_size,
		                     T S_data[], const int S_data_size,
		                     const T B[], const int B_size,
                             const I Cpts[], const int Cpts_size,
		                     const I n,
                             const I num_bad_guys)
{

	// Get splitting of points in one vector, Cpts enumerated in positive,
	// one-indexed ordering and Fpts in negative. 
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

	// Form constraint vector, \hat{B}_c = A_{cc}B_c
	std::vector<T> constraint(num_bad_guys*Cpts_size, 0);
	for (I j=0; j<Cpts_size; j++) {
		for (I k=Acc_colptr[j]; k<Acc_colptr[j+1]; k++) {
			I temp = Acc_rowinds[k];
			I temp2 = Cpts[temp];
			for (I i=0; i<num_bad_guys; i++) {
				constraint[i*Cpts_size+temp] += data[k] * B[i*n+temp2];
			}
		}
	}

	// Get sparse CSC column pointer for submatrix Afc. Final two arguments
	// select F-points for rows (negative) and C-points for columns (positive).
	std::vector<I> Afc_colptr(Cpts_size+1,0);
	int max_rows = get_col_ptr(A_rowptr, A_colinds, n, &splitting[0],
							   &splitting[0], &Afc_colptr[0],
							   Cpts_size, -1, 1);

	// Allocate row-ind and data arrays for sparse submatrix 
	I nnz = Afc_colptr[Cpts_size];
	std::vector<I> Afc_rowinds(nnz,0);
	std::vector<T> Afc_data(nnz,0);

	// Fill in sparse structure for Afc. 
	get_csc_submatrix(A_rowptr, A_colinds, A_data, n, &splitting[0],
					  &splitting[0], &Afc_colptr[0], &Afc_rowinds[0],
					  &Afc_data[0], Cpts_size, 1, -1);

	// Get maximum number of columns selected in sparsity pattern for any row.
	int max_cols = 0;
	for (int i=1; i<S_rowptr_size; i++) {
		int temp = S_rowptr[i]-S_rowptr[i-1]
		if (max_cols < temp) {
			max_cols = temp;
		} 
	}

	// Preallocate storage for submatrices used in minimization process
	// Generally much larger than necessary, but may be needed in certain
	// cases. 
	int max_size = max_cols * (max_rows * max_cols); 
	vector<I> vec_inds(max_size, 0);
	vector<T> submatrix(max_size, 0);

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

	        // Find row indices for all nonzero elements in submatrix
			int f_row = row_P - numCpts;
	        std::set<int> row_inds;

			// Store nonzero row indices of any column for a given
			// sparsity pattern (row of S)
			for (int j=S_rowptr[f_row]; j<S_rowptr[f_row+1]; j++) {
				int temp_col = S_colinds[j];
				for (int i=Afc_colptr[temp_col]; i<Afc_colptr[temp_col+1]; i++) {
					row_inds.insert(Afc_rowinds[i]);
				}
			}
			int submat_m = row_inds.size();
			int submat_n = S_rowptr[f_row+1] - S_rowptr[f_row];
			int submat_size = submat_m * submat_n;

			// Fill in column major data array for submatrix
			int submat_ind = 0;
			for (int j=S_rowptr[f_row]; j<S_rowptr[f_row+1]; j++) {

				int temp_col = S_colinds[j];
				int temp_ind = Afc_colptr[temp_col];

				int num_nnz = Afc_colptr[temp_ind+1] - Afc_colptr[temp_ind];

	            for (auto it=row_inds.begin(); it!=row_inds.end(); ++it) {

	            	if ( (*it) == Afc_rowinds[temp_ind] ) {

	            		submatrix[submat_ind] = 

	            		if
	            		next_nnz
	            	}
	            	else {

	            	}


	            }
			
			}


			// TODO :
			//	- Select submatrix for given row from Afc
			//		+ Submatrix needs to be stored in column major...
			// 	- Get rhs, constraint matrix, constraint rhs, s






			std::vector<double> w_l = constrained_least_squares(submatrix,
																sub_rhs,
																std::vector<double> &Ct,
																std::vector<double> &d,
																submat_m,
																submat_n,
																const int &s)

			// Let w_l := \hat{w}_lA_{cc}.
			// TODO : Make sure w_l has ordered indices?
			//		- Finish setting this up for CSC
			I row_length = 0;
			for (I j=0; j<Cpts_size; j++) {

				T temp_prod = 0;

				// Loop over nonzero indices for this row of Acc and current vector \hat{w}_l.
				for (I k=Acc_colptr[j]; k<Acc_colptr[j+1]; k++) {

					for (I v_ind=0; v_ind<vec_length; v_ind++) {
						// If nonzero, add to dot product 
						if (Acc_colinds[k] == vec_inds[v_ind]) {
							temp_prod += vec_data[v_ind] * Acc_data[k];
						}
					}
				}
				// If dot product of column of Acc and vector \hat{w}_l is nonzero,
				// add to sparse structure of P.
				if (temp_prod != 0) {
					P_colinds[data_ind] = j;
					P_data[data_ind] = temp_prod;
					data_ind += 1;
					row_length += 1;
				}
			}

			// Set row pointer for next row in P
			P_rowptr[row_P+1] = P_rowptr[row_P] + row_length;
		}
	}



	// Check that all C-points were added to P. 
	if (numCpts != Cpts_size) {
		std::cout << "Warning - C-points missed in constructing P.\n";
	}










}

