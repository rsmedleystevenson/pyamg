#include <vector>




// Need some kind of sparsity pattern? --> Use multiple pairwise aggregations for sparsity...
// 		Then stencil stretches in right direction
//	- TODO : Make sure C-points are sorted
//		- Probably don't need to, can use splitting array to determine each point!
//		- Might need it in line 46, temp2 = ...
// TODO : Can add natural filtering after row w_l has been computed.
//		  Instead of pushing straight to P, store in temp vector, filter,
//		  then push to P. 

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

	/* ------ tested ----- */
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

	/* ------------------- */

	// Form constraint vector, \hat{B}_c = A_{cc}B_c
	// TODO - MUST HAVE CPTS SORTED FOR THIS LOOP TO WORK
	//	- Verified if Cpts are sorted...
	std::vector<T> constraint(num_bad_guys*Cpts_size, 0);
	for (I j=0; j<Cpts_size; j++) {
		for (I k=Acc_colptr[j]; k<Acc_colptr[j+1]; k++) {
			for (I i=0; i<num_bad_guys; i++) {
				constraint[i*Cpts_size + Acc_rowinds[k]] += Acc_data[k] * B[i*n + Cpts[j]];
			}
		}
	}

	/* ------ tested ----- */
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

	// Preallocate storage for submatrix used in minimization process
	// Generally much larger than necessary, but may be needed in certain
	// cases. 
	int max_size = max_cols * (max_rows * max_cols); 
	vector<T> submatrix(max_size, 0);

	/* ------------------- */

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

			/* ------ tested ----- */
	        // Find row indices for all nonzero elements in submatrix
			int f_row = row_P - numCpts;
	        std::set<int> row_inds;

	        // Get number of columns in sparsity pattern, create pointer to indices
			const int *col_inds = &S_colinds[S_rowptr[f_row]];
			int submat_n = S_rowptr[f_row+1] - S_rowptr[f_row];

			// Get all nonzero row indices of any column in sparsity pattern
			for (int j=0; j<submat_n; j++) {
				int temp_col = col_inds[j];
				for (int i=Afc_colptr[temp_col]; i<Afc_colptr[temp_col+1]; i++) {
					row_inds.insert(Afc_rowinds[i]);
				}
			}
			int submat_m = row_inds.size();

			// Fill in column major data array for submatrix
			int submat_ind = 0;
			for (int j=0; j<submat_n; j++) {
				int temp_col = col_inds[j];
				int temp_ind = Afc_colptr[temp_col];
				// Loop over rows in sparsity pattern
	            for (auto it=row_inds.begin(); it!=row_inds.end(); ++it) {
	            	
	            	// Initialize matrix entry to zero
					submatrix[submat_ind] = 0.0;

	            	// Check if this row, col pair is in Afc submatrix. Note, both
	            	// sets of indices are ordered and Afc rows a subset of row_inds!
	            	for (int i=temp_ind; i<Afc_colptr[temp_col+1]; i++) {
						if ( (*it) < Afc_rowinds[i] ) {
							break;
						}
						else if ( (*it) == Afc_rowinds[i] ) {
							submatrix[submat_ind] = Afc_data[i];
							temp_ind = i+1;
							break;
						}
	            	}
					submat_ind += 1;
	            }
			}

			// Make right hand side basis vector for this row of P
			// 	- TODO : is row_P definitely what we want, or number F / C -point??
			std::vector<double> subrhs(nrows,0);
			{
				int l=0;
				for (auto it=rows.begin(); it!=rows.end(); it++, l++) {
					if ( (*it) == row_P ) {
						subrhs[l] = 1.0;
					}
				}
			}
			/* ------------------- */

			// Restrict constraint vector to sparsity pattern
			//  - constraint Acc*Bc is actually transpose of what we want
			//    for constrained minimization.
			//	- This is okay - if CLS requires Cx = d, we pass in C^T, 
			//	  which is exactly Acc*Bc. Need in column major form. 



			// Get rhs of constraint - this is just the (row_P)th row of B_f,
			// or (row_P)th column of B_f^T


			std::vector<double> w_l = constrained_least_squares(submatrix,
																subrhs,
																std::vector<double> &Ct,
																std::vector<double> &d,
																submat_m,
																submat_n,
																const int &s)


			/* ------ tested ----- */
			// Let w_l := w_l*Acc
			int row_length = 0;
			for (int j=0; j<Cpts_size; j++) {
				double temp_prod = 0;
				int temp_v0 = 0;
				// Loop over nonzero indices for this column of Acc and vector w_l.
				// Note, both have ordered, unique indices.
				for (int k=Acc_colptr[j]; k<Acc_colptr[j+1]; k++) {
					for (int j=temp_v0; j<submat_n; j++) {
						// Can break here because indices are sorted increasing
						if ( col_inds[j] > Acc_rowinds[k] ) {
							break;
						}
						// If nonzero, add to dot product 
						else if (col_inds[j] == Acc_rowinds[k]) {
							temp_prod += w_l[j] * Acc_data[k];
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
				if (temp_prod != 0) {
					P_colinds[data_ind] = j;
					P_data[data_ind] = temp_prod;
					data_ind += 1;
					row_length += 1;
				}
			}
			/* ------------------- */
			// TODO : Can add filtering option here

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

