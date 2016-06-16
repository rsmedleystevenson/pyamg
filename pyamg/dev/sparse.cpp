#include <iostream>
#include <vector>
#include <set>



// Makes vector enumerating F-points using negative numbers and C-points using positive. 
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
			cind += 1;
		}
	}
	return ind_split;
}


// Returns maximum number of nonzeros in any column
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
					 const int &col_scale = 1
						)
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



std::set<int> get_sub_mat(const std::vector<int> &cols,
						  const std::vector<int> &Afc_colptr,
						  const std::vector<int> &Afc_rowinds,
						  const std::vector<double> &Afc_data,
						  std::vector<double> &submatrix )
{

	// Find row indices for all nonzero elements in submatrix
	std::set<int> row_inds;

	// Store nonzero row indices of any column for a given
	// sparsity pattern (row of S)
	// for (int j=S_rowptr[f_row]; j<S_rowptr[f_row+1]; j++) {
		// int temp_col = S_colinds[j];
	for (int j=0; j<cols.size(); j++) {
		int temp_col = cols[j];
		for (int i=Afc_colptr[temp_col]; i<Afc_colptr[temp_col+1]; i++) {
			row_inds.insert(Afc_rowinds[i]);
		}
	}
	int submat_m = row_inds.size();
	int submat_n = cols.size();

	// Fill in column major data array for submatrix
	int submat_ind = 0;
	for (int j=0; j<cols.size(); j++) {

		int temp_col = cols[j];
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
	return row_inds;
}


int main(int argc, char *argv[]) 
{

#if 0
	// Construct sparse matrix, size 9x9
	std::vector<int> A_rowptr {0,2,5,8,11,14,17,20,23,25};
	std::vector<int> A_colinds {0,1,0,1,2,1,2,3,2,3,4,3,4,5,4,5,6,5,6,7,6,7,8,7,8};
	std::vector<double> A_data {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25};

	// Arbitrarily coarsen
	std::vector<int> Cpts {1,2,3,5,6,7};
	std::vector<int> Fpts {0,4,8};
	// std::vector<int> Cpts {4,5,6,7,8};		// Scale with -1
	// std::vector<int> Fpts {0,1,2,3};			// Scale with +1

	int numCpts = Cpts.size();
	int numFpts = Fpts.size();
	int n = numFpts + numCpts;
	int scale_row = 1;
	int scale_col = 1;
	int num_cols = numCpts;
	int num_rows = numFpts;

	// Get splitting of points in one vector
	std::vector<int> splitting = get_ind_split(&Cpts[0],numCpts,n);

	// Get sparse CSC column pointer for submatrix Acf
	std::vector<int> colptr(num_cols+1,0);
	get_col_ptr(&A_rowptr[0], &A_colinds[0], n, &splitting[0], &splitting[0], &colptr[0], num_cols, scale_row, scale_col);

	// Allocate row-ind and data arrays for sparse submatrix 
	int nnz = colptr[num_cols];
	std::vector<int> rowinds(nnz,0);
	std::vector<double> data(nnz,0);

	// Fill in sparse structure
	get_csc_submatrix(&A_rowptr[0], &A_colinds[0], &A_data[0], n,
					  &splitting[0], &splitting[0], &colptr[0], &rowinds[0],
					  &data[0], num_cols, scale_row, scale_col);


	std::cout << "spltting = \n\t";
	for (int i=0; i<n; i++) {
		std::cout << splitting[i] << ", ";
	}
	std::cout << std::endl;

	std::cout << "colptr = np.array([";
	for (int i=0; i<colptr.size(); i++) {
		std::cout << colptr[i] << ", ";
	}
	std::cout << "],dtype=int)" << std::endl;

	std::cout << "rowinds = np.array([";
	for (int i=0; i<rowinds.size(); i++) {
		std::cout << rowinds[i] << ", ";
	}
	std::cout << "],dtype=int)" << std::endl;

	std::cout << "data0 = np.array([";
	for (int i=0; i<data.size(); i++) {
		std::cout << data[i] << ", ";
	}
	std::cout << "],dtype=int)" << std::endl;
	std::cout << "test_return = csc_matrix((data0,rowinds,colptr))\n\n";


	/* -------------- test forming constraint \hat{Bc} = Acc * Bc -------------- */
	// Bad guys
	std::vector<double> B = {1,2,3,4,5,6,7,8,9,-1,1,-1,1,-1,1,-1,1,-1};
	int num_bad_guys = 2;

	// Form constraint vector, \hat{B}_c = A_{cc}B_c
	std::vector<double> constraint(num_bad_guys*numCpts, 0);
	for (int j=0; j<numCpts; j++) {
		for (int k=colptr[j]; k<colptr[j+1]; k++) {
			for (int i=0; i<num_bad_guys; i++) {
				constraint[i*numCpts + rowinds[k]] += data[k] * B[i*n + Cpts[j]];
			}
		}
	}

	// Get column major submatrix to print out, verify
	std::vector<double> submat(81,0);
	std::vector<int> cols;
	for (int i=0; i<numCpts; i++) {
		cols.push_back(i);
	}
	std::set<int> rows = get_sub_mat(cols, colptr, rowinds, data, submat);
	int ncols = cols.size();
	int nrows = rows.size();

	std::cout << "Columns: \n\t";
	for (int i=0; i<cols.size(); i++) {
		std::cout << cols[i] << ", ";
	}
	std::cout << "\nRows: \n\t";
	for (auto it = rows.begin(); it!=rows.end(); ++it) {
		std::cout << *it << ", ";
	}
	std::cout << "\nSubmatrix: \n\t";
	for (int i=0; i<nrows; i++) {
		for (int j=0; j<numCpts; j++) {
			int ind = j*nrows + i;
			std::cout << submat[ind] << "\t";
		}
		std::cout << "\n\t";
	}

	std::cout << "Bc = \n\t";
	for (int i=0; i<num_bad_guys; i++) {
		for (int j=0; j<numCpts; j++) {
			std::cout << B[i*n+Cpts[j]] << ", ";			
		}
		std::cout << "\n\t";
	}
	std::cout << std::endl;

	std::cout << "Acc*Bc = \n\t";
	for (int i=0; i<num_bad_guys; i++) {
		for (int j=0; j<numCpts; j++) {
			std::cout << constraint[i*numCpts+j] << ", ";			
		}
		std::cout << "\n\t";
	}
	std::cout << std::endl;


	/* -------------- test forming constraint w = w_l * Acc = Acc * Bc -------------- */
	
	// std::vector<double> w_l {1,2,3,4,5,6};
	// std::vector<double> w;
	// std::vector<double> w_full;
	// std::vector<int> col_inds {0,1,2,3,4,5};

	// std::vector<double> submat(81,0);
	// std::vector<int> cols = Cpts;
	// std::set<int> rows = get_sub_mat(cols, colptr, rowinds, data, submat);
	// int ncols = cols.size();
	// int nrows = rows.size();

	// std::cout << "Columns: \n\t";
	// for (int i=0; i<cols.size(); i++) {
	// 	std::cout << cols[i] << ", ";
	// }
	// std::cout << "\nRows: \n\t";
	// for (auto it = rows.begin(); it!=rows.end(); ++it) {
	// 	std::cout << *it << ", ";
	// }
	// std::cout << "\nSubmatrix: \n\t";
	// for (int i=0; i<nrows; i++) {
	// 	for (int j=0; j<ncols; j++) {
	// 		int ind = j*nrows + i;
	// 		std::cout << submat[ind] << "\t";
	// 	}
	// 	std::cout << "\n\t";
	// }
	// std::cout << "\n";

	// // Let w_l := \hat{w}_lA_{cc}.
	// int row_length = 0;
	// for (int j=0; j<numCpts; j++) {

	// 	double temp_prod = 0;
	// 	int temp_v0 = 0;

	// 	// Loop over nonzero indices for this column of Acc and vector w_l.
	// 	// Note, both have ordered, unique indices.
	// 	for (int k=colptr[j]; k<colptr[j+1]; k++) {
	// 		for (int v_ind=temp_v0; v_ind<ncols; v_ind++) {

	// 			// Can break here because indices are sorted increasing
	// 			if ( col_inds[v_ind] > rowinds[k] ) {
	// 				break;
	// 			}
	// 			// If nonzero, add to dot product 
	// 			else if (col_inds[v_ind] == rowinds[k]) {
	// 				temp_prod += w_l[v_ind] * data[k];
	// 				temp_v0 += 1;
	// 				break;
	// 			}
	// 			else {
	// 				temp_v0 += 1;
	// 			}
	// 		}
	// 	}
	// 	// If dot product of column of Acc and vector \hat{w}_l is nonzero,
	// 	// add to sparse structure of P.
	// 	if (temp_prod != 0) {
	// 		row_length += 1;
	// 		w.push_back(temp_prod);
	// 	}
	// 	w_full.push_back(temp_prod);
	// }

	// std::cout << "w_l \t\t= ";
	// for (int i=0; i<w_l.size(); i++) {
	// 	std::cout << w_l[i] << ", ";
	// }
	// std::cout << "\ncol inds \t= ";
	// for (int i=0; i<w_l.size(); i++) {
	// 	std::cout << col_inds[i] << ", ";
	// }
	// std::cout << "\nw \t\t= ";
	// for (int i=0; i<w.size(); i++) {
	// 	std::cout << w[i] << ", ";
	// }
	// std::cout << "\nw_full \t\t= ";
	// for (int i=0; i<numCpts; i++) {
	// 	std::cout << w_full[i] << ", ";
	// }
	// std::cout << "\n";
	// if (row_length == w.size()) {
	// 	std::cout << "sizes agree\n";
	// }


#endif

#if 1
	// Construct sparse matrix, size 9x9
	std::vector<int> A_colptr {0,2,5,8,11,14,17,20,23,25};
	std::vector<int> A_rowinds {0,1,0,1,2,1,2,3,2,3,4,3,4,5,4,5,6,5,6,7,6,7,8,7,8};
	std::vector<double> A_data {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25};
	std::vector<double> submat(81,0);
	std::vector<int> cols = {0,1,2,7,8};

	std::set<int> rows = get_sub_mat(cols, A_colptr, A_rowinds, A_data, submat);

	int ncols = cols.size();
	int nrows = rows.size();

	int row_P = 6;
	std::vector<double> rhs(nrows,0);
	int i=0;
	for (auto it=rows.begin(); it!=rows.end(); it++, i++) {
		if ( (*it) == row_P ) {
			rhs[i] = 1.0;
		}
	}

	std::cout << "Columns: \n\t";
	for (int i=0; i<cols.size(); i++) {
		std::cout << cols[i] << ", ";
	}

	std::cout << "\nRows: \n\t";
	for (auto it = rows.begin(); it!=rows.end(); ++it) {
		std::cout << *it << ", ";
	}

	std::cout << "\nSubmatrix: \n\t";
	for (int i=0; i<nrows; i++) {
		for (int j=0; j<ncols; j++) {
			int ind = j*nrows + i;
			std::cout << submat[ind] << "\t";
		}
		std::cout << "\n\t";
	}

	std::cout << "\nRHS: \n\t";
	for (auto it = rhs.begin(); it!=rhs.end(); ++it) {
		std::cout << *it << ", ";
	}
	std::cout << "\n";


#endif

}


#if 0
rows = np.array([0,2,5,8,11,14,17,20,23,25],dtype=int)
cols = np.array([0,1,0,1,2,1,2,3,2,3,4,3,4,5,4,5,6,5,6,7,6,7,8,7,8],dtype=int)
data = np.arange(0,len(cols),dtype=int)
test = csr_matrix((data,cols,rows))
#endif

// A_colptr = np.array([0,2,5,8,11,14,17,20,23,25])
// A_rowinds = np.array([0,1,0,1,2,1,2,3,2,3,4,3,4,5,4,5,6,5,6,7,6,7,8,7,8])
// A_data = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])



