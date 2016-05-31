#include <iostream>
#include <vector>



// Makes vector enumerating F-points using positive numbers and C-points using negative. 
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
			ind_split[i] = find;
			find += 1;
		}
		else {
			ind_split[i ] = -cind;
			cind += 1;
		}
	}
	return ind_split;
}


// TODO : May need to bring back in row_scale so that I can use a single 
// 		  vector for Acc anf Acf. This msy not be possible though...?

void get_col_ptr(const int A_rowptr[],
				 const int A_colinds[],
				 const int &n,
				 const int is_col_ind[],
				 const int is_row_ind[],
				 std::vector<int> &colptr, 
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
			// std::cout << "row = " << i << ", data_ind = " << k << ", A col = " << A_colinds[k] << "\n";
			if ( ind > 0) {
				// std::cout << "\t is_col_ind = " << is_col_ind[A_colinds[k]] << ", new col_ind = " << ind-1 << "\n";
				colptr[ind] += 1;
			}
		}
	}

	// Cumulative sum column pointer to correspond with data entries
	for (int i=1; i<(num_cols); i++) {
		int temp = colptr[i];
		colptr[i] = colptr[i-1] + temp;
	}
	colptr[num_cols] += colptr[num_cols-1];
}


void get_csc_submatrix(const int A_rowptr[],
					   const int A_colinds[],
					   const double A_data[],
					   const int &n,
					   const int is_col_ind[],
					   const int is_row_ind[],
					   std::vector<int> &colptr, 
					   std::vector<int> &rowinds, 
					   std::vector<int> &data,
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


int main(int argc, char *argv[]) 
{
	// Construct sparse matrix, size 9x9
	std::vector<int> A_rowptr {0,2,5,8,11,14,17,20,23,25};
	std::vector<int> A_colinds {0,1,0,1,2,1,2,3,2,3,4,3,4,5,4,5,6,5,6,7,6,7,8,7,8};
	std::vector<double> A_data {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25};

	// Arbitrarily coarsen
	std::vector<int> Cpts {0,2,4,6,8};
	std::vector<int> Fpts {1,3,5,7};
	// std::vector<int> Cpts {4,5,6,7,8};		// Scale with -1
	// std::vector<int> Fpts {0,1,2,3};		// Scale with +1

	int numCpts = Cpts.size();
	int numFpts = Fpts.size();
	int n = numFpts + numCpts;
	int scale_row = -1;
	int scale_col = 1;
	int num_cols = numFpts;

	// Get splitting of points in one vector
	std::vector<int> splitting = get_ind_split(&Cpts[0],numCpts,n);

	// Get sparse CSC column pointer for submatrix Acf
	std::vector<int> colptr(num_cols+1,0);
	get_col_ptr(&A_rowptr[0], &A_colinds[0], n, &splitting[0], &splitting[0], colptr, num_cols, scale_row, scale_col);

	// Allocate row-ind and data arrays for sparse submatrix 
	int nnz = colptr[num_cols];
	std::vector<int> rowinds(nnz,0);
	std::vector<int> data(nnz,0);

	// Fill in sparse structure
	get_csc_submatrix(&A_rowptr[0], &A_colinds[0], &A_data[0], n,
					  &splitting[0], &splitting[0], colptr, rowinds,
					  data, num_cols, scale_row, scale_col);


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



}

#if 0
rows = [0,2,5,8,11,14,17,20,23,25]
cols = [0,1,0,1,2,1,2,3,2,3,4,3,4,5,4,5,6,5,6,7,6,7,8,7,8]
rows = np.array(rows,dtype=int)
cols = np.array(cols,dtype=int)
data = np.arange(0,len(cols),dtype=int)
test = csr_matrix((data,cols,rows))
#endif
