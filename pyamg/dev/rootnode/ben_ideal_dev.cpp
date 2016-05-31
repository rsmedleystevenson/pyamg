#include <vector>




// For now will assume symmetry of Acc. 
// Need some kind of sparsity pattern? --> Use multiple pairwise aggregations for sparsity...
// 		Then stencil stretches in right direction
//	- TODO : Make sure C-points are sorted
template<class I, class T, class F>
void ben_ideal_interpolation(const I A_rowptr[], const int A_rowptr_size,
		                     const I A_colinds[], const int A_colinds_size,
		                     const T A_data[], const int A_data_size,
		                     const T B[], const int B_size,
                             const I Cpts[], const int Cpts_size,
		                     const I n,
                             const I num_bad_guys)
{

	// Form set of F-points, given set of C-points. 
	vector<I> Fpts(n-Cpts_size);
	get_fpts(Cpts, Cpts_size, Fpts);

	// TODO : Get Acc submatrix in CSC format
	std::vector<I> Acc_colptr(Cpts_size);
	std::vector<I> Acc_rowinds;
	std::vector<I> Acc_data;


	// Form constraint vector, \hat{B}_c = A_{cc}B_c
	// TODO : Adjust to be CSC compatible
	vector<T> constraint(Bc_size, 0);
	for (I row=0; row<num_Cpts; row++) {
		I data_ind0 = Acc_rowptr[row];
		I temp_ind1 = Acc_rowptr[row+1];
		for (I j=data_ind0; j<data_ind1; j++) {
			I col = Acc_colinds[j];
			for (I i=0; i<num_bad_guys; i++) {
				constraint[i*num_bad_guys+row] += Acc_data[j]*B[i*num_bad_guys+j];
		}
	}

	// TODO : Form Afc submatrix in CSC format


	// Form P row-by-row
	I next_Cind = 0;
	I data_ind = 0; 
	for (I row_P=0; i<n; i++) {

		// Check if row is a C-point (assume Cpts ordered).
		// If so, add identity to P.
		if (row_P == Cpts[next_Cind]) {
			P_rowptr[row_P+1] = P_rowptr[row_P] + 1;
			P_colinds[data_ind] = next_Cind;
			P_data[data_ind] = 1.0;
			data_ind += 1;
			next_Cind += 1;
		}

		// If row is an F-point, form row of \hat{W} through constrained
		// minimization and multiply by A_{cc} to get row of P. 
		else {

			vector<I> vec_inds;
			vector<I> vec_data;
			I vec_length;

			// TODO :
			//	- Select submatrix for given row from Afc
			//	- Call CLS routine
			// 	- See if possible to preallocate vec_data and vec_inds for all rows



			// Let w_l := \hat{w}_lA_{cc}.
			// TODO : Change this to CSC format compatible - should be pretty close now
			I row_length = 0;
			for (I row_Acc=0; row_Acc<num_Cpts; row_Acc++) {
				I temp_ind0 = Acc_rowptr[row_Acc];
				I temp_ind1 = Acc_rowptr[row_Acc+1];
				T temp_prod = 0;

				// Loop over nonzero indices for this row of Acc and current vector \hat{w}_l.
				for (I d_ind=temp_ind0; d_ind<temp_ind1; d_ind++) {
					for (I v_ind=0; v_ind<vec_length; v_ind++) {
						// If nonzero, add to dot product 
						if (Acc_colinds[d_ind] == vec_inds[v_ind]) {
							temp_prod += vec_data[v_ind] * Acc_data[d_ind];
						}
					}
				}
				// If dot product of row of Acc and vector \hat{w}_l is nonzero, add to 
				// sparse structure of P.
				if (temp_prod != 0) {
					P_colinds[data_ind] = row_Acc;
					P_data[data_ind] = temp_prod;
					data_ind += 1;
					row_length += 1;
				}
			}

			// Set row pointer for next row in P
			P_rowptr[row_P+1] = P_rowptr[row_P] + row_length;
		}
	}













}

