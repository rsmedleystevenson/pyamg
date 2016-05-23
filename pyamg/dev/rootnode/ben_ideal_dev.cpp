#include <vector>


// Function to get list of F-points given C-points. 
// --> Note, this assumes that C-points are sorted for efficiency.
template<class I>
void get_fpts(const I Cpts[], const int Cpts_size, vector<I> &Fpts)
{
	I temp_Cind = 0;
	I temp_Find = 0;
	I C0 = 0;
	I C1 = Cpts[0];
	for (I i=C0; i<C1; i++) {
		Fpts[temp_Find] = i;
		temp_Find += 1;
	}
	while (temp_Cind < (Cpts_size-1)) {
		C0 = Cpts[temp_Cind]+1;
		C1 = Cpts[temp_Cind+1];
		for (I i=C0; i<C1; i++) {
			Fpts[temp_Find] = i;
			temp_Find += 1;
		}
		temp_Cind += 1;
	}
	C0 = Cpts[temp_Cind]+1;
	for (I i=C0; i<n; i++) {
		Fpts[temp_Find] = i;
		temp_Find += 1;
	}
}


// Note, column indices for A are sorted a-priori - this is yuuuge
void get_submatrix(const I A_rowptr[],
				   const I A_colinds[],
				   const I A_data[],
				   const I row_inds[],
				   const I &row_inds_size,
				   const I col_inds[],
				   const I &col_inds_size,
				   std::vector<I> &rowptr, 
				   std::vector<I> &colinds,
				   std::vector<I> &data )
{

	rowptr.push_back(0);
	for (I i=0; i<row_inds_size; i++) {
		I row = row_inds[i];
		for (I j=A_rowptr[row]; j<A_rowptr[row+1]; j++) {

		}
	}




}


// For now will assume symmetry of Acc. 
// Need some kind of sparsity pattern? --> Use multiple pairwise aggregations for sparsity...
// 		Then stencil stretches in right direction
template<class I, class T, class F>
void ben_ideal_interpolation(const I Acc_rowptr[], const int Acc_rowptr_size,
		                     const I Acc_colinds[], const int Acc_colinds_size,
		                     const T Acc_data[], const int Acc_data_size,
		                     const T Bc[], const int Bc_size,
		                     const T Bf[], const int Bf_size,
                             const I Cpts[], const int Cpts_size,
		                     const I n,

                             const I num_bad_guys)
{

	// Form set of F-points, given set of C-points. 
	vector<I> Fpts(n-Cpts_size);
	get_fpts(Cpts, Cpts_size, Fpts);

	// Form constraint vector, \hat{B}_c = A_{cc}B_c
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

	// Form P row-by-row
	I next_Cind = 0;
	I data_ind = 0; 
	for (I row_P=0; i<n; i++) {

		// Check if row is a C-point (assume Cpts ordered).
		// If so, add identity to P.
		if (row_P == Cpts[next_Cind]) {
			P_rowptr[row_P+1] = P_rowptr[row_P] + 1;
			P_colinds[data_ind] = row_P;
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





			// Let w_l := \hat{w}_lA_{cc}. Assume symmetry and compute A_{cc}\hat{w}_l^T
			// row-by-row. 
			I row_length = 0;
			for (I row_Acc=0; row_Acc<num_Cpts; row_Acc++) {
				I temp_ind0 = Acc_rowptr[row_Acc];
				I temp_ind1 = Acc_rowptr[row_Acc+1];
				I temp_prod = 0;

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

