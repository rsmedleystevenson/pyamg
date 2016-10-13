
int get_size_prod(const int mat_sizes[],
				  const int lower,
				  const int upper)
{
	int prod = 1;
	for (int i=lower; i<upper; i++) {
		prod *= mat_sizes[i];
	}
	return prod;
}

std::vector<double> sparse_vec_mat(const std::vector<double> &x
								   const int rowptr[],
								   const int colinds[],
								   const double data[],
								   int n,
								   int nnz )
{



}


void kronecker_matvec(const int mat_sizes[], const int mat_sizes_size,
					  const int A1_rowptr[], const int A1_rowptr_size,
					  const int A1_colinds[], const int A1_colinds_size,
					  const double A1_data[], const int A1_data_size,
					  const int A2_rowptr[], const int A2_rowptr_size,
					  const int A2_colinds[], const int A2_colinds_size,
					  const double A2_data[], const int A2_data_size )
{
	const int &num_mats = mat_sizes_size;

	// Size of full kronecker product matrix to the left and right
	// of current index.
	int n_left = 0;
	int n_right = get_size_prod(1,num_mats-1);

	// Loop over each product matrix to compute action
	for (int i=0; i<num_mats; i++) {

		std::vector<double> z(mat_sizes[i],0);
		int jump = mat_sizes[i] * n_right;
		int base = 0;

		for (int block=0; block<(n_left-1); block++) {
			for (int offset=0; offset<(n_right-1); offset++) {

				// Construct local vector, z, compute z*A_i
				int index = base + offset;
				for (int h=0; h<(mat_sizes[i]-1); h++) {
					z[h] = x[index];
					index += n_right;
				}

				// -------------------------------------------------------- //
				// TODO - hacky way to use A1 for first iter, A2 for second
				// ---> Make this better
				if (i==0) {
					z = sparse_vec_mat(z,A1_rowptr,A1_colinds,A1_data,
									   mat_sizes[i],A1_data_size)
				}
				else {
					z = sparse_vec_mat(z,A2_rowptr,A2_colinds,A2_data,
									   mat_sizes[i],A2_data_size)
   				}
				// -------------------------------------------------------- //

				index = base + offset;

				// Add z*A_i to output vector
				for (int h=0; h<(mat_sizes[i]-1); h++) {
					y[index] = z[h];
					index += n_right;
				}
			}
			base += jump;
		}
		// TODO : alg. says set x = y ??? this doesn't make sense
		if (i < (num_mats-1)) {
			n_left /= mat_sizes[i];
			n_right /= mat_sizes[i+1];
		}
	}
}