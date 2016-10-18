#include <iostream>
#include <vector>
#include <cmath>


std::vector<double> sparse_vecmat(const std::vector<double> &x,
                                  const int indptr[],
                                  const int indices[],
                                  const double data[],
                                  int m,
                                  int n )
{
    std::vector<double> b(n,0);
    for(int i=0; i<m; i++) { 
        int ind1 = indptr[i],
            ind2 = indptr[i+1];
        for(int j=ind1; j<ind2; j++) {
            b[indices[j]] += x[i]*data[j];
        }
    }
    return b;
}

std::vector<double> sparse_matvec(const std::vector<double> &x,
                                  const int indptr[],
                                  const int indices[],
                                  const double data[],
                                  const int m,
                                  const int n)
{
    std::vector<double> b(m,0);
    for(int i=0; i<m; i++) { 
        int ind1 = indptr[i],
            ind2 = indptr[i+1];
        for(int j=ind1; j<ind2; j++) {
            b[i] += x[indices[j]]*data[j];
        }
    }
    return b;
}


/* Compute product of range of entries in mat_sizes[] for
 * indices lower,...,upper. 
 */
int get_size_prod(const int mat_sizes[],
                  const int lower,
                  const int upper)
{
    int prod = 1;
    for (int i=lower; i<=upper; i++) {
        prod *= mat_sizes[i];
    }
    return prod;
}


/* For CSR matrix A, compute
 *
 * 		y += I_(n_left) \otimes A \otimes I_(n_right),
 *
 * for matrix sizes n_left, n_right. Computing a mat-vec
 * by a Kronecker sum matrix A = \bigoplus_{i=1}^k A_i is then given as
 * 
 *  	y = [0,...,0]
 * 		for i=1,...,k
 *			partial_kronsum_matvec(A_i, x, y, n_1^(i-1), n_(i+1)^k, n_i)
 *
 * where n_i is the size of A_i, and
 *
 *		n_i^k = 1 * \prod_{l=i}^k n_l,
 *
 * for i <= k, and n_i^(i-1) = 1.
 *
 * Parameters
 * ----------
 *		A_rowptr : array<int>
 *			Row pointer for A in csr format.
 *		A_colinds : array<int>
 *			Column indices for A in csr format.
 * 		A_data : array<double>
 *			Data for A in csr format.
 *		x : array<double>
 *			Vector to multiply by, size (n_left * n_right * n).
 * 		y : array<double>
 *			Output vector, size (n_left * n_right * n); solution *added* to y.
 * 		n_left : int
 *			Size of identity on left of A.
 *		n_right : int
 *			Size of identity on right of A.
 *		n : int
 * 			Size of A.
 *
 * Returns
 * -------
 * Nothing, y is modified in place.
 *
 * References
 * ----------
 * [1] Buchholz, Peter, et al. "Complexity of Kronecker operations on sparse
 *     matrices with applications to the solution of Markov models." (1997).
 *
 * TODO : Maybe keep this for the case of square matrices as it is cheaper?
 */
#if 0 
void kronprod_matvec(const int num_cols[], const int num_cols_size,
					 const int num_rows[], const int num_rows_size,
					 const int A1_rowptr[], const int A1_rowptr_size,
					 const int A1_colinds[], const int A1_colinds_size,
					 const double A1_data[], const int A1_data_size,
					 const int A2_rowptr[], const int A2_rowptr_size,
					 const int A2_colinds[], const int A2_colinds_size,
					 const double A2_data[], const int A2_data_size,
				    	   double x[], const int x_size,
				    	   double y[], const int y_size)
{
	const int &num_mats = num_cols_size;
	// std::vector<double> y(x_size,0);

	// Size of full kronecker product matrix to the left and right
	// of current index.
	int n_left = 1;
	int n_right = get_size_prod(num_cols,1,num_mats-1);

	// Loop over each product matrix to compute action
	for (int i=0; i<num_mats; i++) {

		std::vector<double> z(num_rows[i],0);
		int jump = num_cols[i] * n_right;
		int base = 0;

		for (int block=0; block<n_left; block++) {
			for (int offset=0; offset<n_right; offset++) {

				// Construct local vector, z, compute A_i*z
				int index = base + offset;
				int index = base + offset;
				for (int h=0; h<num_cols[i]; h++) {
					z[h] = x[index];
					index += n_right;
				}

				// -------------------------------------------------------- //
				// TODO - hacky way to use A1 for first iter, A2 for second
				// ---> Make this better
				// 		Need to generalize to more matrices anyways...
				if (i==0) {
					z = sparse_matvec(z,A1_rowptr,A1_colinds,A1_data,num_rows[i]);
				}
				else {
					z = sparse_matvec(z,A2_rowptr,A2_colinds,A2_data,num_rows[i]);
   				}
				// -------------------------------------------------------- //

				index = base + offset;

				// Add A_i*z to output vector
				for (int h=0; h<num_rows[i]; h++) {
					y[index] = z[h];
					index += n_right;
				}
			}
			base += jump;
		}
		// Update values of x with new values of y.
		for (int k=0; k<x_size; k++) {
			x[k] = y[k];
		}

		if (i < (num_mats-1)) {
			n_left *= num_cols[i];
			n_right /= num_cols[i+1];
		}

	}
}
#endif



void kronprod_vecmat(const int num_rows[], const int num_rows_size,
					 const int num_cols[], const int num_cols_size,
					 const int A1_rowptr[], const int A1_rowptr_size,
					 const int A1_colinds[], const int A1_colinds_size,
					 const double A1_data[], const int A1_data_size,
					 const int A2_rowptr[], const int A2_rowptr_size,
					 const int A2_colinds[], const int A2_colinds_size,
					 const double A2_data[], const int A2_data_size,
				    	   double x[], const int x_size,
				    	   double y[], const int y_size)
{
	const int &num_mats = num_cols_size;
	std::vector<double> q(std::max(y_size,x_size),0);
	for (int i=0; i<x_size; i++) {
		y[i] = x[i];
	}

	// Size of full kronecker product matrix to the left and right
	// of current index.
	int n_left = 1;
	int n_right = get_size_prod(num_rows,1,num_mats-1);

	// Loop over each product matrix to compute action
	for (int i=0; i<num_mats; i++) {

		int base_i = 0;
		int base_j = 0;

		for (int il=0; il<n_left; il++) {
			for (int ir=0; ir<n_right; ir++) {

				// Construct local vector, z, compute A_i*z
				std::vector<double> z(num_rows[i],0);
				int index_i = base_i + ir;
				for (int row=0; row<num_rows[i]; row++) {
					z[row] = y[index_i];
					std::cout << "z[" << index_i << "] = " << q[index_i] << ", ";
					index_i += n_right;
				}
				std::cout << "\n";

				// -------------------------------------------------------- //
				// TODO - hacky way to use A1 for first iter, A2 for second
				// ---> Make this better
				// 		Need to generalize to more matrices anyways...
				if (i==0) {
					z = sparse_vecmat(z,A1_rowptr,A1_colinds,A1_data,
									  num_rows[i],num_cols[i]);
				}
				else {
					z = sparse_vecmat(z,A2_rowptr,A2_colinds,A2_data,
									  num_rows[i],num_cols[i]);
   				}
				// -------------------------------------------------------- //

				int index_j = base_j + ir;

				// Add A_i*z to output vector
				for (int col=0; col<num_cols[i]; col++) {
					q[index_j] = z[col];
					std::cout << "q[" << index_j << "] = " << q[index_j] << ", ";
					index_j += n_right;
				}
				std::cout << "\n";
			}
			base_i += num_rows[i]*n_right;
			base_j += num_cols[i]*n_right;
		}
		// Update values of x with new values of y.
		for (int k=0; k<y_size; k++) {
			y[k] = q[k];
		}

		if (i < (num_mats-1)) {
			n_left *= num_cols[i];
			n_right /= num_rows[i+1];
		}
	}
}


void kronprod_matvec(const int num_rows[], const int num_rows_size,
					 const int num_cols[], const int num_cols_size,
					 const int A1_rowptr[], const int A1_rowptr_size,
					 const int A1_colinds[], const int A1_colinds_size,
					 const double A1_data[], const int A1_data_size,
					 const int A2_rowptr[], const int A2_rowptr_size,
					 const int A2_colinds[], const int A2_colinds_size,
					 const double A2_data[], const int A2_data_size,
				    	   double x[], const int x_size,
				    	   double y[], const int y_size)
{
	const int &num_mats = num_cols_size;
	std::vector<double> q(std::max(y_size,x_size),0);
	for (int i=0; i<x_size; i++) {
		y[i] = x[i];
	}

	// Size of full kronecker product matrix to the left and right
	// of current index.
	int n_left = 1;
	int n_right = get_size_prod(num_cols,1,num_mats-1);

	// Loop over each product matrix to compute action
	for (int i=0; i<num_mats; i++) {

		int base_i = 0;
		int base_j = 0;

		for (int il=0; il<n_left; il++) {
			for (int ir=0; ir<n_right; ir++) {

				// Construct local vector, z, compute A_i*z
				std::vector<double> z(num_cols[i],0);
				int index_i = base_i + ir;
				for (int col=0; col<num_cols[i]; col++) {
					z[col] = y[index_i];
					std::cout << "z[" << index_i << "] = " << q[index_i] << ", ";
					index_i += n_right;
				}
				std::cout << "\n";

				// -------------------------------------------------------- //
				// TODO - hacky way to use A1 for first iter, A2 for second
				// ---> Make this better
				// 		Need to generalize to more matrices anyways...
				if (i==0) {
					z = sparse_matvec(z,A1_rowptr,A1_colinds,A1_data,
									  num_rows[i],num_cols[i]);
				}
				else {
					z = sparse_matvec(z,A2_rowptr,A2_colinds,A2_data,
									  num_rows[i],num_cols[i]);
   				}
				// -------------------------------------------------------- //

				int index_j = base_j + ir;

				// Add A_i*z to output vector
				for (int row=0; row<num_rows[i]; row++) {
					q[index_j] = z[row];
					std::cout << "q[" << index_j << "] = " << q[index_j] << ", ";
					index_j += n_right;
				}
				std::cout << "\n";
			}
			base_i += num_cols[i]*n_right;
			base_j += num_rows[i]*n_right;
		}
		// Update values of x with new values of y.
		for (int k=0; k<y_size; k++) {
			y[k] = q[k];
		}

		if (i < (num_mats-1)) {
			n_left *= num_rows[i];
			n_right /= num_cols[i+1];
		}
	}
}


void partial_kronsum_matvec(const int A_rowptr[], const int A1_rowptr_size,
                            const int A_colinds[], const int A1_colinds_size,
                            const double A_data[], const int A1_data_size,
                            const double x[], const int x_size,
                                  double y[], const int y_size,
                            const int n_left,
                            const int n_right,
                            const int n,
                            const int left_mult)
{
    int base = 0;
    int jump = n * n_right;
    std::vector<double> z(n,0);

    for (int block=0; block<n_left; block++) {
        for (int offset=0; offset<n_right; offset++) {

            // Construct local vector, z, compute A_i*z
            int index = base + offset;
            for (int h=0; h<n; h++) {
                z[h] = x[index];
                index += n_right;
            }
            z = sparse_matvec(z,A_rowptr,A_colinds,A_data,n,n);
            index = base + offset;

            // Add A_i*z to output
            for (int h=0; h<n; h++) {
                y[index] += z[h];
                index += n_right;
            }
        }
        base += jump;
    }
}


void partial_kronprod_matvec(const int A_rowptr[], const int A_rowptr_size,
							 const int A_colinds[], const int A_colinds_size,
							 const double A_data[], const int A_data_size,
						    	   double y[], const int y_size,
						    	   double q[], const int q_size,
						     const int num_rows,
							 const int num_cols,
							 const int num_left,
							 const int num_right )
{
	int base_i = 0;
	int base_j = 0;

	for (int il=0; il<n_left; il++) {
		for (int ir=0; ir<n_right; ir++) {

			// Construct local vector, z.
			std::vector<double> z(num_cols,0);
			int index_i = base_i + ir;
			for (int col=0; col<num_cols; col++) {
				z[col] = y[index_i];
				index_i += n_right;
			}

			// Compute A_i * z.
			z = sparse_matvec(z,A_rowptr,A_colinds,A_data,
							  num_rows,num_cols);

			// Add A_i * z to output vector.
			int index_j = base_j + ir;
			for (int row=0; row<num_rows; row++) {
				q[index_j] = z[row];
				index_j += n_right;
			}
		}
		// Update indices
		base_i += num_cols*n_right;
		base_j += num_rows*n_right;
	}

	// Update values of y with new values of q.
	for (int k=0; k<y_size; k++) {
		y[k] = q[k];
	}
}


void partial_kronprod_vecmat(const int A_rowptr[], const int A_rowptr_size,
							 const int A_colinds[], const int A_colinds_size,
							 const double A_data[], const int A_data_size,
						    	   double y[], const int y_size,
						    	   double q[], const int q_size,
						     const int num_rows,
							 const int num_cols,
							 const int num_left,
							 const int num_right )
{
	int base_i = 0;
	int base_j = 0;

	for (int il=0; il<n_left; il++) {
		for (int ir=0; ir<n_right; ir++) {

			// Construct local vector, z.
			std::vector<double> z(num_rows,0);
			int index_i = base_i + ir;
			for (int row=0; row<num_rows; row++) {
				z[row] = y[index_i];
				index_i += n_right;
			}

			// Compute z * A_i.
			z = sparse_vecmat(z,A1_rowptr,A1_colinds,A1_data,
							  num_cols,num_rows);

			// Add z * A_i to output vector.
			int index_j = base_j + ir;
			for (int col=0; col<num_cols; col++) {
				q[index_j] = z[col];
				index_j += n_right;
			}
		}
		// Update indices
		base_i += num_rows*n_right;
		base_j += num_cols*n_right;
	}
	// Update values of x with new values of y.
	for (int k=0; k<y_size; k++) {
		y[k] = q[k];
	}
}











int main()
{
	std::vector<int> A1_rowptr = {0,1,2,3,4};
	std::vector<int> A1_colinds = {0,1,2,3,4};
	std::vector<double> A1_data = {1,2,3,4};
	std::vector<int> A2_rowptr = {0,2,5,7,8};
	std::vector<int> A2_colinds = {0,1,0,1,2,1,2,2};
	std::vector<double> A2_data = {1,2,3,4,5,6,7,8};
	std::vector<int> A3_rowptr = {0,2,5,7};
	std::vector<int> A3_colinds = {0,1,1,2,3,2,3};
	std::vector<double> A3_data = {1,2,3,4,5,6,7};

	std::vector<int> num_rows = {4,4};
	std::vector<int> num_cols = {3,3};
	// std::vector<double> x = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
	// std::vector<double> y(16,0);
	std::vector<double> x = {0,1,2,3,4,5,6,7,8};
	std::vector<double> y(16,0);

	kronprod_matvec(&num_rows[0], num_rows.size(),
					 &num_cols[0], num_cols.size(),
					 &A2_rowptr[0], A2_rowptr.size(),
					 &A2_colinds[0], A2_colinds.size(),
					 &A2_data[0], A2_data.size(),
					 &A2_rowptr[0], A2_rowptr.size(),
					 &A2_colinds[0], A2_colinds.size(),
					 &A2_data[0], A2_data.size(),
					 &x[0], x.size(),
					 &y[0], y.size());

	for (int i=0; i<y.size(); i++) {
		std::cout << y[i] << ", ";	
	}
	std::cout << "\n";


	// std::vector<double> y = {0,1,2,3,4,5,6,7,8,9,10,11};
	// std::vector<double> z(16,0);
	// int n_left = 2;
	// int n_right = 2;
	// partial_kronsum_matvec(&A2_rowptr[0], A2_rowptr.size(),
	// 					   &A2_colinds[0], A2_colinds.size(),
	// 					   &A2_data[0], A2_data.size(),
	// 					   &y[0], y.size(),
	// 					   &z[0], z.size(),
	// 					   n_left,
	// 					   n_right,
	// 					   mat_sizes[1],
	// 					   1);

	// for (int i=0; i<z.size(); i++) {
	// 	std::cout << z[i] << ", ";		
	// }
	// std::cout << "\n";


}
