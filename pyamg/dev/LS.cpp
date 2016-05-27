
#include <vector>
#include <iostream>
#include <cmath>
#include <iomanip>



inline int signof(int a) { return (a<0 ? -1 : 1); }

// References may not work here...
inline int row_major(const int &row, const int &col, const int &num_cols) 
{
	// return col*num_cols + row;		// THIS IS WRONG - JUST A TEST
	return row*num_cols + col;
}

// References may not work here...
inline int col_major(const int &row, const int &col, const int &num_rows) 
{
	return col*num_rows + row;
}

void print_mat(std::vector<double> &vec, const int &m, const int &n,
					 const bool &is_col_major=0) 
{
	if (is_col_major) {
		for (int i=0; i<m; i++) {
			for (int j=0; j<n; j++) {
				double val = vec[col_major(i,j,m)];
				if (std::abs(val) < 1e-12) { val = 0; }
				std::cout << std::setprecision(4) << std::setw(8) << val << ", ";
			}
			std::cout << std::endl;
		}
	}
	else {
		for (int i=0; i<m; i++) {
			for (int j=0; j<n; j++) {
				double val = vec[row_major(i,j,n)];
				if (std::abs(val) < 1e-12) { val = 0; }
				std::cout << std::setprecision(4) << std::setw(8) << val << ", ";
			}
			std::cout << std::endl;
		}
	}
}


// QR on matrix A stored in row-major form. A overwritten with R, Q returned.
std::vector<double> QR(std::vector<double> &A,
					   const int &m,
					   const int &n,
					   const bool is_col_major=0)
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


// Backwards substitution solve on upper triangular system. Note, R
// need not be square, the system will be solved over the rank r upper
// triangular block. If remaining entries in solution available, will
// be set to zero. Solution stored in vector reference x.
void upper_tri_solve(const std::vector<double> &R,
					 const std::vector<double> &rhs,
					 std::vector<double> &x,
					 const int &m,
					 const int &n,
					 const bool is_col_major=0)
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
	
	// Check vector is large enough to hold solution
	int rank = std::min(m,n);
	if (x.size() < rank) {
		std::cout << "Warning - vector too small - reallocating.\n";
		x.resize(rank);
	}

	// Backwards substitution
	for (int i=(rank-1); i>=0; i--) {
		double temp = rhs[i];
		for (int j=(i+1); j<rank; j++) {
			temp -= R[get_ind(i,j,*C)]*x[j];
		}
		x[i] = temp / R[get_ind(i,i,*C)];
	}

	// If rank < size of rhs, set elements to zero
	if (rank < x.size()) {
		for (int i=rank; i<x.size(); i++) {
			x[i] = 0;
		}
	}
}


// Least squares minimization using QR. If system is under determined, 
// free entries are set to zero. 
void least_squares(std::vector<double> &A,
				   std::vector<double> &b,
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

	// Multiply right hand side, b:= Q^T*b
	std::vector<double> rhs(m,0);
	for (int i=0; i<m; i++) {
		for (int k=0; k<m; k++) {
			rhs[i] += b[k] * Q[get_ind(k,i,m)];
		}
	}

	// Solve upper triangular system, overwrite b with solution.
	upper_tri_solve(A,rhs,b,m,n,is_col_major);
}


// Assume A is mxn, C is sxn. 
void constrained_least_squares(std::vector<double> &A,
							   std::vector<double> &b,
							   std::vector<double> &C,
							   std::vector<double> &d,
							   const int &m,
							   const int &n,
							   const int &s,
							   const bool is_col_major=0)
{


	// Need to track which vectors are stored in what format...
	// --> TODO : Maybe best to be consistent
	// 		Want QR on C^T, not C...
	std::vector<double> Qc = QR(C,)


	// Form matrix product A*Qc


	// Satisfy constraints R^Tz = d, move to rhs of LS


	// Call LS on reduced system


	// Form x = Q*z

}



int main(int argc, char *argv[]) 
{

/* QR unit tests */
 	// Tall and skinny unit test
 	// std::vector<double> A {1,2,3, 1,1,1, 1,-1,1, 0,0,4, 2,3,-9};
 	// int m = 5;
 	// int n = 3;
 	// Short and fat unit test
 	// std::vector<double> A {1,2,3,4,5, 1,1,1,1,1, 1,-1,1,-1,1};
 	// int m = 3;
 	// int n = 5;
 	// Square unit test
	// std::vector<double> A {1,2,3,4,5, 1,1,1,1,1, 1,-1,1,-1,1, 0,0,0,4,2,0, 2,3,-9,7,1};
	// int m = 5;
	// int n = 5;

	// int is_col_major = 0;
	// std::cout << "A : \n";
	// print_mat(A,m,n,is_col_major);
	// std::vector<double> Q = QR(A,m,n,is_col_major);
	// std::cout << "R : \n";
	// print_mat(A,m,n,is_col_major);
	// std::cout << "Q : \n";
	// print_mat(Q,m,m,is_col_major);

/* Upper triangular solve unit tests */
 	// Tall skinny R 
 	// std::vector<double> R {1,2,3,4, 0,1,-1,1, 0,0,2,0, 0,0,0,3, 0,0,0,0, 0,0,0,0};
 	// std::vector<double> rhs {-2,5,3,0};
 	// std::vector<double> x(4);
 	// int n = 4;
 	// int m = 6;
 	// Short fat R 
 	// std::vector<double> R {1,2,3,4,1,-1, 0,1,-1,1,-1,1, 0,0,2,0,-2,3, 0,0,0,3,1,1};
 	// std::vector<double> rhs {-2,5,3,0};
 	// std::vector<double> x(4);
 	// int n = 6;
 	// int m = 4;

	// std::cout << "R : \n";
	// print_mat(R,m,n,is_col_major);
	// upper_tri_solve(R,rhs,x,m,n);
	// for (int i=0; i<n; i++) {
	// 	std::cout << x[i] << std::endl;
	// }

/* Normal least squares unit tests */
 	// Tall and skiing
	// std::vector<double> A {1,2,3,4, 0,-1,2,1, -3,-2,2,0, 5,5,1,0, 3,2,0,1, 0,0,1,0};
	// std::vector<double> rhs {-2,5,3,0,1,1};
	// int m = 6;
	// int n = 4;
	// Square
	// std::vector<double> A {1,2,3,4,5, 0,-1,2,1,-1, -3,-2,2,0,1, 5,5,1,0,0, 3,2,0,1,1};
	// std::vector<double> rhs {1,0,3,-2,0};
	// int m = 5;
	// int n = 5;
	// Short and fat (under determined)
	// std::vector<double> A {1,2,3,4,5,6, 0,-1,2,1,-1,0, -3,-2,2,0,1,0, 5,5,1,0,0,-1};
	// // std::vector<double> rhs {3,3,0,-1};
	// std::vector<double> rhs {-2,5,3,0,1,1};
	// int m = 4;
	// int n = 6;

	// int is_col_major = 0;
	// std::cout << "A : \n";
	// print_mat(A,m,n,is_col_major);
	// std::cout << "rhs : \n\t";
	// for (int i=0; i<m; i++) {
	// 	std::cout << rhs[i] << ", ";
	// }
	// std::cout << std::endl;
	// least_squares(A,rhs,m,n,is_col_major);
	// std::cout << "x : \n\t";
	// for (int i=0; i<n; i++) {
	// 	std::cout << rhs[i] << ", ";
	// }
	// std::cout << std::endl;




}





