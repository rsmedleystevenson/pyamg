
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

void print_mat(double vec[], const int &m, const int &n,
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


// QR on matrix A stored in row or column major. A is 
// overwritten with R and Q is returned, both in the  
// same storage format that A is passed in as.
std::vector<double> QR(double A[],
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
//	- R is m x n
//  - rhs has length m
//	- x has length n
void upper_tri_solve(const double R[],
					 const double rhs[],
					 double x[],
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


// Forward substitution solve on lower triangular system. Note, L
// need not be square, the system will be solved over the rank r lower
// triangular block. If remaining entries in solution available, will
// be set to zero. Solution stored in vector reference x.
//	- L is m x n
//  - rhs has length m
//	- x has length n
void lower_tri_solve(const double L[],
					 const double rhs[],
					 double x[],
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

	// Backwards substitution
	int rank = std::min(m,n);
	for (int i=0; i<rank; i++) {
		double temp = rhs[i];
		for (int j=0; j<i; j++) {
			temp -= L[get_ind(i,j,*C)]*x[j];
		}
		if (std::abs(L[get_ind(i,i,*C)]) < 1e-12) {
			std::cout << "Warning: Upper triangular matrix near singular.\n"
						 "Dividing by ~ 0.\n";
		}
		x[i] = temp / L[get_ind(i,i,*C)];
	}

	// If rank < size of rhs, set free elements in x to zero
	for (int i=m; i<n; i++) {
		x[i] = 0;
	}
}



// Least squares minimization using QR. If system is under determined, 
// free entries are set to zero. 
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


// Assume A is mxn, C is sxn. 
//	- Pass A and C^T in column major (C in in row major)
std::vector<double> constrained_least_squares(std::vector<double> &A,
											  std::vector<double> &b,
											  std::vector<double> &Ct,
											  std::vector<double> &d,
											  const int &m,
											  const int &n,
											  const int &s)
{

	// Need to track which vectors are stored in what format...
	std::vector<double> Qc = QR(&Ct[0],n,s,1);

	// Form matrix product S = A*Qc in column major. Stored in A.
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
			Ct[col_major(i,j,n)] = Ct[col_major(j,i,n)];
		}
	}

	// Satisfy constraints R^Tz = d, move to rhs of LS
	lower_tri_solve(&Ct[0],&d[0],&temp_vec[0],n,s,1);
	for (int i=0; i<m; i++) {
		double val = 0.0;
		for (int j=0; j<s; j++) {
			val += A[col_major(i,j,m)] * temp_vec[j];
		}
		b[i] -= val;
	}

	// Call LS on reduced system
	// 	- TODO : Either need to define (n-s) or not pass by reference
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
 // 	std::vector<double> R {1,2,3,4, 0,1,-1,1, 0,0,2,0, 0,0,0,1};
 // 	std::vector<double> rhs {-2,5,3,0};
 // 	std::vector<double> x(4);
 // 	int n = 4;
 // 	int m = 4;
 // 	int is_col_major = 0;

	// std::cout << "R : \n";
	// print_mat(R,m,n,is_col_major);
	// std::cout << "rhs : \n\t";
	// for (int i=0; i<n; i++) {
	// 	std::cout << rhs[i] << ", ";
	// }
	// std::cout << "\n";

	// upper_tri_solve(&R[0],&rhs[0],&x[0],m,n);
	// std::cout << "x : \n\t";
	// for (int i=0; i<n; i++) {
	// 	std::cout << x[i] << ", ";
	// }
	// std::cout << "\n";



/* Upper triangular solve unit tests */
 	// Short fat L (column major)
 	// std::vector<double> L {1,2,3,4, 0,1,-1,1, 0,0,2,0, 0,0,0,3, 0,0,0,0, 0,0,0,0};
 	// std::vector<double> rhs {-2,5,3,0};
 	// std::vector<double> x(6);
 	// int n = 6;
 	// int m = 4;
 	// Tall skinny L (column major)
 	// std::vector<double> L {1,2,3,4,1,-1, 0,1,-1,1,-1,1, 0,0,2,0,-2,3, 0,0,0,3,1,1};
 	// std::vector<double> rhs {-2,5,3,0,1,1};
 	// std::vector<double> x(4);
 	// int n = 4;
 	// int m = 6;
 	// Square (row major)
	// std::vector<double> L {1,0,0,0, 2,1,0,0, 3,-1,2,0, 4,1,0,1};
	// std::vector<double> rhs {-2,5,3,0};
	// std::vector<double> x(4);
	// int n = 4;
	// int m = 4;
	// int is_col_major = 0;

	// std::cout << "L : \n";
	// print_mat(L,m,n,is_col_major);
	// std::cout << "rhs : \n\t";
	// for (int i=0; i<m; i++) {
	// 	std::cout << rhs[i] << ", ";
	// }
	// std::cout << "\n";

	// lower_tri_solve(&L[0],&rhs[0],&x[0],m,n,is_col_major);
	// std::cout << "x : \n\t";
	// for (int i=0; i<n; i++) {
	// 	std::cout << x[i] << ", ";
	// }
	// std::cout << "\n";

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


	// std::vector<double> R_row {-2.6557, -0.9431, -0.1570, 0, -1.3163, 1.2563, 0, 0, -0.1619};
	// std::vector<double> R_col { -2.6557, 0, 0, -0.9431, -1.3163, 0, -0.1570, 1.2563, -0.1619};
	// std::vector<double> b {1.8482, 3.6382, -2.2689};
	// std::vector<double> x_row(3,0);
	// std::vector<double> x_col(3,0);
	// int n=3;
	// int m=3;

	// upper_tri_solve(&R_row[0],&b[0],&x_row[0],m,n,0);
	// upper_tri_solve(&R_col[0],&b[0],&x_col[0],m,n,1);
	
	// std::cout << "x_row : \n\t";
	// for (int i=0; i<n; i++) {
	// 	std::cout << x_row[i] << ", ";
	// }
	// std::cout << std::endl;

	// std::cout << "x_col : \n\t";
	// for (int i=0; i<n; i++) {
	// 	std::cout << x_col[i] << ", ";
	// }
	// std::cout << std::endl;


/* FINAL CLS TEST */
	
	// Under determined 
	// Solution = [-0.382384, 0.382384, -0.490184, 0.705872, -0.147064]
	// std::vector<double> A {1,2,1, 0,2,1, 3,0,0, 1,1,0, -1,-2,0};
	// std::vector<double> b { -1, 1, 0};
	// std::vector<double> Ct {1,2,3,4,5};
	// std::vector<double> d {1};
	// int m = 3;
	// int n = 5;
	// int s = 1;

	// Exactly determined
	// Solution = [12, -12, -5.25, 4.5, 1.75]
	// std::vector<double> A {1,2,1, 0,2,1, 3,0,0, 1,1,0, -1,-2,0};
	// std::vector<double> b { -1, 1, 0};
	// std::vector<double> Ct {1,1,1,1,1, 1,2,3,4,5};
	// std::vector<double> d {1, -1};
	// int m = 3;
	// int n = 5;
	// int s = 2;

	// Over determined 
	// Solution = [0.596714, -0.12228, 0.3956, 1.54949, -1.56511]
	// std::vector<double> A {1,-1,2,-2,3, 1,3,1,-2,0, 0,1,0,4,1, 0,0,-1,0,3, 2,0,0,0,3};
	// std::vector<double> b { -2, -1, 0, 1, 2};
	// std::vector<double> Ct {-1,1,-1,1,-1};
	// std::vector<double> d {2};
	// int m = 5;
	// int n = 5;
	// int s = 1;

	// Over determined, singular A
	// Solution = [-1.61035, 0.0202868, 0.280168, 0.579224, -0.0703043]
	std::vector<double> A {1,1,0,0,0, 2,2,0,0,0, 0,1,0,4,1, 0,0,-1,0,3, 2,0,0,0,3};
	std::vector<double> b { -2, -1, 0, 1, 2};
	std::vector<double> Ct {-1,1,-1,1,-1};
	std::vector<double> d {2};
	int m = 5;
	int n = 5;
	int s = 1;



	std::vector<double> x = constrained_least_squares(A, b, Ct, d, m, n, s);

	std::cout << "x : \n\t";
	for (int i=0; i<n; i++) {
		std::cout << x[i] << ", ";
	}
	std::cout << std::endl;

}





