import numpy as np
from scipy.sparse import *
from pyamg import amg_core

# TODO : Overload __add__ and __sub__, and maybe __mul__
#		 for other kronecker products or sums? 


class KronProd():
	"""


	Attributes
	----------
	matrices : list
		List of sparse CSR matrices defining this object.
	num_mats : int 
		Number of matrices.
	mat_sizes : list<int>
		Size of matrices.
	n : int
		Number of columns of Kronecker sum matrix, A. Equal to
		product of elements of num_cols.
	m : int
		Number of rows of Kronecker sum matrix, A. Equal to
		product of elements of num_rows

	Methods
	-------
	__mul__()
		Unassembled matrix-vector multiplication with global matix A.
	__rmul__()
		Unassembled vector-matrix multiplication with global matix A.


	"""
	def __init__(self, matrices):
		""" Class constructor.

		Parameters
		----------
			matrices : list
				List of sparse CSR matrices over which to define
				a Kronecker product, A = bigotimes_{i=1}^k A_i.

		"""
		try:
			assert not isinstance(matrices, basestring)
		except:
			raise TypeError("Must provide list of matrices to constructor.")
	
		self.num_mats = len(matrices)
		self.matrices = matrices
		self.num_rows = []
		self.num_cols = []
		for i in range(0,self.num_mats):
			self.num_rows.append(matrices[i].shape[0])
			self.num_cols.append(matrices[i].shape[1])

		self.m = prod(self.num_rows)
		self.n = prod(self.num_cols)


	def __mul__(self, x):
		""" Left multiplication, A * x.

		Parameters 
		----------
			x : 1d-array, size n, where A is m x n.

		Returns
		-------
			y : 1d-array y = A * x.

		References
		----------
			[1] Dayar, Tuǧrul, and M. Can Orhan. "On Vector-Kronecker
				Product Multiplication with Rectangular Factors." SIAM
				Journal on Scientific Computing 37.5 (2015): S526-S543.
		"""
		mult = amg_core.partial_kronprod_matvec

		# Function to compute product of number of columns for
		# A_{ind1}, ..., A_{ind2}
		def get_prod(ind1, ind2):
			if ind2 < ind1:
				return 1
			else:
				return prod(self.num_cols[ind1:ind2])

		# Check for compatible dimensions, set solution size
		sol_size = self.m
		if x.shape[0] != self.n:
			raise ValueError("Incompatible dimensions.")		

		# Construct temp vectors to be sufficiently large for
		# shuffle multiplication algorithm
		temp_size = 1
		for i in range(0,self.num_mats):
			temp_size *= max(num_rows[i], num_cols[i])

		y = np.zeros(temp_size,)
		q = np.zeros(temp_size,)

		# Set y[i] = x[i], for i=1,...,size(x)
		y[0:x.shape[0]] = x[:]

		# Get initial matrix sizes left and right of A_0
		n_left = 1
		n_right = get_prod(1,self.num_mats)

		# Add contribution from each term in product
		#	A = (A_1\otimes I_2 \otimes ...) * 
		#		(I_1\otimes A_2 \otimes I_3 ...) * ...
		for i in range(0,self.num_mats):
			mult(self.matrices[i].indptr,
				 self.matrices[i].indices,
				 self.matrices[i].data,
				 x,
				 y, 
				 num_rows[i],
				 num_cols[i],
				 n_left,
				 n_right,
				 left_mult)
			# Update size of matrices left and right of current index
			if i < (self.num_mats-1):
				n_left *= num_rows[i];
				n_right /= num_cols[i+1];

		return y[0:sol_size]


	def __rmul__(self, x):
		""" Right multiplication, x * A.

		Parameters 
		----------
			x : 1d-array, size m, where A is m x n.

		Returns
		-------
			y : 1d-array y = x * A.

		References
		----------
			[1] Dayar, Tuǧrul, and M. Can Orhan. "On Vector-Kronecker
				Product Multiplication with Rectangular Factors." SIAM
				Journal on Scientific Computing 37.5 (2015): S526-S543.
		"""
		mult = amg_core.partial_kronprod_vecmat

		# Function to compute product of number of columns for
		# A_{ind1}, ..., A_{ind2}
		def get_prod(ind1, ind2):
			if ind2 < ind1:
				return 1
			else:
				return prod(self.num_rows[ind1:ind2])

		# Check for compatible dimensions, set solution size
		sol_size = self.n
		if x.shape[0] != self.m:
			raise ValueError("Incompatible dimensions.")		

		# Construct temp vectors to be sufficiently large for
		# shuffle multiplication algorithm
		temp_size = 1
		for i in range(0,self.num_mats):
			temp_size *= max(num_rows[i], num_cols[i])

		y = np.zeros(temp_size,)
		q = np.zeros(temp_size,)

		# Set y[i] = x[i], for i=1,...,size(x)
		y[0:x.shape[0]] = x[:]

		# Get initial matrix sizes left and right of A_0
		n_left = 1
		n_right = get_prod(1,self.num_mats)

		# Add contribution from each term in product
		#	A = (A_1\otimes I_2 \otimes ...) * 
		#		(I_1\otimes A_2 \otimes I_3 ...) * ...
		for i in range(0,self.num_mats):
			mult(self.matrices[i].indptr,
				 self.matrices[i].indices,
				 self.matrices[i].data,
				 x,
				 y, 
				 num_rows[i],
				 num_cols[i],
				 n_left,
				 n_right,
				 left_mult)
			# Update size of matrices left and right of current index
			if i < (self.num_mats-1):
				n_left *= num_cols[i];
				n_right /= num_rows[i+1];

		return y[0:sol_size]





class KronSum():
	"""

	Attributes
	----------
	matrices : list
		List of sparse CSR matrices defining this object.
	num_mats : int 
		Number of matrices.
	mat_sizes : list
		Size of matrices (num rows = num cols).
	n : int
		Number of columns and rows of Kronecker sum matrix, A.
		Equal to product of elements of mat_sizes.

	Methods
	-------
	__mul__()
		Unassembled matrix-vector multiplication with global matix A.
	__rmul__()
		Unassembled vector-matrix multiplication with global matix A.

	Notes
	-----
		--- Only defined for square matrices ----

	"""

	def __init__(self, matrices):
		""" Class constructor.

		Parameters
		----------
			matrices : list
				List of sparse, square CSR matrices over which to define
				a Kronecker sum, A = bigoplus_{i=1}^k A_i.

		"""
		try:
			assert not isinstance(matrices, basestring)
		except:
			raise TypeError("Must provide list of matrices to constructor.")
	
		self.num_mats = len(matrices)
		self.matrices = matrices
		self.mat_sizes = []
		for i in range(0,self.num_mats):
			self.mat_sizes.append(matrices[i].shape[0])

		self.n = prod(self.mat_sizes)


	def __mul__(self, x):
		""" Left multiplication, A * x.
		"""
		return self.mult(x, 1)


	def __rmul__(self, x):
		""" Right multiplication, x * A.
		"""
		return self.mult(x, 0)


	def mult(self, x, left_mult):
		"""

		"""
		# Check for compatible dimensions
		if x.shape[0] != self.n:
			raise ValueError("Incompatible dimensions.")		

		# Function to compute product of matrix sizes for
		# A_{ind1}, ..., A_{ind2}
		def get_prod(ind1,ind2):
			if ind2 < ind1:
				return 1
			else:
				return prod(self.num_rows[ind1:ind2])

		# Starting with zero vector, add contribution from each term
		# in summation,
		#	A = A_1\otimes I_2 \otimes ... + I_1\otimes A_2 \otimes I_3 ...
		mult = amg_core.partial_kronsum_matvec
		y = np.zeros((self.n,))
		for i in range(0,num_mats):
			n_left = get_prod(0,i-1)
			n_right = get_prod(i+1,self.num_mats)
			mult(self.matrices[i].indptr,
				 self.matrices[i].indices,
				 self.matrices[i].data,
				 x,
				 y, 
				 n_left,
				 n_right,
				 num_rows[i],
				 left_mult)

		return y






dat1 = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
diags = np.array([0, -1, 2])
A1 = spdiags(dat1, diags, 4, 4, format='csr')
A2 = spdiags(dat1, diags, 5, 5, format='csr')

test = KronSum([A1,A2])

import pdb
pdb.set_trace()

v = np.random.rand(20,1)
v2 = test*v

pdb.set_trace()


