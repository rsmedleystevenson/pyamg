import numpy as np
from scipy.sparse import *
from pyamg import amg_core
import numbers
from copy import deepcopy



# Overloading example
# http://code.activestate.com/recipes/189971-basic-linear-algebra-matrix/

class GenTensor():
""" Class for general tensor-structure matrix, that is as a sum
	of tensor products:

		A = \sum_{i=1}^n \prod_{j=1}^{k_i} A_ij  


	"""

	def __init__(self, products):
	""" Class constructor.

		Parameters
		----------
			products : list
				List of Kronecker products over which to sum.



		"""
		try:
			assert not isinstance(products, basestring)
		except:
			raise TypeError("Must provide list of matrices to constructor.")
	
		self.num_mats = len(matrices)
		self.products = products
		self.num_rows = []
		self.num_cols = []

		# Get data type and matrix type of first element in list
		self.dtype = products[0].dtype
		if issparse(products[0]):
			self.issparse = True
		else:
			self.issparse = False

		# Ensure consistent storage types.
		for i in range(0,self.num_mats):
			if self.issparse:
				try:
					self.products.append(csr_matrix(products[i],dtype=self.dtype))
				except:
					raise TypeError("Need sparse products to be of Scipy CSR "
									"format, or convertible to CSR.")
			else:
				try:
					self.products.append(np.asarray(products[i],dtype=self.dtype))
				except:
					raise TypeError("Need dense arrays to be of Numpy array "
									"format, or convertible to Numpy array.")					

		self.set_shape()
		self.nnz = self.count_nonzero()










# TODO:
#	- Address multiplying by multiple vectors / dense 2d array
#	- Add __get_item__() to get (i,j) element of complete matrix
#	- Addition and subtraction, __add__(), __sub__() will return a
#	  general GenTensor() object when I make that class. 
#	- np.asarray() and csr_matrix() do not throw exceptions when I
#	  want them to. What if we want to have a dense vector kron
#	  with a sparse matrix though? Need to be stored in csr format
#	  for mult algorithm...
class KronProd():
"""


	Attributes
	----------
	matrices : list
		List of sparse CSR matrices defining this object.
	num_mats : int 
		Number of matrices.
	num_rows : list<int>
		Number of rows in product matrices.
	num_cols : list<int>
		Number of columns in product matrices.
	shape : [m, n]
		Shape of Kronecker product matrix. Total rows, m, equal
		to product of elements of num_rows, and similarly, total
		columns, n, equal to product of elements of num_cols.
	ndim : 2
		Currently only supports 2-dimensional arrays
	dtype : type
		Data type of object. All product matrices must have same
		type.
	nnz : int
		Number of nonzeros in global matrix.
	issparse : bool
		Boolean denoting if product matrices are sparse matrices.
		The alternative is a set of dense vectors. 

		TODO : for low-rank work, is it the tensor of a column
			vector with row-vector? If so, how do we account for
			this?

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
		self.matrices = []
		self.num_rows = []
		self.num_cols = []

		# Get data type and matrix type of first element in list
		self.dtype = matrices[0].dtype
		if issparse(matrices[0]):
			self.issparse = True
		else:
			self.issparse = False

		# Ensure consistent storage types.
		for i in range(0,self.num_mats):
			if self.issparse:
				try:
					self.matrices.append(csr_matrix(matrices[i],dtype=self.dtype))
				except:
					raise TypeError("Need sparse matrices to be of Scipy CSR "
									"format, or convertible to CSR.")
			else:
				try:
					self.matrices.append(np.asarray(matrices[i],dtype=self.dtype))
				except:
					raise TypeError("Need dense arrays to be of Numpy array "
									"format, or convertible to Numpy array.")					

		self.set_shape()
		self.nnz = self.count_nonzero()


	def set_shape(self):
	""" Get dimensions of each product matrix and total Kronecker product.
		"""
		self.num_rows = []
		self.num_cols = []
		for i in range(0,self.num_mats):
			self.num_rows.append(self.matrices[i].shape[0])
			self.num_cols.append(self.matrices[i].shape[1])

		self.shape = [np.prod(self.num_rows), np.prod(self.num_cols)]


	def count_nonzero(self):
	""" Count total number of nonzeros *in full matrix.* This is given
		as the product of number of nonzeros in all product matrices. 
		"""
		if self.issparse:
			nnzs = [self.matrices[i].nnz for i in range(0,self.num_mats)]
			return np.prod(nnzs)
		else:
			return np.prod(self.shape)


	def __mul__(self, x):
	""" Overloaded functions to multiply by a scalar, vector,
		or kronecker product on the right. 
		"""
		if isinstance(x, numbers.Number):
			return self.scalar_multiply(x, copy=True)
		elif isinstance(x, list):
			try:
				y = np.asarray(x)
			except:
				raise TypeError("If list-type passed in, must be "
								"convertible to numpy array.")
			if self.issparse:
				return self.sparse_mat_vec(y)
			else:
				return 0 # TODO : dense vector times vector kron prod
		elif isinstance(x, KronProd):
			return self.mat_mul(x, copy=True)
		else:
			raise TypeError("Cannot multiply by type "type(x))


	def __rmul__(self, x):
	""" Overloaded functions to multiply by a scalar, vector,
		or kronecker product on the left (where self is the
		operator on the right). 
		"""
		if isinstance(x, numbers.Number):
			return self.scalar_multiply(x, copy=True)
		elif isinstance(x, list):
			try:
				y = np.asarray(x)
			except:
				raise TypeError("If list-type passed in, must be "
								"convertible to numpy array.")
			if self.issparse:
				return self.sparse_vec_mat(y)
			else:
				return 0 # TODO : dense vector times vector kron prod

		elif isinstance(x, KronProd):
			return self.rmat_mul(x)
		else:
			raise TypeError("Cannot multiply by type "type(x))


	def __imul__(self, x):
	""" Overloaded functions to multiply by a scalar kronecker
		product in place. 
		"""
		if isinstance(x, numbers.Number):
			return self.scalar_multiply(x, copy=False)
		elif isinstance(x, KronProd):
			return self.mat_mul(x, copy=False)
		else:
			raise TypeError("Cannot multiply in place by type "type(x))

	def scalar_multiply(self, C, copy=True):
	""" Scalar multiplication of self by constant. Constant can
		be absorbed by any of product matrices, defaults to first
		in list self.matrices. 

		Parameters
		----------
			C : scalar
				Scale tensor product by C
			copy : bool
				If true, copies tensor object and underlying matrices,
				and scales by constant. If false, scales Kronecker
				product in place.
		"""
		if copy:
			temp = deepcopy(self.matrices)
			temp[0] *= C
			return KronProd(temp)
		else:
			self.matrices[0] *= C
			return self


	def sparse_mat_vec(self, x):
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

		# Check for compatible dimensions, set solution size
		sol_size = self.shape[0]
		if x.shape[0] != self.shape[1]:
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

		# Get initial matrix sizes (in columns) left and right of A_0
		n_left = 1
		n_right = np.prod(self.num_cols[1:self.num_mats])

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


	def sparse_vec_mat(self, x):
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

		# Check for compatible dimensions, set solution size
		sol_size = self.shape[1]
		if x.shape[0] != self.shape[0]:
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

		# Get initial matrix sizes (in rows) left and right of A_0
		n_left = 1
		n_right = np.prod(self.num_rows[1:self.num_mats])

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


	def transpose(self, copy=True):
	""" Transpose Kronecker product by transposing each product matrix.

		Parameters
		----------
			copy : bool
				If true, copies tensor object and underlying matrices,
				returns transpose. If false, takes transpose of each
				product matrix in place. 
		"""

		if copy:
			temp = [self.matrices[i].transpose(copy=True) for i in range(0,self.num_mats)]
			return KronProd(temp)
		else:
			for i in range(0,self.num_mats):
				self.matrices[i] = self.matrices[i].transpose(copy=False)
			return self


	def mat_mul(self, X, copy=True):
	""" Kronecker product multiplication, B = A*X, where this object
		is matrix A and X is passed in.

		"""
		if (self.num_mats != X.num_mats):
			raise ValueError("To take product of Kronecker products, each "
							 "must have same number of product matrices.")

		if copy:
			temp = []
			for i in range(0,self.num_mats):
				if (self.matrices[i].shape[1] != X.matrices[i].shape[0]):
					raise ValueError("Product matrix dimensions do not agree.")
				temp.append(self.matrices[i]*X.matrices[i])
			return KronProd(temp)
		else:
			for i in range(0,self.num_mats):
				if (self.matrices[i].shape[1] != X.matrices[i].shape[0]):
					raise ValueError("Product matrix dimensions do not agree.")
				self.matrices[i] *= X.matrices[i]
			self.issparse = self.issparse * X.issparse
			self.set_shape()
			self.nnz = self.count_nonzero()
			return self


	def rmat_mul(self, X):
	""" Kronecker product multiplication, B = X*A, where this object
		is matrix A and X is passed in. 

		Notes
		-----
			Do not need copy parameter in this function, because in-place
			multiplication will only happen with self as the left-most
			operator.
		"""
		if (self.num_mats != X.num_mats):
			raise ValueError("To take product of Kronecker products, each "
							 "must have same number of product matrices.")

		temp = []
		for i in range(0,self.num_mats):
			if (self.matrices[i].shape[0] != X.matrices[i].shape[1]):
				raise ValueError("Product matrix dimensions do not agree.")
			temp.append(X.matrices[i]*self.matrices[i])
		return KronProd(temp)



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

		self.n = np.prod(self.mat_sizes)


	def __mul__(self, x):
	""" Overloaded functions to multiply by a scalar, vector,
		or kronecker product on the right. 
		"""
		if isinstance(x, numbers.Number):
			return self.scalar_multiply(x)
		elif isinstance(x, list):
			try:
				y = np.array(x)
				return self.mult_vec(y, 0)
			except:
				raise TypeError("If list-type passed in, must be "
								"convertible to numpy array.")
		elif isinstance(x, KronProd):
			return self.mat_mul(x)
		else:
			raise TypeError("Cannot multiply by type "x.dtype)


	def __rmul__(self, x):
	""" Overloaded functions to multiply by a scalar, vector,
		or kronecker product on the left (where self is the
		operator on the right). 
		"""
		if isinstance(x, numbers.Number):
			return self.scalar_multiply(x)
		elif isinstance(x, list):
			try:
				y = np.array(x)
				return self.mult_vec(y, 0)
			except:
				raise TypeError("If list-type passed in, must be "
								"convertible to numpy array.")
		elif isinstance(x, KronProd):
			return self.rmat_mul(x)
		else:
			raise TypeError("Cannot multiply by type "x.dtype)


	def scalar_multiply(self, C):
	""" Scalar multiplication of self by constant.
		"""
		temp = deepcopy(self.matrices)
		for mat in temp:
			mat *= C
		return KronSum(temp)

	def mult_vec(self, x, left_mult):
		"""

		"""
		# Check for compatible dimensions
		if x.shape[0] != self.n:
			raise ValueError("Incompatible dimensions.")		

		# Function to compute product of matrix sizes for
		# A_{ind1}, ..., A_{ind2}
		def get_np.prod(ind1,ind2):
			if ind2 < ind1:
				return 1
			else:
				return np.prod(self.num_rows[ind1:ind2])

		# Starting with zero vector, add contribution from each term
		# in summation,
		#	A = A_1\otimes I_2 \otimes ... + I_1\otimes A_2 \otimes I_3 ...
		mult = amg_core.partial_kronsum_matvec
		y = np.zeros((self.n,))
		for i in range(0,num_mats):
			n_left = get_np.prod(0,i-1)
			n_right = get_np.prod(i+1,self.num_mats)
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


