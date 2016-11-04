import numpy as np
from scipy.sparse import *
from pyamg import amg_core
import numbers
from copy import deepcopy



# Overloading example
# http://code.activestate.com/recipes/189971-basic-linear-algebra-matrix/

# TODO
#	- Add __add__(), __mul__(), ... functions.
class GenTensor:
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
		raise TypeError("Must provide list of tensor products to constructor.")

	self.num_prods = 0
	self.products = []

	# Get data type and matrix type of first element in list
	self.dtype = products[0].dtype
	self.shape = products[0].shape
	if products[0].issparse:
		self.issparse = True
	else:
		self.issparse = False

	for i in range(0,len(products)):
		if products[i].shape != self.shape:
			raise ValueError("All tensor pdocuts must have same dimension.")
		if products[i].dtype != self.dtype:
			raise TypeError("All tensor pdocuts must have same data type.")
		if isinstance(products[i],KronProd):
			self.products.append(products[i])
			self.num_prods += 1
		if isinstance(products[i],GenTensor):
			self.products.extend(products[i].products)
			self.num_prods += len(products[i].products)





# TODO:
#	- Address multiplying by multiple vectors / dense 2d array
#	- Add __get_item__() to get (i,j) element of complete matrix
class KronProd:
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

	Methods
	-------
	__mul__()
		Unassembled matrix-vector multiplication with global matix A.
	__rmul__()
		Unassembled vector-matrix multiplication with global matix A.

	Notes
	-----
	  - Right now a kronecker product must consist of only sparse or
		only dense operators. Of course theoretically this need not
		be the case, but the multiplication algorithms are designed
		that way. 
	  - For low-rank work, do you ever use the tensor of a column
	    vector with row-vector (i.e. a dense rank 2 matrix)? If so,
	    how should we account for this in the class?


	"""

	# Override numpy multiplication b = x*A
	__array_priority__ = 100


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
					self.matrices.append(matrices[i].tocsr())
					self.matrices[i].dtype = self.dtype
				except:
					raise TypeError("Need sparse matrices to be of Scipy CSR "
									"format, or convertible to CSR.")
			else:
				if type(matrices[i]) is 'numpy.ndarray':
					self.matrices.append(matrices[i])
					self.matrices[i].dtype = self.dtype
				else:
					try:
						self.matrices.append(np.array(matrices[i],dtype=self.dtype))
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


	def __add__(self, X):
		""" Add Kronecker product or general tensor to this Kronecker product. 

		Notes
		-----
			Does not accept Kronecker sum, as this will likely muck
			up the design of GenTensor class. 
		"""
		if (not isinstance(X,KronProd)) and (not isinstance(X,GenTensor)):
			raise TypeError("Can only add Kronecker product, or general"
							"tensor to Kronecker product object.")
		if X.shape != self.shape:
			raise ValueError("Cannot add operators of different dimension.")
		kron1 = deepcopy(X)
		kron2 = deepcopy(self)
		return GenTensor([kron1,kron2])


	def __sub__(self, X):
		""" Subtract Kronecker product or general tensor from this Kronecker product. 

		Notes
		-----
			Does not accept Kronecker sum, as this will likely muck
			up the design of GenTensor class. 
		"""
		if (not isinstance(X,KronProd)) and (not isinstance(X,GenTensor)):
			raise TypeError("Can only add Kronecker product, or general"
							"tensor to Kronecker product object.")
		if X.shape != self.shape:
			raise ValueError("Cannot add operators of different dimension.")
		kron1 = deepcopy(X)
		kron1 *= -1
		kron2 = deepcopy(self)
		return GenTensor([kron1,kron2])


	def __rsub__(self, X):
		""" Subtract this Kronecker product froom general tensor or Kronecker product. 

		Notes
		-----
			Does not accept Kronecker sum, as this will likely muck
			up the design of GenTensor class. 
		"""
		if (not isinstance(X,KronProd)) and (not isinstance(X,GenTensor)):
			raise TypeError("Can only add Kronecker product, or general"
							"tensor to Kronecker product object.")
		if X.shape != self.shape:
			raise ValueError("Cannot add operators of different dimension.")
		kron1 = deepcopy(X)
		kron2 = deepcopy(self)
		kron2 *= -1
		return GenTensor([kron1,kron2])


	def __mul__(self, x):
		""" Overloaded functions to multiply by a scalar, vector,
		or kronecker product on the right. 
		"""
		if isinstance(x, numbers.Number):
			return self.scalar_multiply(x, copy=True)
		elif isinstance(x, np.ndarray):
			if self.issparse:
				return self.sparse_mat_vec(x)
			else:
				return 0 # TODO : dense vector times vector kron prod		
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
			raise TypeError("Cannot multiply by type ",type(x))


	def __rmul__(self, x):
		""" Overloaded functions to multiply by a scalar, vector,
		or kronecker product on the left (where self is the
		operator on the right). 
		"""
		if isinstance(x, numbers.Number):
			return self.scalar_multiply(x, copy=True)
		elif isinstance(x, np.ndarray):
			if self.issparse:
				return self.sparse_vec_mat(x)
			else:
				return 0 # TODO : dense vector times vector kron prod
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
			raise TypeError("Cannot multiply by type ",type(x))


	def __imul__(self, x):
		""" Overloaded functions to multiply by a scalar kronecker
		product in place. 
		"""
		if isinstance(x, numbers.Number):
			return self.scalar_multiply(x, copy=False)
		elif isinstance(x, KronProd):
			return self.mat_mul(x, copy=False)
		else:
			raise TypeError("Cannot multiply in place by type ",type(x))


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
			[1] Dayar, Tugrul, and M. Can Orhan. "On Vector-Kronecker
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
			temp_size *= max(self.num_rows[i], self.num_cols[i])

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
				 self.num_rows[i],
				 self.num_cols[i],
				 n_left,
				 n_right)
			# Update size of matrices left and right of current index
			n_left *= self.num_rows[i];
			if i < (self.num_mats-1):
				n_right /= self.num_cols[i+1];

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
			[1] Dayar, Tugrul, and M. Can Orhan. "On Vector-Kronecker
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
			temp_size *= max(self.num_rows[i], self.num_cols[i])

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
				 self.num_rows[i],
				 self.num_cols[i],
				 n_left,
				 n_right)
			# Update size of matrices left and right of current index
			n_left *= self.num_cols[i];
			if i < (self.num_mats-1):
				n_right /= self.num_rows[i+1];

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



# TODO:
#	- count / bound nnz in kron sum (assume nonzero diagonal?)
# 	- What if multiply by dense array, e.g. 3 columns vectors?
#	  Should be able to handle this, need to adjust code.
class KronSum:
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
	shape : [n, n]
		Shape of Kronecker product matrix. Total rows, m, equal
		to product of elements of num_rows, and similarly, total
		columns, n, equal to product of elements of num_cols.
	ndim : 2
		Currently only supports 2-dimensional arrays
	dtype : type
		Data type of object. All product matrices must have same type.
	nnz : int
		Number of nonzeros in global matrix.
	issparse : bool
		Boolean denoting if product matrices are sparse matrices.
		The alternative is a set of dense vectors. 

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

	# Override numpy multiplication b = x*A
	__array_priority__ = 100


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
		self.matrices = []
		self.num_rows = []
		self.num_cols = []

		# Get data type and matrix type of first element in list
		self.dtype = matrices[0].dtype
		self.issparse = True

		# Ensure consistent storage types.
		for i in range(0,self.num_mats):
			try:
				self.matrices.append(matrices[i].tocsr())
				self.matrices[i].dtype = self.dtype
			except:
				raise TypeError("Need sparse matrices to be of Scipy CSR "
								"format, or convertible to CSR.")

		self.set_shape()
		self.nnz = self.count_nonzero()


	def set_shape(self):
		""" Get dimensions of each product matrix and total Kronecker product.
		Check that all matrices are square. 
		"""
		self.num_rows = []
		self.num_cols = []
		for i in range(0,self.num_mats):
			if (self.matrices[i].shape[0] != self.matrices[i].shape[1]):
				raise ValueError("Kronecker sum only defined for square matrices.")
			self.num_rows.append(self.matrices[i].shape[0])
			self.num_cols.append(self.matrices[i].shape[1])
		self.shape = [np.prod(self.num_rows), np.prod(self.num_cols)]


	def count_nonzero(self):
		""" Count total number of nonzeros *in full matrix.* 

		"""
		return 0


	def check_dims(self, X):
		""" Check if dimensions of matrices in self and X are compatible.
		"""
		if X.shape != self.shape:
			raise ValueError("Operators must have same dimensions.")
		if X.num_mats != self.num_mats:
			raise ValueError("Kronecker sums must consist of same number "
							 "of matrices.")
		for i in range(0,self.num_mats):
			if (self.matrices[i].shape != X.matrices[i].shape):
				raise ValueError("All sum-matrices must have equal dimension.")


	def __add__(self, X):
		""" Build Kronecker sum as sum of X and this Kronecker sum 

		Notes
		-----
			Does not accept Kronecker product or General Tensors. 
		"""
		if (not isinstance(X,KronSum)):
			raise TypeError("Can only add Kronecker sum to Kronecker sum.")
		self.check_dims(X)
		temp = []
		for i in range(0,self.num_mats):
			temp.append(self.matrices[i] + X.matrices[i])
		return KronSum(temp)


	def __iadd__(self, X):
		""" Add Kronecker sum to this Kronecker sum. 

		Notes
		-----
			Does not accept Kronecker product or General Tensors. 
		"""
		if (not isinstance(X,KronSum)):
			raise TypeError("Can only add Kronecker sum to Kronecker sum.")
		self.check_dims(X)
		for i in range(0,self.num_mats):
			self.matrices[i] += X.matrices[i]
		self.nnz = self.count_nonzero()
		return self


	def __sub__(self, X):
		""" Build Kronecker sum as difference of this Kronecker sum and X.

		Notes
		-----
			Does not accept Kronecker product or General Tensors. 
		"""
		if (not isinstance(X,KronSum)):
			raise TypeError("Can only subtract Kronecker sum from Kronecker sum.")
		self.check_dims(X)
		temp = []
		for i in range(0,self.num_mats):
			temp.append(self.matrices[i] - X.matrices[i])
		return KronSum(temp)


	def __rsub__(self, X):
		""" Build Kronecker sum as difference of X and this Kronecker sum.

		Notes
		-----
			Does not accept Kronecker product or General Tensors. 
		"""
		if (not isinstance(X,KronSum)):
			raise TypeError("Can only subtract Kronecker sum from Kronecker sum.")
		self.check_dims(X)
		temp = []
		for i in range(0,self.num_mats):
			temp.append(X.matrices[i] - self.matrices[i])
		return KronSum(temp)


	def __isub__(self, X):
		""" Subtract Kronecker sum from this Kronecker sum. 

		Notes
		-----
			Does not accept Kronecker product or General Tensors. 
		"""
		if (not isinstance(X,KronSum)):
			raise TypeError("Can only subtract Kronecker sum from Kronecker sum.")
		self.check_dims(X)
		for i in range(0,self.num_mats):
			self.matrices[i] -= X.matrices[i]
		self.nnz = self.count_nonzero()
		return self


	def __mul__(self, x):
		""" Overloaded functions to multiply by a scalar or vector
		on the right. 
		"""
		if isinstance(x, numbers.Number):
			return self.scalar_multiply(x, copy=True)
		elif isinstance(x, np.ndarray):
			return self.mult_vec(x, 0)
		elif isinstance(x, list):
			try:
				y = np.asarray(x, dtype=self.dtype)
				return self.mult_vec(y, 0)
			except:
				raise TypeError("If list-type passed in, must be "
								"convertible to numpy array.")
		else:
			raise TypeError("Cannot multiply by type ",type(x))


	def __rmul__(self, x):
		""" Overloaded functions to multiply by a scalar or vector
		on the left. 
		"""
		if isinstance(x, numbers.Number):
			return self.scalar_multiply(x, copy=True)
		elif isinstance(x, np.ndarray):
			return self.mult_vec(x, 1)
		elif isinstance(x, list):
			try:
				y = np.asarray(x, dtype=self.dtype)
				return self.mult_vec(y, 1)
			except:
				raise TypeError("If list-type passed in, must be "
								"convertible to numpy array.")
		else:
			raise TypeError("Cannot multiply by type ",type(x))


	def __imul__(self, x):
		""" Overloaded functions to multiply by scalar in place.
		"""
		if isinstance(x, numbers.Number):
			return self.scalar_multiply(x, copy=False)
		else:
			raise TypeError("Cannot multiply in place by type ",type(x))


	def scalar_multiply(self, C, copy=True):
		""" Scalar multiplication of self by constant.
		"""
		if copy:
			temp = deepcopy(self.matrices)
			for mat in temp:
				mat *= C
			return KronSum(temp)
		else:
			for mat in self.matrices:
				mat *= C
			return self


	def mult_vec(self, x, right_mult):
		""" Multiply Kronecker sum by vector.

		Parameters
		----------
			x : array-like
				1d vector to multiply by.
			right_mult : bool
				Boolean on left multiplication. If right_mult = 1,
				return x*A, otherwise return A*x.

		Returns
		-------
			y : numpy array
				Output of multiplication as dense numpy array.

		"""
		# Check for compatible dimensions
		if x.shape[0] != self.shape[0]:
			raise ValueError("Incompatible dimensions.")		
		if (right_mult != 0) and (right_mult != 1):
			raise ValueError("Parameter 'right_mult' must be 0 or 1.")

		# Function to compute product of matrix sizes for
		# A_{ind1}, ..., A_{ind2}
		def get_prod(ind1,ind2):
			if ind2 < ind1:
				return 1
			else:
				return int(np.prod(self.num_rows[ind1:ind2]))

		# Starting with zero vector, add contribution from each term
		# in summation,
		#	A = A_1\otimes I_2 \otimes ... + I_1\otimes A_2 \otimes I_3 ...
		mult = amg_core.partial_kronsum_matvec
		y = np.zeros((self.shape[0],),dtype=self.dtype)
		for i in range(0,self.num_mats):
			n_left = get_prod(0,i)
			n_right = get_prod(i+1,self.num_mats)
			mult(self.matrices[i].indptr,
				 self.matrices[i].indices,
				 self.matrices[i].data,
				 x,
				 y,
				 self.num_rows[i],
				 self.num_cols[i],
				 n_left,
				 n_right,
				 right_mult)

		return y


	def transpose(self, copy=True):
		""" Transpose Kronecker sum by transposing each sum matrix.

		Parameters
		----------
			copy : bool
				If true, copies object and underlying matrices, returns
				transpose. If false, takes transpose of each sum matrix
				in place. 
		"""

		if copy:
			temp = [self.matrices[i].transpose(copy=True) for i in range(0,self.num_mats)]
			return KronProd(temp)
		else:
			for i in range(0,self.num_mats):
				self.matrices[i] = self.matrices[i].transpose(copy=False)
			return self

import pdb


# Gets kronecker product from scipy sparse library
def get_kron_prod(matrices):
	n = len(matrices)
	if n == 1:
		return matrices[0]
	A = kron(matrices[0],matrices[1])
	for i in range(2,n):
		A = kron(A,matrices[i])
	A = csr_matrix(A)
	A.eliminate_zeros()
	return A

# Gets kronecker sum from scipy sparse library
# 	- assumes square matrices
def get_kron_sum(matrices):
	n = len(matrices)
	if n==1:
		return matrices[0]
	sizes = []
	# Get size of matrices
	for i in range(0,n):
		if matrices[i].shape[0] != matrices[i].shape[1]:
			raise ValueError("Kronecker sum only defined for square matrices.")
		sizes.append(matrices[i].shape[0])
	pre_I = [-1]
	post_I = []
	# Get size of identity before and after ith matrix
	for i in range(1,n):
		pre_I.append(int(np.prod(sizes[0:i])))
		post_I.append(int(np.prod(sizes[i:])))
	post_I.append(-1)
	
	# Form Kronecker sum
 	A = kron(matrices[0], eye(post_I[0]))
 	for i in range(1,n-1):
 		A = A + kron( eye(pre_I[i]), kron(matrices[i], eye(post_I[i])) )
 	A = A + kron(eye(pre_I[n-1]), matrices[n-1])
	A = csr_matrix(A)
	A.eliminate_zeros()
	return A


# Notes
#	- TODO : KronProd multiplication overwrites vector. Do we want this??
#	- TODO : Rectangular Kron prod mat-vecs/vec-mats aren't working

A1 = rand(m=5, n=5, density=0.25, format='csr')
A2 = rand(m=4, n=4, density=0.25, format='csr')
A3 = rand(m=8, n=8, density=0.25, format='csr')
A4 = rand(m=10, n=10, density=0.25, format='csr')

# Explicit kronecker sums and products built with Scipy
sp_sum2 = get_kron_sum([A1,A2])
sp_sum3 = get_kron_sum([A1,A2,A3])
sp_sum4 = get_kron_sum([A1,A2,A3,A4])

sp_prod2 = get_kron_prod([A1,A2])
sp_prod3 = get_kron_prod([A1,A2,A3])
sp_prod4 = get_kron_prod([A1,A2,A3,A4])

# Class Kronecker sum and product objects
ksum2 = KronSum([A1,A2])
ksum3 = KronSum([A1,A2,A3])
ksum4 = KronSum([A1,A2,A3,A4])

kprod2 = KronProd([A1,A2])
kprod3 = KronProd([A1,A2,A3])
kprod4 = KronProd([A1,A2,A3,A4])

n2 = ksum2.shape[0]
n3 = ksum3.shape[0]
n4 = ksum4.shape[0]

pdb.set_trace()

# Kron Prod works for two square matrices of different sizes
v2 = np.array([i for i in range(0,n2)],dtype='float64')
spl2 = sp_prod2*v2
spr2 = v2*sp_prod2
kl2 = kprod2*v2
v2 = np.array([i for i in range(0,n2)],dtype='float64')
kr2 = v2*kprod2
if np.max(np.abs(kl2-spl2)) > 1e-10:
	raise ValueError("Mat-vec, A*x, for kronecker product of "
					 "two square matrices was incorrect.")
if np.max(np.abs(kr2-spr2)) > 1e-10:
	raise ValueError("Vec-mat, x*A, for kronecker product of "
					 "two square matrices was incorrect.")

# Kron Prod works for three square matrices of different sizes
v3 = np.array([i for i in range(0,n3)],dtype='float64')
spl3 = sp_prod3*v3
spr3 = v3*sp_prod3
kl3 = kprod3*v3
v3 = np.array([i for i in range(0,n3)],dtype='float64')
kr3 = v3*kprod3
if np.max(np.abs(kl3-spl3)) > 1e-10:
	raise ValueError("Mat-vec, A*x, for kronecker product of "
					 "three square matrices was incorrect.")
if np.max(np.abs(kr3-spr3)) > 1e-10:
	raise ValueError("Vec-mat, x*A, for kronecker product of "
					 "three square matrices was incorrect.")

# Kron Prod works for four square matrices of different sizes
v4 = np.array([i for i in range(0,n4)],dtype='float64')
spl4 = sp_prod4*v4
spr4 = v4*sp_prod4
kl4 = kprod4*v4
v4 = np.array([i for i in range(0,n4)],dtype='float64')
kr4 = v4*kprod4
if np.max(np.abs(kl4-spl4)) > 1e-10:
	raise ValueError("Mat-vec, A*x, for kronecker product of "
					 "four square matrices was incorrect.")
if np.max(np.abs(kr4-spr4)) > 1e-10:
	raise ValueError("Vec-mat, x*A, for kronecker product of "
					 "four square matrices was incorrect.")

# Kron sum on two square matrices of different sizes
v2 = np.array([i for i in range(0,n2)],dtype='float64')
spl2 = sp_sum2*v2
spr2 = v2*sp_sum2
kl2 = ksum2*v2
v2 = np.array([i for i in range(0,n2)],dtype='float64')
kr2 = v2*ksum2
if np.max(np.abs(kl2-spl2)) > 1e-10:
	raise ValueError("Mat-vec, A*x, for kronecker sum of "
					 "two square matrices was incorrect.")
if np.max(np.abs(kr2-spr2)) > 1e-10:
	raise ValueError("Vec-mat, x*A, for kronecker sum of "
					 "two square matrices was incorrect.")

# Kron sum on three square matrices of different sizes
v3 = np.array([i for i in range(0,n3)],dtype='float64')
spl3 = sp_sum3*v3
spr3 = v3*sp_sum3
kl3 = ksum3*v3
v3 = np.array([i for i in range(0,n3)],dtype='float64')
kr3 = v3*ksum3
if np.max(np.abs(kl3-spl3)) > 1e-10:
	raise ValueError("Mat-vec, A*x, for kronecker sum of "
					 "three square matrices was incorrect.")
if np.max(np.abs(kr3-spr3)) > 1e-10:
	raise ValueError("Vec-mat, x*A, for kronecker sum of "
					 "three square matrices was incorrect.")

# Kron sum on four square matrices of different sizes
v4 = np.array([i for i in range(0,n4)],dtype='float64')
spl4 = sp_sum4*v4
spr4 = v4*sp_sum4
kl4 = ksum4*v4
v4 = np.array([i for i in range(0,n4)],dtype='float64')
kr4 = v4*ksum4
if np.max(np.abs(kl4-spl4)) > 1e-10:
	raise ValueError("Mat-vec, A*x, for kronecker sum of "
					 "four square matrices was incorrect.")
if np.max(np.abs(kr4-spr4)) > 1e-10:
	raise ValueError("Vec-mat, x*A, for kronecker sum of "
					 "four square matrices was incorrect.")

pdb.set_trace()

# ----------------------- #
# Rectangular matrices ---> THESE AREN'T WORKING
# ----------------------- #

A5 = rand(m=3, n=5, density=0.25, format='csr')
A6 = rand(m=4, n=8, density=0.25, format='csr')
A7 = rand(m=8, n=4, density=0.25, format='csr')
A8 = rand(m=5, n=3, density=0.25, format='csr')

# Explicit kronecker sums and products built with Scipy
sp_prod6 = get_kron_prod([A5,A3,A6])
sp_prod5 = get_kron_prod([A7,A8])
sp_prod7 = get_kron_prod([A5,A7,A6,A8])

# Class Kronecker sum and product objects
kprod6 = KronProd([A5,A3,A6])
kprod5 = KronProd([A7,A8])
kprod7 = KronProd([A5,A7,A6,A8])

m5 = kprod5.shape[0]
m6 = kprod6.shape[0]
m7 = kprod7.shape[0]
n5 = kprod5.shape[1]
n6 = kprod6.shape[1]
n7 = kprod7.shape[1]

# Kron Prod works for two square matrices of different sizes
vl5 = np.array([i for i in range(0,n5)],dtype='float64')
vr5 = np.array([i for i in range(0,m5)],dtype='float64')
spl5 = sp_prod5*vl5
spr5 = vr5*sp_prod5
kl5 = kprod5*vl5
kr5 = vr5*kprod5
pdb.set_trace()
if np.max(np.abs(kl5-spl5)) > 1e-10:
	raise ValueError("Mat-vec, A*x, for kronecker product of "
					 "two square matrices was incorrect.")
if np.max(np.abs(kr5-spr5)) > 1e-10:
	raise ValueError("Vec-mat, x*A, for kronecker product of "
					 "two square matrices was incorrect.")




pdb.set_trace()







