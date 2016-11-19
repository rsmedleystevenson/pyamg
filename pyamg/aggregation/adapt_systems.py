# What about coarsening? This is key to the bad guys, because
# the bad guy should look locally smooth over the coarsening. 
# Doing independent coarsening in the development of targets
# may not be very effective if we do different coarsening
# after assembling the full hierarchy. 
# 	- Could coarsen (by block or not?) A first, then use
#	  that coarsening for all development of targets.
#	- Could build two V-cycles, one for the block-diagonal
#	  where we can coarsen each independently, and one (or
#	  more?) for the coupling terms.
#	- Is there any kind of theory we can do on if it makes
#	  sense to coarsen A by supernodes or independently?
#	- Is there any kind of theory to show that / when a
#	  scalar problem can be represented locally by one
#	  target vector? What if first-order scalar PDE, like
#	  in FOSLS? 
#	- What about a target for the coupling? If we have a
#	  2x2 block matrix, and one target for each diagonal
#	  block, is one target sufficient for the coupling?
#	  When is it necessary?
#	- What about higher order elements?
#	- 



def build_target(M, init_guess, B=None):

	if init_guess == 'rand':
		x0 = np.random.rand((M.shape[0],1))
	elif init_guess == 'ones':
		x0 = np.random.rand((M.shape[0],1))
	else:
		raise ValueError("Initial guess must be 'rand' or 'ones'.")




# Assume constant guess for blocks, random for off-diagonal
def adapt_systems(A, init_guess=None):

    if not isspmatrix_bsr(A):
		raise TypeError("Expected block BSR matrix.")

	# Get blocksize, convert A to csr to extract sub-matrices
	blocksize = A.blocksize[0]
	A = A.tocsr()
	n = A.shape[0]

	# If list of initial guess not provided, choose ones for
	# diagonal blocks and random for coupling
	if init_guess == None:
		init_guess = []
		for i in range(0,blocksize):
			init_guess.append('ones')
		for i in range(0,np.sum(np.arange(0,(blocksize+1)))):
			init_guess.append('rand')

	# Get number of adaptive cycles
	num_adapt = np.sum(np.arange(0,(blocksize+1)))
	B = np.zeros((n,num_adapt))

	# Get bad guys local to each block
	for i in range(0,blocksize):
		inds = np.arange(i, n, blocksize)
		M = A[inds,:][:,inds]
		B[inds,i] = build_target(M, init_guess[i])

	# Get coupled bad guys (not global, not relevant in 2d)


	# Get global bad guy
	B[:,-1] = build_target(A, init_guess[-1], B[0:(num_adapt-1)])












