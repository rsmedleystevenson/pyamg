import pdb
import numpy as np
from scipy.sparse import csr_matrix


# Form discretization of transport equation given by Jim Morel.
def lumped_transport(num_rows, num_cols, theta=np.pi/4.0, sigt=1.0, size_x=1.0, size_y=1.0, scale=True):

	# Constants used in discretization
	mu = np.cos(theta)
	eta = np.sin(theta)
	hx = size_x/num_cols
	hy = size_y/num_rows
	non_boundary_nodes = []
	scale_node = 1.0
	
	if type(sigt)!=float:
		max_sig = sigt.get_max()
	else:
		max_sig = sigt

	# Form arrays for csr matrix
	mat_size = 4*num_cols*num_rows
	num_nonzeros = 16*(num_cols-1)*(num_rows-1) + 14*(num_cols+num_rows-2) + 12
	row_ptr = np.empty([mat_size+1,],dtype=int)
	col_inds = np.empty([num_nonzeros,],dtype=int)
	data = np.empty([num_nonzeros,])

	# Get indices for interior vs. exterior of domain for sigma values
	in_row_lower = np.floor(num_rows/4)
	in_row_upper = 3*np.floor(num_rows/4)
	in_col_left  = np.floor(num_cols/4)
	in_col_right = 3*np.floor(num_cols/4)

	# Fill in sparse data structure 
	next = 0
	for i in range(0,num_rows):
		for j in range(0,num_cols):
			k = i*num_cols + j

			# Determine if node is in interior or exterior of domain for sigma value
			if type(sigt)!=float:
				mpx = (j+0.5)*hx
				mpy = (i+0.5)*hy
				this_sig = sigt.eval((mpx,mpy))
			else:
				this_sig = sigt

			# Scale node dependent on sigma value
			if scale == True:
				if this_sig >= (1.0/max_sig) :
					scale_node = 1.0 / np.sqrt(max_sig)
				else:
					scale_node = np.sqrt(max_sig)

			# Get list of non-boundary nodes to prune boundary nodes off later
			if i==0 and j==0:
				non_boundary_nodes.append(4*k+3)
			elif i==0:
				non_boundary_nodes.append(4*k+2)
				non_boundary_nodes.append(4*k+3)
			elif j==0:
				non_boundary_nodes.append(4*k)
				non_boundary_nodes.append(4*k+1)
			else:
				non_boundary_nodes.append(4*k)
				non_boundary_nodes.append(4*k+1)
				non_boundary_nodes.append(4*k+2)
				non_boundary_nodes.append(4*k+3)

			# Left bottom corner DOF
			if i==0 and j==0:
				row_ptr[0]     	 = 0
				row_ptr[1]     	 = 3
				col_inds[next]	 = 0
				data[next] 	 	 = scale_node*0.25*(mu*hy + eta*hx + this_sig*hx*hy)
				col_inds[next+1] = 1
				data[next+1] 	 = scale_node*0.25*mu*hy
				col_inds[next+2] = 2
				data[next+2] 	 = scale_node*0.25*eta*hx
				next += 3	
			elif i==0:
				row_ptr[4*k+1]   = row_ptr[4*k] + 4
				col_inds[next]	 = 4*(k-1)+1
				data[next] 		 = -scale_node*0.5*mu*hy
				col_inds[next+1] = 4*k
				data[next+1] 	 = scale_node*0.25*(mu*hy + eta*hx + this_sig*hx*hy)
				col_inds[next+2] = 4*k+1
				data[next+2] 	 = scale_node*0.25*mu*hy
				col_inds[next+3] = 4*k+2
				data[next+3] 	 = scale_node*0.25*eta*hx
				next += 4
			elif j==0:
				row_ptr[4*k+1]   = row_ptr[4*k] + 4
				col_inds[next] 	 = 4*(k-num_cols)+2
				data[next] 	 	 = -scale_node*0.5*eta*hx
				col_inds[next+1] = 4*k
				data[next+1] 	 = scale_node*0.25*(mu*hy + eta*hx + this_sig*hx*hy)
				col_inds[next+2] = 4*k+1
				data[next+2] 	 = scale_node*0.25*mu*hy
				col_inds[next+3] = 4*k+2
				data[next+3] 	 = scale_node*0.25*eta*hx
				next += 4
			else:
				row_ptr[4*k+1]   = row_ptr[4*k] + 5
				col_inds[next] 	 = 4*(k-num_cols)+2
				data[next] 	 	 = -scale_node*0.5*eta*hx
				col_inds[next+1] = 4*(k-1)+1
				data[next+1] 	 = -scale_node*0.5*mu*hy
				col_inds[next+2] = 4*k
				data[next+2] 	 = scale_node*0.25*(mu*hy + eta*hx + this_sig*hx*hy)
				col_inds[next+3] = 4*k+1
				data[next+3] 	 = scale_node*0.25*mu*hy
				col_inds[next+4] = 4*k+2
				data[next+4] 	 = scale_node*0.25*eta*hx
				next += 5

			# Right bottom corner DOF
			if i==0:
				row_ptr[4*k+2]   = row_ptr[4*k+1] + 3
				col_inds[next] 	 = 4*k
				data[next] 		 = -scale_node*0.25*mu*hy
				col_inds[next+1] = 4*k+1
				data[next+1] 	 = scale_node*0.25*(mu*hy + eta*hx + this_sig*hx*hy)
				col_inds[next+2] = 4*k+3
				data[next+2] 	 = scale_node*0.25*eta*hx
				next += 3
			else:
				row_ptr[4*k+2]   = row_ptr[4*k+1] + 4
				col_inds[next] 	 = 4*(k-num_cols)+3
				data[next] 	 	 = -scale_node*0.5*eta*hx
				col_inds[next+1] = 4*k
				data[next+1] 	 = -scale_node*0.25*mu*hy
				col_inds[next+2] = 4*k+1
				data[next+2] 	 = scale_node*0.25*(mu*hy + eta*hx + this_sig*hx*hy)
				col_inds[next+3] = 4*k+3
				data[next+3] 	 = scale_node*0.25*eta*hx
				next += 4

			# Top left corner DOF
			if j==0:
				row_ptr[4*k+3]   = row_ptr[4*k+2] + 3
				col_inds[next] 	 = 4*k
				data[next] 		 = -scale_node*0.25*eta*hx
				col_inds[next+1] = 4*k+2
				data[next+1] 	 = scale_node*0.25*(mu*hy + eta*hx + this_sig*hx*hy)
				col_inds[next+2] = 4*k+3
				data[next+2] 	 = scale_node*0.25*mu*hy
				next += 3
			else:
				row_ptr[4*k+3]   = row_ptr[4*k+2] + 4
				col_inds[next] 	 = 4*(k-1)+3
				data[next] 	 	 = -scale_node*0.5*mu*hy
				col_inds[next+1] = 4*k
				data[next+1] 	 = -scale_node*0.25*eta*hx
				col_inds[next+2] = 4*k+2
				data[next+2] 	 = scale_node*0.25*(mu*hy + eta*hx + this_sig*hx*hy)
				col_inds[next+3] = 4*k+3
				data[next+3] 	 = scale_node*0.25*mu*hy
				next += 4

			# Top right corner DOF
			row_ptr[4*k+4]   = row_ptr[4*k+3] + 3
			col_inds[next] 	 = 4*k+1
			data[next] 	 	 = -scale_node*0.25*eta*hx
			col_inds[next+1] = 4*k+2
			data[next+1] 	 = -scale_node*0.25*mu*hy
			col_inds[next+2] = 4*k+3
			data[next+2] 	 = scale_node*0.25*(mu*hy + eta*hx + this_sig*hx*hy)
			next += 3

	A = csr_matrix((data,col_inds,row_ptr),shape=[mat_size,mat_size])
	non_boundary_nodes.sort()
	A = A[non_boundary_nodes,:][:,non_boundary_nodes]
	return A
