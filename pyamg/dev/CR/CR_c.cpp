


// Note
template<class I, class T>
void cr_helper(const I A_rowptr, const int A_rowptr_size,
			   const I A_colinds, const int A_colinds_size, 
			   const T initial_target[], const int initial_target_size,
			   const T relaxed_vec[], const int relaxed_vec_size,
			   I indices[], const int indices_size,
			   I splitting[], const int splitting_size,
			   I gamma[], const int gamma_size,
			   const I n,
			   const T thetacs  )
{

	// Steps 3.1d, 3.1e in Falgout / Brannick (2010)
	// Compute candidate set measure, pick coarse grid candidates

//----> NEED TO UPDATE TO GENERAL INIDICES ARRAY PASSED IN

	vector<I> Uindex;
	for (I ind=0; ind<Findex_size; ind++) {
		I pt = Findex[ind];
		gamma[pt] = abs(    ) / // ----> FILL THIS IN
		if (gamma[pt] > thetacs) {
			Uindex.push_back(pt);
		}
	}
	I set_size = Uindex.size();

	// Step 3.1f in Falgout / Brannick (2010)
	// Find weights: omega_i = |N_i\C| + gamma_i
	vector<T> omega(n,0);
	for (I ind=0; ind<set_size; ind++) {
		I pt = Uindex[ind];
		I num_neighbors = 0
		I A_ind0 = A_rowptr[pt];
		I A_ind1 = A_rowptr[pt+1];
		for (I j=A_ind0; j<A_ind1; j++) {
			I neighbor = A_colinds[j];
			if (splitting[neighbor] == 0) {
				num_neighbors += 1;
			}
		}
		omega[pt] = num_neighbors + gamma[pt];
	}

	// Form maximum independent set
	while (true) {
		// 1. Add point i in U with maximal weight to C 
		T max_weight = 0;
		I new_pt = -1;
		for (I ind=0; ind<set_size; i++) {
			I pt = Uindex[ind];
			if (omega[pt] > max_weight) {
				max_weight = omega[pt];
				new_pt = pt;
			}
		}
		// If all points have zero weight (index set is empty) break loop
		if (new_pt < 0) {
			break;
		}
		splitting[new_pt] = 1;
		gamma[new_pt] = 0;

		// 2. Remove from candidate set all nodes connected to 
		// new C-point by marking weight zero.
		vector<I> neighbors;
		I A_ind0 = A_rowptr[new_pt];
		I A_ind1 = A_rowptr[new_pt+1];
		for (I j=A_ind0; j<A_ind1; j++) {
			I temp = A_colinds[j];
			neighbors.push_back(temp);
			omega[temp] = 0;
		}

		// 3. For each node removed in step 2, set the weight for 
		// each of its neighbors still in the candidate set +1.
		I num_neighbors = neighbors.size();
		for (I i=0; i<num_neighbors; i++) {
			I pt = neighbors[i];
			I A_ind0 = A_rowptr[pt];
			I A_ind1 = A_rowptr[pt+1];
			for (I j=A_ind0; j<A_ind1; j++) {
				I temp = A_colinds[j];
				if (omega[temp] != 0) {
					omega[temp] += 1;					
				}
			}
		}
	}

	// Reorder indices array, with the first element giving the number
	// of F indices, nf, followed by F indices in elements 1:nf, and 
	// C indices in (nf+1):n
	vector<I> Cpts;
	I &num_Fpts = indices[0];
	num_Fpts = 0;
	I next_ind = 1;
	for (I i=0; i<n; i++) {
		if (splitting[i] == 0) {
			indices[next_ind] = i;
			next_ind += 1;
			num_Fpts += 1;
		}
		else {
			Cpts.push_back(i);
		}
	}
	I num_Cpts = Cpts.size()
	for (I i=0; i<num_Cpts; i++) {
		indices[num_Fpts + i + 1] = Cpts[i];
	}

}















