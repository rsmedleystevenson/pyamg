


// Note
template<class I, class T>
void cr_helper(const I A_rowptr, const int A_rowptr_size,
			   const I A_colinds, const int A_colinds_size, 
			   const T target[], const int target_size,
			   T e[], const int e_size,
			   I indices[], const int indices_size,
			   I splitting[], const int splitting_size,
			   I gamma[], const int gamma_size,
			   const T thetacs  )
{
	I n = splitting_size;

	// Steps 3.1d, 3.1e in Falgout / Brannick (2010)
	// Divide each element in e by corresponding index in initial target vector.
	// Get inf norm of new e.
	T inf_norm = 0;
	for (I i=1; i<(num_Fpts+1); i++) {
		I pt = indices[i];
		e[pt] = abs(e[pt] / target[pt]);
		if (e[pt] > inf_norm) {
			inf_norm = e[pt];
		}	
	}

	// Compute candidate set measure, pick coarse grid candidates.
	I &num_Fpts = indices[0];
	vector<I> Uindex;
	for (I i=1; i<(num_Fpts+1); i++) {
		I pt = indices[i];
		gamma[pt] = e[pt] / inf_norm; 
		if (gamma[pt] > thetacs) {
			Uindex.push_back(pt);
		}
	}
	I set_size = Uindex.size();

	// Step 3.1f in Falgout / Brannick (2010)
	// Find weights: omega_i = |N_i\C| + gamma_i
	vector<T> omega(n,0);
	for (I i=0; i<set_size; i++) {
		I pt = Uindex[i];
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
		for (I i=0; i<set_size; i++) {
			I pt = Uindex[i];
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
		for (I i=A_ind0; i<A_ind1; i++) {
			I temp = A_colinds[i];
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
	// C indices in (nf+1):n. Note, C indices sorted largest to smallest.
	num_Fpts = 0;
	I next_Find = 1;
	I next_Cind = n;
	for (I i=0; i<n; i++) {
		if (splitting[i] == 0) {
			indices[next_Find] = i;
			next_Find += 1;
			num_Fpts += 1;
		}
		else {
			indices[next_Cind] = i;
			next_Cind -= 1;
		}
	}
}















