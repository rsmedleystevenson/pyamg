#ifndef KRYLOV_H
#define KRYLOV_H

#include "linalg.h"
#include <algorithm>
#include <iostream>
#include <vector>
#include <cmath>

/* Apply |start-stop| Householder reflectors in B to z
 *
 * Implements the below python
 *
 * for j in range(start,stop,step):
 *   z = z - 2.0*dot(conjugate(B[j,:]), v)*B[j,:]
 *
 * Parameters
 * ----------
 * z : {float array}
 *  length n vector to be operated on
 * B : {float array}
 *  n x m matrix of householder reflectors
 *  must be in row major form
 * n : {int}
 *  dimensionality of z
 * start, stop, step : {int}
 *  control the choice of vectors in B to use
 *
 * Returns
 * -------
 * z is modified in place to reflect the application of
 * the Householder reflectors, B[:,range(start,stop,step)]
 *
 * Notes
 * -----
 * Principle calling routine is gmres(...) and fgmres(...) in krylov.py
 */
template<class I, class T, class F>
void apply_householders(      T z[], const int z_size,
                        const T B[], const int B_size,
                        const I n,
                        const I start,
                        const I stop,
                        const I step)
{
    I index = start*n;
    I index_step = step*n;
    const T * Bptr;
    for(I i = start; i != stop; i+=step)
    {
        Bptr = &(B[index]);
        T alpha = dot_prod(Bptr, z, n);
        alpha *= -2;
        axpy(z, Bptr, alpha, n);
        index += index_step;
    }
}

/* For use after gmres is finished iterating and the least squares
 * solution has been found.  This routine maps the solution back to
 * the original space via the Householder reflectors.
 *
 * Apply |start-stop| Householder reflectors in B to z
 * while also adding in the appropriate value from y, so
 * that we follow the Horner-like scheme to map our least squares
 * solution in y back to the original space
 *
 * Implements the below python
 *
 * for j in range(inner,-1,-1):
 *  z[j] += y[j]
 *  # Apply j-th reflector, (I - 2.0*w_j*w_j.T)*update
 *  z = z - 2.0*dot(conjugate(B[j,:]), update)*B[j,:]
 *
 * Parameters
 * ----------
 * z : {float array}
 *  length n vector to be operated on
 * B : {float array}
 *  n x m matrix of householder reflectors
 *  must be in row major form
 * y : {float array}
 *  solution to the reduced system at the end of GMRES
 * n : {int}
 *  dimensionality of z
 * start, stop, step : {int}
 *  control the choice of vectors in B to use
 *
 * Returns
 * -------
 * z is modified in place to reflect the application of
 * the Householder reflectors, B[:,range(start,stop,step)],
 * and the inclusion of values in y.
 *
 * Notes
 * -----
 * Principle calling routine is gmres(...) and fgmres(...) in krylov.py
 *
 * See pages 164-167 in Saad, "Iterative Methods for Sparse Linear Systems"
 */
template<class I, class T, class F>
void householder_hornerscheme (      T z[], const int z_size,
                               const T B[], const int B_size,
                               const T y[], const int y_size,
                               const I n,
                               const I start,
                               const I stop,
                               const I step)
{
    I index = start*n;
    I index_step = step*n;
    const T * Bptr;
    for(I i = start; i != stop; i+=step)
    {
        z[i] += y[i];
        Bptr = &(B[index]);

        T alpha = dot_prod(Bptr, z, n);
        alpha *= -2;
        axpy(z, Bptr, alpha, n);

        index += index_step;
    }
}


/* Apply the first nrot Givens rotations in B to x
 *
 * Parameters
 * ----------
 * x : {float array}
 *  n-vector to be operated on
 * B : {float array}
 *  Each 4 entries represent a Givens rotation
 *  length nrot*4
 * n : {int}
 *  dimensionality of x
 * nrot : {int}
 *  number of rotations in B
 *
 * Returns
 * -------
 * x is modified in place to reflect the application of the nrot
 * rotations in B.  It is assumed that the first rotation operates on
 * degrees of freedom 0 and 1.  The second rotation operates on dof's 1 and 2,
 * and so on
 *
 * Notes
 * -----
 * Principle calling routine is gmres(...) and fgmres(...) in krylov.py
 */
template<class I, class T, class F>
void apply_givens(const T B[], const int B_size,
                        T x[], const int x_size,
                  const I n,
                  const I nrot)
{
    I ind1 = 0;
    I ind2 = 1;
    I ind3 = 2;
    I ind4 = 3;
    T x_temp;

    for(I rot=0; rot < nrot; rot++)
    {
        // Apply rotation
        x_temp = x[rot];
        x[rot]   = B[ind1]*x_temp + B[ind2]*x[rot+1];
        x[rot+1] = B[ind3]*x_temp + B[ind4]*x[rot+1];

        // Increment indices
        ind1 +=4;
        ind2 +=4;
        ind3 +=4;
        ind4 +=4;
    }
}


/* Get preconditioner for trace-minimization. Preconditioner
 * given by taking Haddarmard product with D, where 
 *
 *  D_{ij} = 1 / (tau*Aff_ii + K * (BcBc^T)_jj * Mff_ii),
 * 
 * for (i,j) in the fixed sparsity pattern over which we
 * are minimizing. 
 */
template<class I, class T>
void tracemin_preconditioner(const I S_rowptr[], const int S_rowptr_size,
                             const I S_colinds[], const int S_colinds_size,
                             T S_data[], const int S_data_size,
                             const T Aff_diag[], const int Aff_diag_size,
                             const T Bc[], const int Bc_size,
                             const T tau,
                             const T K, 
                             const I nc,
                             const I nb )
{
    // // Get (BcBc^T)_{jj} for j=0,...,nc-1
    std::vector<T> BcBct(nc,0);
    for (I j=0; j<nc; j++) {
        for (I k=0; k<nb; k++) {
            I temp = k*nc + j;
            BcBct[j] += Bc[temp]*Bc[temp];
        }
    }

    // Loop over sparsity pattern and form preconditioner
    for (I i=0; i<(S_rowptr_size-1); i++) {
        for (I k=S_rowptr[i]; k<S_rowptr[i+1]; k++) {
            I j = S_colinds[k];
            S_data[k] = 1.0 / (tau * Aff_diag[i] + K * BcBct[j]);
        }
    }
}

template<class I, class T>
void tracemin_preconditioner(const I S_rowptr[], const int S_rowptr_size,
                             const I S_colinds[], const int S_colinds_size,
                             T S_data[], const int S_data_size,
                             const T Aff_diag[], const int Aff_diag_size,
                             const T Mff_diag[], const int Mff_diag_size,
                             const T Bc[], const int Bc_size,
                             const T tau,
                             const T K, 
                             const I nc,
                             const I nb )
{
    // // Get (BcBc^T)_{jj} for j=0,...,nc-1
    std::vector<T> BcBct(nc,0);
    for (I j=0; j<nc; j++) {
        for (I k=0; k<nb; k++) {
            I temp = k*nc + j;
            BcBct[j] += Bc[temp]*Bc[temp];
        }
    }

    // Loop over sparsity pattern and form preconditioner
    for (I i=0; i<(S_rowptr_size-1); i++) {
        for (I k=S_rowptr[i]; k<S_rowptr[i+1]; k++) {
            I j = S_colinds[k];
            S_data[k] = 1.0 / (tau * Aff_diag[i] + K * BcBct[j] * Mff_diag[i]);
        }
    }
}


#endif
