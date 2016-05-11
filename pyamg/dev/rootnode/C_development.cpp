#include <vector>

 /*  Parameters
 *      num_rows   - number of rows in A
 *      num_cols   - number of columns in A
 *      K1         - BSR row blocksize
 *      K2         - BSR column blocksize
 *      Ap[]       - BSR row pointer
 *      Aj[]       - BSR index array
 *      Ax[]       - BSR data array
 *      B[]        - fine-level near-nullspace candidates (n_row x K2)
 *      R[]        - coarse-level near-nullspace candidates (n_coarse x K2)
 *      tol        - tolerance used to drop numerically linearly dependent vectors
 *
 *  Returns:
 *      Nothing, Ax and R will be modified in places.

 -----------------------
 What I need passed in -
   - sparse matrix K
   - sparse matrix G
   - sparse matrix A
   - sparse aggregation matrix AggOp --> format as CSC??
   - Cpts (we know this from the aggregation routine)
   - size of AggOp --> numPoints x numCpts

   - sparsity template for P, to be filled in in place.
     --> Pass in template as floats/doubles with one in all entries
         then the identity for C points is already set. 
 */
template <class I, class S, class T, class DOT, class NORM>
         // NEED TO FIGURE OUT WHAT THIS TEMPLATE CLASS IS

void isolated_interpolation(const I n_row,
                            const I n_col,
                            const I   K1,
                            const I   K2,
                            const I Ap[], 
                            const I Ai[],
                                  T Ax[],
                            const T  B[],
                                  T  R[],
                            const S  tol,
                            const DOT& dot,
                            const NORM& norm)
{

    // Get set of F-points 
    // vector<int> Fpts(numPts-numCpts);
    // I nextInd = 0,
    //   nextCind = 0,
    //   nextCpt = Cpts[0];
    // for (I j=0; j<numPts; j++) {
    //     if ( j != nextCpt ) {
    //         Fpts[nextInd] = j;
    //         nextInd += 1;
    //     }
    //     else {
    //         nextCind += 1;
    //         nextCpt = Cpts[nextCind];
    //     }
    // }

    // Number of F-points in each aggregate. 
    vector<int> aggregateSizes(numCpts);

    for (I j=0; j<numPts; j++) {

        agg = aggOp_col[j];
        if 
        aggregateSizes
    }

    for (I j=0; j<numAggregates; j++) {

      F sum = 0.0;
    }

    

    // Compute (K + G*Y)Afc. Note that K,G,Y are symmetric, but K + G*Y is not.

}



/* ----------------------------------------------------------------------------- */
/* ----------------------------------------------------------------------------- */
/* Input:
 *      - Sparse matrices Afc, Acf, K, G, bad_guy? (in csr format?)
 *      - Dense arrays Bf, Bc, passed in in column major form
 *      - Sparsity structure for P --> maybe overwrite this with P for return?
 *      - Weighting parameter
 *      - Num points, maybe Fpoints vs. Cpoints?
 * This routine is incomplete, but designed to form P =[ (I+Y)A_{fc}, I ]^T. 
 * Implementation was stopped at forming Y in C++ and P in Python.  
 */

// Need to add input names / output names to amg_core.i
// Need to put function name in amg_core.i

void new_ideal_interpolation(    )
{


// TO DO
//      - Preallocate array to solve SVD systems. Need to make it big enough to
//        store the SVD system for any row... Preallocate singular value array too.
//      - Once this function is finished, make a separate function or case for k=1 (cheaper!)? 
//      - Run test comparing methods of adding sparse row of K + y_r...
//		- Make sure all memory is cleaned up

    // SOULD BE ABLE TO PREALLOCATE RHS AND LEASTSQUARESOP HERE TOO. 
    // --> TEST FIRST WITHOUT PREALLOCATION, THEN CHANGE
    I svdWorkSize = (2.0*maxCols_Row + 1.0) * maxCols_Row;
    T *svdWorkSpace = new T[svdWorkSize]; 
    F *svdSingVals = new F[maxCols_Row]; 

    const I numFpts, numCpts

    const I numBadGuys = 0;
    I nextCind = 0,
      nextCpt = Cpts[0];

    for (I row=0; row<numPts; row++) {

        // Make rows corresponding to coarse points an identity vector
        if (row == nextCpt) {
            // Make row identity in sparse data structure...


            nextCind += 1;
            nextCpt = Cpts[nextCind];
        }

        else {
        // -------------------------------------------------------------------------------------- //
        // -------------------------------------- VERIFIED -------------------------------------- //
        // -------------------------------------------------------------------------------------- //

            // Get sparsity indices for this row - NOTE, WE ASSUME SPARSITY STRUCTURE TO BE SYMMETRIC
// --> For hyperbolic type problems does a symmetric sparsity pattern/SOC matrix make sense though? 
            const I sparseThisRow = sparsityRowPtr[row];
            const I numInd = sparsityRowPtr[row+1] - sparseThisRow;
            std::vector<I> colInd(numInd);
            for (I k=0; k<numInd; k++) {
                colInd[k] = sparsityColInds[sparseThisRow+k]
            }
                // ---> CAN I USE THIS SECTION TO FORM SPARSITY PATTERN FOR THIS ROW OF P?
                //      THEN FILL IN VALUES AFTER I COMPUTE SVD?


        // -------------------------------------------------------------------------------------- //
        // -------------------------------------------------------------------------------------- //
         
            // Form least squares operator in column major form. 
            I opInd = 0;
            I leastSquaresOp[numInd * (numInd+numBadGuys)];
            for (I opCol=0; opCol<numInd; opCol++) {

                I ind1 = AfcRowPtr[opCol];
                I size1 = AfcRowPtr[opCol+1] - ind1;
                I *row1Inds = &AfcColInds[ind1];
                T *row1Vals = &AfcValues[ind1];

                // Use symmetry of AfcAcf to avoid repeating computations.
                for (I opRow=0; opRow<opCol; opRow++) {
                    leastSquaresOp[opInd] = leastSquaresOp[opCol*(numInd+numBadGuys) + opRow];
                    opInd += 1;
                }
                // Get dot product of current row with itself for diagonal entry.
                leastSquaresOp[opInd] = sparse_dot(size1, row1Inds, row1Vals,
                                                   size1, row1Inds, row1Vals);
                opInd += 1;
                // Calculate dot product of remaining elements of AfcAcf in this row.
                for (I opRow=(opCol+1); opRow<numInd; opRow++) {
                    I ind2 = AfcRowPtr[opRow];
                    I size2 = AfcRowPtr[opRow+1] - ind2;
                    I *row2Inds = &AfcColInds[ind2];
                    T *row2Vals = &AfcValues[ind2];  
                    leastSquaresOp[opInd] = sparse_dot(size1, row1Inds, row1Vals,
                                                       size2, row2Inds, row2Vals);
                    opInd += 1;
                }
                // Dot product of ith coarse bad guy w/ jth (current iterate) column of 
                // Acf for i=1,...,numBadGuys. 
                for (I opRow=numInd; opRow<(numInd+numBadGuys); opRow++) {
                    T *badGuyVec = &Bc[opRow*numCpts];
                    leastSquaresOp[opInd] = weighting * sparse_dense_dot(size1, row1Inds,
                                                            row1Vals, numCpts, badGuyVec);
                    opInd += 1;
                }
            }

        // -------------------------------------------------------------------------------------- //
        // -------------------------------------------------------------------------------------- //
            
            // Form right hand side of minimization. Size = |row_ind| + num bad guys

            // -------------------- VERIFIED -------------------- //
            T rightHandSide[numInd+numBadGuys];
            I GThisRow = GRowPtr[row];
            I GnumInds = GRowPtr[row+1] - GThisRow,
              lowerInd = 0;
            // Find intersection between desired sparsity pattern for this row iterate and 
            // sparsity pattern of current row/column of G.
            for (I k=0; k<numInd; k++) {
                for (I compInd=lowerInd; compInd<GnumInds; compInd++) {
                    // Find overlapping sparsity element, set right hand side accordingly
                    if ( colInd[k] == GColInds[GThisRow+compInd] ) {
                        rightHandSide[k] = GValues[GThisRow+compInd];
                        lowerInd = compInd+1;
                        break;
                    }
                    // Sparsity element does not overlap, set right hand side to zero
                    else if ( colInd[k] < GColInds[GThisRow+compInd] ) {
                        rightHandSide[k] = 0;
                        break;
                    }
                }
            }

            // -------------------- VERIFIED -------------------- //
            // To compute Acf*K*e_r, we let k_r = K*e_r be the rth column of K. We then 
            // expance Acf*k_r in terms of columns of Acf, which are rows of Afc.
            T vec_AfcKe[numCpts] = {0};
            I KThisRow = KRowPtr[row];
            I KUpperInd = KRowPtr[row+1];
            // Only loop over nonzero row indices in which k_r(i) != 0.
            for (I k=KThisRow; k<KUpperInd; k++) {
                I tempInd = KColInds[k];
                T constMult = KValues[k];
                I tempLowerInd = AfcRowPtr[tempInd];
                I tempUpperInd = AfcRowPtr[tempInd+1];
                for (I j=tempLowerInd; j<tempUpperInd; j++) {
                    I tempAddInd = AfcColInds[j]
                    vec_AfcKe[tempAddInd] += constMult * AfcValues[j];
                }
            }
          
            // -------------------- NEED TO CHECK -------------------- //
            // Compute right hand side for bad guys --> weighting*(B_f^T - B_c^T*Afc*K)e_row
            for (I k=0; k<numBadGuys; k++) {
                T *badGuyVec = &Bc[k*numCpts];
                T tempBadGuy = linalg.dot_prod(badGuyVec, vec_AfcKe, numCpts);
                rightHandSide[numInd+k] = weighting * (Bf[k*numFpts + row] - tempBadGuy);
            }

        // -------------------------------------------------------------------------------------- //
        // -------------------------------------------------------------------------------------- //
           
            // Solve system from svd_solve in linalg.h.
            svd_solve(leastSquaresOp, (numInd+numBadGuys), numInd, rightHandSide,
                      &(svdSingVals[0]), &(svdWorkSpace[0]), svdWorkSize);
            // Solution above stored in rightHandSide. Give new name accordingly.
            T *newRowY = rightHandSide;

        // -------------------------------------------------------------------------------------- //
        // -------------------------------------------------------------------------------------- //
           
            // Form row of -(K+Y)Afc
            T *rowKY = new T[]

            //  - For each row
            //		+ construct sparse vector of this row of K+Y
            //			~ 
            // 		+ take sparse dot of this addition with ech row of Acf = columns of Afc.
            //	--> CAN I FILL IN P AS IM DOING THIS? NEED TO PREALLOCATE ENOUGH SPACE AND THEN REDUCE WHEN DONE...
            //			~ options include memcpy to a new array, std::copy(), or a dequeue container? 
            for (I k=0; k<numCpts; k++) {

                I thisRow = AcfRowPtr[k];
                I thisSize = AcfRowPtr[k+1] - thisRow;

                T temp = sparse_dot(size1, row1Inds, row1Vals,
                                    thisSize, &(AcfColInds[thisRow]), &(AcfValues[thisRow]) );            
                // If nonzero, add to sparsity pattern... 
                if (temp != 0 ) {

                }

            }

        // -------------------------------------------------------------------------------------- //
        // -------------------------------------------------------------------------------------- //
            // Save result from above as row in sparse data structure for P.


        }
    }

    delete[] svdWorkSpace;
    delete[] svdSingVals;
}



/* Takes dot product of sparse vector and dense vector.                     */
/*      + size1  - number of nonzero elements in sparse vector              */
/*      + ind1   - array of indices of nonzero elements in sparse vector    */
/*      + value1 - nonzero values in sparse vector                          */
/*      + size2  - size of dense vector                                     */
/*      + value2 - list of values in dense vector                           */
void sparse_add(vector<I> &outInd[], vector<T> &outData[], 
                const I &size1, const I ind1[], const T value1[],
                const I &size2, const I ind2[], const T value2[],
                const F &scale = 1.0)
{
    T result = 0.0;
    I lowerInd = 0;

    // Loop over elements in sparse vector, 
    for (I k=0; k<size1; k++) {
        for (I j=lowerInd; j<size2; j++) {
            // If indices overlap, add product to dot product
            if ( ind1[k] == ind2[j] ) {
                result += scale * value1[k] * value2[j];
                lowerInd = j+1;
                break;
            }
            else if( ind1[k] > ind2[j] ) {

            }

            // If inner loop vector index > outer loop vector index (assuming sorted indices),
            // the outer loop vector index is not contained in inner loop indices. 
            else if ( ind1[k] < ind2[j] ) {
                break;
            }
        }
    }
}

// ------------------------------------ VERIFIED ------------------------------------ //
/* Returns dot product of sparse vector and dense vector.                   */
/*      + size1  - number of nonzero elements in sparse vector              */
/*      + ind1   - array of indices of nonzero elements in sparse vector    */
/*      + value1 - nonzero values in sparse vector                          */
/*      + size2  - size of dense vector                                     */
/*      + value2 - list of values in dense vector                           */
T sparse_dot(const I &size1, const I ind1[], const T value1[],
             const I &size2, const I ind2[], const T value2[],
             const F &scale = 1.0)
{
    T result = 0.0;
    I lowerInd = 0;
    // Loop over elements in sparse vector, 
    for (I k=0; k<size1; k++) {
        for (I j=lowerInd; j<size2; j++) {
            // If indices overlap, add product to dot product
            if ( ind1[k] == ind2[j] ) {
                result += scale * value1[k] * value2[j];
                lowerInd = j+1;
                break;
            }
            // If inner loop vector index > outer loop vector index (assuming sorted indices),
            // the outer loop vector index is not contained in inner loop indices. 
            else if ( ind1[k] < ind2[j] ) {
                break;
            }
        }
    }
    return result;
}

// ------------------------------------ VERIFIED ------------------------------------ //
/* Returns dot product of sparse vector and dense vector.                   */
/*      + size1  - number of nonzero elements in sparse vector              */
/*      + ind1   - array of indices of nonzero elements in sparse vector    */
/*      + value1 - nonzero values in sparse vector                          */
/*      + size2  - size of dense vector                                     */
/*      + value2 - list of values in dense vector                           */
T sparse_dense_dot(const I &size1, const I ind1[], const T value1[],
                   const I &size2, const T value2[], const F &scale = 1.0)
{
    T result = 0.0;
    I lowerInd = 0;
    // Loop over elements in sparse vector, 
    for (I k=0; k<size1; k++) {
        for (I j=lowerInd; j<size2; j++) {
            // If indices overlap, add product to dot product
            if ( ind1[k] == j ) {
                result += scale * value1[k] * value2[j];
                lowerInd = j+1;
                break;
            }
            // If inner loop vector index > outer loop vector index (assuming sorted indices),
            // the outer loop vector index is not contained in inner loop indices. 
            else if ( ind1[k] < j ) {
                break;
            }
        }
    }
    return result;
}

