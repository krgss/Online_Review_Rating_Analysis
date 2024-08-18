package utilities;

public class MatrixOperations {


	// Function to get cofactor of mat[p][q] in temp[][]. n is current
	// dimension of mat[][]
	/*void getCofactor(double mat[][], double temp[][], int p, int q, int n)
	{
	    int i = 0, j = 0;
	 
	    // Looping for each element of the matrix
	    for (int row = 0; row < n; row++)
	    {
	        for (int col = 0; col < n; col++)
	        {
	            //  Copying into temporary matrix only those element
	            //  which are not in given row and column
	            if (row != p && col != q)
	            {
	                temp[i][j++] = mat[row][col];
	 
	                // Row is filled, so increase row index and
	                // reset col index
	                if (j == n - 1)
	                {
	                    j = 0;
	                    i++;
	                }
	            }
	        }
	    }
	}
	 
	/* Recursive function for finding determinant of matrix.
	   n is current dimension of mat[][]. 
	double determinantOfMatrix(double mat[][], int n)
	{
	    double D = 0; // Initialize result
	 
	    //  Base case : if matrix contains single element
	    if (n == 1)
	        return mat[0][0];
	 
	    double temp[][]; // To store cofactors
	    temp =  new double[n][n];
	 
	    int sign = 1;  // To store sign multiplier
	 
	     // Iterate for each element of first row
	    for (int f = 0; f < n; f++)
	    {
	        // Getting Cofactor of mat[0][f]
	        getCofactor(mat, temp, 0, f, n);
	        D += sign * mat[0][f] * determinantOfMatrix(temp, n - 1);
	 
	        // terms are to be added with alternate sign
	        sign = -sign;
	    }
	 
	    return D;
	}*/
	
	
	public static double[][] invertMatrix (double[][] matrix) {
		double[][] auxiliaryMatrix, invertedMatrix;
		int[] index;

		auxiliaryMatrix = new double[matrix.length][matrix.length];
		invertedMatrix = new double[matrix.length][matrix.length];
		index = new int[matrix.length];

		for (int i = 0; i < matrix.length; ++i) {
			auxiliaryMatrix[i][i] = 1;
		}

		transformToUpperTriangle (matrix, index);

		for (int i = 0; i < (matrix.length - 1); ++i) {
			for (int j = (i + 1); j < matrix.length; ++j) {
				for (int k = 0; k < matrix.length; ++k) {
					auxiliaryMatrix[index[j]][k] -= matrix[index[j]][i] * auxiliaryMatrix[index[i]][k];
				}
			}
		}

		for (int i = 0; i < matrix.length; ++i) {
			invertedMatrix[matrix.length - 1][i] = (auxiliaryMatrix[index[matrix.length - 1]][i] / matrix[index[matrix.length - 1]][matrix.length - 1]);

			for (int j = (matrix.length - 2); j >= 0; --j) {
				invertedMatrix[j][i] = auxiliaryMatrix[index[j]][i];

				for (int k = (j + 1); k < matrix.length; ++k) {
					invertedMatrix[j][i] -= (matrix[index[j]][k] * invertedMatrix[k][i]);
				}

				invertedMatrix[j][i] /= matrix[index[j]][j];
			}
		}

		return (invertedMatrix);
	}

	public static void transformToUpperTriangle (double[][] matrix, int[] index) {
		double[] c;
		double c0, c1, pi0, pi1, pj;
		int itmp, k;

		c = new double[matrix.length];

		for (int i = 0; i < matrix.length; ++i) {
			index[i] = i;
		}

		for (int i = 0; i < matrix.length; ++i) {
			c1 = 0;

			for (int j = 0; j < matrix.length; ++j) {
				c0 = Math.abs (matrix[i][j]);

				if (c0 > c1) {
					c1 = c0;
				}
			}

			c[i] = c1;
		}

		k = 0;

		for (int j = 0; j < (matrix.length - 1); ++j) {
			pi1 = 0;

			for (int i = j; i < matrix.length; ++i) {
				pi0 = Math.abs (matrix[index[i]][j]);
				pi0 /= c[index[i]];

				if (pi0 > pi1) {
					pi1 = pi0;
					k = i;
				}
			}

			itmp = index[j];
			index[j] = index[k];
			index[k] = itmp;

			for (int i = (j + 1); i < matrix.length; ++i) {
				pj = matrix[index[i]][j] / matrix[index[j]][j];
				matrix[index[i]][j] = pj;

				for (int l = (j + 1); l < matrix.length; ++l) {
					matrix[index[i]][l] -= pj * matrix[index[j]][l];
				}
			}
		}
	}
	
	public static double matrixDeterminant (double[][] matrix) {
		double temporary[][];
		double result = 0;

		if (matrix.length == 1) {
			result = matrix[0][0];
			return (result);
		}

		if (matrix.length == 2) {
			result = ((matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0]));
			return (result);
		}

		for (int i = 0; i < matrix[0].length; i++) {
			temporary = new double[matrix.length - 1][matrix[0].length - 1];

			for (int j = 1; j < matrix.length; j++) {
				for (int k = 0; k < matrix[0].length; k++) {
					if (k < i) {
						temporary[j - 1][k] = matrix[j][k];
					} else if (k > i) {
						temporary[j - 1][k - 1] = matrix[j][k];
					}
				}
			}

			result += matrix[0][i] * Math.pow (-1, (double) i) * matrixDeterminant (temporary);
		}
		return (result);
	}
}
