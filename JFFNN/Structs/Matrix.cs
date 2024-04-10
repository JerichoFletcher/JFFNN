using JFFNN.Utils;
using System;
using System.Threading.Tasks;

namespace JFFNN.Structs {
    /// <summary>
    /// Represents a matrix of real numbers.
    /// </summary>
    public readonly struct Matrix {
        /// <summary>
        /// The number of rows in the matrix.
        /// </summary>
        public int RowCount => values.GetLength(0);

        /// <summary>
        /// The number of columns in the matrix.
        /// </summary>
        public int ColumnCount => values.GetLength(1);

        private readonly double[,] values;

        /// <summary>
        /// Accessor of an element in the matrix.
        /// </summary>
        /// <param name="row">The row index of the accessed element.</param>
        /// <param name="col">The column index of the accessed element.</param>
        /// <returns>The element at the position (<paramref name="row"/>, <paramref name="col"/>).</returns>
        public double this[int row, int col] {
            get => values[row, col];
            set => values[row, col] = value;
        }

        /// <summary>
        /// Creates a matrix with the given size.
        /// </summary>
        /// <param name="rowCount">The number of rows in the matrix.</param>
        /// <param name="colCount">The number of columns in the matrix.</param>
        public Matrix(int rowCount, int colCount) {
            values = new double[rowCount, colCount];
        }

        /// <summary>
        /// Multiplies a vector with a matrix. The vector is treated as a right operand column matrix.
        /// </summary>
        /// <param name="mat">The matrix.</param>
        /// <param name="vec">The vector.</param>
        /// <returns>The result vector, treated as a column matrix.</returns>
        public static Vector operator *(Matrix mat, Vector vec) {
            if(mat.ColumnCount != vec.Size) throw new ArgumentException($"Incompatible matrix and vector size: matrix column is {mat.ColumnCount}, vector is {vec.Size}");

            Vector result = new Vector(mat.RowCount);
            Parallel.For(0, mat.RowCount, row => {
                double d = 0d;

                Parallel.For(0, mat.ColumnCount, col => {
                    double p = mat[row, col] * vec[col];
                    Interlockedf.Add(ref d, p);
                });

                result[row] = d;
            });

            return result;
        }
    }
}
