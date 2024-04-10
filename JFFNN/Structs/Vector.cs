using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

namespace JFFNN.Structs {
    /// <summary>
    /// Represents a vector of real numbers.
    /// </summary>
    public readonly struct Vector : IEnumerable<double>, IEnumerable {
        private readonly double[] values;

        /// <summary>
        /// The size of the vector.
        /// </summary>
        public int Size => values.Length;

        /// <summary>
        /// Accessor of an element in the vector.
        /// </summary>
        /// <param name="i">The index of the accessed element.</param>
        /// <returns>The <paramref name="i"/>-th element of the vector.</returns>
        public double this[int i] {
            get => values[i];
            set => values[i] = value;
        }

        /// <summary>
        /// Creates a vector with the given size.
        /// </summary>
        /// <param name="size">The size of the vector.</param>
        public Vector(int size) {
            values = new double[size];
        }

        /// <summary>
        /// Adds two vectors by summing each corresponding element of the vectors.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>The result vector.</returns>
        /// <exception cref="ArgumentException">The two vectors do not have the same size.</exception>
        public static Vector operator +(Vector left, Vector right) {
            if(left.Size != right.Size) throw new ArgumentException($"Mismatched vector size: left vector is {left.Size}; right vector is {right.Size}");

            Vector result = new Vector(left.Size);
            Parallel.For(0, left.Size, i => {
                result[i] = left[i] + right[i];
            });

            return result;
        }

        /// <summary>
        /// Prepends the vector with an additional element.
        /// </summary>
        /// <param name="left">The new element to add.</param>
        /// <param name="right">The vector to prepend.</param>
        /// <returns>The result vector.</returns>
        public static Vector operator &(double left, Vector right) {
            Vector result = new Vector(right.Size + 1);
            result[0] = left;

            Parallel.For(1, result.Size, i => {
                result[i] = right[i - 1];
            });

            return result;
        }

        IEnumerator IEnumerable.GetEnumerator() {
            return GetEnumerator();
        }

        public IEnumerator<double> GetEnumerator() {
            return (IEnumerator<double>)values.GetEnumerator();
        }

        public override string ToString() {
            StringBuilder str = new StringBuilder("[");

            for(int i = 0; i < Size; ++i) {
                str.Append(values[i].ToString());
                if(i < Size - 1) str.Append(", ");
            }

            return str.Append("]").ToString();
        }
    }
}
