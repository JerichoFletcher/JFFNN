using System.Threading;

namespace JFFNN.Utils {
    /// <summary>
    /// Provides additional methods for atomic operations using the methods in <see cref="Interlocked"/>.
    /// </summary>
    public static class Interlockedf {
        /// <summary>
        /// Atomically adds two doubles (64-bit floating-point numbers) and replaces the first double with the result.
        /// </summary>
        /// <param name="x">The variable containing the first value, and where the result will be stored.</param>
        /// <param name="y">The value to be added to the first value.</param>
        /// <returns>The new value stored at <paramref name="x"/>.</returns>
        public static double Add(ref double x, double y) {
            double z = x;

            while (true) {
                double curZ = z;
                double newZ = curZ + y;
                
                z = Interlocked.CompareExchange(ref x, newZ, curZ);
                if(z.Equals(curZ)) return newZ;
            }
        }
    }
}
