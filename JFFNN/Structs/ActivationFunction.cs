using JFFNN.Utils;
using System;
using System.Threading.Tasks;

namespace JFFNN.Structs {
    /// <summary>
    /// Encapsulates a method describing an activation function, which maps values in a logit vector.
    /// </summary>
    /// <param name="logit">The input vector, in which elements to be mapped are stored.</param>
    /// <returns>A vector containing the mapping result.</returns>
    public delegate Vector ActivationFunction(Vector logit);

    /// <summary>
    /// Contains common activation functions to be used in fully connected network layers (see also <seealso cref="NN.NetworkLayerType.FullyConnected"/>).
    /// </summary>
    public static class ActivationFunctions {
        /// <summary>
        /// Defines a pass-through activation function, where every logit vector maps to itself.
        /// </summary>
        public static ActivationFunction Passthrough => logit => logit;

        /// <summary>
        /// Defines a sigmoid activation function, where every element in a logit vector maps through the sigmoid function.
        /// </summary>
        public static ActivationFunction Sigmoid => logit => {
            Vector result = new Vector(logit.Size);

            Parallel.For(0, logit.Size, i => {
                result[i] = 1d / (1d + Math.Exp(-logit[i]));
            });

            return result;
        };

        /// <summary>
        /// Defines a rectifier activation function, where every element in a logit vector is rectified to its positive part.
        /// </summary>
        public static ActivationFunction ReLU => logit => {
            Vector result = new Vector(logit.Size);

            Parallel.For(0, logit.Size, i => {
                result[i] = Math.Max(0d, logit[i]);
            });

            return result;
        };

        /// <summary>
        /// Defines a softmax activation function, where every element in a logit vector is normalized such that all elements are
        /// non-negative and sums up to 1, and larger elements are given greater weight relative to smaller elements.
        /// </summary>
        public static ActivationFunction Softmax => logit => {
            Vector result = new Vector(logit.Size);
            double sum = 0d;

            Parallel.For(0, logit.Size, i => {
                double d = Math.Exp(logit[i]);
                result[i] = d;
                Interlockedf.Add(ref sum, d);
            });

            Parallel.For(0, logit.Size, i => {
                result[i] /= sum;
            });

            return result;
        };
    }
}
