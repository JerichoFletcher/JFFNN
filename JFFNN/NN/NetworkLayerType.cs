using JFFNN.Structs;

namespace JFFNN.NN {
    /// <summary>
    /// Contains common types of neural network layers.
    /// </summary>
    public static class NetworkLayerType {
        /// <summary>
        /// Describes a fully connected neural network layer, in which every neuron in a preceding layer is connected to each neuron in this layer.
        /// </summary>
        public class FullyConnected : INetworkLayerType {
            /// <summary>
            /// The activation function used in this layer.
            /// </summary>
            public ActivationFunction ActivationFunction { get; private set; }

            /// <summary>
            /// The number of neurons contained in this layer.
            /// </summary>
            public int NeuronCount { get; private set; }

            /// <summary>
            /// The rate of dropout of the layer. This value is the probability of a neuron being dropped out during the forward propagation phase of training.
            /// After training, weights are also scaled down proportional to this value.
            /// </summary>
            public double Dropout { get; private set; }

            /// <summary>
            /// The weights of the neuron connections in this layer.
            /// </summary>
            public Matrix Weights { get; private set; }

            /// <summary>
            /// Creates a fully connected neural network layer with the given hyperparameters and initial weights.
            /// </summary>
            /// <param name="activationFunction">The activation function to use.</param>
            /// <param name="dropout">The dropout rate of the layer.</param>
            /// <param name="weights">A matrix containing the weights of each connection to use.</param>
            public FullyConnected(ActivationFunction activationFunction, double dropout, Matrix weights) {
                ActivationFunction = activationFunction;
                NeuronCount = weights.RowCount;
                Dropout = dropout;
                Weights = weights;
            }

            public Vector Feed(Vector input) => ActivationFunction(Weights * (1d & input));
        }
    }
}
