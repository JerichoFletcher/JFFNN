using JFFNN.Structs;

namespace JFFNN.NN {
    /// <summary>
    /// Describes a layer in a neural network.
    /// </summary>
    public class NetworkLayer {
        /// <summary>
        /// The type of the layer, describing its behavior and parameters.
        /// </summary>
        public INetworkLayerType Type { get; private set; }

        /// <summary>
        /// Creates a layer of the given type.
        /// </summary>
        /// <param name="type">The type of the layer.</param>
        public NetworkLayer(INetworkLayerType type) {
            Type = type;
        }

        /// <summary>
        /// Processes an input vector according to the type of the layer.
        /// </summary>
        /// <param name="input">The input vector.</param>
        /// <returns>The result vector.</returns>
        public Vector Feed(Vector input) => Type.Feed(input);
    }
}
