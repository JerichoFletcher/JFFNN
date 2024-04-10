using JFFNN.Structs;

namespace JFFNN.NN {
    /// <summary>
    /// Describes a type of neural network layer.
    /// </summary>
    public interface INetworkLayerType {
        /// <summary>
        /// Processes an input vector according to the layer parameters.
        /// </summary>
        /// <param name="input">The input vector.</param>
        /// <returns>The result vector.</returns>
        Vector Feed(Vector input);
    }
}
