using JFFNN.Structs;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using System.Threading.Tasks.Dataflow;

namespace JFFNN.NN {
    /// <summary>
    /// Describes a neural network.
    /// </summary>
    public class Network {
        /// <summary>
        /// The size of the input vector, i.e. the number of neurons in the input layer.
        /// </summary>
        public int InputSize { get; private set; }

        private readonly CancellationTokenSource cancellationSource = new CancellationTokenSource();
        private NetworkLayer[] layers;

        private Network(int inputSize) {
            InputSize = inputSize;
        }

        private List<TransformBlock<Vector, Vector>> CreatePipeline() {
            var pipeline = new List<TransformBlock<Vector, Vector>>();

            for(int i = 0; i < layers.Length; ++i) {
                NetworkLayer layer = layers[i];
                var block = new TransformBlock<Vector, Vector>(v => layer.Feed(v),
                    new ExecutionDataflowBlockOptions() {
                        CancellationToken = cancellationSource.Token
                    }
                );

                pipeline.Add(block);
                if(i > 0) pipeline[i - 1].LinkTo(block,
                    new DataflowLinkOptions() {
                        PropagateCompletion = true
                    }
                );
            }

            return pipeline;
        }

        /// <summary>
        /// Processes an input vector through the layers in the network.
        /// </summary>
        /// <param name="input">The input vector.</param>
        /// <returns>The output vector.</returns>
        public Vector Feed(Vector input) {
            if(layers == null || layers.Length == 0) return input;

            Vector current = input;
            foreach(NetworkLayer layer in layers) current = layer.Feed(current);

            return current;
        }

        /// <summary>
        /// Iterates through an enumerable of input vectors and returns the result of each of them.
        /// </summary>
        /// <param name="input">The input vectors.</param>
        /// <returns>The output vectors.</returns>
        /// <exception cref="AggregateException">One or more exception was thrown in the feed-forward pipeline.</exception>
        public IEnumerable<Vector> Feed(IEnumerable<Vector> input) {
            if(layers == null || layers.Length == 0) foreach(Vector v in input) yield return v;

            var pipeline = CreatePipeline();
            var firstBlock = pipeline[0];
            var lastBlock = pipeline[pipeline.Count - 1];

            try {
                foreach(Vector v in input) if(!firstBlock.Post(v)) break;
            } finally {
                firstBlock.Complete();
            }

            try {
                while(lastBlock.OutputAvailableAsync().Result)
                    while(lastBlock.TryReceive(out Vector output)) yield return output;
            } finally {
                cancellationSource.Cancel();
            }

            lastBlock.Completion.Wait();
        }

        /// <summary>
        /// Obtains a neural network builder for use in creating a neural network instance.
        /// </summary>
        /// <param name="inputSize">The size of the input vector of the model.</param>
        /// <returns>A neural network builder instance.</returns>
        public static Builder Create(int inputSize) => new Builder(inputSize);

        /// <summary>
        /// Describes a builder for use in creating a neural network instance.
        /// </summary>
        public class Builder {
            private readonly Network network;
            private readonly List<NetworkLayer> networkLayers;

            internal Builder(int inputSize) {
                network = new Network(inputSize);
                networkLayers = new List<NetworkLayer>();
            }

            /// <summary>
            /// Adds a layer of the given type to the neural network instance.
            /// </summary>
            /// <param name="type">The type of the layer to be added.</param>
            /// <returns>The builder instance.</returns>
            public Builder AddLayer(INetworkLayerType type) {
                networkLayers.Add(new NetworkLayer(type));
                return this;
            }

            /// <summary>
            /// Builds the neural network instance using the given parameters.
            /// </summary>
            /// <returns>The resulting neural network instance.</returns>
            public Network Build() {
                network.layers = networkLayers.ToArray();
                return network;
            }
        }
    }
}
