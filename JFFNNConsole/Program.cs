using JFFNN.NN;
using JFFNN.Structs;
using System;
using System.Collections.Generic;

using static JFFNN.NN.NetworkLayerType;
using static JFFNN.Structs.ActivationFunctions;

namespace JFFNNConsole {
    internal class Program {
        static void Main() {
            Network network = Network.Create(3)
                .AddLayer(new FullyConnected(Sigmoid, 0, new Matrix(2, 4) {
                    [0, 0] = 0.6,   [0, 1] = -1.2,  [0, 2] = 1.4,   [0, 3] = -0.7,
                    [1, 0] = -1.2,  [1, 1] = -1.7,  [1, 2] = -1.6,  [1, 3] = 1.1
                }))
                .AddLayer(new FullyConnected(Sigmoid, 0, new Matrix(4, 3) {
                    [0, 0] = -0.4,  [0, 1] = -0.0,  [0, 2] = 2.1,
                    [1, 0] = 1.6,   [1, 1] = 0.0,   [1, 2] = -0.2,
                    [2, 0] = 1.6,   [2, 1] = -1.5,  [2, 2] = 0.0,
                    [3, 0] = -1.5,  [3, 1] = 0.7,   [3, 2] = 1.8
                }))
                .Build();

            List<Vector> inputs = new List<Vector>() {
                new Vector(3) { [0] = -0.6, [1] = 1.6, [2] = -1.0 },
                new Vector(3) { [0] = -1.4, [1] = 0.9, [2] = 1.5 },
                new Vector(3) { [0] = 0.2, [1] = -1.3, [2] = -1.0 },
                new Vector(3) { [0] = -0.9, [1] = -0.7, [2] = -1.2 },
                new Vector(3) { [0] = 0.4, [1] = 0.1, [2] = 0.2 }
            };
            
            foreach(Vector output in network.Feed(inputs)) {
                Console.WriteLine(output);
            }

            Console.ReadLine();
        }
    }
}
