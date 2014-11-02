using Encog.MathUtil.Matrices;
using Encog.ML.Data;
using Encog.Neural;
using Encog.Neural.Thermal;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TrafficSigns.Encog.Neural.Thermal
{
    public class PseudoinverseHopfieldNetwork : HopfieldNetwork
    {
        public PseudoinverseHopfieldNetwork(int size) : base(size)
        {
        }

        public void AddPseudoinversePattern(IMLData pattern)
        {
            if (pattern.Count != NeuronCount)
            {
                throw new NeuralNetworkError("Network with " + NeuronCount
                                             + " neurons, cannot learn a pattern of size "
                                             + pattern.Count);
            }

            // Create a row matrix from the input, convert boolean to bipolar
            //Matrix m2 = Matrix.CreateRowMatrix(pattern);
            //// Transpose the matrix and multiply by the original input matrix
            //Matrix m1 = MatrixMath.Transpose(m2);
            //Matrix m3 = MatrixMath.Multiply(m1, m2);

            //// matrix 3 should be square by now, so create an identity
            //// matrix of the same size.
            //Matrix identity = MatrixMath.Identity(m3.Rows);

            //// subtract the identity matrix
            //Matrix m4 = MatrixMath.Subtract(m3, identity);

            // now add the calculated matrix, for this pattern, to the
            // existing weight matrix.
            ConvertHopfieldMatrix(m4);
        }

        /// <summary>
        /// Update the Hopfield weights after training.
        /// </summary>
        ///
        /// <param name="delta">The amount to change the weights by.</param>
        private void ConvertHopfieldMatrix(Matrix delta)
        {
            // add the new weight matrix to what is there already
            for (int row = 0; row < delta.Rows; row++)
            {
                for (int col = 0; col < delta.Rows; col++)
                {
                    AddWeight(row, col, delta[row, col]);
                }
            }
        }

    }
}
