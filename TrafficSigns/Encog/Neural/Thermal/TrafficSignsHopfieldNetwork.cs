using Encog.MathUtil.Matrices;
using Encog.ML.Data;
using Encog.ML.Data.Specific;
using Encog.Neural;
using Encog.Neural.Thermal;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TrafficSigns.Encog.Neural.Thermal
{
    public class TrafficSignsHopfieldNetwork : HopfieldNetwork
    {
        public double LearningRate { get; set; }

        public TrafficSignsHopfieldNetwork(int size) : base(size)
        {
            LearningRate = 0.8d;
        }

        public Matrix GenerateWeightMatrix()
        {
            Matrix result = new Matrix(NeuronCount, NeuronCount);
            for (int i = 0; i < NeuronCount; i++)
            {
                for (int j = 0; j < NeuronCount; j++)
                {
                    result[i, j] = GetWeight(i, j);
                }
            }
            return result;
        }

        public void TrainHebbian(IList<BiPolarMLData> patterns, int iterations = 1)
        {
            for (int i = 0; i < iterations; i++)
            {
                foreach (BiPolarMLData pattern in patterns)
                {
                    AddPattern(pattern);
                }
            }
        }

        public void TrainDelta(IList<BiPolarMLData> patterns)
        {
            foreach (BiPolarMLData pattern in patterns)
            {
                Matrix weightMatrix = GenerateWeightMatrix();
                Matrix patternMatrixT = Matrix.CreateRowMatrix(pattern);
                Matrix patternMatrix = MatrixMath.Transpose(patternMatrixT);

                Matrix deltaMatrix = MatrixMath.Multiply(weightMatrix, patternMatrix);
                deltaMatrix = MatrixMath.Subtract(patternMatrix, deltaMatrix);
                deltaMatrix = MatrixMath.Multiply(deltaMatrix, patternMatrixT);
                deltaMatrix = MatrixMath.Multiply(deltaMatrix, LearningRate / (double)NeuronCount);

                AddMatrixWeights(deltaMatrix);
            }
        }

        public void TrainPseudoinverse(IList<BiPolarMLData> patterns)
        {
            double[,] oneOverQ = new double[patterns.Count, patterns.Count];
            for (int n = 0; n < patterns.Count; n++)
            {
                for (int m = 0; m < patterns.Count; m++)
                {
                    double sumQ = 0d;
                    for (int k = 0; k < NeuronCount; k++)
                    {
                        sumQ += patterns[n][k] * patterns[m][k];
                    }
                    oneOverQ[n, m] = 1d / (sumQ / (double)NeuronCount);
                }
            }

            for (int i = 0; i < NeuronCount; i++)
            {
                for (int j = 0; j < NeuronCount; j++)
                {
                    if (i == j)
                    {
                        SetWeight(i, j, 0d);
                    }
                    SetWeight(i, j, CalculateWeightPseudoinverse(i, j, patterns, oneOverQ));
                }
            }
        }

        protected double CalculateWeightPseudoinverse(int i, int j, IList<BiPolarMLData> patterns, double[,] oneOverQ)
        {
            double sum = 0d;
            for (int n = 0; n < patterns.Count; n++)
            {
                for (int m = 0; m < patterns.Count; m++)
                {
                    sum += patterns[n][i] * oneOverQ[n, m] * patterns[m][j];
                }
            }

            return sum / (double)NeuronCount;
        }

        /// <summary>
        /// Update the Hopfield weights after training.
        /// </summary>
        ///
        /// <param name="delta">The amount to change the weights by.</param>
        private void AddMatrixWeights(Matrix delta)
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
