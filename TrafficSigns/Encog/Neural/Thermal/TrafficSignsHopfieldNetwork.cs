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
using TrafficSigns.Utils;

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

        public void SetCurrentBiPolarState(BiPolarMLData data)
        {
            var currentState = new BiPolarMLData(data.Count);
            for (int i = 0; i < data.Count; i++)
            {
                currentState.SetBoolean(i, (data[i] > 0));
            }
            CurrentState = currentState;
        }

        protected bool CheckCorrectness(IList<BiPolarMLData> patterns, int maxCycles)
        {
            var sw = new Stopwatch();
            sw.Start();
            foreach (BiPolarMLData pattern in patterns)
            {
                SetCurrentBiPolarState(pattern);
                RunUntilStable(maxCycles);
                
                if (!pattern.Data.SequenceEqual(CurrentState.Data))
                {
                    return false;
                }
            }
            sw.Stop();

            return true;
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

        public void TrainDelta(IList<BiPolarMLData> patterns, int limit = 100)
        {
            for (int i = 0; i < limit; i++)
            {
                if (i % 10 == 0 && CheckCorrectness(patterns, 100))
                {
                    break;
                }

                BiPolarMLData pattern = patterns[MathUtils.Random.Next(patterns.Count)];

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
            Matrix patternsMatrix = GeneratePatternsMatrix(patterns);
            Matrix patternsMatrixT = MatrixMath.Transpose(patternsMatrix);

            Matrix middleProductMatrix = MatrixMath.Multiply(patternsMatrixT, patternsMatrix);
            Matrix inversedMatrix = middleProductMatrix.Inverse();

            Matrix productWithInverseMatrix = MatrixMath.Multiply(patternsMatrix, inversedMatrix);
            Matrix weightMatrix = MatrixMath.Multiply(productWithInverseMatrix, patternsMatrixT);

            SetWeightMatrix(weightMatrix);
        }

        protected Matrix GeneratePatternsMatrix(IList<BiPolarMLData> patterns)
        {
            var sourceMatrix = new double[patterns.Count][];

            for (int i = 0; i < patterns.Count; i++)
            {
                sourceMatrix[i] = patterns[i].Data;
            }
            var resultT = new Matrix(sourceMatrix);

            return MatrixMath.Transpose(resultT);
        }

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

        public void SetWeightMatrix(Matrix weightMatrix)
        {
            for (int row = 0; row < weightMatrix.Rows; row++)
            {
                for (int col = 0; col < weightMatrix.Cols; col++)
                {
                    SetWeight(row, col, weightMatrix[row, col]);
                }
            }
        }

    }
}
