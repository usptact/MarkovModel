using System;
using System.Linq;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;

namespace MarkovModel
{
    class Program
    {
        static void Main(string[] args)
        {
            TestMarkovModel();
        }

        static void TestMarkovModel()
        {
            // Problem dimensions
            int T = 200;
            int K = 3;

            // Define true parameters explicitly for the generator
            double[] trueInit = GetTrueInitial(K);
            double[][] trueTrans = GetTrueTransition(K);

            Console.WriteLine("True parameters:");
            PrintParameters(trueInit, trueTrans);
            Console.WriteLine();

            // Generate data explicitly (no Infer.NET sampling)
            int seed = 12347;
            int[] actualStates = GenerateStates(T, trueInit, trueTrans, seed);

            Console.WriteLine("sample data:");
            Console.WriteLine(string.Join(",", actualStates));
            Console.WriteLine();

            // Fit the Markov model with uninformative priors
            MarkovModel model = new MarkovModel(T, K);
            model.SetUninformedPriors();
            model.ObserveData(actualStates);
            model.InferPosteriors();
            Console.WriteLine("model likelihood: " + Math.Exp(model.ModelEvidencePosterior.LogOdds));
            Console.WriteLine();

            // Convert posteriors to means for comparison
            var inferredInitMean = model.ProbInitPosterior.GetMean().ToArray();
            var inferredTransMean = model.CPTTransPosterior.Select(row => row.GetMean().ToArray()).ToArray();

            Console.WriteLine("Inferred parameter means:");
            PrintParameters(inferredInitMean, inferredTransMean);
            Console.WriteLine();

            // Print comparison of true vs inferred
            PrintComparison(trueInit, trueTrans, inferredInitMean, inferredTransMean);
        }

        /// <summary>
        /// Returns explicit true initial distribution over K states.
        /// </summary>
        private static double[] GetTrueInitial(int K)
        {
            if (K == 3)
            {
                return new double[] { 0.55, 0.30, 0.15 };
            }
            // Default: uniform
            return Enumerable.Repeat(1.0 / K, K).ToArray();
        }

        /// <summary>
        /// Returns explicit true transition matrix (row-stochastic) of size KxK.
        /// </summary>
        private static double[][] GetTrueTransition(int K)
        {
            if (K == 3)
            {
                return new double[][]
                {
                    new double[] { 0.70, 0.20, 0.10 },
                    new double[] { 0.10, 0.65, 0.25 },
                    new double[] { 0.20, 0.25, 0.55 }
                };
            }
            // Default: near-diagonal with small off-diagonals
            double stay = 0.7;
            double off = (1.0 - stay) / (K - 1);
            double[][] trans = new double[K][];
            for (int i = 0; i < K; i++)
            {
                trans[i] = Enumerable.Range(0, K)
                    .Select(j => j == i ? stay : off)
                    .ToArray();
            }
            return trans;
        }

        /// <summary>
        /// Generates a sequence of states using explicit categorical sampling from the given parameters.
        /// </summary>
        private static int[] GenerateStates(int T, double[] init, double[][] trans, int seed)
        {
            int[] states = new int[T];
            var rng = new Random(seed);

            states[0] = SampleCategorical(init, rng);
            for (int t = 1; t < T; t++)
            {
                states[t] = SampleCategorical(trans[states[t - 1]], rng);
            }
            return states;
        }

        /// <summary>
        /// Samples an index from a categorical distribution represented by probabilities.
        /// </summary>
        private static int SampleCategorical(double[] probs, Random rng)
        {
            double r = rng.NextDouble();
            double c = 0.0;
            for (int i = 0; i < probs.Length; i++)
            {
                c += probs[i];
                if (r <= c) return i;
            }
            // Fallback due to potential floating-point accumulation
            return probs.Length - 1;
        }

        /// <summary>
        /// Prints initial and transition parameters with 4 decimal precision.
        /// </summary>
        private static void PrintParameters(double[] init, double[][] trans)
        {
            Console.WriteLine(string.Join(" ", init.Select(x => x.ToString("0.####"))));
            for (int i = 0; i < trans.Length; i++)
            {
                Console.WriteLine("[" + i + "]" + string.Join(" ", trans[i].Select(x => x.ToString("0.####"))));
            }
        }

        /// <summary>
        /// Prints a comparison of true vs inferred parameters and reports L1 errors.
        /// </summary>
        private static void PrintComparison(double[] trueInit, double[][] trueTrans, double[] infInit, double[][] infTrans)
        {
            Console.WriteLine("Comparison (true vs inferred means):");
            Console.WriteLine("Init:");
            Console.WriteLine("  true:     " + string.Join(" ", trueInit.Select(x => x.ToString("0.####"))));
            Console.WriteLine("  inferred: " + string.Join(" ", infInit.Select(x => x.ToString("0.####"))));

            Console.WriteLine("Transitions:");
            for (int i = 0; i < trueTrans.Length; i++)
            {
                Console.WriteLine("  row " + i + " true:     " + string.Join(" ", trueTrans[i].Select(x => x.ToString("0.####"))));
                Console.WriteLine("  row " + i + " inferred: " + string.Join(" ", infTrans[i].Select(x => x.ToString("0.####"))));
            }

            double l1Init = trueInit.Zip(infInit, (a, b) => Math.Abs(a - b)).Sum();
            double l1Trans = trueTrans.Select((row, i) => row.Zip(infTrans[i], (a, b) => Math.Abs(a - b)).Sum()).Average();

            Console.WriteLine();
            Console.WriteLine("L1 error (init): " + l1Init.ToString("0.####"));
            Console.WriteLine("Avg L1 error (rows of trans): " + l1Trans.ToString("0.####"));
        }
    }
}
