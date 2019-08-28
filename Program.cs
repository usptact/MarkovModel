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
            Rand.Restart(12347);

            int T = 100;
            int K = 3;

            // set non-informative hyperparameters
            Dirichlet ProbInitPriorObs = Dirichlet.Uniform(K);
            Dirichlet[] CPTTransPriorObs = Enumerable.Repeat(Dirichlet.Uniform(K), K).ToArray();

            // sample model parameters
            double[] init = ProbInitPriorObs.Sample().ToArray();
            double[][] trans = new double[K][];
            for (int i = 0; i < K; i++)
            {
                trans[i] = CPTTransPriorObs[i].Sample().ToArray();
            }

            // print parameters
            MarkovModel modelForPrinting = new MarkovModel(T, K);
            modelForPrinting.SetParameters(init, trans);
            Console.WriteLine("parameters:");
            modelForPrinting.PrintParameters();
            Console.WriteLine();

            // create distributions for sampling
            Discrete initDist = new Discrete(init);
            Discrete[] transDist = new Discrete[K];
            for (int i = 0; i < K; i++)
            {
                transDist[i] = new Discrete(trans[i]);
            }

            // sample state data
            int[] actualStates = new int[T];
            actualStates[0] = initDist.Sample();
            for (int i = 1; i < T; i++)
            {
                actualStates[i] = transDist[actualStates[i-1]].Sample();
            }
            Console.WriteLine("sample data:");
            Console.WriteLine(string.Join(",", actualStates));
            Console.WriteLine();

            // infer model parameters, states and model evidence given priors and state data
            MarkovModel model = new MarkovModel(T, K);
            //model.SetPriors(ProbInitPriorObs, CPTTransPriorObs);
            model.SetUninformedPriors();
            model.ObserveData(actualStates);
            model.InferPosteriors();
            Console.WriteLine("model likelihood: " + Math.Exp(model.ModelEvidencePosterior.LogOdds));
            Console.WriteLine();

            model.PrintPosteriors();
        }
    }
}
