using System;
using System.Linq;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Utilities;

namespace MarkovModel
{
    class MarkovModel
    {
        private int[] StateData;

        private Range K;
        private Range T;

        // Set up the main model variables
        private Variable<int> ZeroState;
        private VariableArray<int> States;

        // Set up model parameters
        private Variable<Vector> ProbInit;
        private VariableArray<Vector> CPTTrans;

        // Set up prior distributions
        private Variable<Dirichlet> ProbInitPrior;
        private VariableArray<Dirichlet> CPTTransPrior;

        // Set up model evidence (likelihood of model given data)
        private Variable<bool> ModelEvidence;

        private InferenceEngine Engine;

        // Set up posteriors
        public Dirichlet ProbInitPosterior;
        public Dirichlet[] CPTTransPosterior;
        public Bernoulli ModelEvidencePosterior;

        public MarkovModel(int ChainLength, int NumStates)
        {
            ModelEvidence = Variable.Bernoulli(0.5);
            using (Variable.If(ModelEvidence))
            {
                K = new Range(NumStates);
                T = new Range(ChainLength);

                ProbInitPrior = Variable.New<Dirichlet>();                  // init state priors
                ProbInit = Variable<Vector>.Random(ProbInitPrior);
                ProbInit.SetValueRange(K);

                CPTTransPrior = Variable.Array<Dirichlet>(K);               // transition state priors
                CPTTrans = Variable.Array<Vector>(K);
                CPTTrans[K] = Variable<Vector>.Random(CPTTransPrior[K]);
                CPTTrans.SetValueRange(K);

                ZeroState = Variable.Discrete(ProbInit);                    // main model variables
                States = Variable.Array<int>(T);

                using (var block = Variable.ForEach(T))
                {
                    var t = block.Index;

                    using (Variable.If(t == 0))                             // init state
                    {
                        States[t] = Variable.DiscreteUniform(K);
                        /*
                        using (Variable.Switch(ZeroState))
                        {
                            States[t] = Variable.Discrete(ProbInit);
                            //States[t] = Variable.DiscreteUniform(K);
                        }
                        */
                    }

                    using (Variable.If(t > 0))                              // transition states
                    {
                        using (Variable.Switch(States[t-1]))
                        {
                            States[t] = Variable.Discrete(CPTTrans[States[t-1]]);
                        }
                    }                         
                }
            }

            DefineInferenceEngine();
        }

        public void DefineInferenceEngine()
        {
            Engine = new InferenceEngine(new ExpectationPropagation());
            //Engine = new InferenceEngine(new VariationalMessagePassing());
            //Engine = new InferenceEngine(new GibbsSampling());
            Engine.ShowFactorGraph = false;
            Engine.ShowWarnings = true;
            Engine.ShowProgress = true;
            Engine.Compiler.WriteSourceFiles = true;
            Engine.NumberOfIterations = 50;
            Engine.ShowTimings = true;
            Engine.ShowSchedule = false;
        }

        public void ObserveData(int[] stateData)
        {
            StateData = new int[stateData.Length];
            Array.Copy(stateData, StateData, stateData.Length);
            States.ObservedValue = StateData;
        }

        public void InferPosteriors()
        {
            CPTTransPosterior = Engine.Infer<Dirichlet[]>(CPTTrans);
            ProbInitPosterior = Engine.Infer<Dirichlet>(ProbInit);
            ModelEvidencePosterior = Engine.Infer<Bernoulli>(ModelEvidence);
        }

        public void SetUninformedPriors()
        {
            ProbInitPrior.ObservedValue = Dirichlet.Uniform(K.SizeAsInt);
            CPTTransPrior.ObservedValue = Util.ArrayInit(K.SizeAsInt, k => Dirichlet.Uniform(K.SizeAsInt)).ToArray();
        }

        public void SetPriors(Dirichlet ProbInitPriorParamObs, Dirichlet[] CPTTransPriorObs)
        {
            ProbInitPrior.ObservedValue = ProbInitPriorParamObs;
            CPTTransPrior.ObservedValue = CPTTransPriorObs;
        }

        public void SetParameters(double[] init, double[][] trans)
        {
            // fix parameters
            ProbInit.ObservedValue = Vector.FromArray(init);
            Vector[] v = new Vector[trans.Length];
            for (int i = 0; i < trans.Length; i++)
            {
                v[i] = Vector.FromArray(trans[i]);
            }
            CPTTrans.ObservedValue = v;
        }

        public void SetParametersToMAPEstimates()
        {
            Vector[] trans = new Vector[K.SizeAsInt];
            for (int i = 0; i < K.SizeAsInt; i++)
            {
                trans[i] = CPTTransPosterior[i].PseudoCount;
            }
            ProbInit.ObservedValue = ProbInitPosterior.PseudoCount;
            CPTTrans.ObservedValue = trans;
        }

        public void PrintPrior()
        {
            Console.WriteLine(ProbInitPrior.ObservedValue);
            for (int i = 0; i < K.SizeAsInt; i++)
            {
                Console.WriteLine("[" + i + "]" + CPTTransPrior.ObservedValue[i]);
            }
        }

        public void PrintParameters()
        {
            Console.WriteLine(ProbInit.ObservedValue);
            for (int i = 0; i < K.SizeAsInt; i++)
            {
                Console.WriteLine("[" + i + "]" + CPTTrans.ObservedValue[i]);
            }
        }

        public void PrintPosteriors()
        {
            Console.WriteLine(ProbInitPosterior);
            for (int i = 0; i < K.SizeAsInt; i++)
            {
                Console.WriteLine("[" + i + "]" + CPTTransPosterior[i]);
            }
        }

        public string HyperparametersToString()
        {
            string returnString = "";

            // init
            returnString += ProbInitPrior.ObservedValue.PseudoCount + "\n";
            // trans
            for (int i = 0; i < K.SizeAsInt; i++)
            {
                returnString += CPTTransPrior.ObservedValue[i].PseudoCount + "\n";
            }
            returnString += "\n";

            return returnString;
        }

        public override string ToString()
        {
            string output = "";
            Boolean PrintInit = true;
            Boolean PrintTrans = true;

            if (PrintInit)
            {
                output += "ProbInitPosterior" + ProbInitPosterior + "\n";
            }

            if (PrintTrans)
            {
                for (int i = 0; i < K.SizeAsInt; i++)
                {
                    output += "CPTTransPosterior[" + i + "]" + CPTTransPosterior[i] + "\n";
                }
            }

            return output;
        }
    }
}
