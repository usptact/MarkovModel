# Markov Model (Bayesian Markov Chain with Infer.NET)

This project fits a finite-state Markov chain to an observed sequence of discrete states using Bayesian inference with Infer.NET.

- **What it models**: a K-state first-order Markov chain over hidden states that, in this example, are directly observed. The model learns the initial state probabilities and the transition matrix from data under Dirichlet priors.

- **Model structure**:

```
   initial ~ Dirichlet(K)
   for each row k in 0..K-1:
       trans[k] ~ Dirichlet(K)

   s[0] ~ Categorical(initial)
   for t = 1..T-1:
       s[t] ~ Categorical(trans[s[t-1]])
```

- **ASCII diagram** (example K=3):

```
   [S0] ---> [S1] ---> [S2] ---> [S0] ---> ...
     ^         |          ^
     |         v          |
     +---------+----------+

initial: P(s0 in {S0,S1,S2})
transitions: P(next | current)
```

## Requirements

- .NET 8 SDK

## Build and Run

```bash
dotnet restore
dotnet build -c Release
dotnet run -c Release
```

The program will:
- simulate a length-T state sequence from randomly sampled parameters
- infer posteriors over the initial distribution and transition rows
- print the model likelihood and posterior summaries

## Notes

- Inference uses Expectation Propagation via `InferenceEngine` from Infer.NET.
- Priors are set to be uninformative (uniform Dirichlet) by default; see `SetUninformedPriors()`.
