# Bayesian Optimization and Hyperparameter Tuning
[Bayesian optimization](https://en.wikipedia.org/wiki/Bayesian_optimization) is a strategy for optimizing black-box functions. Generally, Bayesian optimization is useful when the function you want to optimize is not differentiable, or each function evaluation is expensive. As such, it is a natural candidate for hyperparameter tuning.

## Bayesian optimization
The outline of Bayesian optimization is as follows:

- Compute the value of your black-box function at a point
- Store this point and function value in your history of points previously sampled
- Use this history to decide what point to inspect next
- Repeat

As such, Bayesian optimization is a 'sequential' strategy: you compute function values at points one after the other. In the case of hyperparameter tuning, this is often referred to as Sequential Model Based Optimization (SMBO).

We can restate this general strategy more precisely: start by placing a prior distribution over your function (the prior distribution can be uniform). Use the prior distribution to choose a point to sample - ideally, the goal is to sample a point with a high probability of maximizing (or minimizing) your function. Compute the function value at this point, and incorporate this data into your prior to create a posterior distribution. Begin again: your posterior is your new prior. 

There are many algorithms for how to create distributions, and how to choose what point to sample (see references).

## Bayesian optimization for hyperparameter tuning

In the case of hyperparameter tuning, the 'black-box function' generally consists of two steps:

1. train a model
2. compute a performance metric

This black-box function takes values of hyperparameters as inputs, and returns a performance metric. The performance metric can be anything (f1-score, AUC-ROC, accuracy, etc.), and it can take into account penalties for undesirable features (training time, evaluation time, memory use, etc). 

## Code Examples
- `Examples/Affairs.ipynb` shows how to use hyperopt with a tree of parzen estimators for hyperparameter tuning, and for model selection. 

## References
- [Algorithms for Hyper-Parameter Optimization](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)
  - Authors: Bergstra, Bardenet, Bengio, Kégl
- [Auto-WEKA: Combined Selection and Hyperparameter Optimization of Classification Algorithms](https://arxiv.org/pdf/1208.3719.pdf)
  - Authors: Thornton, Hutter, Hoos, Brown
- [Practical Bayesian Optimization of Machine Learning Algorithms](https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf)
  - Authors: Snoek, Larochelle, Adams
- [Bayesian Optimization on Wikipedia](https://en.wikipedia.org/wiki/Bayesian_optimization)
- [SigOpt](https://sigopt.com/)
