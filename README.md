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
- `Examples/Hyperopt_Predict_Affairs.ipynb` shows how to use hyperopt with a tree of parzen estimators for hyperparameter tuning, and for model selection. 
- `Examples/Scikit-optimize_Predict_Affairs.ipynb` shows how to use scikit-optimize with Gaussian processes and trees for hyperparameter tuning (using expected improvement). 

## References
### Papers

- [Algorithms for Hyper-Parameter Optimization](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)
  - Authors: Bergstra, Bardenet, Bengio, Kégl
- [Automatic Model Construction with Gaussian Processes](http://www.cs.toronto.edu/~duvenaud/thesis.pdf)
  - Author: Duvenaud
- [Auto-WEKA: Combined Selection and Hyperparameter Optimization of Classification Algorithms](https://arxiv.org/pdf/1208.3719.pdf)
  - Authors: Thornton, Hutter, Hoos, Brown
- [Bayesian Hyperparameter Optimization for Ensemble Learning](https://www.arxiv.org/abs/1605.06394)
  - Authors: Lévesque, Gagné, Sabourin
- [Practical Bayesian Optimization of Machine Learning Algorithms](https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf)
  - Authors: Snoek, Larochelle, Adams
- [Sequential Model-Based Optimization for General Algorithm Configuration](http://www.cs.ubc.ca/~hutter/papers/11-LION5-SMAC.pdf)
  - Authors: Hutter, Hoos, Leyton-Brown
- [Towards an Empirical Foundation for Assessing Bayesian Optimization of Hyperparameters](https://www.cs.ubc.ca/~hoos/Publ/EggEtAl13.pdf)
  - Authors: Eggensperger, Feurer, Hutter, Bergstra, Snoek, Hoos, Leyton-Brown
- [Modular mechanisms for Bayesian optimization](http://mlg.eng.cam.ac.uk/hoffmanm/papers/hoffman:2014b.pdf)
  - Authors: Hoffman, Shahriari

### Books
- [Gaussian Processes for Machine Learning](http://www.gaussianprocess.org/gpml/)
   - Authors: Rasmussen, Williams

### Software (list curated primarily for Python)
- [Hyperopt](https://github.com/hyperopt/hyperopt)
- [Scikit-Optimize](https://scikit-optimize.github.io/)
- [Spearmint](https://github.com/JasperSnoek/spearmint)
- [MOE](https://github.com/Yelp/MOE)
- [SigOpt](https://sigopt.com/)
- [GPyOpt](https://sheffieldml.github.io/GPyOpt/)
- [BayesOpt](http://rmcantin.bitbucket.org/html/)

### Other 
- [Bayesian Optimization on Wikipedia](https://en.wikipedia.org/wiki/Bayesian_optimization)
