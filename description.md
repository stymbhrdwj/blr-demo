In Bayesian linear regression, we typically consider the model

| Prior | Likelihood | Posterior | Posterior Predictive |
|:-----------------------------------------------------:|:------------------------------------------------------:|:------------------------------------------------------:|:------------------------------------------------------:|
| $p(\boldsymbol{\theta}) = \mathcal{N}(\mathbf{m}_0, \mathbf{S}_0)$ | $p(y \| x, \boldsymbol{\theta}) = \mathcal{N}(\boldsymbol{\theta}^T x, \sigma^2)$ | $p(\boldsymbol{\theta} \mid \mathcal{X}, \mathcal{Y})=\mathcal{N}\left(\boldsymbol{\theta} \mid \boldsymbol{m}_N, \boldsymbol{S}_N\right)$ | $p\left(y_* \mid \mathcal{X}, \mathcal{Y}, \boldsymbol{x}_*\right)=\mathcal{N}\left(y_* \mid \boldsymbol{\phi}^{\top}\left(\boldsymbol{x}_*\right) \boldsymbol{m}_N, \boldsymbol{\phi}^{\top}\left(\boldsymbol{x}_*\right) \boldsymbol{S}_N \boldsymbol{\phi}\left(\boldsymbol{x}_*\right)+\sigma^2\right)$ |


where $\mathbf{m}_0$ and $\mathbf{S}_0$ are the mean and covariance of the prior distribution, respectively. $\boldsymbol{m}_N=\boldsymbol{S}_N\left(\boldsymbol{S}_0^{-1} \boldsymbol{m}_0+\sigma^{-2} \boldsymbol{\Phi}^{\top} \boldsymbol{y}\right)$ and $\boldsymbol{S}_N=\left(\boldsymbol{S}_0^{-1}+\sigma^{-2} \boldsymbol{\Phi}^{\top} \boldsymbol{\Phi}\right)^{-1}$ are the mean and covariance of the posterior distribution, respectively.

In this app, we allow the user to change the prior distribution from a Gaussian to a Laplace distribution or Uniform distribution. Likewise for the Likelihood function. The user can also select from 3 different datasets. Since the distributions don't always form a conjugate pair, we rely on ```numpyro``` library for it's fast implementation of MCMC using the NUTS algorithm. 

References:
1. Bayesian regression using numpyro—Numpyro documentation. (n.d.). Retrieved November 15, 2023, from https://num.pyro.ai/en/stable/tutorials/bayesian_regression.html
2. Mathematics for machine learning. (n.d.). Mathematics for Machine Learning. Retrieved November 15, 2023, from https://mml-book.com/