# causalML# Causal ML

## The theory

The classical approach to estimating treatment effect is  **linear regression** (OLS). Given the outcome $Y_i$, the treatment $W_i$ and covariates $X_i$, fit:

$Y_i = W_i\tau + X_i\beta + \epsilon_i$

The coefficient $\hat{\tau_{OLS}}$ (notice no $_i$) is the **estimated treatment effect**. 

$ATE: \tau = E [(Y_i|D_i = 1) − (Y_i|D_i = 0)]$

The assumptions here:

* There are **no unmeasured confounders**: $W_i$ is random once we control for $X_i$
* The relationship betweeen $X_i$ and $Y_i$ is **linear**
* The treatment is effect $\tau$ is **constant** (in reality, we often know this is not the case, but we are satisfied with estimating the average treatment effect, ATE)

We can relax the second and third assumptions. The relationship does not have to be linear (unless we think it is!) and the treatment effect does not have to be constant.

### Double Machine learning

To relax the assumption of linear relationship between $X_i$ and $Y_i$ we can use the partially linear model.

$Y_i = W_i\tau + f(X) + \epsilon_i$

ML models are really good at prediction problems, so we could try to predict $Y$ using $W$ and $X$. Using a regularised linear regression gives good estimates of $Y$, but biased estiamtes of the parameters, including $\tau$ which is the parameter of interest. This bias is similar to the omitted variable bias, and it's called regularosation bias. 

Instead, to address the regularisation bias we orthogonalize, i.e. we solve two proediction pronblems instead of one:
1. $Y$ using $X$
2. $D$ using $X$
You can use any ML method you want here, does not have to be linear. 
3. then you calucate teh residuals from both regressions:

$\hat{Z} = Y - \widehat{E[Y|X]}$ 

$\hat{D} = W - \widehat{E[W|X]}$ This is the conditional probability of treatment

And regress $\hat{Z}$ on $\hat{D}$ to get $\hat{\tau}$. 

This gives a much less biased (i.e. close to be centered at zero) estimate of the parameter $\tau$ (which compared to the naive approach has larger variance). We can also calcualte a confidence interval. It can actually be proven that this estimator is $\sqrt{n}$ consistent and approximately centered normal, as long as the models in step 1 and 2 converge quickly enough (which good ML models will do). 

The orthoginalization approach partials out the association between $X$ and $D$ and between $Y$ and $X$ conditional on $D$. This is similar, and in a way is a generalisation of, the Frisch-Waugh-Lovell Theorem.

Double ML also have another form of bias, due to overfitting (i.e. modelling the noise from our data). To address this, we rely on sample splitting and cross-fitting. This reduces complexity and removes the bias from overfitting. Turns out you can do it without sample splitting, but it's much more complicated.

* **sample-splitting** to avoid using the same indiviudal observations to estimate the functions in steps 1 and 2 and $\hat{\tau}$
* **cross-fitting**, within the training data, split the data into 2 folds, use a fold to fit steps 1 and 2, and get an estimate of $\hat{\tau_1}$ from the other fold. Do the same thing for the second fold to get a second estimate $\hat{\tau_2}$, and average the two estimates of $\hat{\tau}$.

#### Frisch-Waugh-Lovell Theorem 

The theorem says you can recover the OLS estimate for a parameter in linear regression using residuals-on-residuals OLS regression. This is:
 
1. Get the resudual from the linear regression of $W_i$ on $X_i$ - this 'purges' the effects of the covariates on the treatment 
2. Get the residual from the linear regression of $Y_i$ on $X_i$ - this 'purges' the effects of the covariates on the outcome
3. Estimate $\tau$ by residual-on-residual regression: regress the residual from step 2 on the residual from step 1. This estimates the effect of $W_i$ on $Y_i$ 'without the effects 'cleaned' of the effects of the effects of $X_i$ on them. 

The Frisch-Waugh-Lovell Theorem states that $hat{\tau}$ obtained in this way is the same as that estimated via simple OLS. 

TWL theorem is a way to orthogonalize. Robinson's innovation on this: you can use kernel regression at steps 1 and 2. In step 3 you still use a rlinear regression of the residuals on the residuals.

### Heterogeneity in treatment effect

We can relax the assumption of constant treatment effect (or our acceptance of an estimate of the ATE). This does not mean we can estimate individual treatment effects, but the conditional average treatment effect (CATE). Given some observables $x$, we can obtain an estimate of the average treatment effect for those with those observables. We are interested in how treatment effects vary with the observed covariates $X_i$.

$CATE: \tau(x) = E [(Y_i|D_i = 1, X_i = x) − (Y_i|D_i = 0, X_i = x)]$

Causal forests estimate treatment heterogeneity by fitting local versions of a partially linear model. It uses the structure of a random forest to define a relevant notion of a neighborhood. 

It uses Orthoginalisation to partial out the treatment propensity and simplify the problem. This means that first, causal forest estimates $e(x) = E[D|X=x]$ and $m(x) = E[Y|X=x]$ though separate regression forests. Then calculates the residual treatment $D - e(x)$ and residual outcome $Y - m(x)$, and trains a random forest on these residuals. The outcome is a treatment effect function $\tau(x)$. 

#### Splitting

When choosing a split, the algorithm seeks to maximize the difference in treatment effect between the two nodes. So it computes the treatment effect of each side of the potential split, assuming homogeneous leaf-effects, and chooses the split that maximises the weighted (by $n$) difference in the treatment effects

#### Honesty

Causal forest uses 'honest trees', to reduce bias in the preediction, by using different observations for constructing the tree and making predictions from it. 
In honest forests observations are randomly split into two halfs. Only the first is used for splitting. The second is then used to populate the tree’s leaf nodes: each new example is ‘pushed down’ the tree, and added to the leaf in which it falls. In a sense, the leaf nodes are ‘repopulated’ after splitting using a fresh set of examples.

## Data

I used the data from the GSS surveys. I got it from [Stanford's Digital Business Initiative repo](https://github.com/gsbDBI/ExperimentData/tree/master/Welfare/ProcessedData).

The particular question I use is to test whether support for welfare spending by the government is influenced by the working we use to describe it.

## Packages

I'm using Microsoft's EconML version 0.9.0b1. At time of writing, this version is only available on git:

```
pip install -e git+https://github.com/microsoft/EconML.git@master#egg=econml
```
You need Python 3.8 to get it to work. 

# Readings

* Wager' S.'s [Causal Inference lecture notes](https://web.stanford.edu/~swager/stats361.pdf)
* On partially linear models and Double ML:
  - [Victor Chernozhukov video lecture at UChicago](https://www.youtube.com/watch?v=eHOjmyoPCFU&ab_channel=BeckerFriedmanInstituteatUChicago-BFI) 
  - [these slides from Princeton Uni](https://scholar.princeton.edu/sites/default/files/bstewart/files/chern.handout.pdf) are a easy intro
* On causal forest the [reference in the grf package](https://grf-labs.github.io/grf/REFERENCE.html#table-of-contents-1) is great. 
