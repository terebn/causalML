import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import plotly.express as px

# For processing data 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# For the model(s)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from econml.dml import CausalForestDML
from econml.score import RScorer

# Load cleaned data 

df = pd.read_parquet("../data/interim/welfareprep.pq")

# Prep data

y = df["y"]
w = df["w"]
X = df[df.columns.difference(["y", "id", "w"])]

# For the linear model we need numeric to be standardised
# categorical are turned into dummies (including a dummy for missing data)

numeric_cols = ['age', 'childs', 'educ']

scaler=StandardScaler()
X_numeric_cols = pd.DataFrame(scaler.fit_transform(X[numeric_cols]), 
                               columns = numeric_cols)

categ_cols = X.select_dtypes(include=object).columns.tolist()
X_categ_cols = pd.get_dummies(X[categ_cols], drop_first=True)

X = X_numeric_cols.merge(X_categ_cols, left_index=True, right_index=True)

# Split y, w and X in train and test 

X_train, X_test, w_train, w_test, y_train, y_test = train_test_split(X, w, y,
                                                                     test_size=.3,
                                                                     random_state=123)

# Fit causal forest

# We want to optimise the two first models (Y|X and W|X).
# I use Extra TRee for both with some hyperparameter tuning. 
# the Y|X model automatically uses regression, but in this case i have a 0-1 outcome 

chosen_params = {"max_depth": [10, 30, 50],
                 "min_samples_leaf": [5, 10, 20],
                 "max_features": ["sqrt", 0.25, 0.5]}

first_stage = lambda: GridSearchCV(estimator=ExtraTreesClassifier(),
                                   param_grid=chosen_params,
                                   cv=5)

model_y = first_stage().fit(X_train, y_train).best_estimator_
model_w = first_stage().fit(X_train, w_train).best_estimator_

# Third step is to use the two residual models to estimate the treatment effect 
# with causal forest.
# As split criterion I use 'het', the one from Athey, Tibshirani, Wager 
# described in https://arxiv.org/pdf/1610.01271.pdf 

est = CausalForestDML(model_y = model_y,
                      model_t = model_w,
                      discrete_treatment=False,
                      cv=3,
                      n_estimators=100,
                      criterion="het",
                      random_state=234) 

est.fit(Y=y_train, T=w_train, X=X_train)

# Measure performance of the residual models with RScorer
# Returns an analogue of the R-square score for regression::
#        score = 1 - loss(cate) / base_loss
# This corresponds to the extra variance of the outcome explained by introducing heterogeneity
# in the effect as captured by the cate model, as opposed to always predicting a constant effect.
# A negative score, means that the cate model performs even worse than a constant effect model
# and hints at overfitting during training of the cate model.

scorer = RScorer(model_y = model_y,
                 model_t = model_w,
                 discrete_treatment=False,
                 cv=3,
                 random_state=234)

scorer.fit(y_test, w_test, X_test)

rscore = scorer.score(est)

print(f"RScore: {rscore:.2%}")

# Use the model to estimate treatment effects

# estimate the ATE

ATE = est.ate(X_test)

print(f"Average treatment effect: {ATE:.2%}")

# estimate CATE

CATE = est.effect(X_test)

# CATE 95% confidence interval

lb, ub = est.effect_interval(X_test, alpha=0.05)

# merge CATE to confidence interval

CATE = pd.DataFrame(CATE)

CATE["ci_lower"] = lb
CATE["ci_upper"] = ub

CATE = CATE.rename(columns={0: "treatment_effect"})

# there's also a good CATE df summary function

res = est.effect_inference(X_test)

CATE2 = res.summary_frame()

print(f"CATE:{CATE2.head(10)}")

# and a good population summary function (which also has ATE)

pop_summary = res.population_summary()

# Plot of the distribution of treatment effects

fig = px.histogram(CATE, x="treatment_effect")
fig.add_vline(x=ATE, line_width=2, line_color="black", annotation_text="ATE", 
              annotation_position="top right")
fig.show()

# plot dist of estimates
# see: https://github.com/microsoft/EconML/blob/master/notebooks/Causal%20Forest%20and%20Orthogonal%20Random%20Forest%20Examples.ipynb

import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))
plt.title("CausalForest")
plt.plot(X_test["educ"], CATE["treatment_effect"], label='CF estimate')
plt.fill_between(X_test["educ"], CATE["ci_lower"], CATE["ci_upper"], label="95% CI", alpha=0.3)
plt.ylabel("Treatment Effect")
plt.xlabel("x")
plt.legend()
plt.show()

