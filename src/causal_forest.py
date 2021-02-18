import pandas as pd
import numpy as np
import pyarrow.parquet as pq
# For step 1
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.proportion import power_proportions_2indep
# For step 2
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
# For step 3
from sklearn.model_selection import train_test_split
from econml.dml import CausalForestDML
from sklearn.preprocessing import OneHotEncoder
import plotly.express as px

# Load prepped data 

df = pd.read_parquet("data/interim/welfareprep.pq")

# Some descriptives

# Years of the surveys and number of respondants

df.groupby("year").size()

# Outcome and treatment
# y = 1 if respondent says 'too much'
# w is the treatment, w = 1 means individual was shown 'assistance to the poor'  

df[["y", "w"]].head(10)

def describe_by(var):

     dfescribe = df.groupby(var).size().to_frame("count")

     if len(var)==1 :
         dfescribe["prop"] = dfescribe / dfescribe.sum()
     else:
         dfescribe["prop"] = dfescribe / dfescribe.groupby(level = 0).sum()
     
     dfescribe = dfescribe.reset_index()

     return dfescribe

describe_y = describe_by("y")
describe_w = describe_by("w")

# Step 1: compare outcome conditional on treatment
# From this we can already see that a much smaller proportion of those who were 
# shown the variation said that 'too much'

df1 = describe_by(["w", "y"])
print(df1)

successes = np.array(df1[df1.y ==1]["count"])
trials = np.array(df1.groupby("w")["count"].sum())
props = successes / trials

diff = props[1] - props[0]
print(f"Difference between test and control: {diff:.2%}")

lift = props[1] / props[0]
print(f"Lift: {lift:.2%}")

_, pval = proportions_ztest(count=successes,
                            nobs=trials,
                            alternative='two-sided')
print(f"P-value of two-sided test: {pval:.3%}")

power = power_proportions_2indep(diff=abs(diff),
                                 prop2=props[1],
                                 nobs1=trials[0],
                                 ratio=trials[1]/trials[0],
                                 alpha=0.05,
                                 alternative='two-sided')
print(f"Power of the test: {power.power:.3%}")

# Step 2: ATE 

X = df[df.columns.difference(["y", "id"])]

y = np.array(df["y"])

# Prep X

numeric_cols = ['age', 'childs', 'educ']

scaler=StandardScaler()
X_numeric_cols = pd.DataFrame(scaler.fit_transform(X[numeric_cols]), 
                               columns = numeric_cols)

categ_cols = X.select_dtypes(include=object).columns.tolist()
X_categ_cols = pd.get_dummies(X[categ_cols], drop_first=True)

X = X_numeric_cols.merge(X_categ_cols, left_index=True, right_index=True)

X["w"] = df["w"]

# add a constant

X = sm.add_constant(X)

# fit the model

model = sm.OLS(y,X)
results = model.fit()

# get results

res = pd.DataFrame(results.summary().tables[1])

# tidy  up results

res.columns = res.iloc[0]
res = res[1:]
res.rename(columns={ res.columns[0]: "feature" }, inplace = True)

# print ATE

print(res[70:71])

# Step 3: Double ML


# Step 4 : Causal forest

#X = df[df.columns.difference(["y", "id"])]
y = df["y"]

x_numeric_cols = df[['age', 'childs', 'educ']]

categ_cols = X.select_dtypes(include=object).columns.tolist()
X_categ_cols = pd.get_dummies(X[categ_cols], drop_first=True)

X = X_numeric_cols.merge(X_categ_cols, left_index=True, right_index=True)

X["w"] = df["w"]

# Pipeline
# as we're doing linear regression, we need to scale numeric feature and 
# one-hot encode categorical features

numeric_cols =  X.select_dtypes(include=[float, int]).columns.tolist()
numeric_transformer = Pipeline(steps = [('scaler', StandardScaler())])

categ_cols = X.select_dtypes(include=object).columns.tolist()
categ_transformer = Pipeline(steps= [('onehot', OneHotEncoder(drop="first"))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categ_transformer, categ_cols)])

pipe = Pipeline(steps=[("preprocessor", preprocessor),
                       ("regression", LinearRegression())])


# Prep data

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.33, 
                                                    random_state=42)

w_train = X_train["w"]
w_test = X_test["w"]

X_train = X_train.drop(["w"], axis=1)
X_test = X_test.drop(["w"], axis=1)

# fit causal forest and predict

est = CausalForestDML()

est.fit(Y=y_train, T=w_train, X=X_train)

# re-calculate the ATE

ATE = est.ate(X_test)

# estimate CATE

CATE = est.effect(X_test)

# CI
# lb = lower bound 
# up = upper bound

lb, ub = est.effect_interval(X_test, alpha=0.05)

# merge CATE to confidence interval

CATE = pd.DataFrame(CATE)

CATE["lb"] = lb
CATE["ub"] = ub

CATE = CATE.rename(columns={0: "treatment_effect"})

# plot 

fig = px.histogram(CATE, x="treatment_effect")
fig.show()
fig.write_image(fig, "out/hist_het_effect.png")