import pandas as pd
import numpy as np
import pyarrow.parquet as pq

# For hypothesis testing and power analysis
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.proportion import power_proportions_2indep

# For data prepping and OLS model
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# Load cleaned data 

df = pd.read_parquet("../data/interim/welfareprep.pq")

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

# Compare outcome conditional on treatment

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

# Calculate the Average Treatment effect (ATE)

X = df[df.columns.difference(["y", "id", "w"])]
y = np.array(df["y"])

# For the linear model we need numeric to be standardised
# categorical are turned into dummies (including a dummy for missing data)

numeric_cols = ['age', 'childs', 'educ']

scaler=StandardScaler()
X_numeric_cols = pd.DataFrame(scaler.fit_transform(X[numeric_cols]), 
                               columns = numeric_cols)

categ_cols = X.select_dtypes(include=object).columns.tolist()
X_categ_cols = pd.get_dummies(X[categ_cols], drop_first=True)

X = X_numeric_cols.merge(X_categ_cols, left_index=True, right_index=True)

X["w"] = df["w"]

# add a constant (needed in stats model)

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
