import pandas as pd
import numpy as np
import pyarrow.parquet as pq

# Data

# The data is from GSS surveys. 
# I got it from here: https://github.com/gsbDBI/ExperimentData/tree/master/Welfare/ProcessedData 
# The particular question we use is to test whether support for welfare spending 
# by the government is influenced by the working we use to describe it. 
# The question is "are we spending too much, too little or about the right amount
# on w ?"
# Where w can be either "welfare" or "assistance to the poor".

df = pd.read_csv("data/raw/welfarelabel.csv")

## Nas are filled with averages, I don't like that because, especially if there 
# are many missing values, this gives the model what seem very certain relationships 

# Refill nas using the column dummies

for var in df.columns:

    if var+"_miss" in df.columns:

        df.loc[df[var+"_miss"]==1, var] = np.nan

# Keep a subset of interpretable columns

cols = ["year", "id", "w","y",
        "wrkstat","marital","childs","age","educ",
        "sex","race","family16","res16","income","partyid","polviews"] 	

df = df[cols]

# There are some non-numeric values in numeric columns

df.loc[df["childs"] == "eight or more", "childs"] = "8"
df.loc[df["age"] == "89 or older", "age"] = "89"

df[["childs", "age"]] = df[["childs", "age"]].apply(pd.to_numeric)

# And random numbers in categorical columns

num_in_string = ["2.900022", "2.4010744", "22.382875", "1.8806012", "13.151037",
                 "1.9609094", "3.5032499", "10.55557", "2.8216343", "4.1220088"]

categ_cols = df.select_dtypes(include=object).columns.tolist()

# sub random numbers with nan

df[categ_cols] = df[categ_cols].replace(num_in_string, np.nan)

# year as object (I want it as dummies in the models, not as continuous variable)

df["year"] = df["year"].astype(object)

# save

df.to_parquet("data/interim/welfareprep.pq")
