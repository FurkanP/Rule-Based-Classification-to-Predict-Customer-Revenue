# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Load the dataset
df = pd.read_csv('/kaggle/input/persona/persona.csv')
df.head()

# **1. Overview of the Data**

# Display general information about the dataset
print("##################### Shape #####################")
print(df.shape)
print("##################### Types #####################")
print(df.dtypes)
print("##################### Head #####################")
print(df.head())
print("##################### Tail #####################")
print(df.tail())
print("##################### NA #####################")
print(df.isnull().sum())
print("##################### Quantiles #####################")
print(df.describe([0, 0.25, 0.50, 0.75, 1]).T)
print("##################### Columns #####################")
print(df.columns)

# **Exploratory Data Analysis**

# Number of unique values in the SOURCE variable and what these values are
print(df["SOURCE"].nunique())  # 2 unique values
print(df["SOURCE"].unique())  # ['android' 'ios']

# Number of unique values in the PRICE variable and what these values are
print(df["PRICE"].nunique())  # 6 unique values
print(df["PRICE"].unique())  # [39 49 29 19 59  9]

# Number of sales made at each price
df["PRICE"].value_counts()

# Number of sales made from each country
df["COUNTRY"].value_counts()

# Total sales amount by country
df.groupby("COUNTRY").agg({"PRICE": "sum"})

# Average sales price by COUNTRY and SOURCE
df.groupby(["COUNTRY", "SOURCE"]).agg({"PRICE": "mean"})

# **2. Data Manipulation**

# Create a new dataframe (agg_df) by aggregating the data based on COUNTRY, SOURCE, SEX, and AGE
agg_df = df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"})
agg_df = agg_df.sort_values("PRICE", ascending=True)
agg_df = agg_df.reset_index()
agg_df.head()

# Convert the AGE variable into categorical bins
agg_df["range_age"] = pd.cut(agg_df["AGE"], [0,18,23,30,40,70], labels=["0_18", "19_23", "24_30", "31_40", "41_70"])
agg_df.head()

# Create the customer_level_based variable by concatenating COUNTRY, SOURCE, SEX, and range_age
agg_df["customer_level_based"] = [(agg_df.loc[i, "COUNTRY"] + "_" +
                                  agg_df.loc[i, "SOURCE"] + "_" +
                                  agg_df.loc[i, "SEX"] + "_" +
                                  agg_df.loc[i, "range_age"]).upper() for i in range(len(agg_df))]
agg_df.head()

# Check the unique values of customer_level_based
agg_df["customer_level_based"].value_counts()

# Group by customer_level_based and calculate the mean PRICE
agg_df = agg_df.groupby("customer_level_based").agg({"PRICE": "mean"})
agg_df

# Segment the data into 4 segments based on PRICE
agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=['D', 'C', 'B', 'A'])
agg_df = agg_df.reset_index()
agg_df.head()

# Display the average, maximum, and total PRICE for each segment
agg_df.groupby("SEGMENT").agg({"PRICE": ["mean", "max", "sum"]})

# **3. Potential Revenue Prediction**

# Define a new user
new_user = "TUR_ANDROID_FEMALE_31_40"
print(agg_df[agg_df["customer_level_based"] == new_user].loc[:, "PRICE"].mean())
print(agg_df[agg_df["customer_level_based"] == new_user].loc[:, "SEGMENT"].mode())
print("-----------------------------------------------------")

# Segment: A
# Expected earning income: 41.83

# Function to estimate segment and potential earning income based on user information
def estimate(nation, os, sex, age):
    if age <= 18:
        user = nation.upper() + "_" + os.upper() + "_" + sex.upper() + "_0_18"
    elif (age > 18) & (age < 24):
        user = nation.upper() + "_" + os.upper() + "_" + sex.upper() + "_19_23"
    elif (age >= 24) & (age <= 30):
        user = nation.upper() + "_" + os.upper() + "_" + sex.upper() + "_24_30"
    elif (age > 30) & (age <= 40):
        user = nation.upper() + "_" + os.upper() + "_" + sex.upper() + "_31_40"
    else:
        user = nation.upper() + "_" + os.upper() + "_" + sex.upper() + "_41_70"

    print("Segment: ", agg_df[agg_df["customer_level_based"] == user].loc[:, "SEGMENT"].mode()[0])
    print("Expected earning income: ", agg_df[agg_df["customer_level_based"] == user].loc[:, "PRICE"].mean())

# Example usage of the estimate function
estimate("tur", "android", "female", 33)
