import pandas as pd

# Exploring the census data

df = pd.read_csv("data/census.csv")

print(df.head())

print(df.columns.tolist())
