# Import LinearRegression
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
import numpy as np

sales_df = pd.read_csv("sales.csv")

X = sales_df.drop("sales", axis=1).values
y = sales_df["sales"].values

# Import the necessary modules


# Create a KFold object
kf = KFold(n_splits=6, shuffle=True, random_state=5)

reg = LinearRegression()

# Compute 6-fold cross-validation scores
cv_scores = cross_val_score(reg, X, y, cv=kf)

# Print scores
print(cv_scores)

cv_results = cv_scores

# Print the mean
print(np.mean(cv_results))

# Print the standard deviation
print(np.std(cv_results))

# Print the 95% confidence interval
print(np.quantile(cv_results, [0.025, 0.975]))