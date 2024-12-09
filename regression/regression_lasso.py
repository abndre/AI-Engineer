# Import Lasso
from sklearn.linear_model import Lasso
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
sales_df = pd.read_csv("sales.csv")

X = sales_df.drop("sales", axis=1).values
y = sales_df["sales"].values
# Instantiate a lasso regression model
lasso = Lasso(alpha=0.3)

# Fit the model to the data
lasso.fit(X,y)

# Compute and print the coefficients
lasso_coef = lasso.coef_
print(lasso_coef)
plt.bar(sales_columns, lasso_coef)
plt.xticks(rotation=45)
plt.show()