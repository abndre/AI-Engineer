# Import LinearRegression
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
sales_df = pd.read_csv("sales.csv")
# Create X from the radio column's values
X = sales_df["Radio"].values

# Create y from the sales column's values
y = sales_df["Sales"].values

# Reshape X
X = X.reshape(-1, 1)
# Create the model
reg = LinearRegression()

# Fit the model to the data
reg.fit(X,y)

# Make predictions
predictions = reg.predict(X)

print(predictions[:5])


# Create scatter plot
plt.scatter(X, y, color="blue")

# Create line plot
plt.plot(X, predictions, color="red")
plt.xlabel("Radio Expenditure ($)")
plt.ylabel("Sales ($)")

# Display the plot
plt.show()