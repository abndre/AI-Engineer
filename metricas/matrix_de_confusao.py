# Import confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
# Import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier 
import pandas as pd
from sklearn.model_selection import train_test_split


churn_df = pd.read_csv('crurn.csv')

y = churn_df["churn"].values
X = churn_df[["account_length", "customer_service_calls"]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


knn = KNeighborsClassifier(n_neighbors=6)

# Fit the model to the training data
knn.fit(X_train, y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))