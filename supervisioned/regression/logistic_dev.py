# Import LogisticRegression
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('Diabets.csv')
y = df["diabetes"].values
X = df.drop("diabetes", axis=1).values
# Instantiate the model
logreg = LogisticRegression()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)

logreg.fit(X_train, y_train)

# Predict probabilities
y_pred_probs = logreg.predict_proba(X_test)[:, 1]

print(y_pred_probs[:10])

# Import roc_curve
from sklearn.metrics import roc_curve

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)

plt.plot([0, 1], [0, 1], 'k--')

# Plot tpr against fpr
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Diabetes Prediction')
plt.show()

#Well done! The ROC curve is above the dotted line, so the model performs better than randomly guessing the class of each observation.