# Load the required libraries
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load the iris dataset
iris = load_iris()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=123)

# Train the XGBoost model
xgb_model = xgb.XGBClassifier(objective="multi:softmax", n_estimators=10, seed=123)
xgb_model.fit(X_train, y_train)

# Make predictions on the test set
predictions = xgb_model.predict(X_test)

# Evaluate the model performance
print(classification_report(y_test, predictions))
