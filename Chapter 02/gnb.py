from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

# Load the iris dataset from the datasets package
iris = load_iris()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=123)

# Train a Gaussian Naive Bayes model on the training data
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)

# Check the accuracy of the predictions
print(classification_report(y_test, predictions))
