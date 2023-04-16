# Load the housing dataset from CSV file
import pandas as pd
url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
housing = pd.read_csv(url)

# Encode categorical features numerically
housing = pd.get_dummies(housing, columns=['ocean_proximity'])

# Remove rows with missing values
housing.dropna(inplace=True)

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(housing.drop(columns=['median_house_value']), housing['median_house_value'], test_size=0.2, random_state=123)

# Fit a Lasso regression model to the training set
from sklearn.linear_model import LassoCV
model = LassoCV(alphas=[0.001, 0.01, 0.1, 1], cv=5)
model.fit(X_train, y_train)

# Evaluate the model on the testing set
from sklearn.metrics import mean_squared_error
predictions = model.predict(X_test)
rmse = mean_squared_error(y_test, predictions, squared=False)
print(f"RMSE: {rmse}")
