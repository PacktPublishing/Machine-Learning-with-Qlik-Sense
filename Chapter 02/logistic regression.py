# Load the dataset
import pandas as pd
data = pd.read_csv("customer_data.csv")

# Fit the logistic regression model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(data[['age', 'income']], data['purchased'])

# Predict the probability of purchase for a new customer
new_customer = pd.DataFrame({'age': [35], 'income': [50000]})
prob_purchase = model.predict_proba(new_customer)[:, 1]

# Make a classification decision based on the probability
if prob_purchase >= 0.5:
    print("The customer is predicted to purchase the product.")
else:
    print("The customer is predicted not to purchase the product.")
