# Load the dataset
data <- read.csv('C:\\Users\\bip\\OneDrive - QlikTech Inc\\Desktop\\Machine Learning with Qlik Sense\\Chapters\\Code\\Chapter 02\\customer_data.csv')

# Fit the logistic regression model
model <- glm(purchased ~ age + income, data = data, family = binomial())

# Predict the probability of purchase for a new customer
new_customer <- data.frame(age = 35, income = 50000)
prob_purchase <- predict(model, new_customer, type = "response")

# Make a classification decision based on the probability
if (prob_purchase >= 0.5) {
  print("The customer is predicted to purchase the product.")
} else {
  print("The customer is predicted not to purchase the product.")
}
