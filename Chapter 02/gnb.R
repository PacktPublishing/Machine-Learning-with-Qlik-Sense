library(e1071) # Load the e1071 package for the naiveBayes function

# Load the iris dataset from the datasets package
data(iris)

# Split the data into training and test sets
set.seed(123)
trainIndex <- sample(nrow(iris), 0.7 * nrow(iris))
train <- iris[trainIndex, ]
test <- iris[-trainIndex, ]

# Train a Gaussian Naive Bayes model on the training data
model <- naiveBayes(Species ~ ., data = train)

# Make predictions on the test data
predictions <- predict(model, test)

# Check the accuracy of the predictions
cfm <- table(predictions, test$Species)
print(cfm)