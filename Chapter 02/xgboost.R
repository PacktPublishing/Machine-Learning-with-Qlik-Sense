# Load the required libraries
library(xgboost)
library(caret)

# Load the iris dataset
data(iris)

# Split the dataset into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(iris$Species, p = 0.8, list = FALSE)
train <- iris[trainIndex, ]
test <- iris[-trainIndex, ]

# Convert the species column to a factor
train$Species <- as.factor(train$Species)
test$Species <- as.factor(test$Species)

# Map factor levels to integer labels starting from 0
train$label <- as.integer(train$Species) - 1
test$label <- as.integer(test$Species) - 1

# Train the XGBoost model
xgb_model <- xgboost(data = as.matrix(train[, 1:4]), 
                     label = train$label, 
                     nrounds = 10, 
                     objective = "multi:softmax", 
                     num_class = 3, 
                     eval_metric = "mlogloss")

# Make predictions on the test set
predictions <- predict(xgb_model, as.matrix(test[, 1:4]))

# Map integer labels back to factor levels
predictions <- factor(predictions, levels = 0:2, labels = levels(iris$Species))

# Evaluate the model performance
confusionMatrix(predictions, test$Species)
