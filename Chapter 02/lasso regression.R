# Load the housing dataset from CSV file
url <- "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
housing <- read.csv(url)

# Remove rows with missing values
housing <- na.omit(housing)

# Split the dataset into training and testing sets
set.seed(123)
train_index <- sample(nrow(housing), nrow(housing) * 0.8)
train <- housing[train_index, ]
test <- housing[-train_index, ]

# Fit a Lasso regression model to the training set
library(glmnet)
x <- model.matrix(median_house_value ~ ., train)[,-1]
y <- train$median_house_value
model <- cv.glmnet(x, y, alpha = 1)

# Evaluate the model on the testing set
x_test <- model.matrix(median_house_value ~ ., test)[,-1]
y_test <- test$median_house_value
predictions <- predict(model, newx = x_test)
rmse <- sqrt(mean((predictions - y_test)^2))
print(paste0("RMSE: ", rmse))
