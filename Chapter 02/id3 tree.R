# Load the dataset
data <- data.frame(
  Animal = c("Dog", "Cat", "Parrot", "Eagle", "Snake"),
  Has_fur = c("Yes", "Yes", "No", "No", "No"),
  Has_feathers = c("No", "No", "Yes", "Yes", "No"),
  Eats_meat = c("Yes", "Yes", "No", "Yes", "Yes"),
  Classification = c("Mammal", "Mammal", "Bird", "Bird", "Reptile")
)

# Load the 'rpart' package for building decision trees
library(rpart)
library(rpart.plot)

# Build the decision tree using the ID3 algorithm
tree <- rpart(Classification ~ Has_fur + Has_feathers + Eats_meat, data = data, method = "class", control = rpart.control(minsplit = 1))

# Visualize the decision tree
rpart.plot(tree, type=5)

# Test the model with new values
new_data <- data.frame(
  Has_fur = "Yes",
  Has_feathers = "No",
  Eats_meat = "Yes"
)

predicted <- predict(tree, new_data, type = "class") # Output: "Mammal"
print(predicted)
