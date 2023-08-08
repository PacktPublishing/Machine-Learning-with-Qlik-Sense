dataset <- data.frame(id = 1:20, fact = letters[1:20])
set.seed(123)

sample <- dataset[sample(1:nrow(dataset), size=5), ]