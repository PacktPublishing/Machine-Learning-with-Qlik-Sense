import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load the dataset
data = pd.DataFrame({
    'Animal': ['Dog', 'Cat', 'Parrot', 'Eagle', 'Snake'],
    'Has_fur': ['Yes', 'Yes', 'No', 'No', 'No'],
    'Has_feathers': ['No', 'No', 'Yes', 'Yes', 'No'],
    'Eats_meat': ['Yes', 'Yes', 'No', 'Yes', 'Yes'],
    'Classification': ['Mammal', 'Mammal', 'Bird', 'Bird', 'Reptile']
})

# Convert categorical variables to numeric using one-hot encoding
data_encoded = pd.get_dummies(data[['Has_fur', 'Has_feathers', 'Eats_meat']])

# Build the decision tree using the ID3 algorithm
clf = DecisionTreeClassifier(criterion='entropy', min_samples_split=2)
clf.fit(data_encoded, data['Classification'])

# Visualize the decision tree
plt.figure(figsize=(8, 6))
plot_tree(clf, feature_names=data_encoded.columns, class_names=np.unique(data['Classification']), filled=True)
plt.show()

# Test the model with new values
new_data = pd.DataFrame({
    'Has_fur_No': [0],
    'Has_fur_Yes': [1],
    'Has_feathers_No': [1],
    'Has_feathers_Yes': [0],
    'Eats_meat_No': [0],
    'Eats_meat_Yes': [1]
})

predicted = clf.predict(new_data) # Output: "Mammal"
print(predicted)
