import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Create a sample DataFrame
data = {
    'ID': [1, 2, 3, 4, 5],
    'Category': ['Mammal', 'Bird', 'Mammal', 'Reptile', 'Reptile'],
    'Carnivore': ['No', 'No', 'Yes', 'No', 'Yes']
}
df = pd.DataFrame(data)

# Perform one-hot encoding on the 'Category' column
encoder = OneHotEncoder(sparse=False)
category_encoded = encoder.fit_transform(df[['Category']])

# Create new columns for each category
category_names = encoder.get_feature_names_out(['Category'])
df[category_names] = category_encoded

# Drop the original 'Category' column
df.drop('Category', axis=1, inplace=True)

# Print the updated DataFrame
print(df)
