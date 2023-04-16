import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Create a sample DataFrame
data = {
    'ID': [1, 2, 3, 4, 5],
    'Age': [25, 30, 35, 40, 45],
    'Income': [50000, 60000, 70000, 80000, 90000]
}
df = pd.DataFrame(data)

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Perform feature scaling on 'Age' and 'Income' columns
df[['Age', 'Income']] = scaler.fit_transform(df[['Age', 'Income']])

# Print the updated DataFrame
print(df)
