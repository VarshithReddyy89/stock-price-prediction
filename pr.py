import zipfile
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

zip_file_path = 'C:/Users/varsh/OneDrive/Desktop/MInor Project(Corizo)/Samsung electronics dataset.zip'
extracted_path = 'C:/Users/varsh/OneDrive/Desktop/MInor Project(Corizo)/Samsung_dataset/'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_path)

extracted_files = os.listdir(extracted_path)
print(f'Extracted files: {extracted_files}')

csv_file_path = os.path.join(extracted_path, extracted_files[0])
data = pd.read_csv(csv_file_path)

print(f'First few rows of the dataset:\n{data.head()}')

data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data['50_MA'] = data['Close'].rolling(window=50).mean()
data.dropna(inplace=True)

print(f'Dataset after preprocessing:\n{data.head()}')

X = data[['50_MA']]
y = data['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')

plt.figure(figsize=(14, 7))
plt.plot(y_test.index, y_test, label='Actual Prices', color='blue', linewidth=2)
plt.plot(y_test.index, y_pred, label='Predicted Prices', color='red', linestyle='--', linewidth=2)
plt.title('Actual vs. Predicted Stock Prices', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Stock Price', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()
