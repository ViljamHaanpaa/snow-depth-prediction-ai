import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
import os

# Load the data from the CSV file
df = pd.read_csv('stockholm_snow_depth.csv', header=0)

# Display the first few rows of the data to understand its structure
print(df.head())

# Check for missing values in the dataset
if df.isnull().values.any():
    print("Missing values detected. Filling missing values with the mean for each column.")
    for column in df.columns:
        if df[column].isnull().any():
            mean_value = df[column].mean()
            print(f"Filling missing values in column '{column}' with mean value {mean_value:.2f}")
            df[column].fillna(mean_value, inplace=True)

# Normalize the features (snow depth values)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['Max Snow Depth (December)', 'Max Snow Depth (December-April)']])

# Prepare the data for LSTM (sequence data for prediction)
def create_sequences(data, sequence_length=5):
    sequences = []
    labels = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
        labels.append(data[i + sequence_length][1])  # Predict max_snow_depth_dec_april
    return np.array(sequences), np.array(labels)

# Create sequences for training (use the last 5 years to predict the next year's max snow depth)
sequence_length = 5
X, y = create_sequences(scaled_data, sequence_length)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model path
model_path = 'snow_depth_model.h5'

# Check if a saved model exists
if os.path.exists(model_path):
    # Load the existing model if it exists
    print("Loading saved model...")
    model = load_model(model_path)
else:
    # Build the LSTM model if no saved model is found
    print("No saved model found, building a new model...")
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(units=1))  # We predict a single value: max_snow_depth_dec_april

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))

    # Save the trained model
    model.save(model_path)
    print(f"Model saved to {model_path}")

# Predict for the next 10 years
predictions = []
last_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, scaled_data.shape[1])

for _ in range(10):
    predicted_snow_depth = model.predict(last_sequence)
    predictions.append(predicted_snow_depth[0, 0])
    
    # Update the last_sequence with the new prediction
    new_sequence = np.append(last_sequence[:, 1:, :], [[[predicted_snow_depth[0, 0], predicted_snow_depth[0, 0]]]], axis=1)
    last_sequence = new_sequence

# Inverse transform the predictions to get them back to the original scale
predictions = np.array(predictions).reshape(-1, 1)
predictions_original_scale = scaler.inverse_transform(np.concatenate((np.zeros((10, 1)), predictions), axis=1))[:, 1]

print("Predicted snow depths for the next 10 years:")
for i, prediction in enumerate(predictions_original_scale, start=1):
    print(f"Year {i}: {prediction:.2f}")