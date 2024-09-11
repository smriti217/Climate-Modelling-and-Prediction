import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error
from cleaning import data_cleaned

# Load the cleaned data
data = pd.read_csv(r"D:/hpc/cleaned.csv")

# Ensure no null values
data.isnull().sum()

# Assuming data_cleaned is the cleaned dataset
data = data_cleaned

# Extracting year, month, and day from the date
data['dt'] = pd.to_datetime(data['dt'])  # Ensure 'dt' column is in datetime format
data['year'] = data['dt'].dt.year
data['month'] = data['dt'].dt.month
data['day'] = data['dt'].dt.day

# Encoding categorical variables
label_encoder_city = LabelEncoder()
data['City'] = label_encoder_city.fit_transform(data['City'].astype(str))

label_encoder_country = LabelEncoder()
data['Country'] = label_encoder_country.fit_transform(data['Country'].astype(str))

# Defining features and target variable
features = data[['City', 'Country', 'Latitude', 'Longitude', 'year', 'month', 'day']]
target = data['AverageTemperature']

# Scaling the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Display the first few rows of the processed features and target
print("Features (first few rows):\n", features.head())
print("\nScaled Features (first few rows):\n", features_scaled[:5])
print("\nTarget (first few rows):\n", target.head())
print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Step 2: Modeling
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=5, batch_size=32, verbose=2)
model.save('model.h5')
# Save the scaler
np.save('scaler.npy', scaler.scale_)

# Save the label encoders
np.save('city_classes.npy', label_encoder_city.classes_)
np.save('country_classes.npy', label_encoder_country.classes_)

print("Model and encoders saved successfully.")

# Step 3: Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Example prediction function
def predict_temperature(city, country, latitude, longitude, year, month, day):
    city_encoded = label_encoder_city.transform([city])[0]
    country_encoded = label_encoder_country.transform([country])[0]
    features = np.array([[city_encoded, country_encoded, latitude, longitude, year, month, day]])
    features_scaled = scaler.transform(features)
    return model.predict(features_scaled)[0][0]

# Example prediction
city = 'Århus'
country = 'Denmark'
latitude = 57.05
longitude = 10.33
year = 2024
month = 7
day = 17

# Ensure the input city and country are encoded properly
try:
    predicted_temp = predict_temperature(city, country, latitude, longitude, year, month, day)
    print(f'Predicted temperature for {city} on {year}-{month}-{day} is {predicted_temp:.2f}°C')
except Exception as e:
    print(f'Error in prediction: {e}')
