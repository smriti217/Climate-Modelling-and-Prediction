
import pandas as pd
file_path = r"D:/hpc/GlobalLandTemperaturesByCity.csv"

def clean_data(file_path):
    # Load the CSV file
    data = pd.read_csv(file_path)

    # Handle missing values by removing rows where average temperature is NaN
    data_cleaned = data.dropna(subset=['AverageTemperature'])

    # Convert the 'dt' column to datetime format
    data_cleaned['dt'] = pd.to_datetime(data_cleaned['dt'])

    # Parse the Latitude and Longitude to numeric values
    def parse_coordinate(coord):
        if pd.isnull(coord):
            return None
        direction = coord[-1]
        value = float(coord[:-1])
        if direction in ['S', 'W']:
            value = -value
        return value

    data_cleaned['Latitude'] = data_cleaned['Latitude'].apply(parse_coordinate)
    data_cleaned['Longitude'] = data_cleaned['Longitude'].apply(parse_coordinate)

    # Remove duplicates if any
    data_cleaned = data_cleaned.drop_duplicates()

    return data_cleaned

data_cleaned = clean_data(file_path)

# Save the cleaned data to a new CSV file
cleaned_file_path = r"D:/hpc/cleaned.csv"
data_cleaned.to_csv(cleaned_file_path, index=False)

print(f"Cleaned data saved to {cleaned_file_path}")
