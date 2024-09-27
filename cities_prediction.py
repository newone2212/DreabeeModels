import numpy as np
import joblib
import pandas as pd

# Load the trained model
model_path = 'ml_models/models/cities_model/city_percentage_model.pkl'
model = joblib.load(model_path)

# Load the encoder used during training
encoder_path = 'ml_models/models/cities_model/encoder.pkl'
encoder = joblib.load(encoder_path)

# Function to introduce randomness and normalize percentages unevenly
def uneven_normalize(predictions, total_percentage):
    # Introduce some randomness to each prediction
    randomness = np.random.uniform(0.8, 1.2, size=len(predictions))
    predictions = predictions * randomness
    
    # Normalize to ensure the sum equals total_percentage
    total_pred = sum(predictions)
    normalized_percentages = [p / total_pred * total_percentage for p in predictions]
    
    # Ensure the last percentage adjusts to meet the total percentage exactly
    difference = total_percentage - sum(normalized_percentages)
    normalized_percentages[-1] += difference
    
    return normalized_percentages

# Example new data for prediction
new_data = pd.DataFrame({
    'Brand': ['Lotus Herbals'] * 5,
    'Brand Category': ['Shows'] * 5,
    'Country': ['Iran'] * 5,
    'City': ['Tehran', 'Mashhad', 'Isfahan', 'Shiraz', 'Tabriz'],
    'Country Percentage': [45] * 5
})

# Encode the new data
encoded_new_data = encoder.transform(new_data)

# Predict city percentages
predicted_city_percentages = model.predict(encoded_new_data)
print("Predicted City Percentages (before normalization):", predicted_city_percentages)

# Apply uneven normalization
normalized_city_percentages = uneven_normalize(predicted_city_percentages, total_percentage=45)
print("Normalized City Percentages (uneven):", normalized_city_percentages)


# Output the predicted and normalized city percentages
for city, percentage in zip(new_data['City'], normalized_city_percentages):
    print(f"City: {city}, Predicted Percentage: {percentage:.2f}%")