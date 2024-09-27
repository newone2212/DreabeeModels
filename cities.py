from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and encoder
model = joblib.load('ml_models/models/cities_model/sgd_view_percentage_model.pkl')
encoder = joblib.load('ml_models/models/cities_model/encoder.pkl')

# List of major cities in India
cities = [
    'Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Ahmedabad', 'Chennai', 'Kolkata', 'Surat', 'Pune', 'Jaipur',
    # Add more cities as needed
]

# Function to predict view percentages for a new brand
def predict_view_percentages(brand_name, brand_category, cities, model, encoder):
    try:
        # Create a DataFrame with the new brand data
        new_data = pd.DataFrame({
            'Brand': [brand_name] * len(cities),
            'Brand Category': [brand_category] * len(cities),
            'City': cities
        })

        # Encode the categorical features
        encoded_features = encoder.transform(new_data)
        
        # Predict view percentages
        predictions = model.predict(encoded_features)
        
        # Create a DataFrame with the results
        result_df = pd.DataFrame({
            'City': cities,
            'View Percentage': predictions
        })
        
        return result_df
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        brand_name = data['brand_name']
        brand_category = data['brand_category']
        
        result_df = predict_view_percentages(brand_name, brand_category, cities, model, encoder)
        
        if result_df is not None:
            return jsonify(result_df.to_dict(orient='records'))
        else:
            return jsonify({'error': 'Prediction failed.'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
