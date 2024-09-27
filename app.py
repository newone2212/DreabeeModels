from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Load the model from the file
with open('Fake_Follower_predictor.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json(force=True)

    # Convert the data into a DataFrame
    df = pd.DataFrame(data)

    # Select the required features
    features = ['followers_count', 'follows_count', 'posts_count', 'total_comments', 'engagement_per_post', 'engagement_normalized']
    df = df[features]

    # Make prediction using the loaded model
    prediction = model.predict(df)

    # Return the prediction as JSON
    return jsonify({'fake_follower_percentage': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
