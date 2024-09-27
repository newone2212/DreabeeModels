# from flask import Flask, jsonify
# import pandas as pd
# import mysql.connector
# import pickle
# from threading import Thread
# from datetime import datetime
# import numpy as np
# import math

# app = Flask(__name__)

# # Database configuration for reading and writing
# read_db_config = {
#     'host': '192.155.100.47',
#     'user': 'youtube',
#     'password': '!qR%xf|L3@',
#     'database': 'insta_scraper'
# }

# write_db_config = {
#     'host': '192.155.100.47',
#     'user': 'youtube',
#     'password': '!qR%xf|L3@',
#     'database': 'fakefollowPrediction'
# }

# # Load the fake follower prediction model
# model_path = 'Fake_Follower_predictor.pkl'  # Update this path as necessary
# try:
#     with open(model_path, 'rb') as model_file:
#         model = pickle.load(model_file)
# except FileNotFoundError:
#     print(f"Error: Model file not found at path: {model_path}")
#     model = None
# except Exception as e:
#     print(f"Error loading model: {e}")
#     model = None

# # Connect to the database using mysql-connector-python
# def connect_db(config):
#     try:
#         conn = mysql.connector.connect(**config)
#         return conn
#     except mysql.connector.Error as err:
#         print(f"Error: Could not connect to database: {err}")
#         return None

# # Function to fetch and process data
# def fetch_and_process_data():
#     conn = connect_db(read_db_config)
#     if not conn:
#         return pd.DataFrame()  # Return empty dataframe on error

#     query = """
#     SELECT 
#         ips.id as profile_id,
#         ips.username,
#         ips.followers_count,
#         ips.follows_count,
#         ips.posts_count,
#         up.post_id as post_id,
#         up.like_count,
#         up.comment_count,
#         up.video_view_count
#     FROM 
#         insta_profile_scraper ips
#     JOIN 
#         user_posts up ON ips.id = up.user_id
#     """

#     try:
#         df = pd.read_sql(query, conn)
#     except Exception as e:
#         print(f"Error: Could not execute query: {e}")
#         return pd.DataFrame()  # Return empty dataframe on error
#     finally:
#         conn.close()

#     df_grouped = df.groupby(['profile_id', 'username', 'followers_count', 'follows_count', 'posts_count']).agg(
#         total_likes=pd.NamedAgg(column='like_count', aggfunc='sum'),
#         total_comments=pd.NamedAgg(column='comment_count', aggfunc='sum'),
#         total_views=pd.NamedAgg(column='video_view_count', aggfunc='sum')
#     ).reset_index()

#     df_grouped['engagement_per_post'] = (df_grouped['total_likes'] + df_grouped['total_comments']) / df_grouped['posts_count']
#     df_grouped['engagement_normalized'] = (df_grouped['total_likes'] + df_grouped['total_comments']) / (df_grouped['followers_count'] * df_grouped['posts_count'])
#     threshold = 0.1
#     df_grouped['expected_genuine_followers'] = df_grouped['total_likes'] / (threshold * df_grouped['posts_count'])
#     df_grouped['fake_followers'] = df_grouped['followers_count'] - df_grouped['expected_genuine_followers']
#     df_grouped['fake_followers'] = df_grouped['fake_followers'].apply(lambda x: max(x, 0))
#     df_grouped['fake_follower_percentage'] = (df_grouped['fake_followers'] / df_grouped['followers_count']) * 100

#     df_grouped.replace([np.inf, -np.inf], np.nan, inplace=True)
#     df_grouped.dropna(inplace=True)

#     return df_grouped

# # Function to insert data in batches using threading
# def insert_data_batch(df_batch):
#     conn = connect_db(write_db_config)
#     if not conn:
#         return

#     cursor = conn.cursor()

#     for _, row in df_batch.iterrows():
#         cursor.execute("""
#             INSERT INTO fake_follow_predictions (
#                 profile_id, username, followers_count, follows_count, posts_count, 
#                 total_likes, total_comments, total_views, engagement_per_post, 
#                 engagement_normalized, expected_genuine_followers, fake_followers, 
#                 fake_follower_percentage, prediction, status, timestamp
#             ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
#         """, (
#             row['profile_id'], row['username'], row['followers_count'], row['follows_count'], row['posts_count'],
#             row['total_likes'], row['total_comments'], row['total_views'], row['engagement_per_post'],
#             row['engagement_normalized'], row['expected_genuine_followers'], row['fake_followers'],
#             row['fake_follower_percentage'], 0, 0, datetime.utcnow()
#         ))

#     conn.commit()
#     cursor.close()
#     conn.close()

# # Function to move data to FakeFollowPrediction table in batches
# def move_data_to_fakefollow():
#     df_grouped = fetch_and_process_data()

#     if df_grouped.empty:
#         print("No data to process.")
#         return

#     num_batches = math.ceil(len(df_grouped) / 1000)

#     threads = []
#     for i in range(num_batches):
#         df_batch = df_grouped.iloc[i*1000:(i+1)*1000]
#         thread = Thread(target=insert_data_batch, args=(df_batch,))
#         thread.start()
#         threads.append(thread)

#     for thread in threads:
#         thread.join()

# # Function to update fake follower predictions in batches
# def update_predictions_batch(df_batch):
#     conn = connect_db(write_db_config)
#     if not conn:
#         return

#     try:
#         predictions = model.predict(df_batch[['followers_count', 'follows_count', 'posts_count', 'total_comments', 'engagement_per_post', 'engagement_normalized']])
#     except Exception as e:
#         print(f"Error during prediction: {e}")
#         conn.close()
#         return

#     cursor = conn.cursor()
#     for i, prediction in enumerate(predictions):
#         cursor.execute("""
#             UPDATE fake_follow_predictions 
#             SET prediction = %s, status = 1, timestamp = %s 
#             WHERE id = %s
#         """, (prediction, datetime.utcnow(), int(df_batch.iloc[i]['id'])))

#     conn.commit()
#     cursor.close()
#     conn.close()

# # Function to update fake follower predictions
# def update_fake_follower_predictions():
#     conn = connect_db(write_db_config)
#     if not conn:
#         return

#     query = "SELECT * FROM fake_follow_predictions WHERE status = 0"
#     df = pd.read_sql(query, conn)
#     conn.close()

#     if df.empty:
#         return

#     num_batches = math.ceil(len(df) / 1000)

#     threads = []
#     for i in range(num_batches):
#         df_batch = df.iloc[i*1000:(i+1)*1000]
#         thread = Thread(target=update_predictions_batch, args=(df_batch,))
#         thread.start()
#         threads.append(thread)

#     for thread in threads:
#         thread.join()

# @app.route('/api/data', methods=['GET'])
# def get_data():
#     conn = connect_db(write_db_config)
#     if not conn:
#         return jsonify({"error": "Could not connect to database"}), 500

#     query = "SELECT * FROM fake_follow_predictions"
#     df = pd.read_sql(query, conn)
#     conn.close()

#     return jsonify(df.to_dict(orient='records'))

# @app.route('/api/insert_data', methods=['POST'])
# def insert_data():
#     move_data_to_fakefollow()
#     return jsonify({"message": "Data insertion started"}), 200

# @app.route('/api/update_predictions', methods=['POST'])
# def update_predictions():
#     update_fake_follower_predictions()
#     return jsonify({"message": "Prediction update started"}), 200

# if __name__ == '__main__':
#     app.run(debug=True)
import pandas as pd
import joblib

# Load the saved pipeline
pipeline = joblib.load(r'models/fake_follower_prediction_pipeline.pkl')

# Example input data (this should be in the same format as used during training)
input_data = pd.DataFrame({
    'rank': [1, 2, 3],
    'influence_score': [92, 91, 90],
    'posts': ['3.3k', '6.9k', '0.89k'],  # This will be converted to numerical values by the pipeline
    'followers': [475800000.0, 366200000.0, 357300000.0],
    'total_likes': [2.9e10, 5.74e10, 6e9],
    'country': ['Spain', 'United States', 'Argentina'],
    'calculated_er': [6094.997898, 15674.494812, 1679.261125]
})

# Use the pipeline to predict fake follower percentage
predictions = pipeline.predict(input_data)

# Calculate the number of fake followers and real followers
input_data['fake_follower_percentage'] = predictions
input_data['fake_followers'] = (input_data['fake_follower_percentage'] / 100) * input_data['followers']
input_data['real_followers'] = input_data['followers'] - input_data['fake_followers']

# Print the detailed results
print(input_data[['channel_info', 'followers', 'fake_follower_percentage', 'fake_followers', 'real_followers']])

# Calculate the total number of influencers, real followers, and fake followers
total_influencers = len(input_data)
total_fake_followers = input_data['fake_followers'].sum()
total_real_followers = input_data['real_followers'].sum()

# Print the aggregate results
print(f"Total Influencers: {total_influencers}")
print(f"Total Fake Followers: {total_fake_followers}")
print(f"Total Real Followers: {total_real_followers}")
