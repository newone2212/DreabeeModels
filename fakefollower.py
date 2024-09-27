from flask import Flask, jsonify
import pandas as pd
import mysql.connector
import pickle
from threading import Thread
from datetime import datetime
import numpy as np

app = Flask(__name__)

# Database configuration for reading and writing
read_db_config = {
    'host': '192.155.100.47',
    'user': 'youtube',
    'password': '!qR%xf|L3@',
    'database': 'insta_scraper'
}

write_db_config = {
    'host': '192.155.100.47',
    'user': 'youtube',
    'password': '!qR%xf|L3@',
    'database': 'fakefollowPrediction'
}

# Load the fake follower prediction model
model_path = 'Fake_Follower_predictor.pkl'  # Update this path as necessary
try:
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    print(f"Error: Model file not found at path: {model_path}")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Connect to the database using mysql-connector-python
def connect_db(config):
    try:
        conn = mysql.connector.connect(**config)
        return conn
    except mysql.connector.Error as err:
        print(f"Error: Could not connect to database: {err}")
        return None

# Function to create the table if it does not exist
def create_table():
    conn = connect_db(write_db_config)
    if not conn:
        return

    cursor = conn.cursor()
    create_table_query = """
    CREATE TABLE IF NOT EXISTS fake_follow_predictions (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        profile_id BIGINT,
        username VARCHAR(255),
        followers_count BIGINT,
        follows_count BIGINT,
        posts_count BIGINT,
        total_likes BIGINT,
        total_comments BIGINT,
        total_views BIGINT,
        engagement_per_post FLOAT,
        engagement_normalized FLOAT,
        expected_genuine_followers FLOAT,
        fake_followers FLOAT,
        fake_follower_percentage FLOAT,
        prediction FLOAT DEFAULT 0,
        status INT DEFAULT 0,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """
    try:
        cursor.execute(create_table_query)
        conn.commit()
    except mysql.connector.Error as err:
        print(f"Error: Could not create table: {err}")
    finally:
        cursor.close()
        conn.close()

# Function to alter the table if it already exists to adjust column types
def alter_table():
    conn = connect_db(write_db_config)
    if not conn:
        return

    cursor = conn.cursor()
    alter_table_query = """
    ALTER TABLE fake_follow_predictions
    MODIFY COLUMN profile_id BIGINT,
    MODIFY COLUMN followers_count BIGINT,
    MODIFY COLUMN follows_count BIGINT,
    MODIFY COLUMN posts_count BIGINT,
    MODIFY COLUMN total_likes BIGINT,
    MODIFY COLUMN total_comments BIGINT,
    MODIFY COLUMN total_views BIGINT;
    """
    try:
        cursor.execute(alter_table_query)
        conn.commit()
    except mysql.connector.Error as err:
        print(f"Error: Could not alter table: {err}")
    finally:
        cursor.close()
        conn.close()

# Function to fetch and process data
def fetch_and_process_data():
    conn = connect_db(read_db_config)
    if not conn:
        return pd.DataFrame()  # Return empty dataframe on error

    # SQL query to join insta_profile_scraper and user_posts tables
    query = """
    SELECT 
        ips.id as profile_id,
        ips.username,
        ips.followers_count,
        ips.follows_count,
        ips.posts_count,
        up.post_id as post_id,
        up.like_count,
        up.comment_count,
        up.video_view_count
    FROM 
        insta_profile_scraper ips
    JOIN 
        user_posts up ON ips.id = up.user_id
    """

    # Execute the query and fetch the data
    try:
        df = pd.read_sql(query, conn)
    except Exception as e:
        print(f"Error: Could not execute query: {e}")
        return pd.DataFrame()  # Return empty dataframe on error
    finally:
        conn.close()

    # Group and calculate metrics
    df_grouped = df.groupby(['profile_id', 'username', 'followers_count', 'follows_count', 'posts_count']).agg(
        total_likes=pd.NamedAgg(column='like_count', aggfunc='sum'),
        total_comments=pd.NamedAgg(column='comment_count', aggfunc='sum'),
        total_views=pd.NamedAgg(column='video_view_count', aggfunc='sum')
    ).reset_index()
    df_grouped['engagement_per_post'] = (df_grouped['total_likes'] + df_grouped['total_comments']) / df_grouped['posts_count']
    df_grouped['engagement_normalized'] = (df_grouped['total_likes'] + df_grouped['total_comments']) / (df_grouped['followers_count'] * df_grouped['posts_count'])
    threshold = 0.1
    df_grouped['expected_genuine_followers'] = df_grouped['total_likes'] / (threshold * df_grouped['posts_count'])
    df_grouped['fake_followers'] = df_grouped['followers_count'] - df_grouped['expected_genuine_followers']
    df_grouped['fake_followers'] = df_grouped['fake_followers'].apply(lambda x: max(x, 0))
    df_grouped['fake_follower_percentage'] = (df_grouped['fake_followers'] / df_grouped['followers_count']) * 100

    # Remove rows with NaNs or Infs
    df_grouped.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_grouped.dropna(inplace=True)

    return df_grouped

# Function to move data to FakeFollowPrediction table
def move_data_to_fakefollow():
    df_grouped = fetch_and_process_data()

    if df_grouped.empty:
        print("No data to process.")
        return

    conn = connect_db(write_db_config)
    if not conn:
        return

    cursor = conn.cursor()

    for _, row in df_grouped.iterrows():
        cursor.execute("""
            INSERT INTO fake_follow_predictions (
                profile_id, username, followers_count, follows_count, posts_count, 
                total_likes, total_comments, total_views, engagement_per_post, 
                engagement_normalized, expected_genuine_followers, fake_followers, 
                fake_follower_percentage, prediction, status, timestamp
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            row['profile_id'], row['username'], row['followers_count'], row['follows_count'], row['posts_count'],
            row['total_likes'], row['total_comments'], row['total_views'], row['engagement_per_post'],
            row['engagement_normalized'], row['expected_genuine_followers'], row['fake_followers'],
            row['fake_follower_percentage'], 0, 0, datetime.utcnow()
        ))

    conn.commit()
    cursor.close()
    conn.close()

# Function to update fake follower predictions
def update_fake_follower_predictions():
    conn = connect_db(write_db_config)
    if not conn:
        return

    query = "SELECT * FROM fake_follow_predictions WHERE status = 0"
    df = pd.read_sql(query, conn)

    if df.empty:
        conn.close()
        return

    # Assuming the model expects a DataFrame with certain columns for prediction
    try:
        predictions = model.predict(df[['followers_count', 'follows_count', 'posts_count', 'total_comments', 'engagement_per_post', 'engagement_normalized']])
    except Exception as e:
        print(f"Error during prediction: {e}")
        conn.close()
        return

    cursor = conn.cursor()
    for i, prediction in enumerate(predictions):
        cursor.execute("""
            UPDATE fake_follow_predictions 
            SET prediction = %s, status = 1, timestamp = %s 
            WHERE id = %s
        """, (prediction, datetime.utcnow(), int(df.iloc[i]['id'])))

    conn.commit()
    cursor.close()
    conn.close()

@app.route('/api/data', methods=['GET'])
def get_data():
    conn = connect_db(write_db_config)
    if not conn:
        return jsonify({"error": "Could not connect to database"}), 500

    query = "SELECT * FROM fake_follow_predictions"
    df = pd.read_sql(query, conn)
    conn.close()

    return jsonify(df.to_dict(orient='records'))

@app.route('/api/insert_data', methods=['POST'])
def insert_data():
    move_data_to_fakefollow()
    return jsonify({"message": "Data insertion started"}), 200

@app.route('/api/update_predictions', methods=['POST'])
def update_predictions():
    update_fake_follower_predictions()
    return jsonify({"message": "Prediction update started"}), 200

if __name__ == '__main__':
    # Create the table if it does not exist
    # create_table()
    # Alter the table if necessary to adjust column types
    # alter_table()
    # move_data_to_fakefollow()
    # Run the Flask app
    update_fake_follower_predictions()
    app.run(debug=True)
