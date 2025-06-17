from flask import Flask, request, jsonify
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load dataset
data = pd.read_csv('chatbot_dataset.csv')

corpus_pairs = list(zip(data['pattern'].str.lower(), data['response']))
user_inputs = [pair[0] for pair in corpus_pairs]
responses = [pair[1] for pair in corpus_pairs]

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(user_inputs)

def chatbot_response(user_input):
    user_input_processed = user_input.lower()
    user_vector = vectorizer.transform([user_input_processed])
    similarity_scores = cosine_similarity(user_vector, vectors)
    most_similar_index = similarity_scores.argmax()
    max_similarity = similarity_scores[0, most_similar_index]
    threshold = 0.3

    if max_similarity < threshold:
        return "I'm sorry, I don't understand. Can you please rephrase?"
    else:
        return responses[most_similar_index]

# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Chatbot API is running!"

@app.route('/get-response', methods=['POST'])
def get_response():
    data = request.get_json()
    user_message = data.get('message', '')
    bot_reply = chatbot_response(user_message)
    return jsonify({'reply': bot_reply})

if __name__ == '__main__':
    app.run(debug=True)
