import json
import re

from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# ---- Config: must match training exactly ----
MAXLEN = 200
NUM_WORDS = 10000
MODEL_PATH = 'sentiment_model.keras'      # or 'sentiment_model.keras'
WORD_INDEX_PATH = 'word_index.json'

# ---- Load model + word index once, at startup ----
model = load_model(MODEL_PATH)

with open(WORD_INDEX_PATH) as f:
    word_index = json.load(f)


def encode_review(text, word_index, num_words=NUM_WORDS, maxlen=MAXLEN):
    """Turns a raw review string into the padded integer sequence
    the model expects. Mirrors how imdb.load_data() encodes text:
    0 = padding, 1 = start, 2 = out-of-vocabulary, real words = index + 3."""
    tokens = re.findall(r"[a-z']+", text.lower())  # strips punctuation, keeps words
    seq = [1]  # start token
    for t in tokens:
        idx = word_index.get(t)
        if idx is not None and idx + 3 < num_words:
            seq.append(idx + 3)
        else:
            seq.append(2)  # unknown / out-of-vocab
    return pad_sequences([seq], maxlen=maxlen, padding='post')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or {}
    review = data.get('review', '').strip()

    if not review:
        return jsonify({'error': 'No review text provided'}), 400

    encoded = encode_review(review, word_index)
    probability = float(model.predict(encoded, verbose=0)[0][0])

    return jsonify({'probability': probability})


import os

if __name__ == '__main__':
    # host='0.0.0.0' makes it reachable from other devices on your network,
    # not just localhost -- useful if "other users" means people on the same LAN.
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)