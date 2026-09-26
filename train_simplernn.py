import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

NUM_WORDS = 10000
MAXLEN = 200

print(f"TensorFlow Version: {tf.__version__}")
print(f"GPUs Available: {tf.config.list_physical_devices('GPU')}")

# 1. Load dataset with num_words cap
print("Loading IMDB data...")
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=NUM_WORDS)

# 2. Pad sequences with padding='pre' (CRITICAL for SimpleRNN)
print("Padding sequences ('pre')...")
X_train = pad_sequences(X_train, maxlen=MAXLEN, padding='pre')
X_test = pad_sequences(X_test, maxlen=MAXLEN, padding='pre')

# 3. Build SimpleRNN architecture
print("Building SimpleRNN model...")
model = Sequential([
    Input(shape=(MAXLEN,)),
    Embedding(NUM_WORDS, 128),
    SimpleRNN(64, dropout=0.2),
    Dense(1, activation='sigmoid')
])

model.summary()

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# 4. Train with EarlyStopping to avoid overfitting
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True,
    verbose=1
)

print("Training SimpleRNN...")
history = model.fit(
    X_train, y_train,
    epochs=6,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop]
)

# 5. Evaluate on test set
print("Evaluating on test set...")
test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=64)
print(f"Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

# 6. Save model and word index
print("Saving model to sentiment_model.keras...")
model.save('sentiment_model.keras')

print("Saving word_index.json...")
word_index = imdb.get_word_index()
with open('word_index.json', 'w') as f:
    json.dump(word_index, f)

print("Training completed successfully!")
