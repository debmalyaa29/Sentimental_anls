# IMDB Movie Review Sentiment Analysis

A web app that classifies movie reviews as positive or negative using a Simple RNN trained on the IMDB dataset.

**Live demo:** https://debmalyaa29-sentimental-anls-app-bd47r6.streamlit.app/

## What it does

Paste any movie review into the text box and the model predicts whether the sentiment is positive or negative, along with the raw prediction score (0 to 1 — closer to 1 means more positive).

## How it works

- The model is a Simple RNN built with TensorFlow/Keras, trained on the IMDB movie reviews dataset (50,000 labeled reviews).
- Text is tokenized, mapped to the same word-index vocabulary the model was trained on, and padded to a fixed length before being fed to the model.
- The output is a single sigmoid score between 0 and 1, thresholded at 0.5 to produce a Positive/Negative label.

## Tech stack

- **Model:** TensorFlow / Keras (Embedding + SimpleRNN + Dense)
- **Dataset:** Keras IMDB dataset
- **App:** Streamlit
- **Deployment:** Streamlit Community Cloud

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The trained model file must be present in the project directory for the app to load it.
