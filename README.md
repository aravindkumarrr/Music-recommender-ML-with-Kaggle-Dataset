# 🎧 Music Recommendation System using FAISS

A fast, content-based music recommendation engine built using Spotify-style metadata, audio features, and vector search powered by **FAISS**.

---

## 🚀 Features

- Clean and preprocess music metadata
- Extract features from:
  - 🎵 Track names (TF-IDF)
  - 💿 Album names (TF-IDF)
  - 👨‍🎤 Artist names (HashingVectorizer)
  - 🎚️ Audio features (StandardScaler)
  - 🏷️ Track genres (OneHotEncoder or CountVectorizer)
- Dimensionality reduction using **TruncatedSVD**
- Fast similarity search using **FAISS**
- Simple command-line query interface using track name

---

## 🧠 What is FAISS?

**FAISS (Facebook AI Similarity Search)** is an open-source library developed by Facebook AI Research for efficient similarity search and clustering of dense vectors.

In this project:
- Track feature vectors are reduced to 256D using TruncatedSVD.
- Normalized with L2 norm to simulate cosine similarity.
- Indexed with `IndexFlatIP` for fast inner product search.
- Returns top-k similar tracks in milliseconds.

---

## 🛠️ Tech Stack

- Python
- FAISS
- Scikit-learn (TruncatedSVD, TF-IDF, scaling)
- Pandas / NumPy / SciPy
- Optional: Streamlit (for UI), Jupyter Notebook (for experimentation)

---

## 📈 How It Works

1. Add the path of `dataset.csv` to dataset_PATH variable.
2. Clean text fields (track, artist, album, genre)
3. Vectorize and scale features
4. Concatenate features using `hstack`
5. Reduce dimensionality with `TruncatedSVD`
6. Normalize vectors for cosine similarity
7. Build FAISS index using `IndexFlatIP`
8. Query for similar tracks using song name

---

## ✅ Example

```python
recommend_by_name("Talking to the Moon", k=10)
