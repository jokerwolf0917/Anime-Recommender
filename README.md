# 🎬 Anime Recommender: Two-Stage Hybrid Recommendation System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)
![Machine Learning](https://img.shields.io/badge/ML-Surprise%20%7C%20Scikit--Learn-yellow)

A production-style anime recommendation system implementing a **Two-Stage Hybrid Architecture**:

* Stage 1: Collaborative Filtering (Recall)
* Stage 2: Content-Based Re-ranking (Precision Optimization)

Designed to demonstrate scalable recommender system design, offline evaluation, and reproducible deployment.

---

# 🌐 Live Online Demo (Toy Mode)

👉 **HuggingFace Space (Public Demo):**

```
https://huggingface.co/spaces/AlyxWang/Anime-Recommender
```

### Why Toy Mode?

Due to Kaggle dataset size (~57M interactions), the public demo runs in **toy mode** with a small synthetic dataset.

The full system (CF + Hybrid + Evaluation) works locally with the full Kaggle dataset.

---

# 🧠 System Architecture

## Two-Stage Recommendation Pipeline

```
User
  ↓
Collaborative Filtering (Top-M Recall)
  ↓
Content-Based Re-Ranking
  ↓
Final Top-N Recommendations
```

---

## Stage 1: Candidate Generation (Recall)

* Algorithm: Item-Item KNN (Surprise)
* Input: User-Item interaction matrix
* Output: Top-M candidate items
* Goal: High recall, fast filtering

---

## Stage 2: Content-Based Re-Ranking

* Text Features:

  * TF-IDF (ngram 1–2)
  * Synopsis + metadata
* Dimensionality Reduction:

  * Truncated SVD
* Numeric Features:

  * Score
  * Members
  * Episodes
* Ranking:

  * Cosine similarity

Goal: Precision optimization on candidate set

---

# ✨ Technical Highlights

## Robust Data Handling

* Automatic fallback to toy dataset
* Safe handling of missing values
* Numeric coercion + NaN replacement
* Dynamic TF-IDF parameter adaptation

---

## Evaluation Framework

Offline ranking metrics implemented:

* Precision@10
* Recall@10
* HitRatio@10
* nDCG@10

Evaluation strategy:

* Per-user holdout
* Remove training interactions from test
* Reproducible random seed

---

## Engineering Features

* Streamlit interactive UI
* Model caching
* Configurable parameters
* Dockerized deployment
* HuggingFace Spaces deployment
* Binary file handling

---

# 📊 Evaluation Results

| Model                  | Precision@10 | Recall@10  | HitRatio@10 |
| ---------------------- | ------------ | ---------- | ----------- |
| Content-Based          | 0.3000       | 0.1100     | 0.6154      |
| CF Only                | 0.1778       | 0.0500     | 0.4815      |
| **Hybrid (Two-Stage)** | **0.5750**   | **0.1916** | **1.0000**  |

Hybrid architecture significantly improves ranking quality.

---

# 📸 UI Preview

### System Interface

![UI](docs/page.png)

### Content-Only Mode

![Content](docs/Content-only.png)

### Hybrid Mode

![Hybrid](docs/Hybrid.png)

---

# 📂 Project Structure

```
Anime-Recommender/
│
├── src/
│   ├── app.py
│   ├── content_model.py
│   ├── cf_model.py
│   ├── data_loader.py
│   └── evaluation.py
│
├── data/ (excluded)
├── docs/
├── reports/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# 🚀 Getting Started

## 1️⃣ Dataset Preparation

Dataset:
Anime Recommendation Database 2020 (Kaggle)

Place into:

```
data/anime.csv
data/anime_with_synopsis.csv
data/rating_complete.csv
```

---

## 2️⃣ Local Installation

```bash
git clone https://github.com/jokerwolf0917/Anime-Recommender.git
cd Anime-Recommender

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt

streamlit run src/app.py
```

---

## 3️⃣ Docker Deployment

```bash
docker compose up --build
```

Access:

```
http://localhost:8501
```

---

# ⚙️ Configurable Parameters

From UI:

* Hybrid / Content-only
* Like threshold
* CF candidate size (Top-M)
* Top-N output
* Sample training size

---

# 🧪 Offline Evaluation

Run evaluation script:

```bash
python src/evaluation.py
```

Metrics printed to console.

---

# 📌 Why Two-Stage Matters

Single CF:

* Poor explainability
* Cold start issues

Single Content:

* No collaborative signals

Two-Stage:

* High recall
* High precision
* Balanced personalization

This architecture is widely used in:

* Netflix
* Amazon
* YouTube
* Spotify

---

# 🛠 Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Surprise
* Streamlit
* Docker
* HuggingFace Spaces

---

# 🎯 Learning Outcomes

This project demonstrates:

* Recommender system design
* Feature engineering
* Ranking metrics
* Offline evaluation protocol
* Docker deployment
* Production-style ML app architecture

---

# 📜 License

MIT License

---

# 👤 Author

GitHub:
[https://github.com/jokerwolf0917](https://github.com/jokerwolf0917)

HuggingFace Demo:
[https://huggingface.co/spaces/AlyxWang/Anime-Recommender](https://huggingface.co/spaces/AlyxWang/Anime-Recommender)

---

# 📌 Final Note

This project is structured as a production-style ML system, not just a notebook experiment.

It covers:

* Algorithm design
* Evaluation
* Engineering
* Deployment
* UI interaction
