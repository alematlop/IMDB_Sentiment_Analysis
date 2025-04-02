# IMDB Sentiment Analysis

## 📌 Project Overview
This project performs sentiment analysis on IMDB movie reviews using Natural Language Processing (NLP). It classifies reviews as **positive** or **negative** using a **Logistic Regression model** trained on **TF-IDF features**.

## 📂 Project Structure
```
IMDB-Sentiment-Analysis/
│── data/                      # Dataset storage
│   ├── raw/                   # Raw IMDB dataset
│   ├── processed/              # Preprocessed data (after cleaning)
│
│── src/                        # Source code
│   ├── data_loader.py          # Load and preprocess dataset
│   ├── preprocess.py           # Text cleaning & tokenization
│   ├── train.py                # Model training
│   ├── predict.py              # Make predictions on new text
│
│── models/                     # Saved trained models
│   ├── logistic_regression.pkl
│   ├── tfidf_vectorizer.pkl
│
│── README.md                   # Project documentation
│── requirements.txt            # Dependencies for the project
│── .gitignore                   # Ignore unnecessary files
```

## 🛠️ Installation
1. **Clone the Repository**
   ```bash
   git clone https://github.com/alematlop/IMDB_Sentiment_Analysis.git
   cd IMDB_Sentiment_Analysis
   ```
2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 How to Run
1. **Load & Process Data**
   ```bash
   python src/data_loader.py
   ```
2. **Train the Model**
   ```bash
   python src/train.py
   ```
3. **Run Predictions**
   ```bash
   python src/predict.py
   ```
   Example input in `predict.py`:
   ```python
   predict_sentiment("This movie was fantastic! I loved every moment.")
   ```
   Output:
   ```
   Predicted Sentiment: Positive
   ```

## 📊 Model Performance
- **Feature Extraction:** TF-IDF (max 5000 features)
- **Classifier:** Logistic Regression
- **Expected Accuracy:** ~85% on IMDB dataset

## 🎯 Future Enhancements
- Train a **Deep Learning model** (LSTM/BERT) for better accuracy
- Deploy as a **Flask/FastAPI API**
- Add a **web-based UI** using Streamlit

## 📝 License
This project is open-source and available under the MIT License.

---
Made with ❤️ by [Alexandros Matinopoulos Lopez](https://github.com/alematlop)

