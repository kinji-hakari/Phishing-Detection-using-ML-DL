# Phishing Email Detection using Machine Learning and Deep Learning

This project implements a comprehensive phishing email detection system using various Machine Learning and Deep Learning techniques.

## Project Description

Phishing emails are fraudulent messages designed to trick recipients into revealing sensitive information. This project builds and compares multiple classification models to automatically detect phishing emails with high accuracy.

## Dataset

The project uses the `Phishing_Email.csv` dataset containing:
- Email text content
- Labels (Phishing vs Safe Email)

The dataset undergoes preprocessing including:
- Removal of duplicates and null values
- Text cleaning (removing hyperlinks, punctuations, lowercasing)
- Label encoding

## Project Structure

```
.
├── notebooks/
│   ├── 01_data_loading_and_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_machine_learning_models.ipynb
│   ├── 05_deep_learning_models.ipynb
│   └── 06_evaluation_and_results.ipynb
├── data/
│   └── Phishing_Email.csv
├── models/
│   └── (saved models)
├── results/
│   └── (plots and metrics)
├── README.md
└── requirements.txt
```

## Technologies Used

### Programming Language
- Python 3.8+

### Machine Learning
- scikit-learn (Logistic Regression, Random Forest, SVM, Naive Bayes, KNN, XGBoost, MLP)

### Deep Learning
- TensorFlow/Keras (SimpleRNN, LSTM, Bidirectional LSTM, GRU)

### Feature Engineering
- TF-IDF Vectorization
- BERT Embeddings (transformers)
- Doc2Vec (gensim)

### Explainable AI
- LIME (Local Interpretable Model-agnostic Explanations)

### Visualization
- Matplotlib, Seaborn, Plotly

## How to Run the Project


### Notebook Execution Order

Run the notebooks in the following sequence:

1. **01_data_loading_and_exploration.ipynb**
   - Load the dataset
   - Explore data distribution
   - Visualize class balance

2. **02_data_preprocessing.ipynb**
   - Clean the data
   - Remove duplicates and nulls
   - Preprocess text (remove hyperlinks, punctuations, etc.)
   - Generate word clouds

3. **03_feature_engineering.ipynb**
   - Create TF-IDF features
   - Generate BERT embeddings
   - Create Doc2Vec embeddings
   - Split data into train/test sets

4. **04_machine_learning_models.ipynb**
   - Train ML models (Random Forest, XGBoost, Logistic Regression, SVM, Naive Bayes, KNN, MLP)
   - Compare performance across different vectorization methods
   - Apply LIME for model interpretability

5. **05_deep_learning_models.ipynb**
   - Train Deep Learning models (SimpleRNN, LSTM, BiLSTM, GRU)
   - Evaluate and visualize training history
   - Generate confusion matrices


## Key Features

- Multiple vectorization techniques (TF-IDF, BERT, Doc2Vec)
- Comprehensive ML model comparison
- Deep Learning architectures for sequence modeling
- Explainable AI with LIME
- Cross-validation and robust evaluation metrics
- Visualization of results and confusion matrices

