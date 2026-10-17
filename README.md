# Multilingual Language Detection Using Machine Learning

## Project Overview

This project compares three machine learning algorithms for identifying languages from text using Natural Language Processing (NLP).

The objective is to explore how character-level text features can be used to distinguish between multiple languages and evaluate different classification models based on their accuracy and training time.

## Dataset

The dataset contains **22,000 text samples across 22 languages**, with 1,000 samples per language.

- Total samples: 22,000
- Languages: 22
- Features: Text
- Target: Language
- Missing values: 0
- Duplicate texts identified: 141

The dataset includes languages such as English, Urdu, Arabic, Chinese, Japanese, Korean, French, Spanish, Hindi, and others.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- NLP
- TF-IDF
- Matplotlib and Seaborn
- Jupyter Notebook

## Methodology

1. Explored the dataset and examined class distribution, missing values, and duplicate texts.
2. Extracted character-level TF-IDF features using `char_wb` with n-grams (1–3) and a maximum of 50,000 features.
3. Applied a stratified 67/33 training-test split.
4. Trained and compared three machine learning models.
5. Evaluated model performance using classification accuracy and training time.
6. Generated a classification report and confusion matrix for Linear SVC.
7. Implemented an interactive prediction function to classify user-entered text.

## Machine Learning Models

Three classification algorithms were compared:

- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Classifier (Linear SVC)

## Results

The following results were obtained from the current notebook:

| Model | Test Accuracy |
|---|---:|
| Naive Bayes | 97.92% |
| Logistic Regression | 98.46% |
| Linear SVC | 98.77% |

**Best-performing model: Linear SVC (98.77% accuracy).**

Naive Bayes achieved the fastest training time among the three models.

*Note: These results are from the current notebook and should be re-evaluated after moving TF-IDF fitting inside the training split.*

## Model Evaluation

Linear SVC was further evaluated using:

- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix

## Interactive Language Prediction

The notebook includes a simple prediction function that accepts user-entered text and identifies its predicted language.

Example:

Input: 안녕하세요! 저는 한국어를 배우고 있어요.

Output: Korean

## How to Run

1. Download the repository.
2. Install the required Python libraries.
3. Place `language.csv` in the project directory.
4. Open `Language Detection v2.ipynb` in Jupyter Notebook.
5. Run the cells sequentially.
6. Enter sample text to test language predictions.

## Future Improvements

- Improve the preprocessing and evaluation pipeline to prevent data leakage.
- Evaluate performance on unseen multilingual text.
- Explore additional feature extraction techniques.
- Develop a simple interface for language prediction.
