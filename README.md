# Multilingual Language Detection Using Machine Learning

## Project Overview

This project compares three machine learning models for identifying the language of a given text across **22 languages**. It uses character-level TF-IDF features and evaluates Multinomial Naive Bayes, Logistic Regression, and Linear SVC on the same stratified test split.

## Dataset

- Input file: `language.csv` with `Text` and `language` columns.
- Original size: **22,000 samples** (1,000 per language), covering **22 languages**.
- Missing values: **0**.
- Duplicate texts detected and removed: **141**.
- Remaining samples: **21,859**.
- Stratified split (`random_state=42`): **14,645 training** and **7,214 testing** samples (33% test size).

The dataset includes Arabic, Chinese, Dutch, English, Estonian, French, Hindi, Indonesian, Japanese, Korean, Latin, Persian, Portugese, Pushto, Romanian, Russian, Spanish, Swedish, Tamil, Thai, Turkish, and Urdu. The spellings `Portugese` and `Pushto` follow the dataset labels.

## Technologies Used

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Jupyter Notebook.

## Methodology

1. Inspected dataset shape, data types, missing values, class distribution, and duplicate texts.
2. Removed duplicate texts and prepared the text and language-label columns.
3. Created a stratified training/test split.
4. Fitted character-level TF-IDF **only on training text** and transformed the test text separately to avoid data leakage. Settings: `analyzer='char_wb'`, 1–3 character n-grams, maximum 50,000 features, lowercase conversion, and sublinear term frequency.
5. Trained and compared Multinomial Naive Bayes, Logistic Regression, and Linear SVC.
6. Evaluated the best-performing model with a classification report, confusion matrix, sample predictions, and error analysis.

## Results

The following values come from the notebook's **shared model-comparison training loop**. Training times are run-dependent.

| Model | Test Accuracy | Training Time |
|---|---:|---:|
| Multinomial Naive Bayes | 97.88% | 0.67 s |
| Logistic Regression | 98.43% | 46.68 s |
| **Linear SVC** | **98.82%** | **10.71 s** |

**Best test accuracy:** Linear SVC, **98.8217355142778%** (approximately **98.82%**) on **7,214** test samples. **Fastest training in the comparison loop:** Multinomial Naive Bayes.

### Linear SVC Classification Report (summary)

| Metric | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Macro average | 0.989 | 0.988 | 0.988 | 7,214 |
| Weighted average | 0.989 | 0.988 | 0.988 | 7,214 |

The notebook also includes a full per-language classification report and a 22-class confusion matrix. English had **0.853 precision** and **1.000 recall** in that report; this illustrates why per-class metrics matter alongside overall accuracy.

**Timing note:** The notebook separately retrains each model for learning purposes, producing slightly different training times (Naive Bayes 0.61 s, Logistic Regression 40.06 s, Linear SVC 10.46 s). The table above consistently uses the original comparison-loop measurements, not a mixture of both runs.

## Interactive Prediction

The notebook defines `predict_language(text)`, which transforms user-entered text using the fitted TF-IDF vectorizer and returns the Linear SVC prediction. It also includes example sentences and an `input()` prompt.

The classifier can only predict one of its **22 trained languages**. For example, German and Italian are not included; the notebook's sample predictions for them should not be treated as correct language identification.

## Limitations and Error Analysis

The notebook inspects the first ten misclassified test examples. Several contain text that appears inconsistent with their assigned language labels, suggesting **possible** labeling issues; these have not been independently verified. The model also predicts a trained class even for unsupported languages and has not been specifically evaluated on code-switched text.

## What I Learned

- How to inspect and clean a multilingual text dataset before training.
- How character-level TF-IDF n-grams represent language patterns across different writing systems.
- Why the vectorizer must be fitted on training data only to avoid test-data leakage.
- How to compare classifiers using both accuracy and training time.
- How precision, recall, F1-score, confusion matrices, and manual error analysis reveal issues that overall accuracy can miss.
- How dataset-label quality and unsupported languages affect real-world predictions.

## Future Work

- **Deployment:**  I plan to deploy this model as an interactive web application using **Streamlit** and **Hugging Face Spaces** to provide a live demo.
- **Deep Learning:** Explore Transformer-based models such as **XLM-RoBERTa** to improve handling of code-switching (mixed-language text), with evaluation needed to establish whether they outperform the current model.
- Investigate suspected label inconsistencies and test performance on shorter texts and unsupported-language inputs.

## How to Run

1. Place `language.csv` in the same working directory as the notebook.
2. Install the Python libraries listed above.
3. Open `Language Detection v2 (1)(2).ipynb` in Jupyter Notebook.
4. Run the cells in order to preprocess the dataset, train the models, evaluate Linear SVC, and try the interactive prediction function.

**Note:** The notebook requires the separate `language.csv` dataset; it is not embedded in the notebook.
