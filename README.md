# Message Classification 

This project represents my first exploration into Machine Learning (ML), specifically centered on text classification. The aim is to construct a model that can predict whether a given message is authored by me or not.

## Demo

![Demo](/demo.gif)

## Prerequisites
Before running the code, ensure you have the following libraries installed:
- pandas
- scikit-learn
- spacy
- French language model for spaCy (`fr_core_news_sm`)

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/message-classification.git
   ```
2. Install the required Python packages:
    ```bash
    pip3 install -U spacy scikit-learn pandas
    ```
3. Download the French language model for spaCy:
    ```bash
    python3 -m spacy download fr_core_news_sm
    ```

## Usage
Run the script main.py:
    ```bash
    python3 main.py
    ```

## Model
The model used for classification is a Logistic Regression model. The text data is preprocessed using spaCy for lemmatization and stop word removal. The features are extracted using TF-IDF vectorization.

## Useful link
- [spacy](https://spacy.io/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [pandas](https://pandas.pydata.org/)