import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import spacy
nlp = spacy.load("fr_core_news_sm")

raw_message_data = pd.read_csv(filepath_or_buffer='message.csv',sep=';')
message_data = raw_message_data.where((pd.notnull(raw_message_data)),'')
# print(message_data)

def preprocess_text(text):
    doc = nlp(text)
    lemmatized_text = " ".join([token.lemma_ for token in doc if not token.is_stop])
    return lemmatized_text

X = message_data['Message']
Y = message_data['IsMine']
X_preprocessed = [preprocess_text(text) for text in X]

X_train, X_test, Y_train, Y_test = train_test_split(X_preprocessed, Y, test_size=0.2, random_state=3)
# print(X_train.shape)
# print(X_test.shape)
feature_extraction = TfidfVectorizer(min_df = 1, lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

model = LogisticRegression()
model.fit(X_train_features, Y_train)

prediction_on_training_data = model.predict(X_train_features)
prediction_on_test_data = model.predict(X_test_features)

# accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
# print('Accuracy on training data : ', accuracy_on_training_data)
# accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
# print('Accuracy on test data : ', accuracy_on_test_data)

def predict_message(message):
    input_message = [preprocess_text(message)]
    input_data_features = feature_extraction.transform(input_message)
    prediction = model.predict(input_data_features)
    print("Message class : " + "Other" if prediction[0] == 0 else "Mine")
    
def input_messsage():
    message = input("Enter your message : ")
    predict_message(message)

input_messsage()
