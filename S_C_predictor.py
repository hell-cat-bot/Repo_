import joblib 
from keras.saving import load_model
import re

model=load_model('sentiment_classifier.h5')
vectorizer= joblib.load('tfidf_vectorizer.pkl')

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text= re.sub(r'[^a-zA-Z]', ' ',text)
    return text.lower()

def predict(text):
    text = clean_text(text)
    vectorized= vectorizer.transform([text]).toarray()
    y_predict = model.predict(vectorized)
    y_predicted = y_predict.argmax(axis=1)[0]
    y_map={0:'Negative', 1:'Neutral', 2:'Positive'}
    prediction= y_map[y_predicted]
    print(f'Predicted Sentiment is {prediction}')

predict('life is miserable right now')