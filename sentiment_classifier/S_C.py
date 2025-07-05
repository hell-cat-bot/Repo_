import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Input
from keras.regularizers import l2
from keras.utils import Sequence
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np
import joblib


#A Keras-compatible generator that loads text data in small batches, applies TF-IDF, and feeds it to the model without using too much RAM
class TfidfDataGenerator(Sequence):
    def __init__(self, texts, labels, vectorizer, batch_size=2048):
        self.texts = texts
        self.labels = labels
        self.vectorizer = vectorizer
        self.batch_size = batch_size
        self.indices = np.arange(len(texts))

    def __len__(self):                                                                          #Tells keras how many batches per epoch / no of times keras calls this fn per epoch  
        return int(np.ceil(len(self.texts) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]         #for idx=0 ,batch_idices varies from [0,batch_size]
        batch_texts = [self.texts[i] for i in batch_indices]
        batch_labels = np.array([self.labels[i] for i in batch_indices])

        X_batch = self.vectorizer.transform(batch_texts).toarray()                              #Vectorizing here
        return X_batch, batch_labels


def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return text.lower()

df = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding='latin-1', header=None)
df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']
df = df[['target', 'text']]
df['target'] = df['target'].map({0: 0, 2: 1, 4: 2})  # Map to 0: neg, 1: neutral, 2: pos
df.head()
df['text'] = df['text'].apply(clean_text)

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'], df['target'], test_size=0.2, random_state=42)

np.unique(test_labels, return_counts=True)


# Fit vectorizer on training data only -- We are just fitting the data here as it is getting vectorized while being fed to the model 
vectorizer = TfidfVectorizer(max_features=3000)
vectorizer.fit(train_texts)

train_gen = TfidfDataGenerator(train_texts.tolist(), train_labels.tolist(), vectorizer, batch_size=2048)
test_gen = TfidfDataGenerator(test_texts.tolist(), test_labels.tolist(), vectorizer, batch_size=2048)

model = Sequential([
    Input(shape=(3000,)),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dense(64, activation='relu',  kernel_regularizer=l2(0.001)),
    Dense(3, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(train_gen, validation_data=test_gen, epochs=5)

#saving the model
model.save('sentiment_classifier.h5')
#saving the vectorizer
joblib.dump( vectorizer , 'tfidf_vectorizer.pkl')

# Plot accuracy
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

y_pred = model.predict(test_gen).argmax(axis=1)                            #predicting in small batches using test_gen
print(classification_report(test_labels, y_pred))