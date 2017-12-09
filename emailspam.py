import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt
import seaborn as sns
data =  pd.read_csv("C:/Users/Chetan.Chougle/Downloads/sms-spam-collection-dataset/spam.csv" ,  encoding='latin-1')
#print(data.head())

data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
data = data.rename(columns={"v1":"label", "v2":"text"})
data["label_num"] =  data.label.map({'ham' :0 , 'spam' : 1 })

print(data.head())
print(data.label.value_counts())

from sklearn.model_selection import train_test_split

X_train, X_test , y_train , y_test =  train_test_split(data["text"] ,data["label"] ,  test_size = 0.2 , random_state = 10 )
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.feature_extraction.text import TfidfVectorizer
vector =  TfidfVectorizer()
vector.fit(X_train)
print(vector.get_feature_names()[0:20])
print(vector.get_feature_names()[-20:])

X_train_df =  vector.transform(X_train)
print(X_train)
print("transformed")
print(X_train_df)
X_test_df = vector.transform(X_test)
print(X_test)
print(X_test_df)
type(X_test_df)
print(data[data.label_num == 1])

ham_words = ''
spam_words = ''
spam = data[data.label_num == 1]
ham = data[data.label_num ==0]
import nltk
from nltk.corpus import stopwords
for val in spam.text:
    text = val.lower()
    tokens = nltk.word_tokenize(text)
    #tokens = [word for word in tokens if word not in stopwords.words('english')]
    for words in tokens:
        spam_words = spam_words + words + ' '

for val in ham.text:
    text = val.lower()
    tokens = nltk.word_tokenize(text)
    for words in tokens:
        ham_words = ham_words + words + ' '

from wordcloud import  WordCloud
spam_wordcloud = WordCloud(width=600, height=400).generate(spam_words)
ham_wordcloud = WordCloud(width=600, height=400).generate(ham_words)

plt.figure( figsize=(10,8), facecolor='k')
plt.imshow(spam_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

prediction = dict()
from sklearn.naive_bayes  import MultinomialNB
model =  MultinomialNB()
model.fit(X_train_df ,  y_train)
prediction["Multinomial"] = model.predict(X_test_df)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print(accuracy_score(y_test,prediction["Multinomial"]))

##########################################################################################
