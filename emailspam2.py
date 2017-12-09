import numpy as np
import pandas as pd


data =  pd.read_csv("C:/Users/Chetan.Chougle/Downloads/sms-spam-collection-dataset/spam.csv" ,  encoding='latin-1' )
data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
data = data.rename(columns={"v1":"label" , "v2":"text"})
data["label_num"] = data.label.map({"ham":0 , "spam": 1 })


#print(data.head())
#print(data.label.value_counts)

from sklearn.model_selection import train_test_split
X_train,y_train,X_test,y_test =  train_test_split(data["text"] , data["label"] , test_size = 0.2 , random_state = 10)

negatives = {}
positives ={}
alpha = 1
pA = 1
pNotA = 1
positivesTotal = 0
negativesTotal = 0
# for index, row in data.iterrows():
#     print("tell 11")
#
#     print(row['text'])
#     print(row['label'])
#     print("end here")
#


def train():

    total = 0
    numSpam = 0
    for index, row in data.iterrows():
        if row['label'] == "spam":
            numSpam+=1
        total+=1
        processEmail(row['text'],row['label'])
    pA = numSpam/float(total)
    pNotA =  (total - numSpam)/float(total)


def processEmail(body,label):
    global positivesTotal , negativesTotal
    for word in body:
        if label == "spam":
            positives[word] =  positives.get(word , 0 ) + 1
            positivesTotal += 1
        else:
            negatives[word] =  negatives.get(word , 0 ) + 1
            negativesTotal += 1
#    print("negativesTotal")
#    print(negativesTotal)
#    print("positivesTotal")
#    print(positivesTotal)

def typeofWord(word , spam):
    if spam:
        return (positives.get(word,0)+alpha)/(float) (positivesTotal   )
    return (negatives.get(word,0)+alpha)/(float)(negativesTotal   )

def typeofEmail(body,spam):
    result =1.0
    for word in body:
        result *= typeofWord(word,spam)

    return result    

def classify(email):
    isSpam =  pA * typeofEmail(email , True)
    isNotSpam =  pNotA * typeofEmail(email , False)

    return isSpam > isNotSpam

train()
print(classify("Profits Money Free"))
print(classify("Hi Hello , Pizza was delicious"))
