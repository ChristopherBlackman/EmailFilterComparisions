import numpy as np
import pandas as pd
from sklearn import model_selection as ms
from sklearn import feature_extraction as fe
from modules.data_extractor import extractor
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.metrics import precision_recall_fscore_support
import time

validation_list = []
precision_list = []
time_list = []

for i in range(1,10):
    # init
    start = 0.0
    end   = 0.0
    prediction = dict()
    kwarg = {'test_size':i*0.1}
    ((x_train,x_test),(y_train,y_test)) = extractor(file='data/emails.csv',**kwarg).get()

    model = MultinomialNB()
    model.fit(x_train,x_test)
    

    #prediction
    start = time.time()
    prediction["naive_bayes"] = model.predict(y_train)
    end = time.time()

    precision,recall,fscore,support = precision_recall_fscore_support(  y_test,
                                                                        prediction['naive_bayes'],
                                                                        average='macro')
    precision_list.append(precision)
    validation_list.append(kwarg['test_size'])
    time_list.append((end-start)/y_test.shape[0])
    print(precision)


name = __file__.split(".")[0]
df = pd.DataFrame({ 'accuracy':precision_list,
                    'time_per_iteration(s)':time_list,
                    'validation_list':validation_list})

df.to_csv("output/{0}.csv".format(name))

print(validation_list)
print(precision_list)
print(time_list)

