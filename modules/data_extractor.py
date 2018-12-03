'''
authour : Christopher Blackman
'''

import numpy as np
import pandas as pd

from sklearn import model_selection as ms
from sklearn import feature_extraction as fe

'''
name : extractor
purpose : file reader for email csv files, 
transforms labeled files into a document frequency matrix of users choosing, 
boiler plate to keep testing data consistent for the rest of the project.
'''
class extractor:
    def __init__(self,**kwargs):
        self.file = kwargs.get('file',None)
        if self.file == None:
            raise FileNotFoundError()

        self.test_size = kwargs.get('test_size',0.2)

        self.vector = kwargs.get('vector',fe.text.CountVectorizer)

        self.seed = kwargs.get('seed',42)

        self.stop_words = kwargs.get('stop_words','english')

    def get(self):
        # retrieve data set
        dataset = pd.read_csv(self.file)
        x_train, x_test, y_train, y_test = ms.train_test_split(dataset['text'],dataset['spam'],test_size = self.test_size, random_state = self.seed)

        #create a courpus model for our data
        vect = self.vector(stop_words=self.stop_words).fit(x_train)
        
        #transform data into matrix 
        x_train_df = vect.transform(x_train)
        y_train_df  = vect.transform(x_test)

        return ((x_train_df, y_train),(y_train_df, y_test))

    def nomalized(self):
        print("Normalizing")
        ((x_train,x_test),(y_train,y_test)) = self.get()

        trX = np.array(x_train.A.astype(float),copy=True)
        teX = np.array(y_train.A.astype(float),copy=True)

        for row in range(trX.shape[0]):
            trX[row,:] = trX[row,:]/np.max(trX[row,:])

        for row in range(teX.shape[0]):
            teX[row,:] = teX[row,:]/np.max(teX[row,:])
        
        trY = self.__one_hot(x_test.values)
        teY = self.__one_hot(y_test.values)

        return ((trX, trY),(teX, teY))

    def __one_hot(self,x):
        m = np.unique(x).size
        matrix = np.zeros([x.shape[0],m])
        for i in range(x.shape[0]):
            matrix[i,x[i]] = 1
        return matrix
