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

