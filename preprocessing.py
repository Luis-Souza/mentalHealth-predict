import pandas as pd
import numpy as np
from sklearn.preprocessing import  LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class Processing:
    def __init__(self, data):
        self.data = data
        
    def passResponseTextsToNumbers(self):
        cols = ['Marriage','Depression','Panic','Anxiety','Treatment']
        for i in cols:
            self.data[i] = self.data[i].apply(lambda x:1 if x =='Yes' else 0)
        return self.data

    def passYersToNumber(self):
        self.data['Year'] = self.data['Year'].apply(lambda x: int(x[-1:]))
        return self.data

    # puts weight in values to column CGPA
    def change_cgpa(self, x):
        if (x == '3.50 - 4.00' or x == '3.50 - 4.00 '):
            x = 5
            return x
        elif x=='3.00 - 3.49' :
            x = 4
            return x
        elif x == '2.50 - 2.99':
            x = 3
            return x
        elif x== '2.00 - 2.49':
            x = 2
            return x
        else:
            x=1
            return x

    def putWeightsCGPA(self):
        self.data['CGPA'].value_counts().sort_values()
        self.data['CGPA'] = self.data['CGPA'].apply(lambda x:self.change_cgpa(x))
        return self.data

    def defineNumeberToRepresentationSex(self):
        self.data['Gender'] = self.data['Gender'].apply(lambda x: 0 if x == 'Female' else 1)
        return self.data
    
    def encodeLabels(self):
        le = LabelEncoder()
        self.data['Major'] = le.fit_transform(self.data['Major'])
        return self.data

    def dropColumn(self, columnName):
        self.data.drop(columnName,axis=1,inplace=True)
    
    def trainProcess(self):
        X=self.data.drop(['Depression'],axis=1).values
        y=self.data['Depression'].values

        X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.2)
        X_train.shape,X_test.shape,y_train.shape,y_test.shape

        scaler = MinMaxScaler()
        norm_X_train = scaler.fit_transform(X_train).astype(np.integer)
        norm_X_test = scaler.transform(X_test).astype(np.integer)

        return (norm_X_train, X_train, y_train, norm_X_test, y_test, X_test)
