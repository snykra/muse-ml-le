
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import pandas as pd
import numpy as np

class MLModels:

    x_train = []
    x_test = []
    y_train = []
    y_test = []

    performance = dict()

    def __init__(self, data, y_col, test_size = 0.2):
        x = data.drop(y_col, axis = 1) # just features
        x = self.clean_dataset(x)
        y = data[y_col] # just one column
        y = self.encode_labels(y)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=test_size)

    def encode_labels(self, y_col): 
        labels = {
            '0.0': 0,
            'very low': 1, 
            'low': 2,
            'neutral': 3, 
            'high': 4, 
            'very high': 5
            }

        for i in range(len(y_col)):
            y_col[i] = labels[y_col[i]]
        
        return list(y_col)

    def clean_dataset(self, df):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        return df

    def get_accuracy(self, model):
        y_pred = model.predict(self.x_test)
        acc = accuracy_score(self.y_test, y_pred)
        return acc

    def gaussian_naive_bayes(self):
        print("Fitting a Gaussian naive bayes model...")

        gnb_model = GaussianNB()
        gnb_model.fit(self.x_train, self.y_train)

        acc = self.get_accuracy(gnb_model)
        self.performance['gaussian naive bayes'] = acc
        
        print("Accuracy (Gaussian Naive Bayes): " + str(acc))

        return gnb_model
    
    def logistic_regression(self):
        print("Fitting a logistic regression model...")

        lr_model  = LogisticRegression(solver = 'lbfgs', multi_class = 'auto')
        lr_model.fit(self.x_train, self.y_train)

        acc = self.get_accuracy(lr_model)
        self.performance['logistic regression'] = acc

        print("Accuracy (Logistic Regression): " + str(acc))

        return lr_model

    def decision_tree(self): 
        print("Fitting a decision tree classifier...") 

        dt_model = DecisionTreeClassifier()
        dt_model.fit(self.x_train, self.y_train)

        acc = self.get_accuracy(dt_model)
        self.performance['decision tree'] = acc

        print("Accuracy (Decision Tree Classifier): " + str(acc))

        return dt_model

    def k_nearest_neighbors(self):
        print("Fitting a K-Nearest Neighbors model...") 

        knn_model = KNeighborsClassifier(n_neighbors = 3, algorithm = 'ball_tree')
        knn_model.fit(self.x_train, self.y_train)

        acc = self.get_accuracy(knn_model)
        self.performance['knn'] = acc
        
        print("Accuracy (K-Nearest Neighbors): " + str(acc))

        return knn_model
    
    def support_vector_machine(self):
        print("Fitting a Support Vector Machine...")

        svm_model = SVC(C=50, kernel='rbf', gamma=1)
        svm_model.fit(self.x_train, self.y_train)

        acc = self.get_accuracy(svm_model)
        self.performance['svm'] = acc

        print("Accuracy (SVM): " + str(acc))

    def all_models(self):
        print("Fitting all available types of models...")
        gnb = self.gaussian_naive_bayes()
        lr = self.logistic_regression()
        dt = self.decision_tree()
        knn = self.k_nearest_neighbors()
        svm = self.support_vector_machine()

        print(self.performance)

        return gnb, lr, dt, knn, svm
        


