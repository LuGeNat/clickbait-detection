#!/usr/bin/python3
import numpy as np
from sklearn import svm
import sklearn.metrics as skm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import scipy.sparse
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def normalized_mean_squared_error(truth, predictions):
    norm = skm.mean_squared_error(truth, np.full(len(truth), np.mean(truth)))
    return skm.mean_squared_error(truth, predictions) / norm


class ClickbaitModel(object):
    __regression_measures = {'Explained variance': skm.explained_variance_score,
                             'Mean absolute error': skm.mean_absolute_error,
                             'Mean squared error': skm.mean_squared_error,
                             'Median absolute error': skm.median_absolute_error,
                             'R2 score': skm.r2_score,
                             'Normalized mean squared error': normalized_mean_squared_error}

    __classification_measures = {'Accuracy': skm.accuracy_score,
                                 'Precision': skm.precision_score,
                                 'Recall': skm.recall_score,
                                 'F1 score': skm.f1_score}

    def __init__(self):
        self.models = {"LogisticRegression": LogisticRegression(),
                       "MultinomialNB": MultinomialNB(),
                       "RandomForestClassifier": RandomForestClassifier(),
                       "SVR_linear": svm.SVR(kernel='linear'),
                       "SVR": svm.SVR(),
                       "Ridge": Ridge(alpha=1.0, solver="auto"),
                       "Lasso": Lasso(),
                       "ElasticNet": ElasticNet(),
                       "SGDRegressor": SGDRegressor(),
                       "RandomForestRegressor": RandomForestRegressor()}
        self.model_trained = None

    def classify(self, x, y, model, evaluate=True, cross_val=False, confusion=False):
        if isinstance(model, str):
            self.model_trained = self.models[model]
        else:
            self.model_trained = model
        if evaluate:
            x_train, x_test, y_train, y_test = train_test_split(x, y.T, random_state=42)
        else:
            x_train = x
            y_train = y

        '''
        # remove some of the dominating class
        nr_ones = np.sum(y_train)
        nr_zeroes = len(y_train) - np.sum(y_train)
        keep_indices = list(range(len(y_train)))
        remove_indices = []
        while abs(nr_ones - nr_zeroes) - len(remove_indices) != 0:
            index = np.random.randint(len(y_train))
            if (y_train[index] == 0 and nr_ones < nr_zeroes) or \
                (y_train[index] == 1 and nr_ones > nr_zeroes):
                y_train = np.delete(y_train, index)
                remove_indices.append(index)
                del keep_indices[index]

        x_train = x_train[keep_indices]

        # supersample clickbait class
        while np.sum(y_train) != len(y_train) - np.sum(y_train):
            index = np.random.randint(len(y_train))
            if (y_train[index] == 1 and nr_ones < nr_zeroes) or \
                (y_train[index] == 0 and nr_ones > nr_zeroes):
                y_train = np.append(y_train, y_train[index])
                x_train = scipy.sparse.vstack((x_train, x_train[index]))
        print(y_train.shape)
        print(x_train.shape)'''

        self.model_trained.fit(x_train, y_train)

        if evaluate:
            self.eval_classify(y_test, self.model_trained.predict(x_test))

        if cross_val:
            scores = cross_val_score(self.model_trained, x, y, cv=10)
            print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        if confusion:
            names = np.asanyarray(['male', 'female', 'non-binary'])
            scores = cross_val_predict(self.model_trained, x, y, cv=10)
            conf_mat = confusion_matrix(y_train, scores)
            print(conf_mat)
            self.plot_confusion_matrix(y_train, scores, names, normalize=False, title=None, cmap=plt.cm.Blues)
            np.set_printoptions(precision=2)
            plt.show()

    def regress(self, x, y, model, evaluate=True, cross_val=False):
        if isinstance(model, str):
            self.model_trained = self.models[model]
        else:
            self.model_trained = model
        if evaluate:
            x_train, x_test, y_train, y_test = train_test_split(x, y.T, random_state=42)
        else:
            x_train = x
            y_train = y

        self.model_trained.fit(x_train, y_train)

        if evaluate:
            y_predicted = self.model_trained.predict(x_test)
            for rm in self.__regression_measures:
                print("{}: {}".format(rm, self.__regression_measures[rm](y_test, y_predicted)))

        if cross_val:
            scores = cross_val_score(self.model_trained, x, y, cv=10)
            print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


    def predict(self, x):
        return self.model_trained.predict(x)

    def eval_classify(self, y_test, y_predicted):
        for cm in self.__classification_measures:
            print("{}: {}".format(cm, self.__classification_measures[cm](y_test, y_predicted)))
        print("ROC-AUC: {}".format(skm.roc_auc_score(y_test, y_predicted)))

    def eval_regress(self, y_test, y_predicted):
        for rm in self.__regression_measures:
            print("{}: {}".format(rm, self.__regression_measures[rm](y_test, y_predicted)))

    def save(self, filename):
        joblib.dump(self.model_trained, filename)

    def load(self, filename):
        self.model_trained = joblib.load(filename)

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, classes,
                              normalize=False,
                              title=None,
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        timestamp = time.time()
        dt_object = datetime.fromtimestamp(timestamp)git

        if not title:
            if normalize:
                title = 'Normalized confusion matrix'  + str(dt_object)
            else:
                title = 'Confusion matrix, without normalization' + str(dt_object)

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        classes = classes[unique_labels(y_true, y_pred)]

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return ax


if __name__ == "__main__":
    pass
