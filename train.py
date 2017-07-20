#coding:utf-8
import codecs
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.model_selection import learning_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import column_or_1d
from sklearn.utils import shuffle
import Preprocessor
sns.set_style("whitegrid")


class SentimentClassifier:
    def __init__(self, filename='./corpus/train.csv'):
        if os.path.exists(filename):
            data = pd.read_csv(filename)
            self.data = shuffle(data)
            X_data = pd.DataFrame(data.drop('sentiment', axis=1))
            Y_data = column_or_1d(data[:]['sentiment'], warn=True)
            self.X_train, self.X_val,\
            self.y_train, self.y_val = train_test_split(X_data, Y_data, test_size=0.3, random_state=1)
            self.model = None
            self.load_model()
            self.preprocessor = Preprocessor.Preprocessor()
        else:
            print('No Source!')
            self.preprocessor.process_data()

    def load_model(self, filename='./model/model.pickle'):
        if os.path.exists(filename):
            with codecs.open(filename, 'rb') as f:
                f = open(filename, 'rb')
                self.model = pickle.load(f)
        else:
            self.train()

    def save_model(self, filename='./model/model.pickle'):
        with codecs.open(filename, 'wb') as f:
            pickle.dump(self.model, f)

    def train(self):
            self.model = LogisticRegression(random_state=3)
            self.model.fit(self.X_train, self.y_train)
            self.save_model()
            print('Accuracy: ' + str(round(self.model.score(self.X_val, self.y_val), 2)))

    def predict(self, sentence):
        vec = self.preprocessor.sentence2vec(sentence)
        return self.model.predict(vec)

    def predict_test_set(self, sentences, pos_file='./test/pos_test.txt', neg_file='./test/neg_test.txt'):
        pos_set = []
        neg_set = []
        for each in sentences:
            score = self.predict(each)
            if score == 1:
                pos_set.append(each)
            elif score == -1:
                neg_set.append(each)
        with codecs.open(pos_file, 'w', 'utf-8') as f:
            for each in pos_set:
                f.write(each + '\n')
            f.close()
        with codecs.open(neg_file, 'w', 'utf-8') as f:
            for each in neg_set:
                f.write(each + '\n')
            f.close()

    def show_heat_map(self):
            pd.set_option('precision', 2)
            plt.figure(figsize=(20, 6))
            sns.heatmap(self.data.corr(), square=True)
            plt.xticks(rotation=90)
            plt.yticks(rotation=360)
            plt.suptitle("Correlation Heatmap")
            plt.show()

    def show_heat_map_to(self, target='sentiment'):
            correlations = self.data.corr()[target].sort_values(ascending=False)
            plt.figure(figsize=(40, 6))
            correlations.drop(target).plot.bar()
            pd.set_option('precision', 2)
            plt.xticks(rotation=90, fontsize=7)
            plt.yticks(rotation=360)
            plt.suptitle('The Heatmap of Correlation With ' + target)
            plt.show()

    def plot_learning_curve(self):
        # Plot the learning curve
        plt.figure(figsize=(9, 6))
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, X=self.X_train, y=self.y_train,
            cv=3, scoring='neg_mean_squared_error')
        self.plot_learning_curve_helper(train_sizes, train_scores, test_scores, 'Learning Curve')
        plt.show()

    def plot_learning_curve_helper(self, train_sizes, train_scores, test_scores, title, alpha=0.1):
        train_scores = -train_scores
        test_scores = -test_scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        plt.plot(train_sizes, train_mean, label='train score', color='blue', marker='o')
        plt.fill_between(train_sizes, train_mean + train_std,
                         train_mean - train_std, color='blue', alpha=alpha)
        plt.plot(train_sizes, test_mean, label='test score', color='red', marker='o')
        plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, color='red', alpha=alpha)
        plt.title(title)
        plt.xlabel('Number of training points')
        plt.ylabel(r'Mean Squared Error')
        plt.grid(ls='--')
        plt.legend(loc='best')
        plt.show()

    # def feature_reduction(self, X_train, y_train, X_val):
    #     thresh = 5 * 10 ** (-3)
    #     # model = XGBRegressor()
    #     model.fit(X_train, y_train)
    #     selection = SelectFromModel(model, threshold=thresh, prefit=True)
    #     select_X_train = selection.transform(X_train)
    #     select_X_val = selection.transform(X_val)
    #     return select_X_train, select_X_val

    def choose_best_model(self):
        seed = 7
        pipelines = []
        pipelines.append(
            ('SVC',
             Pipeline([
                 ('Scaler', StandardScaler()),
                 ("SVC", SVC(random_state=seed))
             ])
             )
        )
        pipelines.append(
            ('AdaBoostClassifier',
             Pipeline([
                 ('Scaler', StandardScaler()),
                 ('AdaBoostClassifier', AdaBoostClassifier(random_state=seed))
             ]))
        )
        pipelines.append(
            ('RandomForestClassifier',
             Pipeline([
                 ('Scaler', StandardScaler()),
                 ('RandomForestClassifier', RandomForestClassifier(random_state=seed))
             ]))
        )
        pipelines.append(
            ('RandomForestClassifier',
             Pipeline([
                 ('Scaler', StandardScaler()),
                 ('RandomForestClassifier', RandomForestClassifier(random_state=seed))
             ]))
        )
        pipelines.append(
            ('LinearSVC',
             Pipeline([
                 ('Scaler', StandardScaler()),
                 ('LinearSVC', LinearSVC(random_state=seed))
             ]))
        )
        pipelines.append(
            ('KNeighborsClassifier',
             Pipeline([
                 ('Scaler', StandardScaler()),
                 ('KNeighborsClassifier', KNeighborsClassifier())
             ]))
        )

        pipelines.append(
            ('GaussianNB',
             Pipeline([
                 ('Scaler', StandardScaler()),
                 ('GaussianNB', GaussianNB())
             ]))
        )

        pipelines.append(
            ('Perceptron',
             Pipeline([
                 ('Scaler', StandardScaler()),
                 ('Perceptron', Perceptron(random_state=seed))
             ]))
        )
        pipelines.append(
            ('SGDClassifier',
             Pipeline([
                 ('Scaler', StandardScaler()),
                 ('SGDClassifier', SGDClassifier(random_state=seed))
             ]))
        )
        pipelines.append(
            ('DecisionTreeClassifier',
             Pipeline([
                 ('Scaler', StandardScaler()),
                 ('DecisionTreeClassifier', DecisionTreeClassifier(random_state=seed))
             ]))
        )
        pipelines.append(
            ('LogisticRegression',
             Pipeline([
                 ('Scaler', StandardScaler()),
                 ('LogisticRegression', LogisticRegression(random_state=seed))
             ]))
        )
        scoring = 'accuracy'
        n_folds = 10
        results, names = [], []
        for name, model in pipelines:
            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            cv_results = cross_val_score(model, self.X_train, self.y_train, cv=kfold,
                                         scoring=scoring, n_jobs=-1)
            names.append(name)
            results.append(cv_results)
            msg = "%s: %f (+/- %f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)


if __name__ == '__main__':
    # DataSpider.Preprocessor().get_new_data()
    classifier = SentimentClassifier()
    classifier.train()
    classifier.plot_learning_curve()
    # classifier.show_heat_map()
    # classifier.show_heat_map_to()
    classifier.choose_best_model()
    # classifier.predict(text)
    # test_set = []
    # with codecs.open('./test/test.txt', 'r', 'utf-8') as f:
    #     for each in f.readlines():
    #         test_set.append(each)
    # classifier.predict_test_set(test_set)
