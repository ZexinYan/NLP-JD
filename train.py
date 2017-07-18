#coding:utf-8
import pandas as pd
import os
import codecs
import pickle
import DataSpider
from sklearn.utils import shuffle
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectFromModel
# # from xgboost import XGBRegressor
from sklearn.utils import column_or_1d
sns.set_style("whitegrid")


class SentimentClassifier:
    def __init__(self, filename='JD_train.csv'):
        self.model = None
        self.load_model()
        self.preprocessor = DataSpider.Preprocessor()
        if os.path.exists(filename):
            data = pd.read_csv(filename)
            self.data = shuffle(data)
            X_data = pd.DataFrame(data.drop('sentiment', axis=1))
            Y_data = column_or_1d(data[:]['sentiment'], warn=True)
            self.X_train, self.X_val,\
            self.y_train, self.y_val = train_test_split(X_data, Y_data, test_size=0.3, random_state=1)
        else:
            print('No Source!')
            self.preprocessor.get_new_data()

    def load_model(self, filename='model.pickle'):
        if os.path.exists(filename):
            with codecs.open(filename, 'rb') as f:
                f = open(filename, 'rb')
                self.model = pickle.load(f)
        else:
            self.train()

    def save_model(self, filename='model.pickle'):
        with codecs.open(filename, 'wb') as f:
            pickle.dump(self.model, f)

    def train(self):
            self.model = LinearSVC(random_state=7)
            self.model.fit(self.X_train, self.y_train)
            self.save_model()
            print('Accuracy: ' + str(round(self.model.score(self.X_val, self.y_val), 2)))

    def predict(self, sentence):
        vec = self.preprocessor.sentence2vec(sentence)
        print(self.model.predict(vec))

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
        scoring = 'f1'
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
    text = '东西收到了，小巧轻便，看着挺不错的，还没有使用，应该没问题。'
    classifier = SentimentClassifier()
    # classifier.show_heat_map()
    # classifier.show_heat_map_to()
    classifier.choose_best_model()
    # classifier.predict(text)
