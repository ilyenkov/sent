# -*- coding: utf-8 -*-
__author__ = 'xead'
from sklearn.externals import joblib
import os


class SentimentClassifier(object):
    def __init__(self):
        self.model = joblib.load("./w6-model.pkl")
        self.transformer = joblib.load("./w6-transformer.pkl")
        self.vectorizer = joblib.load("./w6-vectorizer.pkl")
        self.classes_dict = {0: u'негативный отзыв', 1: u'положительный отзыв', -1: u'ошибка классификации'}

    @staticmethod
    def get_probability_words(probability):
        #классификатор не даёт вероятностей, к сожалению
        if probability < 0.55:
            return "neutral or uncertain"
        if probability < 0.7:
            return "probably"
        if probability > 0.95:
            return "certain"
        else:
            return ""

    def predict_text(self, text):
        try:
            vectorized = self.vectorizer.transform([text.replace('\r', ' ').replace('\n', ' ')])
            transformed = self.transformer.transform(vectorized)
            return self.model.predict(transformed)[0]
        except:
            print u'Ошибка классификации'
            return -1       

    def predict_list(self, list_of_texts):
        try:
            vectorized = self.vectorizer.transform(list_of_texts)
            transformed = self.transformer.transform(vectorized)
            return self.model.predict(transformed)
        except:
            print u'Ошибка классификации'
            return None

    def get_prediction_message(self, text):
        prediction = self.predict_text(text)
        class_prediction = prediction
        return self.classes_dict[class_prediction]