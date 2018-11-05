import os
import time
import random
import jieba  #处理中文
import collections
import pickle
import sklearn
from Tex_classfication.settings import setting
from sklearn.naive_bayes import MultinomialNB  #多项式分类
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class Predict:
    def __init__(self,**kwargs):
        self.text = kwargs.get('text')
        self.predict_features = []
        with open(setting.TRAIN_FEATURES_WORDS, 'rb') as f:
            self.features_words = pickle.load(f)
        with open(setting.TRAIN_FEATURES, 'rb') as f:
            self.train_features = pickle.load(f)
            self.train_class = ['1','0']

    def text_processing(self, text):
        text_words_list = list(jieba.cut(text, cut_all=True))
        text_words_dict = dict(collections.Counter(text_words_list))
        self.predict_features.extend([text_words_dict[word] if word in text_words_dict else 0 for word in self.features_words])

    def predict(self):
        self.text_processing(self.text)
        classifier = MultinomialNB().fit(self.train_features, self.train_class)
        predict_ret = classifier.predict_proba([self.predict_features])
        ret = classifier.predict([self.predict_features])[0]
        predict_acc = str(predict_ret.max()*100)[:4] + '%'
        return '预测类别:{0}，精确度:{1}'.format(ret,predict_acc)