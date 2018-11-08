import pandas as pd
from Tex_classfication.settings import setting
import jieba
import pickle
from sklearn.naive_bayes import MultinomialNB  # 多项式分类
import collections

class Test:
    def __init__(self,**kwargs):
        self.test_features = []
        with open(setting.TRAIN_FEATURES_WORDS, 'rb') as f:
            self.features_words = pickle.load(f)
        with open(setting.TRAIN_FEATURES, 'rb') as f:
            self.train_features = pickle.load(f)
        self.train_class =[1,0]
        df = pd.read_csv(setting.TEST_FILES,encoding='gbk')
        self.test_text= list(df['text'])
        self.test_lables = list(df['lables'])
        self.text_processing()


    def features(self,text):
        text_dict = dict(collections.Counter(text))
        features = [text_dict[word] if word in text_dict else 0 for word in self.features_words]
        return features


    def text_processing(self):
        text_words_list = [list(jieba.cut(text, cut_all=True)) for text in self.test_text]
        self.test_features.extend([self.features(text1) for text1 in text_words_list])

    def test(self):
        classifier = MultinomialNB().fit(self.train_features, self.train_class)
        test_accuracy = classifier.score(self.test_features, self.test_lables)
        return test_accuracy


if __name__ == '__main__':
    # df = pd.read_csv(setting.TEST_FILES,encoding='gbk')
    # print(df)
    p = Test()
    ret=p.test()
    print(ret)
