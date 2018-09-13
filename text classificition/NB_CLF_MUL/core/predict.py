import pickle
import jieba  # 处理中文
from NB_CLF_MUL.settings import setting
from sklearn.naive_bayes import MultinomialNB  # 多项式分类
import numpy as np
import collections

class Predict:
    def __init__(self,**kwargs):
        self.text = kwargs.get('text')
        self.predict_features = []
        with open(setting.TRAIN_FEATURES_WORDS, 'rb') as f:
            self.features_words = pickle.load(f)
        with open(setting.TRAIN_FEATURES, 'rb') as f:
            self.train_features = pickle.load(f)
        with open(setting.TRAIN_CLASS, 'rb') as f:
            self.train_class = pickle.load(f)

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


if __name__ == '__main__':
    text = """
    双卡双待在中国市场是一个比较广泛的需求，目前大多数安卓手机基本具备这一功能。此前早就有声音呼吁苹果针对中国市场推出双卡双待功能，此次发布会前也有猜测苹果将推出这一功能，而最终传闻成真。
    """
    # p = Predict(text=text)
    p = Predict(**{'text':text})
    ret = p.predict()
    print(ret)
