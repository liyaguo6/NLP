import pickle
import jieba  # 处理中文
from NB_CLF.settings import setting
from sklearn.naive_bayes import MultinomialNB  # 多项式分类


class Predict:
    def __init__(self):
        self.predict_features = []
        with open(setting.TRAIN_FEATURES_WORDS, 'rb') as f:
            self.features_words = pickle.load(f)
        with open(setting.TRAIN_FEATURES, 'rb') as f:
            self.train_features = pickle.load(f)
        with open(setting.TRAIN_CLASS, 'rb') as f:
            self.train_class = pickle.load(f)

    def text_processing(self, text):
        text_words_list = list(set(list(jieba.cut(text, cut_all=True))))
        self.predict_features.extend([1 if word in text_words_list else 0 for word in self.features_words])

    def text_classifier(self, text):
        self.text_processing(text)
        classifier = MultinomialNB().fit(self.train_features, self.train_class)
        predict_ret = classifier.predict_proba([self.predict_features])
        ret = classifier.predict([self.predict_features])
        return predict_ret, ret


if __name__ == '__main__':
    text = """
    
    """
    p = Predict()
    ret = p.text_classifier(text)
    print(ret)
