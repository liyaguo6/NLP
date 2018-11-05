import os
import re
import jieba  #处理中文
import collections
import pickle
from Tex_classfication.settings import setting


class Train:
    def __init__(self,path1,path0,**kwargs):
        self.features_dim =kwargs.get('features_dim',10000)
        self.path1=path1
        self.path0=path0
        self.all_words_list =[]
        self.train_data_list =[]
        self.train_class_list=["1","0"]
        self.feature_words = []
        self.flag =kwargs.get('flag','sklearn')
        self.text_processing()
        self.words_dict()
    def text_processing(self):

        with open(self.path0, 'r', encoding='gbk') as fp:
            raw0 = fp.read()
        with open(self.path1, 'r', encoding='gbk') as hp:
            raw1 =hp.read()
        raw=raw1+raw0
        self.word_list1 = list(jieba.cut(raw1,cut_all=True))
        self.word_list0 = list(jieba.cut(raw0,cut_all=True))
        self.train_data_list.append(self.word_list1)
        self.train_data_list.append(self.word_list0)
        word_list = list(jieba.cut(raw, cut_all=True))
        all_words_dict = collections.Counter(word_list)
        #key函数利用词频进行降序排序
        #方法一：
        # all_words_list=all_words_dict.most_common(1000)
        # all_words_list = list(list(zip(*all_words_list))[0])
        #方法二：
        all_words_tuple_list =sorted(all_words_dict.items(),key=lambda f:f[1],reverse=True)
        self.all_words_list.extend(list(list(zip(*all_words_tuple_list))[0]))


    def words_dict(self):
        """
        # 选取特征词
        :param stopwords_set:
        :return:
        """
        with open(setting.STOPWORDS,'rb') as f:
            stopwords_set = set(pickle.load(f))
        n = 1
        for t in range(0,len(self.all_words_list)):
            if n > self.features_dim:  # feature_words的维度1000
                break
            if not self.all_words_list[t].isdigit() and self.all_words_list[t] not in stopwords_set and 1 < len(
                    self.all_words_list[t]) < 6 and not re.search('\n\d+|\s|\d\n|[^\u4e00-\u9fff]{1,}',self.all_words_list[t]):
                self.feature_words.append(self.all_words_list[t])
                n += 1
        with open(setting.TRAIN_FEATURES_WORDS,'wb') as f:
            pickle.dump(self.feature_words, f)
        return self.feature_words

    def features(self,text):
        text_dict = dict(collections.Counter(text))
            ## sklearn特征 list
        features = [text_dict[word] if word in text_dict else 0 for word in self.feature_words]
        return features

    def train(self):
        train_feature_list = [self.features(text) for text in self.train_data_list]
        with open(setting.TRAIN_FEATURES,'wb') as f:
            pickle.dump(train_feature_list,f)

if __name__ == '__main__':
    t =Train('./database/NB_train_1.csv','./database/NB_train_0.csv',features_dim=10000)
    # t.train()
    t.train()