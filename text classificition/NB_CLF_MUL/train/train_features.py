import jieba  #处理中文
import nltk  #处理英文
import collections
import pickle
import sklearn
import os
from NB_CLF_MUL.settings import setting

class Train:
    def __init__(self,**kwargs):
        self.features_dim =kwargs.get('features_dim',5000)
        self.all_words_list =[]
        self.train_data_list =[]
        self.train_class_list=[]
        self.folder_list = os.listdir(setting.DB_DIRS)
        self.feature_words = []
        self.flag =kwargs.get('flag','sklearn')
        self.text_processing()
        self.words_dict()
    def text_processing(self):
        for folder in self.folder_list:
            new_folder_path = os.path.join(setting.DB_DIRS, folder)
            files = os.listdir(new_folder_path)
            # 读取文件
            j =1
            for file in files:
                if j > 100:  # 怕内存爆掉，只取100个样本文件，你可以注释掉取完
                    break
                with open(os.path.join(new_folder_path, file), 'r', encoding='utf-8') as fp:
                    raw = fp.read()
                word_list = list(jieba.cut(raw,cut_all=True))
                self.train_data_list.append(word_list)
                self.train_class_list.append(folder)
                j +=1


        with open(setting.TRAIN_CLASS,'wb') as f:
            pickle.dump(self.train_class_list,f)


        word_list=[]
        for text_list in self.train_data_list:
            word_list.extend(text_list)
        all_words_list = collections.Counter(word_list)

        #key函数利用词频进行降序排序
        #方法一：
        # all_words_list=all_words_dict.most_common(1000)
        # all_words_list = list(list(zip(*all_words_list))[0])
        #方法二：
        all_words_tuple_list =sorted(all_words_list.items(),key=lambda f:f[1],reverse=True)
        self.all_words_list.extend(list(list(zip(*all_words_tuple_list))[0]))
        # print(self.all_words_list)
        return self.all_words_list,self.train_data_list,self.train_class_list

    def words_dict(self,deleteN=0):
        """
        # 选取特征词
        :param stopwords_set:
        :return:
        """
        with open(setting.STOPWORDS,'rb') as f:
            stopwords_set = set(pickle.load(f))
        n = 1
        for t in range(deleteN, len(self.all_words_list), 1):
            if n > self.features_dim:  # feature_words的维度1000
                break
            if not self.all_words_list[t].isdigit() and self.all_words_list[t] not in stopwords_set and 1 < len(
                    self.all_words_list[t]) < 5:
                self.feature_words.append(self.all_words_list[t])
                n += 1
        with open(setting.TRAIN_FEATURES_WORDS,'wb') as f:
            pickle.dump(self.feature_words, f)
        return self.feature_words

    def features(self,text):
        text_dict = dict(collections.Counter(text))
        ## -----------------------------------------------------------------------------------
        if self.flag == 'nltk':
            ## nltk特征 dict
            features = {word: text_dict[word] if word in text_dict else 0 for word in self.feature_words}
        elif self.flag == 'sklearn':
            ## sklearn特征 list
            features = [text_dict[word] if word in text_dict else 0 for word in self.feature_words]
        else:
            features = []
            ## -----------------------------------------------------------------------------------
        return features

    def train(self):
        train_feature_list = [self.features(text) for text in self.train_data_list]
        with open(setting.TRAIN_FEATURES,'wb') as f:
            pickle.dump(train_feature_list,f)




if __name__ == '__main__':
    t =Train(features_dim=5000)
    t.train()