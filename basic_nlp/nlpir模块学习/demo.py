import pynlpir
from ctypes import *
import os

os.path.dirname(__file__)

pynlpir.open()
# 分词与词性标注
# s = '因为我比较懒,所以我就只是修改了这句话,代码还是原博客的'
# segments = pynlpir.segment(s)
# segments = pynlpir.segment(s,pos_names='all')
# for segment in segments:
#     print(segment)


# 关键词提取
# key_words = pynlpir.get_key_words(s, weighted=True)
# for key_word in key_words:
#     print (key_word[0], '\t', key_word[1])

# 段落摘要提取
# 为什么天体都是球形的


# 关键词提取
# key_words = pynlpir.get_key_words(paragraph , weighted=True)
# for key_word in key_words:
#     print (key_word[0], '\t', key_word[1])
# # print(key_words)

##仅支持英文分词
# import nltk
# result=nltk.word_tokenize(paragraph)
# for item in result:
#     print(item)

#
# def getWordCounts(segments):
#     """
#     # 计算文本的词频，生成一个列表，比如[(10,'the'), (3,'language'), (8,'code')...]
#     :param segments:
#     :return:
#     """
#     word_dict = {}
#     wordFrequences = []
#     for word in segments:
#         word_dict[word.strip("\n")] = word_dict.get(word.strip("\n"), 0) + 1
#     for key, val in word_dict.items():
#         wordFrequences.append((val, key))
#     return wordFrequences


# def filtStopWords(paragraph, segments):
#     """
#     #过滤虚词,提取关键字 按照词频的大小进行排序，形成的列表为['code', 'language'...]
#     :param segments:
#     :return:
#     """
#     key_words = pynlpir.get_key_words(paragraph, weighted=False)
#     print(key_words)
#     wordFrequences = getWordCounts(segments)
#     contentWordFrequences = []
#     contentWordsSortbyFreq = []
#     for word in wordFrequences:
#         if word[1] in key_words:
#             contentWordFrequences.append(word)
#     # contentWordsSortbyFreq = sorted(contentWordFrequences,key=lambda x:x[0] )
#     print(contentWordFrequences)
#     d = sorted(contentWordFrequences, reverse=True)
#     for word in d:
#         contentWordsSortbyFreq.append(word[1])
#     print(contentWordsSortbyFreq)
#     return contentWordsSortbyFreq





# def getSentences(paragraph):
#     """
#     # # 将文章分成句子
#     :param paragraph:
#     :return:
#     """
#     sentenceList = []
#     sentenceList1 = re.split("[\?\。\!]", paragraph)
#     for sentence in sentenceList1:
#         if sentence != "\n":
#             sentenceList.append(sentence.replace("\n", ""))
#     return sentenceList


# print(getSentences(paragraph))


#########################结巴########################
import jieba
import jieba.analyse

# 关键词提取
# tf = jieba.analyse.extract_tags(paragraph,topK=30,withWeight=True,allowPOS=())
# print(tf)

# jieba.load_userdict('userdict.txt')
# def stopwordslist(filepath):
#     stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
#     return stopwords

# 文本相似度分析
import nltk

# from nltk import FreqDist
# crops = 'this is my sentence '\
#      'this is my life '\
#      'this is the end '
# tokens = nltk.word_tokenize(crops)

# fdist = FreqDist(tokens)
# print(fdist.values())
# standard_freq_vector =fdist.most_common(20)  #最常用单词统计
# print(standard_freq_vector)
# # print(standard_freq_vector)
# def position_vlookup(v):
#     res={}
#     count = 0
#     for word in v:
#         res[word[0]] =count
#         count += 1
#     return res
# standard_word_position = position_vlookup(standard_freq_vector)
# print(position_vlookup(standard_freq_vector))  #记录每个单词位置{'the': 3, 'my': 1, 'end': 4, 'sentencethis': 5, 'is': 0, 'life': 6, 'this': 2}
# freq_vector = [0]*len(standard_freq_vector)
# print(freq_vector)
# new_sentensce = 'this is a test sentence apple boy girl!'
# #
# tokens = nltk.word_tokenize(new_sentensce)
# #
# for word in tokens:
#     try:
#         freq_vector[standard_word_position[word]] +=1
#     except KeyError:
#         continue
# #
# print(freq_vector)

import numpy as np
from numpy import linalg
import heapq
import re

class Summarizer:
    """
    概括文章主旨大意，固定输出段落首句，主要利用文本相似度的方法做关键句子选择。
    """
    def __init__(self, paragraph, maxSumarySize=2):
        self.paragraph = paragraph
        self.maxSumarySize = maxSumarySize
        self.segments = pynlpir.segment(paragraph, pos_names='all', pos_tagging=False)
        self.key_words = pynlpir.get_key_words(paragraph, weighted=False, max_words=20)
        self.new_sentence_wordlist = [0] * len(self.key_words)
        key_words = pynlpir.get_key_words(paragraph, max_words=20, weighted=True)
        self.key_weight = [item[1] for item in key_words]
        self.sentence_simlarity = {}
        self.result = []

    # 分割句子
    def sentence(self):
        sentenceList = []
        sentenceList1 = re.split("[\?\。\!]", self.paragraph)
        for sentence in sentenceList1:
            if sentence != "\n":
                sentenceList.append(sentence.replace("\n", ""))
        return sentenceList

    # 记录关键词位置
    def position_vlookup(self):
        res = {}
        count = 0
        for word in self.key_words:
            res[word] = count
            count += 1
        return res

    # 计算两个两个向量的余弦值
    def cosin(self, list1, list2):
        v1 = np.array(list1)
        v2 = np.array(list2).T
        denom = linalg.norm(v1) * linalg.norm(v2)
        v1v2 = sum(v2 * v1)
        return v1v2 / denom
    #计算文本相似度
    def cal_text_simliarity(self):
        for seentence in self.sentence():
            tokens = pynlpir.segment(seentence, pos_names='all', pos_tagging=False)
            for word in tokens:
                try:
                    # self.position_vlookup()[word] 关键词位置参数信息
                    self.new_sentence_wordlist[self.position_vlookup()[word]] += 1
                except KeyError:
                    continue
            cos = self.cosin(self.new_sentence_wordlist, self.key_weight)
            self.sentence_simlarity[seentence] = cos
            self.new_sentence_wordlist = [0] * len(self.key_words)
        return self.sentence_simlarity
    #输出摘要
    def get_result(self):
        sentence_dict = self.cal_text_simliarity()
        keys = list(sentence_dict.keys())
        val = list(sentence_dict.values())
        temp = sorted(list(map(val.index, heapq.nlargest(self.maxSumarySize, val))))
        for i in temp:
            if keys[i]  != self.sentence()[0]:
                self.result.append(keys[i])
        self.result.insert(0, self.sentence()[0])
        return ",".join(self.result)

if __name__ == '__main__':
    paragraph = u"太阳像一个无比炽热的大火球，每时每刻都在发光发热。它的亮度，是其他任何天体都无法与之相匹敌的，它比肉眼能见到的最暗星要亮10多万亿倍。如果把一层12米厚的冰壳覆盖在太阳表面，那么1分钟后，太阳发出的热量，就能将这层冰壳完全融化。而在人类有史可查的漫长岁月中，人们未曾发现太阳的光和热有丝毫的减弱。那么，如此巨大而持久的能量究竟是从哪里来的呢？原来太阳中的燃料是氢，它燃烧后的余烬则是氦，氢的聚变反应产生了太阳能。太阳的表面是厚达500千米的热气沸腾的“海洋”，而不像地球那样坚固。太阳中心核反应释放出的能量，经过几千年缓慢而费力的旅途，最后突破光球层，发出耀眼的光芒。在光球层上，气体开始变得透明，使太阳光线可以射向宇宙空间。所以，在太阳上所发生的燃烧过程并非如一 般人想象的那样是太阳内部的物质燃烧的结果。太阳内部进行着的氢转变为氦的热核反应才是其产生巨大能量的源泉。太阳上贮藏的氢至少还可以供给太阳像现在这样继续辉煌地闪耀50亿年！即使太阳上的氢全部燃 烧完毕，也还会有其他的热核反应继续发生，因此太阳还是可以继续发射出它那巨大的光和热来!"
    summar = Summarizer(paragraph, maxSumarySize=2)
    # result = summar.cal_text_simliarity()
    # result=summar.position_vlookup()
    result = summar.get_result()
    print(result)







pynlpir.close()
