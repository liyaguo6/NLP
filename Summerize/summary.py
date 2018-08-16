import pynlpir
from ctypes import *
import os
import numpy as np
from numpy import linalg
import heapq
import re



pynlpir.open()


class Summarizer:
    """
    概括文章主旨大意，固定输出段落首句，主要利用文本相似度的方法做关键句子选择。
    """

    def __init__(self, length, maxSumarySize=2):
        self.length = length
        self.maxSumarySize = maxSumarySize
        self.sentence_simlarity = {}
        self.result = []

    # 分割句子
    def sentence(self):
        sentenceList = []
        sentenceList1 = re.split("[\？\。\！\......\!\.\?]", self.paragraph)
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

    # 计算文本相似度
    def cal_text_simliarity(self):
        position = self.position_vlookup()
        for seentence in self.sentence():
            tokens = pynlpir.segment(seentence, pos_names='all', pos_tagging=False)
            for word in tokens:
                try:
                    # self.position_vlookup()[word] 关键词位置参数信息
                    self.new_sentence_wordlist[position[word]] += 1
                except KeyError:
                    continue
            cos = self.cosin(self.new_sentence_wordlist, self.key_weight)
            self.sentence_simlarity[seentence] = cos
            self.new_sentence_wordlist = [0] * len(self.key_words)
        return self.sentence_simlarity

    # 输出摘要
    def get_result(self, paragraph):
        self.paragraph = paragraph
        self.segments = pynlpir.segment(self.paragraph, pos_names='all', pos_tagging=False)
        self.key_words = pynlpir.get_key_words(self.paragraph, weighted=False, max_words=20)
        self.new_sentence_wordlist = [0] * len(self.key_words)
        key_words = pynlpir.get_key_words(self.paragraph, max_words=20, weighted=True)
        self.key_weight = [item[1] for item in key_words]
        sentence_dict = self.cal_text_simliarity()
        keys = list(sentence_dict.keys())
        val = list(sentence_dict.values())
        temp = sorted(list(map(val.index, heapq.nlargest(self.maxSumarySize, val))))
        for i in temp[:2]:
            if keys[i] != self.sentence()[0]:
                self.result.append(keys[i])
        self.result.insert(0, self.sentence()[0])
        if len(",".join(self.result)) < self.length:
            self.result.append(keys[temp[2]])
        return ",".join(self.result)


pynlpir.close()
