import jieba
import logging
import os

s=jieba.load_userdict("test")
import multiprocessing

sentences = ["中华人民共和国中国人民解放军新巴特科技智能"]

# ret = jieba.cut(sentences)
# print(list(ret))

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence,PathLineSentences
# segment_file = open( '_segment', 'a', encoding='utf8')
# with open('input.text', encoding='utf8') as f:
#     for sentence in f.readlines():
#         sentence = list(jieba.cut(sentence))
#         segment_file.write(" ".join(sentence))

# model = Word2Vec(LineSentence('_segment'), size=12, window=5, min_count=1,
#                  workers=multiprocessing.cpu_count())
# model.save('test_model')
# model.wv.save_word2vec_format("test_vector_model", binary=True)
# #
# model =Word2Vec.load('test_model')
# print(model.most_similar('历史'))
# with open('test_model','rb') as f:
#     data =f.read()
#     print(data)



