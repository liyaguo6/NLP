from nltk.corpus import brown
from nltk.corpus import stopwords
import nltk

data = brown.categories()
# print(data)
# print(len(brown.sents()))


# sentence = 'may I help you ? you of at in'
# tokens = nltk.word_tokenize(sentence)
# print(tokens)
# filtered_words = [word for word in tokens if word not in stopwords.words('english')]
# print(filtered_words)


# 情感分析
# from nltk.classify import NaiveBayesClassifier
# s1 = 'this is a good book'
# s2 = 'this is a awesome book'
# s3 = 'this is a bad book'
# s4 = 'this is a terrible book'
# def preprocess(s):
#     return {word :True for word in s.lower().split()}
#
# # print(preprocess(s4))
# training_data =[[preprocess(s1),'pos'],
#                 [preprocess(s2),'pos'],
#                 [preprocess(s3),'neg'],
#                 [preprocess(s4),'neg']
#                 ]
# model = NaiveBayesClassifier.train(training_data)
#
# print(model.classify(preprocess('good terrible a bad boy !')))


# 文本相似度分析
# from nltk import FreqDist
# crops = 'this is my sentence '\
#      'this is my life '\
#      'this is the end '
#
# tokens = nltk.word_tokenize(crops)
# fdist = FreqDist(tokens)
# # print(fdist.values())
# standard_freq_vector =fdist.most_common(20)  #最常用单词统计
# size = len(standard_freq_vector)
# # print(size)
# print(standard_freq_vector)
# def position_vlookup(v):
#     res={}
#     count = 0
#     for word in v:
#         res[word[0]] =count
#         count += 1
#     return res
# standard_word_position = position_vlookup(standard_freq_vector)
# # print(position_vlookup(standard_freq_vector))  #记录每个单词位置{'the': 3, 'my': 1, 'end': 4, 'sentencethis': 5, 'is': 0, 'life': 6, 'this': 2}
# freq_vector = [0]*size
# # print(freq_vector)
# new_sentensce = 'this is a test sentence !'
#
# tokens = nltk.word_tokenize(new_sentensce)
#
# for word in tokens:
#     try:
#         freq_vector[standard_word_position[word]] +=1
#     except KeyError:
#         continue
#
# print(freq_vector)


# 文本分类（NLtk 实现tf-idf)
# from nltk.text import TextCollection
# corpus = TextCollection(['this is my sentence ',
#      'this is my life ',
#      'this is the end '])
# #计算ftidff
# print(corpus.tf_idf('sentence','this is sentence four'))


# new_sentence = 'this is sentence five'
# for word in starand_vocab:  # 文本预处理后word_list
#     print(corpus.tf_idf(word,new_sentence))


# 分词
import jieba
import jieba.posseg as psg

l = '习近平同塞内加尔总统萨勒共同出席塞内加尔竞技摔跤场项目移交仪式的在了么'
seg_list = jieba.cut(l,cut_all=True)  #全模式
# print(seg_list)
print('/'.join(seg_list))

# seg_list = jieba.cut(l,cut_all=False)  #精确模式 ,默认是精确模式
# print('/'.join(seg_list))


# seg_list = jieba.cut_for_search(l)  #搜所引擎模式
# print('/'.join(seg_list))


# 看词性
# data = [(k.word,k.flag) for k in psg.cut(l)]
# print(data)

# 只想获取分词结果列表中的名词，
# print([(k.word,k.flag) for k in psg.cut(l) if k.flag.startswith('n') or k.flag.startswith('v')])

#去除虚词
# from nltk.corpus import stopwords
# data = [k.word for k in psg.cut(l)]
# data_list = [ v for v in data if v not in stopwords.words['chinese']]  #不支持中文
# print(data_list)
"""
默认格式下的用户词典"dict/userdict：
世界经济论坛 n
达沃斯论坛 n
World Economic Forum n
Davos Forum n

"""

# 加载用户词典

# jieba.load_userdict("dict/userdict")

# 测试用户词典


# data=[
#    "世界经济论坛也叫达沃斯论坛。",
#    "The World Economic Forum is also called the Davos Forum."
#     ]
#
# for d in data:
#    seg_list = jieba.cut(d)
#    #词与词之间用","连接
#    print(",".join(seg_list))


# 默认格式用户词典下中英文分词结果

# 世界经济论坛,也,叫,达沃斯论坛,。
# The, ,World, ,Economic, ,Forum, ,is, ,also, ,called, ,the, ,Davos, >,Forum,.

# 关键词提取

# import jieba.analyse
#
# kWords = jieba.analyse.extract_tags(
#     "此外，公司拟对全资子公司吉林欧亚置业>有限公司增资4.3亿元，增资后，吉林欧亚置业注册资本由7000万元增加到5亿元。>吉林欧亚置业主要经营范围为房地产开发及百货零售等业务。目前在建吉林欧亚城>市商业综合体项目。2013年，实现营业收入0万元，实现净利润-139.13万元。",
#     topK=10, withWeight=True)
# for word, weight in kWords:
#     # print(word + ":" + weight)
#     print(word, ":", weight)
