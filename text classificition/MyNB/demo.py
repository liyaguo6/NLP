import numpy as np
def loadDataSet():
    '''

    创建实验样本，真实样本可能差很多，需要对真实样本做一些处理，如

    去停用词(stopwords)，词干化(stemming)等等，处理完后得到更"clear"的数据集，

    方便后续处理

    '''

    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],

                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],

                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],

                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],

                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],

                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

    classVec = [0, 1, 0, 1, 0, 1]  # 1代表存在侮辱性的文字，0代表不存在

    return postingList, classVec


def createVocabList(dataSet):
    '''

    将所有文档所有（去重的）词都存到一个列表中，可用set()函数去重。

    #用上set()函数操作符号|，取并集，或者写两重循环用vocabSet.add()

    return list(set([word for doc in dataSet for word in doc])

    [word for doc in dataSet for word in doc]: 用列表推导式将dataSet转为1维列表，

    set(XXX)： 将这个列表去重转为集合

    list(set(XXX)): 又转回来

    '''

    vocabSet = set([])

    for document in dataSet:
        vocabSet = vocabSet | set(document)

    return list(vocabSet)


# if __name__ == '__main__':
#     postingList, classVec = loadDataSet()
#     print(createVocabList(postingList))


def setOfWords2Vec(vocabList, inputSet):
    '''  词1,词2,XXX，词n    #词表vocabList

    doc1:  1, 0,...,1        #inputSet的输出结果，

    doc2:  0, 1,...,0

    '''

    returnVec = [0] * len(vocabList)  # 创建同vocabList同样长度的全0列表，也可[0 for i in range(len(vocabList))]

    for word in inputSet:  # 针对某篇inpustSet处理

        if word in vocabList:

            returnVec[vocabList.index(word)] = 1  # 找到某篇文档的词，其在词表中出现的位置，将其改为1

        else:

            print("the word: %s is not in my Vocabulary!" % word)

    return returnVec

# if __name__ == '__main__':
#     postingList, classVec = loadDataSet()
#     vocabList=createVocabList(postingList)
#     inputSet = ['star','yello','dalmation','worthless','love']
#     print(setOfWords2Vec(vocabList,inputSet))



def trainNB0(trainMatrix, trainCategory):
    """

    :param trainMatrix: [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1], [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0]]
    :param trainCategory:[0,1,0,1,0,1]
    :return:
    """
    numTrainDocs =len(trainMatrix)  # 文档的个数6

    numWords = len(trainMatrix[0])  # 词表的个数32

    pAbusive = sum(trainCategory) / float(numTrainDocs)  # sum([0,1,0,1,0,1])=3,也即是trainCategory里面1的个数 #先验概率

    p0Num = np.zeros(numWords)  # type(p0Num)为numpy.array类型  负例词在词表中的分布

    p1Num = np.zeros(numWords)   #正例词在词表中的分布

    p0Denom = 0.0     #负例词的个数

    p1Denom = 0.0      #正例词的个数

    for i in range(numTrainDocs):  # 对于每个文档进行处理

        if trainCategory[i] == 1:  # 对正例进行处理

            p1Num += trainMatrix[i]  # 将正例的训练集的实例按照行相加，

            p1Denom += sum(trainMatrix[i])  # 某篇正例文档，词的个数叠加。

        else:

            p0Num += trainMatrix[i]

            p0Denom += sum(trainMatrix[i])

    '''将正反例中的实例词向量分别叠加，分别除以正反例中词的个数，计算p(w0|ci),...,p(wn|ci)'''

    p1Vect = p1Num / p1Denom  # 所有正例实例按行叠加的向量p1Num，除以所有正例文档中词的个数p1Denom  正例词的概率分布

    p0Vect = p0Num / p0Denom  # p0Num为numpy.array类型，故可以array类型除以数字   负例词的概率分布

    return p0Vect, p1Vect, pAbusive





def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    '''

    待分类的向量vec2Classify，矩阵向量对应相乘piVec,与其相似的那个值必定大

    log(p(w|ci)*p(ci))=log(p(w0|ci)*p(w1|ci)*...p(wn|ci)*p(ci) = sum(log(p(wi|ci))) + log(p(ci))

    '''
    p1_vec=vec2Classify * p1Vec
    p0_vec=vec2Classify * p0Vec
    if np.sum(p1_vec)==0.0 and np.sum(p0_vec) !=0.0:
        return 0
    elif np.sum(p0_vec) == 0.0 and np.sum(p1_vec)!=0.0:
        return 1
    elif np.sum(p1_vec) ==0.0 and np.sum(p0_vec) ==0.0:
        return None
    else:
        p1 = sum(p1_vec) + np.log(pClass1)
        print('p1:', p1)
        p0 = sum(p0_vec) + np.log(1.0 - pClass1)
        print('p0:', p0)

        if p1 > p0:

            return 1

        else:

            return 0


def testingNB():
    '''

    封装所有操作

    '''

    listOPosts, listClasses = loadDataSet()

    myVocabList = createVocabList(listOPosts)

    trainMat = []

    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))

    p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    # print(p0V)  #[ 0.04166667  0.08333333  0.04166667  0.04166667  0.04166667  0.04166667 0.04166667  0.04166667  0.04166667  0.04166667  0. 0. ...]

    # print(p1V)  #[ 0.          0.05263158  0.          0.10526316  0.05263158  0.          0. ...]

    testEntry = ['love', 'my', 'dalmation', 'like']
    #     print(myVocabList)

    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))  #[0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0]
    print(testEntry, "classified as: ", classifyNB(thisDoc, p0V, p1V, pAb))  #['love', 'my', 'dalmation', 'like'] classified as:  0

    testEntry = ["li", "zii",'heh','buying', 'problems','buying','buying','worthless']

    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))  #['stupid', 'garbage'] classified as:  1
    #
    print(testEntry, "classified as: ", classifyNB(thisDoc, p0V, p1V, pAb))

if __name__ == '__main__':
    testingNB()
    """
    贝叶斯分类器：
    1读取所有文档，进行分词处理，去掉虚词，构建词表
    2构建每个类别的先验概率，和每一个类别中词汇在总词表中概率分布情况
        2.1去重处理
        2.2 不去重处理
    3 计算新文档分类：
        3.1 构建新文档在总词表中的频数分布
            3.1.1 去重
            3.1.2 不去重
        3.2 将先文档词表频数分布与之前每一个类别词表概率分布相乘
            log(p(ci|w))∝
             log(p(w|ci)*p(ci))=log(p(w0|ci)*p(w1|ci)*...p(wn|ci)*p(ci) 
             = sum(log(p(wi|ci))) + log(p(ci))
    """