import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer

#导入数据
test_df = pd.read_csv(r"D:\MyProj\Company\day5\data\test.csv")
train_df = pd.read_csv(r"D:\MyProj\Company\day5\data\train.csv")
# 评估准则
def evaluate_recall(y, y_test, k=1):
    num_examples = float(len(y))
    num_correct = 0
    # print(y, y_test)
    for predictions, label in zip(y, y_test):
        if label in predictions[:k]:
            num_correct += 1
    return num_correct / num_examples


# 随机预测
def predict_random(context, utterances):
    return np.random.choice(len(utterances), 10, replace=False)



# y_random =[predict_random(test_df.Context[x],test_df.iloc[x,1:].values) for x in range(len(test_df) )]
# y_test = np.zeros(len(y_random))
# print(y_random)
# print(y_test)


#随机预测模型
# for n in [1,2,5,9,10]:
#     print("Recall@({},10):{:g}".format(n,evaluate_recall(y_random,y_test,n)))


#tf-idf模型
class TFIDPredictor:
    def __init__(self):
        self.vectorsizer = TfidfVectorizer()
    def train(self,data):
        self.vectorsizer.fit(np.append(data.Context.values,data.Utterance.values))
    def predict(self,context,utterances):

        #把文本内容转化成tf-idf向量
        vector_context = self.vectorsizer.transform([context])
        vector_doc =self.vectorsizer.transform(utterances)
        #评估向量之间的相近程度
        result = np.dot(vector_doc,vector_context.T).todense()
        result = np.asarray(result).flatten()
        #排序输出
        return np.argsort(result,axis=0)[::-1]


pred = TFIDPredictor()
pred.train(train_df)
y_random =[pred.predict(test_df.Context[x],test_df.iloc[x,1:].values) for x in range(len(test_df) )]
y_test = np.zeros(len(y_random))
for n in [1,2,5,9,10]:
    print("Recall@({},10):{:g}".format(n,evaluate_recall(y_random,y_test,n)))






#test zip func
# df = np.array([[2, 3, 3],
#                [1, 2, 1],
#                [3, 3, 3],
#                [4, 4, 4]])
#
# dt = np.zeros(4)
# # print(zip(df,dt))
# for k,y in zip(df,dt):
#     print(k,y)