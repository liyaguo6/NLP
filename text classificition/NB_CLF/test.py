import pickle
# from NB_CLF.settings import setting
import pandas as pd
data =[]
# with open(r'database\stop_words.txt','r',encoding='utf-8') as f:
#     words = f.readlines()
#     for line in words:
#         data.append(line.replace('\n',''))
#     data.extend(['\n','nbsp','','\t'])
# with open('database\stop_words.pkl','wb') as f:
#     pickle.dump(data,f)
#
# with open('database\stop_words.pkl','rb') as f:
#     data=pickle.load(f)
# print(type(data))



df = pd.read_csv(r'D:\MyProj\NLP\text classificition\NB_CLF\database\test.csv',encoding='gbk')

print(list(df['text']))
print(list(df['lables']))