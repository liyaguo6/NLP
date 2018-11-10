import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
matplotlib.style.use('ggplot')
# train_df = pd.read_csv(r"D:\MyProj\Company\day5\data\train.csv")
# train_df.Label = train_df.Label.astype("category")
# dsc = train_df.describe()



test_df =pd.read_csv(r"D:\MyProj\Company\day5\data\test.csv")
# test_df.Label = test_df.Label.astype("category")
# dsc = test_df.describe()
# print(dsc)
pd.options.display.max_colwidth = 50
thead=test_df.head()
print(thead)
print(test_df.Context[0])  #context内容

print((test_df.iloc[0,1:].values))  #所有回答以<class 'numpy.ndarray'>


# train_df.Label.hist()
# plt.title("Saple distrbution")
# plt.show()
# pd.options.display.max_colwidth = 50
# thead=train_df.head()
# print(thead)