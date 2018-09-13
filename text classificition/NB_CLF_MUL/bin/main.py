from NB_CLF_MUL.core.acc_test import Test
from NB_CLF_MUL.core.predict import Predict
from NB_CLF_MUL.train.train_features import Train
from NB_CLF_MUL.settings import setting



def text_classifier(mode='predict',**kwargs):
    if mode == 'test':
        obj =Test(**kwargs)
    elif mode == 'predict':
        obj = Predict(**kwargs)
    elif mode == 'train':
        obj = Train(**kwargs)
    else:
        raise ArithmeticError("{}方法错误".format(mode))
    ret = getattr(obj, mode)
    return ret()




if __name__ == '__main__':
    text = '双卡双待在中国市场是一个比较广泛的需求，目前大多数安卓手机基本具备这一功能。此前早就有声音呼吁苹果针对中国市场推出双卡双待功能，此次发布会前也有猜测苹果将推出这一功能，而最终传闻成真。'
    result=text_classifier(flag='sklearn',mode='predict',text=text)
    print(result)
