import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


STOPWORDS = os.path.join(BASE_DIR,'database\\features\\stop_words.pkl')
TRAIN_FEATURES_WORDS = os.path.join(BASE_DIR,'database\\features\\train_features_words.pkl')
TRAIN_FEATURES = os.path.join(BASE_DIR,'database\\features\\train_features.pkl')
TRAIN_CLASS = os.path.join(BASE_DIR,'database\\features\\train_class.pkl')
TEST_FILES =os.path.join(BASE_DIR,'database\\NB_test.csv')