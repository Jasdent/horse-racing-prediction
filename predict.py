from utils_global import *
#import preprocess

from model import *

import os
import pdb

from preprocessing import df_prob as df
# FILE_PATH = "./data/HR200709to201901.csv"
# class rf_model(object):
#     def __init__(
#         self, 
#         Xtrain : np.array = None, 
#         ytrain : np.array = None, 
#         Xtest: np.array = None,
#         ytest: np.array = None
#     ):
#         self.clf = RandomForestClassifier( 
#             n_estimators = 130,
#             max_depth =  13,
#             max_features = 'log2', # with 11 features
#             random_state = 0,
#             n_jobs = 5
#         )

#         self.Xtrain : np.array = Xtrain
#         self.ytrain : np.array = ytrain
#         self.Xtest : np.array = Xtest
#         self.ytest : np.array = ytest
#     def accuracy_info(self):
#         print("Accuracy on test set: {}".format(self.clf.score(self.Xtest, self.ytest)))

def custom_feature_eng(df : pd.DataFrame):
    # further feature engineering that might vary in each submission
    '''
    1. make use of jockey and trainer, quantifying it into more meaningful figures
    2. ranking re-define class
    '''
    # df.drop(df['rank'][(df['rank'] == 0) | (df['rank'] > 10)].index,inplace = True)
    # print(set(df['rank']))
    
    # for ranking 4 - 10, it is no longer too important to be very specific

    # if os.path.exists("./data/trainer.json") and os.path.exists('./data/jockey.json'):
    #     print("Load data from history.")
    #     trainer_dict = json.load(open("./data/trainer.json"))
    #     jockeys_dict = json.load(open("./data/jockey.json"))
    # else:
    #     # df_source = pd.read_csv('./data/HR200709to201901.csv')
    #     # df_source.drop(df_source[(df_source['tname'].isna()) | (df_source['jname'].isna())].index, inplace = True)
    #     # ONLY RUN WITH TRAINING 
    #     trainers = list(set(df['tname']))
    #     jockeys = list(set(df['jname']))
    #     # scores of three types
    #     trainer_dict = dict(zip(trainers, [0] * len(trainers)))
    #     jockeys_dict = dict(zip(jockeys, [0] * len(jockeys)))

    #     # score = central tendency + performance fluctuation, consider data skewing in the future
    #     for t in trainers:
    #         trainer_dict[t] = df['rank'][df['tname'] ==t].mean()
    #     for j in jockeys:
    #         jockeys_dict[j] = df['rank'][df['jname'] == j].mean() + np.log2(1 + df['rank'][df['jname'] ==j].var())
    #     # compute the score contributed by trainer and jocker
    #     json.dump(trainer_dict, open('./data/trainer.json','w'))
    #     json.dump(jockeys_dict, open('./data/jockey.json', 'w'))
    
    # tjscore = [0] * len(df['rank'])
    # pdb.set_trace()
    # for i in range(len(df['rank'])):
    #     if i % 10000 == 0:
    #         print("checkpoint {}".format(i))
    #     try:
    #         tjscore[i] += trainer_dict[df.loc[i]['tname']]
    #     except:
    #         tjscore[i] += 5
    #     try:
    #         tjscore[i] += jockeys_dict[df.loc[i]['jname']]
    #     except:
    #         tjscore[i] += 5 + np.log2(10)
    
    df['rank'][(df['rank'] >= 4) & (df['rank'] <= 6)] = 4
    df['rank'][df['rank'] >= 7] = 5
    # onehot for categorical
    # cat = 'track going'.split()
    # onehot_list = [pd.get_dummies(df[i]) for i in cat]
    # for onehot in onehot_list:
    #     df[onehot.columns] = pd.DataFrame(onehot)
    # df.drop(cat, axis = 1, inplace = True)
    # df['t_j_score'] = pd.Series(tjscore)
    
    print(df.info())

custom_feature_eng(df)

only_test = ['index']

# data splitting and cross validation
X_df, y_df = df.drop(['rank']+only_test,axis=1), df['rank']

models = model(None,None,X_df,y_df)

if os.path.exists(MODEL_PATH[0]):
    with open(MODEL_PATH[0], 'rb') as f:
        models.clf_rf = pickle.load(f)
    with open(MODEL_PATH[1], 'rb') as f:
        models.clf_gb = pickle.load(f)
else:
    print("set mode in utils to 'train' then run the train.py")
models.accuracy_info()


win_prob = (models.clf_rf.predict_proba(X_df.values)[:,0] + models.clf_gb.predict_proba(X_df.values)[:,0]) / 2
place_prob = (np.sum(models.clf_rf.predict_proba(X_df.values)[:,:3], axis = 1)  + np.sum(models.clf_gb.predict_proba(X_df.values)[:,:3], axis = 1)) / 2
df['win_prob'] = pd.Series(win_prob)
df['place_prob'] = pd.Series(place_prob)

# Model Loading or Training

# 新的数据 给到 test_x 的位置

# output_prob = output_prob_tab(W, test_x)
fixratio = 0.01
mthresh = 9
df['winstake'] = fixratio * (df['win_prob']  > 0.9333)
# df['plastake'] = fixratio * (df['place_prob'] * df['place_t5'] > mthresh)

df['plastake'] = fixratio * (df['place_prob']  > prob_thres)

results = df[
    ['index','win_prob','place_prob','winstake','plastake']
]

print("finish getting 4 columns")
# res 就是 第一和前三 可能性
data = pd.read_csv(FILEPATH)

data[results.columns] = pd.DataFrame(results)
data.to_csv("./data/results_test.csv")
res = data
res['win_prob'].fillna(0)
res['place_prob'].fillna(0)
res['winstake'].fillna(0)
res['plastake'].fillna(0)
