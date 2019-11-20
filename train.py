
from utils_global import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

from preprocessing import df_prob as df

from model import *

print("============================Prediction stage============================")



def custom_feature_eng(df : pd.DataFrame):
    # further feature engineering that might vary in each submission
    '''
    1. make use of jockey and trainer, quantifying it into more meaningful figures
    2. ranking re-define class
    '''
    
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
    


    df['rank'][(df['rank'] >= 4) & (df['rank'] <= 6) | df['rank'] == 0] = 4
    df['rank'][df['rank'] >= 7] = 5
    # onehot for categorical
    # cat = 'track venue going'.split()
    # onehot_list = [pd.get_dummies(df[i]) for i in cat]
    # for onehot in onehot_list:
    #     df[onehot.columns] = pd.DataFrame(onehot)
    # df.drop(cat, axis = 1, inplace = True)
    # df['t_j_score'] = pd.Series(tjscore)
    
    # df.drop(['jname','tname'], axis = 1,inplace = True)
    print(df.info())

def acc(ylabel, y_pred):
    total = len(ylabel)
    acc = 0
        
    for rank in range(1,4):
        # strict
        acc += len(set(np.where(ylabel == rank)[0]) & set(np.where(y_pred == rank)[0])) / total
    return acc

custom_feature_eng(df)
print(df.info())

only_test = ['index']
# data splitting and cross validation
print("==================",df.columns)
X_df, y_df = df.drop(['rank']+only_test,axis=1), df['rank']
print(X_df.shape, y_df.shape)
X_train, X_test, y_train, y_test =  train_test_split(
    X_df.values,
    y_df.values,
    test_size = 0.3,
    random_state = 0
)


models = model(X_train, y_train, X_test, y_test)


if os.path.exists(MODEL_PATH[0]):
    with open(MODEL_PATH[0], 'rb') as f:
        models.clf_rf = pickle.load(f)
    with open(MODEL_PATH[1], 'rb') as f:
        models.clf_gb = pickle.load(f)
else:
    models.train()
    with open(MODEL_PATH[0], 'wb') as f:
        pickle.dump(models.clf_rf, f)
    with open(MODEL_PATH[1], 'wb') as f:
        pickle.dump(models.clf_gb, f)
    
models.accuracy_info()
print('============================================')
# models.cross_validation_performance(X_df.values, y_df.values, 5)
# models.cross_validation_performance(X)

win_prob = models.clf_rf.predict_proba(X_df.values)[:,0] # + models.clf_gb.predict_proba(X_df.values)[:,0]) / 2
place_prob = np.sum(models.clf_rf.predict_proba(X_df.values)[:,:3], axis = 1) #   + np.sum(models.clf_gb.predict_proba(X_df.values)[:,:3], axis = 1)) / 2

df['win_prob'] = pd.Series(win_prob)
df['place_prob'] = pd.Series(place_prob)

# Model Loading or Training

# 新的数据 给到 test_x 的位置

# output_prob = output_prob_tab(W, test_x)
fixratio = 0.1
mthresh = 4
# df['winstake'] = 0 * (df['win_prob'] * df['win_t5'] > mthresh)
# df['plastake'] = fixratio * (df['place_prob']  > prob_thres)
fixratio = 0.01
mthresh = 9
df['winstake'] = fixratio * (df['win_prob']  > 0.9)
# df['plastake'] = fixratio * (df['place_prob'] * df['place_t5'] > mthresh)

df['plastake'] = fixratio * (df['place_prob']  > prob_thres)


# df['plastake'] = fixratio * (df['place_prob'] * df['place_t5'] > mthresh)

results = df[
    ['index','win_prob','place_prob','winstake','plastake']
]

print("finish getting 4 columns")
# res 就是 第一和前三 可能性
if FILEPATH == MODIFIED_PATH:
    data = pd.read_csv(SOURCE_PATH)
else:
    data = pd.read_csv(FILEPATH)

res = data.merge(results,left_on = 'index', right_on = 'index')

res.to_csv("./data/results_train.csv")
res['win_prob'].fillna(0)
res['place_prob'].fillna(0)
res['winstake'].fillna(0)
res['plastake'].fillna(0)

# res = pd.DataFrame(
#     {
#         'win_prob' : win_prob,
#         "place_prob" : place_prob,
#     }
# )

# res['winstake'] = fixratio * (res['win_prob'] * df['win_t5'] > mthresh)
# res['plastake'] = fixratio * (res['place_prob'] * df['place_t5'] > mthresh)

