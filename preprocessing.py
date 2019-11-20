"""
MSBD 5013
CHEN, Mo
YANG, Zhuokun
"""

from utils_global import *

class preprocess(object):
    def __init__(self):
        pass

    @staticmethod
    def features_select(df : pd.DataFrame ,conditions : list = []):
        # features for probablistic model 
        if not conditions: 
            """
            NOTICE:
                ratechg has 76651 missing
                horseweightchg has 76651 missing
                age has 76651 missing
                lastsix has 76651 missing
                win_t5 has 2850 missing
                place_t5 has 11328 missing
            """
        else:
            prob_cond = conditions
        
        return df[prob_cond]
    
    @staticmethod
    def imputation(mode: str,
                   d_series : pd.Series,
                   inplace : bool = False        
        ):
        """
        :param mode: the mode of imputation selected ['nn', 'avg']
        :param d_series: the pd.series to be filled na with
        :param inplace: whether changes take place in the series
        :param rg: range of number of preference
        :return: None or pd.Series created from scratch
        """
        modified = None
        if mode == "nn":
            # use previous value to fillna
            modified = d_series.fillna(method = 'pad', inplace = inplace)
        elif mode == 'avg':
            modified = d_series.fillna(d_series.mean() + 0, inplace = inplace)
        else:
            # more methods to be discovered
            pass
            modified = None
        return modified


    @staticmethod
    def encode(convert : dict = {},
               dseries : pd.Series = None,
               mode: str = "",
               inplace : bool = False,
               rg : tuple = (0,100)
    ):
        """

        :param convert: dict, mapping non-numerics to numerics, identify nominal or ordinal attr
        :param dseries: pd.Series, the series to be encoded
        :param mode: char, binary, or more other modes
        :param inplace: bool, whether to modify
        :param rg: range of mapping
        :return:
        """
        if mode == 'b':
            d = {}
            s = list(set(dseries))
            if len(s) != 2:
                print('not suitable for binary encoding')
                return
            else:
                d[s[0]] = 1
                d[s[1]] = 2
                if not convert:
                    convert = d

        return dseries.map(convert)

    @staticmethod
    def derive_difference(df: pd.DataFrame, val_attr: str, val_chg_attr: str, id:int):
        '''
        this method should run after imputation.
        Calculate param difference using param.
        df modified inplace because pd.DataFrame and pd.Series are reference based

        rtype: None
        '''
        
        minidf = df[df['horsenum'] == id][[val_attr, val_chg_attr]]
        indexes = minidf.index
        val = minidf[val_attr].tolist()
        valdiff = minidf[val_chg_attr].tolist()
        for i in range(len(indexes)-1, 0, -1):
            idx = indexes[i]
            valdiff[i] = val[i] - val[i-1]
            df[val_chg_attr][idx] = valdiff[i]
    
# features for probablistic model 
prob_cond = [
    "index",
    "horsenum",
    "rdate",
    "track", # track, use neighbour to inpution
    "venue",
    "distance", # track, use neighbour for inputation 
    "going", # use neighbour for imputation
    "exweight", # no nan
    # "bardraw", # ???
    "rating", # ??? figure out how to fill
    # "ratechg", # future use, figure out how to fillna
    "horseweight", # avg for imputation
    # "horseweightchg", # no nan ,3 ranges [20,inf], [10,20], [0,10] mapped to 2,1,0, respectively, using the convention of disease detection
    # "age", # [3,7] 9 to be determined, whether the range [3,7] can provide further information related to ranking, ? how to impute
    # "lastsix", # future use; Q: how to deal with empty list? | just for portfolio, not for probablistic model
    # 'tname',
    # 'jname',
    "win_t5", # use avg to impute? 
    "place_t5", # use avg to impute?
    ##### labels ####
    "rank"
]

df_source = pd.read_csv(FILEPATH)

df_prob = df_source[prob_cond]
print("----- First Investigation -----")
for col in df_prob.columns:
    countna = df_prob[col].isna().sum() 
    if  countna == 0:
        print("attr {} is cleaned".format(col))
    else:
        if countna > 1000:
            pass
        else:
            print("attr {} has {} missing".format(col, countna))
'''
注意 imputation，训练集是空的，新的数据集不一定为空！！
imputation
'''

print("---- Cleaning and imputation and calculating features ----")
horse_id = set(df_prob['horsenum'])
# categorical value
for attr in ["track", "distance", "going","venue"]:
    preprocess.imputation("nn",df_prob[attr],inplace=True)


# numeric and continuous
for attr in ["horseweight","rating",'win_t5','place_t5','exweight']:
    df_prob[attr].fillna(df_prob[attr].mean(), inplace = True)


count = 0
# if mode == 'test' or FILEPATH == SOURCE_PATH:
#     print("Start to calculate features. Please wait.")
#     for identity in horse_id:
#         if count % 200 == 0:
#             print("Step: {}".format(count))
#         # for attr in ["horseweight","rating",'win_t5','place_t5','exweight']:
#         #     df_prob[df_prob['horsenum'] == id][attr].fillna(df_prob[df_prob['horsenum'] == id][attr].mean(), inplace = True)
#         # print("finished basic imputations")
#         for attr, attrchg in [
#             ('horseweight','horseweightchg'),
#             ('rating','ratechg')
#         ]:
#             preprocess.derive_difference(
#                 df_prob, attr, attrchg, identity
#             )
#             df_prob[attrchg].fillna(np.mean(df_prob[attrchg]),inplace = True)
#         count += 1
#     print("Finished calculation ****************************8")
#     if FILEPATH == SOURCE_PATH:
#         df_prob.to_csv(MODIFIED_PATH)
        


'''
tree-based methods should be able to deal with values of different range
'''


# df_prob['venue'] = preprocess.encode(
#     convert = dict(
#         zip(
#             list(set(df_prob['venue'])),
#             [i for i in range(1, len(list(set(df_prob['venue'])))+1)])
#     ),
#     dseries = df_prob['venue']
# )

# convert to numeric
# df_prob['going'] = preprocess.encode(
#     convert = dict(
#         zip(
#             list(set(df_prob['going'])),
#             [i for i in range(1, len(list(set(df_prob['going'])))+1)]
#         )
#     ), dseries = df_prob['going']
# )
# df_prob['track'] = preprocess.encode(
#     convert = dict(
#         zip(
#             list(set(df_prob['track'])),
#             [i for i in range(1, len(list(set(df_prob['track'])))+1)]
#         )
#     ), dseries = df_prob['track']
# )

# one-hot
df_prob[GOING_SPACE] = pd.DataFrame(
    dict(
        zip(
            GOING_SPACE,
            [[0] * len(df_prob['index'])] * len(GOING_SPACE)        )
    )
)
df_prob[TRACK_SPACE] = pd.DataFrame(
    dict(
        zip(
            TRACK_SPACE,
            [[0] * len(df_prob['index'])] * len(TRACK_SPACE)
        )
    )
)
df_prob[VENUE_SPACE] = pd.DataFrame(
    dict(
        zip(
            VENUE_SPACE,
            [[0] * len(df_prob['index'])] * len(VENUE_SPACE)
        )
    )
)

df_prob[list(set(df_prob['going']))] = pd.get_dummies(df_prob['going'], columns= GOING_SPACE)
df_prob[list(set(df_prob['track']))] = pd.get_dummies(df_prob['track'], columns = TRACK_SPACE)
df_prob[list(set(df_prob['venue']))] = pd.get_dummies(df_prob['venue'])


df_prob.drop(["venue","track","going","horsenum","rdate"],axis = 1, inplace = True)
print("check going")

# Report back
# for col in prob_cond:
#     countna = df_prob[col].isna().sum() 
#     if  countna == 0:
#         print("attr {} is cleaned".format(col))
#     else:
#         if countna > 1000:
#             bigmiss.append((col, countna))
#         else:
#             print("attr {} has {} missing".format(col, countna))
# print("==========================")
# for i in bigmiss:
#     print("BIGMISS: attr {} has {} missing".format(i[0], i[1]))
# print("check")



# df_prob.drop(df_prob[(df_prob['tname'].isna()) | (df_prob['jname'].isna())].index, inplace = True)


df_prob.dropna(axis=1, inplace = True)
print(df_prob.info())
print("finished!")