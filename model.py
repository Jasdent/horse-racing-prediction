from utils_global import *

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score


class model(object):
    def __init__(
        self, 
        Xtrain : np.array, 
        ytrain : np.array, 
        Xtest: np.array,
        ytest: np.array
    ):
        self.clf_rf = RandomForestClassifier( 
            n_estimators = 165,
            max_depth =  13,
            max_features = 'log2', # with 11 features
            random_state = 0,
            n_jobs = 8
        )

        self.clf_gb = GradientBoostingClassifier(
            learning_rate = 0.01,
            max_depth = 4,
            random_state= 0
        )
        # ensemble of classifier
        self.clf = [self.clf_rf,self.clf_gb]
        self.Xtrain : np.array = Xtrain
        self.ytrain : np.array = ytrain
        self.Xtest : np.array = Xtest
        self.ytest : np.array = ytest

    def accuracy_info(self):
        if FILEPATH == './data/HR200709to201901.csv':
            # training mode
            print("Accuracy on training set: {}".format(
                (self.clf_rf.score(self.Xtest, self.ytest) + self.clf_gb.score(self.Xtest, self.ytest)) / 2
            ))
        print("Accuracy on test set: {}".format( 
            (self.clf_rf.score(self.Xtest, self.ytest) + self.clf_gb.score(self.Xtest, self.ytest)) / 2
        ))

    def cross_validation_performance(self, X, Y, n_cv = 8):
        scores = 0
        scores += cross_val_score(self.clf_rf, X,Y, cv = n_cv)
        scores += cross_val_score(self.clf_gb, X,Y, cv = n_cv)
        print("Cross validation accuracy: {} (+/- {})".format((scores / 2).mean(), (scores/2).std() * 2))

    def train(self, Xtrain = None , ytrain = None):
        if not Xtrain:
            Xtrain,ytrain =  self.Xtrain, self.ytrain
        self.clf_gb.fit(Xtrain,ytrain)
        self.clf_rf.fit(Xtrain,ytrain)
        print("finished fitting")
