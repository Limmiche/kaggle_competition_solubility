import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

#importing (imbalanced) data
df=pd.read_csv("Temp/df_proc_train_set.csv",index_col="smiles")
df.drop(["Id"], axis=1, inplace=True)
df.head()
#define y as label and x as features from dataframe
x=df.drop(["sol_category"], axis=1)
y=df["sol_category"]



x_train_0, x_test_0, y_train_0, y_test_0=train_test_split(x,y,test_size=6/7, random_state=12)

print(y_train_0.value_counts(normalize=True))
print(y_test_0.value_counts(normalize=True))


x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=1/7, random_state=12, stratify=y) #6/7 before
y_train
print(y_train.value_counts())
print(y_test.value_counts())
print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))


import warnings
warnings.filterwarnings('ignore')

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy import stats
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline #first transform then estimate data
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier


from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score

from sklearn.linear_model import LogisticRegression
from numpy import linspace

from xgboost import XGBRegressor


feature_sets = {
    'MorganFP': df.columns[pd.Series(df.columns).str.startswith('ecfp_')],
    'RDKit': df.columns[pd.Series(df.columns).str.startswith('rdkit_desc')],
    'ExampleDescriptors': df.columns[pd.Series(df.columns).str.startswith('example_')],
}


estimators = {

    'nn': MLPClassifier(),

    'xgb': XGBClassifier(),

    'rf': RandomForestClassifier(),
    
    'lg': LogisticRegression()
}

params = {
    'rf': {
        'rf__n_estimators': np.arange(50, 1050, 50),
        'rf__max_features': np.arange(0.1, 1.0, 0.1)
    },
    'nn': {
        'nn__hidden_layer_sizes': [(n,) for n in np.arange(5, 200, 5)],
        'nn__activation': ['tanh', "relu"],
        'nn__alpha': 10.0 ** -np.arange(1, 7),
        'nn__max_iter': [500, 1000]
        #'nn__batch_size': [1000] #now it only uses 6 batches for whole dataset
    },


    'xgb': {
        'xgb__learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5],
        'xgb__max_depth': np.arange(1, 11, 2),
        'xgb__n_estimators': np.arange(50, 550, 50),
        'xgb__subsample': [0.5, 1]
    },

    'lg': {
        'max_iter' : range(1000, 2000),
        'solver' : ['newton-cg', 'lbfgs', 'sag', 'saga'],
        'penalty' : ['l2', 'none'],
        'C' : linspace(0.1, 1.0, num=50)
    }
}


best_params = {}
cv_scores = {}
test_score = {}


for f in tqdm(feature_sets): #tqdm for progress bar
    
    print(f'Using {f} features...')

    features = x

    best_params[f] = {} #each value is another dictonary 
    cv_scores[f] = {}
    test_score[f] = {}

    for e in tqdm(estimators):

        print(f'\tRandom search optimisation for {e} estimator...') #n_iter: number hyperparameter setting, normally 100
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(x_train, y_train)
        pipe = Pipeline([('scaler', StandardScaler()), (e, estimators[e])])
        model = RandomizedSearchCV(pipe, param_distributions=params[e], cv=cv, refit=True, n_iter=5, n_jobs=64, verbose=2, random_state=42).fit(x_train, y_train) # verbose=1: information on hyperparameter search
        #cv_score[e] = model.best_score_


        #have best parameters in model.best_estimator_
        # get pearson correlation for each cv fold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(x_train, y_train)
        scores = cross_val_score(model.best_estimator_, x_train, y_train, cv=cv)
        cv_scores[f][e] = [s**0.5 for s in scores] # convert r-squared to pearsonr #t2 keys?

        best_params[f][e] = model.best_params_

        #test model on test set and get scores:
        test_score[f][e] = {}
        
        pred = model.predict(x_test)
        rp = cohen_kappa_score(y_test, pred, weights="quadratic")
        test_score[f][e] = rp


        # JSON encoder for np.int64 #what is this part doing? return int of a number. if it cannot return error
def default(o):
    if isinstance(o, np.integer):
        return int(o)
    raise TypeError   


with open('./results/random_search_best_params.json', 'w') as f:
    json.dump(best_params, f, default=default)
    
with open('./results/random_search_best_cv_scores.json', 'w') as f:
    json.dump(cv_scores, f, default=default) 


with open('./results/random_search_best_cv_scores.json') as f:
    cv_scores = json.load(f)


mean_cv_score = {f: {e: np.mean(cv_scores[f][e]) for e in cv_scores[f]} for f in cv_scores}


row_order = ['RDKit', 'Vina', 'Vina + RDKit', 'RF-Score', 'RF-Score + RDKit', 'RF-Score v3', 'RF-Score v3 + RDKit', 'NNScore 2.0', 'NNScore 2.0 + RDKit']
column_order = ['Neural Network', 'Logistic Regression', 'XGBoost', 'Random Forest']
mapper = {
    
    'lg': 'Logistic Regression',
    'xgb': 'XGBoost',
    'rf': 'Random Forest',
    'nn': 'Neural Network',
}
df = pd.DataFrame(mean_cv_score).T
df = df.rename(mapper=mapper, axis='columns')
df = df.loc[row_order, column_order]

fig, ax = plt.subplots(1,1,figsize=(6,6))
sns.heatmap(df, annot=True, cmap='viridis', cbar_kws={'label': r"Mean Pearson Correlation Coefficient"}, ax=ax)
ax.set_xlabel('Algorithm')
ax.set_ylabel('Features Used')
ax.hlines([1, 3, 5, 7], *ax.get_xlim(), linestyle=':', linewidth=2)
fig.savefig('./figures/algorithm_comparison.jpg', dpi=350, bbox_inches='tight')
