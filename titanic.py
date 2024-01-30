#%%

import pandas as pd
import numpy as np
import tarfile
import matplotlib.pyplot as plt
import seaborn as sns


with tarfile.open('titanic.tgz', 'r:gz') as tar:
    tar.extractall(path='data')

test = pd.read_csv(r'data/titanic/test.csv')
train = pd.read_csv(r'data/titanic/train.csv')
# %%

train.hist(bins=50, figsize=(12,8))
# fare looks like it needs log or exp scaling
# drop passengerid, name, cabin probably

# %%

train.corr(numeric_only=True)['Survived']
# fare is the only current valid numeric feature at this point. Low correlation w/ response.

# %%

train['age_cat'] = pd.cut(train['Age'],
                          bins=[0, 15, 25, 40, 60, np.inf],
                          labels=[1,2,3,4,5])

#%%

# MISSINGS 

train.isna().sum()

train = train.drop(columns='Cabin')
train = train.dropna()

# %%

y = train[['Survived']]
X = train.drop(columns=[
    'Survived', 
    'Name', 
    'Ticket', 
    'PassengerId', 
    'Parch',
    'Age'
    ])


#%%

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

log_transformer = FunctionTransformer(
    lambda x: np.log(x+1), inverse_func=np.exp, feature_names_out='one-to-one'
    )

log_pipeline = make_pipeline(
    log_transformer,
    StandardScaler()
)

num_pipeline = make_pipeline(
    StandardScaler()
)

cat_pipeline = make_pipeline(
    OneHotEncoder(drop='if_binary')
)

preprocessing = ColumnTransformer([
   ("log", log_pipeline, ['Fare']),
    ("categorical", cat_pipeline, ['Pclass', 'Sex', 'Embarked', 'age_cat'])
    ],
remainder=num_pipeline)


# %%

from sklearn.linear_model import LogisticRegression

log_reg = make_pipeline(preprocessing, LogisticRegression())
log_reg.fit(X, y)


# %%

from sklearn.metrics import confusion_matrix

preds = log_reg.predict(X)
cm = confusion_matrix(y, preds)
cm


# %%

log_reg.score(X,y)

# %%

from sklearn.ensemble import RandomForestClassifier

rf = make_pipeline(preprocessing,
                   RandomForestClassifier(n_jobs=-1))
rf.fit(X,y)
preds = rf.predict(X)
cm = confusion_matrix(y,preds)
cm

# %%

feature_importance = list(zip(rf.named_steps['columntransformer'].get_feature_names_out(),
          rf.named_steps['randomforestclassifier'].feature_importances_))
feature_importance

# %%
