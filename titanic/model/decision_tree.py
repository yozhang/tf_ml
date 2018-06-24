from sklearn import tree, svm
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV

def preprocess(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    test['Survived'] = 1001
    dframe = pd.concat([train, test])
    dframe = dframe.fillna({'Age':18, 'Embarked':'N', 'Fare':6.6})
    encode_dat = pd.get_dummies(dframe, columns=['Cabin', 'Embarked', 'Sex', 'Ticket'])
    train_dat = encode_dat[encode_dat['Survived']<1001].drop(['Name', 'PassengerId'], axis=1)
    test_dat = encode_dat[encode_dat['Survived']==1001].drop(['Name','Survived'], axis=1)
    return train_dat, test_dat

def train_model(dat):
    y = dat['Survived']
    X = dat.drop(['Survived'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    params = [
        {'n_estimators':[100], 'learning_rate':[1.0], 'max_depth': [10], 'random_state':[0]},
        {'n_estimators':[120], 'learning_rate': [0.1], 'max_depth': [30], 'random_state':[0]},
        {'n_estimators':[60], 'learning_rate': [0.5], 'max_depth': [50], 'random_state':[0]}
    ]
    scores = ['precision', 'recall']
    clf = GridSearchCV(GradientBoostingClassifier(), params, cv=5, scoring='accuracy')
    # clf = GradientBoostingClassifier(n_estimators=100, 
    # learning_rate=0.2, max_depth=10, random_state=0).fit(X_train, y_train)
    clf.fit(X, y)
    print(clf.score(X_test, y_test))
    print(clf.best_params_)
    return clf

def output_predict(model, test, out_path):
    results = pd.DataFrame(test, columns=['PassengerId'])
    features = test.drop('PassengerId', axis = 1)
    surv = model.predict(features)
    results['Survived'] = surv
    results.to_csv(out_path, index=False)


if __name__ == '__main__':
    r_path = '/Users/zhangyong/projects/tf-exe/titanic/data/train_pro.csv'
    t_path = '/Users/zhangyong/projects/tf-exe/titanic/data/test_pro.csv'
    out_path = '/Users/zhangyong/Downloads/gbdt.csv'

    train, test = preprocess(r_path, t_path)
    model = train_model(train)
    output_predict(model, test, out_path)


