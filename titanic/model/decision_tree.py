from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd 

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
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]
    clf = GradientBoostingClassifier(n_estimators=100, 
    learning_rate=0.1, max_depth=10, random_state=0).fit(X_train, y_train)
    print(clf.score(X_test, y_test))
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


