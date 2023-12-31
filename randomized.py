import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

if __name__ == "__main__":

    dataset = pd.read_csv('/mnt/c/Users/JULIAN/class_sklearn/data/felicidad_corrupt.csv')

    print(dataset)
    
    X = dataset.drop(['country', 'rank', 'score'], axis=1)
    y = dataset[['score']]

    reg = RandomForestRegressor()

    parametros = {
        'n_estimators' : range(4,16),
        'criterion' : ['absolute_error', 'squared_error'],
        'max_depth' : range(2,11)
    }

    rand_est = RandomizedSearchCV(reg, parametros , n_iter=10, cv=3, scoring='neg_mean_absolute_error').fit(X,y.values.ravel())

    print(rand_est.best_estimator_)
    print(rand_est.best_params_)
    print(rand_est.predict(X.loc[[0]]))

    #implmentacion_randomizedSearchCV