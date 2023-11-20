import pandas as pd
from sklearn.cluster import MeanShift
if __name__ == "__main__":

    dataset = pd.read_csv ('/mnt/c/Users/JULIAN/class_sklearn/data/candy.csv')
    print (dataset.head(10))
    
    X = dataset.drop ('competitorname', axis = 1)
    meanshft = MeanShift().fit(X)
    print (meanshft.labels_)
    print ('='* 64)
    print(meanshft.cluster_centers_)
    
    dataset['meanshift'] = meanshft.labels_
    print ('='* 64)
    print(dataset)
