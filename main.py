import pandas as pd
from kmean.algo import kmean

if __name__ == "__main__":
    data = pd.read_csv('data.csv')
    data.set_index(data['point'], inplace=True)

    kmean_euclidean = kmean(data, 2, centroids=[[4, 2], [2, 1]])
    kmean_euclidean.run(1)