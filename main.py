import pandas as pd
from kmean.algo import kmean
import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file = sys.argv[1]
        n_centroid = int(sys.argv[2])
        distance = sys.argv[3]
    else:
        raise Exception("Wrong argument")

    data = pd.read_csv(file)
    data.set_index('point', inplace=True, drop=True)
    centroids = [[4, 2], [2, 1]]

    kmean_euclidean = kmean(data,
                            n_centroid, 
                            centroids=centroids if len(centroids) else [], 
                            distance=distance)
    kmean_euclidean.run(2)