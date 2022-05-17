import pandas as pd
from kmean.algo import kmean
import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file = sys.argv[1]
        n_centroid = int(sys.argv[2])
        distance = sys.argv[3]
        max_iteration = int(sys.argv[4])
        visualized = True if sys.argv[5] == 'visualized' else False
    else:
        raise Exception("Wrong argument")

    data = pd.read_csv(file)
    data.set_index('point', inplace=True, drop=True)
    centroids = [[4, 2], [2, 1]]

    kmean_euclidean = kmean(data,
                            n_centroid, 
                            centroids=centroids if len(centroids) else [], 
                            distance=distance)
    kmean_euclidean.run(max_iteration, visualized=visualized)