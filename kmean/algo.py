import pandas as pd
import numpy as np
import math
from statistics import mean
from kmean.visualized import scatplot


class kmean(object):
    """
    K-mean initialization: only 2 Dimensions (x, y)
    """

    def __init__(self, 
                data: pd.DataFrame,
                n_centroid: int,
                centroids=[],
                distance='euclidean', 
                *args, **kargs):

        super().__init__(*args, **kargs)
        self.data = data
        self.n_centroid = n_centroid
        self.centroids = centroids

        if centroids and len(centroids) != n_centroid:
            raise Exception("Centroids given != n given!")
        elif not centroids:
            self.init_centroid()

        self.centroids = {i: cent for i, cent in enumerate(centroids)}
        self.data['type'] = ['point' for _ in range(self.data.shape[0])]

        if distance == 'euclidean':
            self.distance = self.euclidean
        elif distance == 'manhattan':
            self.distance = self.manhattan
        elif distance == 'cosine':
            self.distance = self.cosine
        else:
            raise Exception("Wrong distance option given!")

    def init_centroid(self) -> None:
        max_x = max(self.data['x'])
        min_x = min(self.data['x'])
        max_y = max(self.data['y'])
        min_y = min(self.data['y'])

        for _ in range(self.n_centroid):
            ran_x = np.random.choice(range(min_x, max_x+1))
            ran_y = np.random.choice(range(min_y, max_y+1))
            self.centroids.append([ran_x, ran_y])

    def re_centroids(self) -> None:
        for k, cent in self.centroids.items():
            tdf = self.data[self.data['grouping'] == k]
            mid_x = mean(tdf['x']) if len(tdf) else cent[0]
            mid_y = mean(tdf['y']) if len(tdf) else cent[1]
            self.centroids[k] = [mid_x, mid_y]

    def euclidean(self, x, y) -> None:
        min_dist = [math.inf, None]
        for k, cent in self.centroids.items():
            dist = (cent[0] - x) ** 2 + (cent[1] - y) ** 2
            if dist < min_dist[0]:
                min_dist = [dist, k]
        return min_dist[1]

    def manhattan(self, x, y) -> None:
        min_dist = [math.inf, None]
        for k, cent in self.centroids.items():
            dist = abs(cent[0] - x) + abs(cent[1] - y)
            if dist < min_dist[0]:
                min_dist = [dist, k]
        return min_dist[1]

    def cosine(self, x, y) -> None:
        min_dist = [math.inf, None]
        for k, cent in self.centroids.items():
            nume = (cent[0] * x) + (cent[1] * y)
            deno = ((x ** 2 + y ** 2) ** 0.5) * \
                    ((cent[0] ** 2 + cent[1] ** 2) ** 0.5)
            dist = nume / deno
            if dist < min_dist[0]:
                min_dist = [dist, k]
        return min_dist[1]

    def get_group(self) -> None:
        list_group = []
        for ind, row in self.data.iterrows():
            list_group.append(self.distance(row['x'], row['y']))
        self.data['grouping'] = list_group

    def run(self, n, visualized=True) -> None:
        for _ in range(n):
            self.get_group()
            if visualized:
                scatplot(self.data, self.centroids)
            self.re_centroids()
        print(self.data)
        print(self.centroids)


