import pandas as pd
import numpy as np
import math
from statistics import mean
from kmean.utils import *


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
        self.converge = False
        self.cost = math.inf

        if centroids and len(centroids) != n_centroid:
            raise Exception("Centroids given != n given!")
        elif not centroids:
            self.init_centroid()

        self.centroids = {i: cent for i, cent in enumerate(centroids)}
        self.data['type'] = ['point' for _ in range(self.data.shape[0])]

        if distance == 'euclidean':
            self.distance = kmean.euclidean
        elif distance == 'manhattan':
            self.distance = kmean.manhattan
        elif distance == 'cosine':
            self.distance = kmean.cosine
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
        conv = True
        for k, cent in self.centroids.items():
            tdf = self.data[self.data['grouping'] == k]
            mid_x = mean(tdf['x']) if len(tdf) else cent[0]
            mid_y = mean(tdf['y']) if len(tdf) else cent[1]

            if self.centroids[k] != [mid_x, mid_y]:
                conv = False
            self.centroids[k] = [mid_x, mid_y]
        self.converge = conv

    def euclidean(x: int, y: int, centroids: dict) -> list:
        min_dist = [None, math.inf]
        for k, cent in centroids.items():
            dist = (cent[0] - x) ** 2 + (cent[1] - y) ** 2

            print(f"Distance from {(x, y)} to Centroid {k}: {dist}")
            if dist < min_dist[1]:
                min_dist = [k, dist]
        return min_dist

    def manhattan(x: int, y: int, centroids: dict) -> list:
        min_dist = [None, math.inf]
        for k, cent in centroids.items():
            dist = abs(cent[0] - x) + abs(cent[1] - y)

            print(f"Distance from {(x, y)} to Centroid {k}: {dist}")
            if dist < min_dist[1]:
                min_dist = [k, dist]
        return min_dist

    def cosine(x: int, y: int, centroids: dict) -> list:
        min_dist = [None, math.inf]
        for k, cent in centroids.items():
            nume = (cent[0] * x) + (cent[1] * y)
            deno = ((x ** 2 + y ** 2) ** 0.5) * \
                    ((cent[0] ** 2 + cent[1] ** 2) ** 0.5)
            dist = 1 - (nume / deno)

            print(f"Distance from {(x, y)} to Centroid {k}: {dist}")
            if dist < min_dist[1]:
                min_dist = [k, dist]
        return min_dist

    def get_group(self) -> None:
        list_group = []
        for ind, row in self.data.iterrows():
            list_group.append(self.distance(row['x'], row['y'], self.centroids))
        self.data['grouping'] = [x[0] for x in list_group]
        self.data['distance'] = [x[1] for x in list_group]

    def run(self, n, visualized) -> None:
        for i in range(n):
            print(f"ITERATION {i}")
            self.get_group()

            data = combineSaved(self.data, self.centroids)
            print(f"Data: {[[val['x'], val['y']] for i, val in self.data[['x', 'y']].iterrows()]}")
            print(f"Centroids: {self.centroids}")
            print(f"Group: {self.data['grouping'].values}")
            print(f"Distance: {self.data['distance'].values}")
            print(f"SSE: {sum(self.data['distance'].values)}\n")

            if visualized:
                scatplot(data, 
                        iterate=i+1, 
                        converge=self.converge)
            self.re_centroids()
        print(self.data)
        print(self.centroids)