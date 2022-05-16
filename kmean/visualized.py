import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def scatplot(data: pd.DataFrame, centroids: dict, iterate=None) -> None:
    tdf = pd.DataFrame({
            'x': [cent[0] for cent in centroids.values()],
            'y': [cent[1] for cent in centroids.values()],
            'grouping': [group for group in centroids.keys()],
            'type': ['centroid' for _ in range(len(centroids))]})

    data = pd.concat([data, tdf], axis=0)
    title = 'Iteration {}'.format(iterate) if iterate else 'Result clustering'

    plt.figure(figsize = (15,8))
    scat = sns.scatterplot(data=data, x='x', y='y', hue='grouping', style='type', s=200)
    scat.set_title(title)
    plt.show();