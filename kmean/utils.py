from xmlrpc.client import Boolean
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import signal


def combineSaved(data: pd.DataFrame, centroids: dict) -> None:
    tdf = pd.DataFrame({
            'x': [cent[0] for cent in centroids.values()],
            'y': [cent[1] for cent in centroids.values()],
            'grouping': [group for group in centroids.keys()],
            'type': ['centroid' for _ in range(len(centroids))],
            'distance': [0 for _ in range(len(centroids))]})

    data = pd.concat([data, tdf], axis=0)
    data.to_csv('result.csv', index=False)
    return data

def scatplot(data: pd.DataFrame, iterate: int, converge: bool) -> None:
    signal.signal(signal.SIGINT, signal.SIG_DFL) # to exit plotting
    
    title = 'Iteration {}'.format(iterate)
    title = title + '\nConverge Achieved' if converge else title

    plt.figure(figsize = (15,8))
    scat = sns.scatterplot(data=data, 
                           x='x', 
                           y='y', 
                           hue='grouping', 
                           style='type', 
                           s=200,
                           palette='deep')
    scat.set_title(title)
    plt.show();