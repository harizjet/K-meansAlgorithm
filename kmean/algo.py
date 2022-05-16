import pandas as pd


class kmean(Object):
    """
    K-mean initialization
    """

    def __init__(self, 
                data: pd.DataFrame, 
                distance='euclidean', 
                *args, **kargs):

        super().__init__(*args, **kargs)
        self.data = data
        self.distance = distance

    def euclidean(self):



