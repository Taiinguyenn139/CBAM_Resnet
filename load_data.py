import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

class DataLoader:
    def __init__(self, df_train_path, df_test_path, file_train, file_test):
        self.df_train_path = df_train_path
        self.df_test_path = df_test_path
        self.file_train = file_train
        self.file_test = file_test

    def train_loader(self, major_rate=0.5):
        df = pd.read_csv(self.df_train_path)
        df_0 = df[df['target']==0].sample(int(len(df[df['target']==0])*major_rate)) #sample method specifies fraction of
                                                                                    #rows to return in the random sample
        df_1 = df[df['target']==1]
        train = pd.concat([df_0, df_1])
        train = train.reset_index()

        labels = []
        data = []
        for i in range(train.shape[0]):
            data.append(self.file_train + train['image_name'].iloc[i] + '.jpg')
            labels.append(train['target'].iloc[i])
        df = pd.DataFrame(data)
        df.columns = ['images']
        df['target'] = labels

        X_train, X_val, y_train, y_val = train_test_split(df['images'], df['target'], test_size=0.2, random_state=1234)

        return list(X_train), list(X_val), list(y_train), list(y_val)

    def test_loader(self):
        test = pd.read_csv(self.df_test_path)
        test_data = []
        for i in range(test.shape[0]):
            test_data.append(self.file_test + test['image_name'].iloc[i] + '.jpg')
        df_test = pd.DataFrame(test_data)
        df_test.columns = ['images']

        return df_test





