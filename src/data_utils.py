import os
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from torch.utils.data import DataLoader, random_split


def minmax_helper(df):
    mmax = max(df.max())
    mmin = min(df.min())
    df = (df - mmin) / (mmax - mmin)
    return df, mmax, mmin


class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir="data", config=None):
        super().__init__()
        self.config = config
        self.data_dir = data_dir
        self.num_workers = os.cpu_count() // 2

        self.stat_price = None

        self.batch_size = config["batch_size"]

    def setup(self, stage: Optional[str] = None):
        if self.config["data"] == "all":
            X = pd.read_csv(os.path.join(self.data_dir, "x_3d.csv"))
            Y = pd.read_csv(os.path.join(self.data_dir, "y_3d.csv"))
        else:
            period = self.config["data"]
            X = pd.read_csv(os.path.join(self.data_dir, f"x_3d_{period}.csv"))
            Y = pd.read_csv(os.path.join(self.data_dir, f"y_3d_{period}.csv"))

        X, Y = self.data_preprocess(X, Y)
        if self.config["only_price"]:
            X = X.reshape(-1, 3, 5)[:, :, 0].reshape(-1, 3)

        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, shuffle=False
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, shuffle=False
        )

        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float()
        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test).float()
        X_val = torch.from_numpy(X_val).float()
        y_val = torch.from_numpy(y_val).float()

        self.train = TensorDataset(X_train, y_train)
        self.val = TensorDataset(X_val, y_val)
        self.test = TensorDataset(X_test, y_test)

    @property
    def train_dataset(self):
        return self.train

    @property
    def val_dataset(self):
        return self.val

    @property
    def test_dataset(self):
        return self.test

    def train_dataloader(self, batch_size=None):
        if not batch_size:
            batch_size = self.batch_size
        return DataLoader(
            self.train,
            batch_size=batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self, batch_size=None):
        if not batch_size:
            batch_size = self.batch_size
        return DataLoader(
            self.val,
            batch_size=batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self, batch_size=None):
        if not batch_size:
            batch_size = 1
        return (
            self.train_dataloader(batch_size),
            self.val_dataloader(batch_size),
            DataLoader(
                self.test,
                batch_size=batch_size,
                num_workers=self.num_workers,
                persistent_workers=True,
            ),
        )

    def data_preprocess(self, X, Y):
        category_cols = [
            "Week-1",
            "Week-2",
            "Last-Week-day",
            "t-1",
            "t-2",
            "Last-Week-t",
        ]

        # data processing

        X = X.drop(columns=category_cols)

        X, stat_price = self.scale(X)
        Y = (Y - stat_price["min"]) / (stat_price["max"] - stat_price["min"])

        X = X.values
        Y = Y.values
        self.stat_price = stat_price
        return X, Y

    @staticmethod
    def scale(df):
        stat_price = {"min": None, "max": None}
        df_price = df.drop(columns=["Vol-1", "Vol-2", "Last-Week-Vol"]).select_dtypes(
            "float64"
        )

        df_price, stat_price["max"], stat_price["min"] = minmax_helper(df_price)
        df[df_price.columns] = df_price

        df_vol = df[["Vol-1", "Vol-2", "Last-Week-Vol"]]
        df_vol, _, _ = minmax_helper(df_vol)
        df[df_vol.columns] = df_vol

        return df, stat_price
