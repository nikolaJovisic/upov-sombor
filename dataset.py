import re

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

INPUT_COLUMNS = [
    "Temperatura_(C˚)_ulaz",
    "Temperatura_(C˚)_dubinska",
    "Temperatura_(C˚)_površinska",
    "Temperatura_(C˚)_izlaz",
    "pH_ulaz",
    "HPK_(mg/l)_ulaz",
    "HRT",
    "O2_(mg/l)_dubinska",
    "O2_(mg/l)_površinska",
    "BPK5_(mg/l)_ulaz",
    "N_(mg/l)_ulaz",
    "P_(mg/l)_ulaz",
    "Susp.materije_(mg/l)_ulaz",
]

OUTPUT_COLUMNS = [
    "pH_izlaz",
    "HPK_(mg/l)_izlaz",
    "BPK5_(mg/l)_izlaz",
    "N_(mg/l)_izlaz",
    "P_(mg/l)_izlaz",
    "Susp.materije_(mg/l)_izlaz",
]


def load():
    df = pd.read_csv("data.csv")

    columns = []
    for column in df.columns:
        if column == "Unnamed: 0":
            continue
        columns.append(column)

    new_columns = []
    valid_index = 0
    steping_index = 0
    for c in df.iloc[0]:
        if c == "Datum":
            new_columns.append("Datum")
            continue
        if "Unnamed" not in columns[steping_index]:
            valid_index = steping_index
        c = f"{columns[valid_index]}_{c}"
        c = c.strip()
        c = c.replace(" ", "_")
        c = re.sub(r"_+", "_", c)
        c = c.replace("_nan", "")
        new_columns.append(c)
        steping_index += 1

    print(*new_columns, sep='\n')

    df.drop(index=0, inplace=True)
    col_name_dict = {
        old_name: new_name for old_name, new_name in zip(df.columns, new_columns)
    }
    df = df.rename(columns=col_name_dict)
    return df


def preprocess(df: pd.DataFrame):
    df.drop(columns=["Datum"], inplace=True)
    df.dropna(subset=INPUT_COLUMNS, inplace=True)
    df.dropna(subset=OUTPUT_COLUMNS, inplace=True)

    def format_strings(element):
        if not isinstance(element, str):
            return element
        element = element.replace(",", ".")
        element = element.strip(". ")
        try:
            element = float(element)
        except:
            element = 0.0
        return element

    df = df.applymap(format_strings)
    df = df[~(df == 0).any(axis=1)]

    scaler = StandardScaler()

    df[INPUT_COLUMNS] = scaler.fit_transform(df[INPUT_COLUMNS])

    return df


class WaterDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        data = load()
        self.df = preprocess(data)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        input_vector = row[INPUT_COLUMNS].to_numpy(dtype=np.float32)
        output_vector = row[OUTPUT_COLUMNS].to_numpy(dtype=np.float32)

        if self.target_transform:
            input_vector = self.target_transform(input_vector)

        if self.transform:
            output_vector = self.transform(output_vector)

        return torch.from_numpy(input_vector), torch.from_numpy(output_vector)
