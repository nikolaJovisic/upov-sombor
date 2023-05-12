import re

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

FEATURE_COLUMNS = [
    "Temperatura_(C˚)_ulaz",
    "pH_ulaz",
    "HPK_(mg/l)_ulaz",
    "BPK5_(mg/l)_ulaz",
    "N_(mg/l)_ulaz",
    "P_(mg/l)_ulaz",
    "Susp.materije_(mg/l)_ulaz",
    # "Q_(m3/dan)_protok",
    # "HRT",
]

OUTPUT_COLUMNS = [
    # "HPK_(mg/l)_izlaz",
    "BPK5_(mg/l)_izlaz",
    # "N_(mg/l)_izlaz",
    # "P_(mg/l)_izlaz",
    # "Susp.materije_(mg/l)_izlaz",
]
SEQUENCE_CHANNELS = [
    OUTPUT_COLUMNS[0],
    OUTPUT_COLUMNS[0][: -len("izlaz")] + "ulaz",
    "weekday",
]
SEQUENCE_COLUMNS = []
TRANSFORM_COLUMNS = []

SEQUENCE_LENGTH = 10


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

    print(*new_columns, sep="\n")

    df.drop(index=0, inplace=True)
    col_name_dict = {
        old_name: new_name for old_name, new_name in zip(df.columns, new_columns)
    }
    df = df.rename(columns=col_name_dict)
    return df


def preprocess(df: pd.DataFrame, use_seq: bool):
    df.dropna(subset=["Datum"], inplace=True)
    df = df[~df["Datum"].str.endswith("(2)")]
    df.dropna(subset=FEATURE_COLUMNS, inplace=True)
    df.dropna(subset=OUTPUT_COLUMNS, inplace=True)

    def format_date(date: str) -> str:
        date = date.replace("/", ".")
        date = date.replace(",", ".")
        return date.strip(".")

    df = df.applymap(lambda x: format_date(x) if isinstance(x, str) else x)
    df["Datum"] = pd.to_datetime(df["Datum"], format="%d.%m.%Y")
    if 'weekday' in SEQUENCE_CHANNELS:
        FEATURE_COLUMNS.append("weekday")
        df["weekday"] = df["Datum"].apply(lambda dt: dt.weekday())
    df.set_index("Datum", inplace=True)

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

    df = df.loc[:, [*FEATURE_COLUMNS, *OUTPUT_COLUMNS]]
    df = df[~(df == 0).any(axis=1)]

    df = df[df[OUTPUT_COLUMNS[0]] < 300]

    df = df.groupby("Datum").mean().asfreq("D").interpolate(method="time")

    # plt.plot(df[OUTPUT_COLUMNS[0]])
    # plt.xlabel('merenja')
    # plt.ylabel(OUTPUT_COLUMNS[0])
    # plt.show()

    numeric_columns = [i for i in FEATURE_COLUMNS if i != 'weekday']

    log_cols = [f"{i}_ln" for i in numeric_columns]
    df[log_cols] = np.log(df[numeric_columns].values)
    TRANSFORM_COLUMNS.extend(log_cols)

    scaler = StandardScaler()
    columns_to_scale = [*FEATURE_COLUMNS, *TRANSFORM_COLUMNS]
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    sq_cols = [f"{i}_sq" for i in numeric_columns]
    df[sq_cols] = np.square(df[numeric_columns].values)
    TRANSFORM_COLUMNS.extend(sq_cols)

    # exp_cols = [f'{i}_exp' for i in numeric_columns]
    # df[exp_cols] = np.exp(df[numeric_columns].values)
    # TRANSFORM_COLUMNS.extend(exp_cols)

    FEATURE_COLUMNS.extend(TRANSFORM_COLUMNS)

    def create_sequence_column_for(column: str, days: int):
        shifted_col_name = f"{column}_{days}"
        SEQUENCE_COLUMNS.append(
            shifted_col_name
        ) if use_seq else FEATURE_COLUMNS.append(shifted_col_name)
        df[shifted_col_name] = df[column].shift(days)

    for i in range(1, SEQUENCE_LENGTH + 1):
        for channel in SEQUENCE_CHANNELS:
            create_sequence_column_for(channel, i)


    df = df.drop(df.index[:SEQUENCE_LENGTH])

    columns_to_average = [f'{OUTPUT_COLUMNS[0]}_{i}' for i in range(1, SEQUENCE_LENGTH + 1)]

    df['average'] = df[columns_to_average].mean(axis=1)

    FEATURE_COLUMNS.append('average')

    corr_matrix = df.corr()
    print(corr_matrix)

    return df


class WaterDataset(Dataset):
    def __init__(self, use_seq):
        data = load()
        self.use_seq = use_seq
        self.df = preprocess(data, self.use_seq)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        features_vector = row[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        seq_vector = np.reshape(
            row[SEQUENCE_COLUMNS].to_numpy(dtype=np.float32),
            (-1, len(SEQUENCE_CHANNELS)),
        )
        outputs_vector = row[OUTPUT_COLUMNS].to_numpy(dtype=np.float32)

        if self.use_seq:
            return (
                (torch.from_numpy(features_vector), torch.from_numpy(seq_vector)),
                torch.from_numpy(outputs_vector),
            )
        else:
            return torch.from_numpy(features_vector), torch.from_numpy(outputs_vector)
