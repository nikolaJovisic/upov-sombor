import re
from pathlib import Path

DATA_CSV = Path(__file__).resolve().parent / "data.csv"

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import seaborn as sns

FEATURE_COLUMNS = [
    "Temperature_(C˚)_in",
    "pH_in",
    "COD_(mg/l)_in",
    "BOD5_(mg/l)_in",
    "N_(mg/l)_in",
    "P_(mg/l)_in",
    "TSS_(mg/l)_in",
]

OUTPUT_COLUMN = "BOD5_(mg/l)_out"  # "COD_(mg/l)_out"
OUTPUT_COLUMNS = [OUTPUT_COLUMN]

SEQUENCE_CHANNELS = [
    OUTPUT_COLUMN,
    OUTPUT_COLUMN[: -len("out")] + "in",
    # "weekday",
]
SEQUENCE_COLUMNS = []
TRANSFORM_COLUMNS = []

SEQUENCE_LENGTH = 10


def format_date(date: str) -> str:
    date = date.replace("/", ".")
    date = date.replace(",", ".")
    return date.strip(".")
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
def generate_violin_plots(df):
    num_columns = len(df.columns)
    num_rows = (num_columns // 6) + (num_columns % 6 > 0)

    # Create the figure and subplots
    fig, axes = plt.subplots(num_rows, 6, figsize=(15, num_rows * 5))

    # Loop through each column in the DataFrame
    for i, column in enumerate(df.columns):
        row = i // 6
        col = i % 6

        # Select the subplot to plot on
        ax = axes[row, col] if num_rows > 1 else axes[col]

        # Create the violin plot using seaborn
        sns.violinplot(data=df[column], orient="v", ax=ax)

        # Set labels and title
        ax.set_ylabel(column)

    # Remove unused subplots if necessary
    if num_columns < num_rows * 6:
        for i in range(num_columns, num_rows * 6):
            fig.delaxes(axes.flatten()[i])

    # Adjust spacing between subplots
    fig.tight_layout()

    # Show the plot
    plt.show()
def load():
    df = pd.read_csv(DATA_CSV)

    columns = []
    for column in df.columns:
        if column == "Unnamed: 0":
            continue
        columns.append(column)

    new_columns = []
    valid_index = 0
    steping_index = 0
    for c in df.iloc[0]:
        if c == "Date":
            new_columns.append("Date")
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
    df.dropna(subset=["Date"], inplace=True)
    df = df[~df["Date"].str.endswith("(2)")]

    df.dropna(subset=FEATURE_COLUMNS, inplace=True)
    df.dropna(subset=OUTPUT_COLUMNS, inplace=True)

    df = df.map(lambda x: format_date(x) if isinstance(x, str) else x)
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y")

    if 'weekday' in SEQUENCE_CHANNELS:
        FEATURE_COLUMNS.append("weekday")
        df["weekday"] = df["Date"].apply(lambda dt: dt.weekday())

    df.set_index("Date", inplace=True)

    df = df.map(format_strings)

    df = df.loc[:, [*FEATURE_COLUMNS, OUTPUT_COLUMN]]
    df = df[~(df == 0).any(axis=1)]

    if OUTPUT_COLUMN == 'COD_(mg/l)_out':
        df = df[df['COD_(mg/l)_out'] < 125]
    if OUTPUT_COLUMN == 'BOD5_(mg/l)_out':
        df = df[df['BOD5_(mg/l)_out'] < 180]

    df = df[df['COD_(mg/l)_in'] < 2000]
    df = df[df['BOD5_(mg/l)_in'] < 900]
    df = df[df['N_(mg/l)_in'] < 200]
    df = df[df['P_(mg/l)_in'] < 35]
    df = df[df['TSS_(mg/l)_in'] < 5000]

    df = df.groupby("Date").mean()
    df_interp = df.asfreq("D").interpolate(method="time")
    rolling_mean = df_interp.rolling(window=50, min_periods=1).mean()
    df = df.combine_first(rolling_mean)

    # for column_to_plot in FEATURE_COLUMNS:
    #     plt.plot(df[column_to_plot])
    #     plt.xlabel('measurements')
    #     plt.ylabel(column_to_plot)
    #     plt.show()

    numeric_columns = [i for i in FEATURE_COLUMNS if i != 'weekday']

    # generate_violin_plots(df)

    # a = df.describe()
    # a.to_csv('stats.csv', index=True)

    # log_cols = [f"{i}_ln" for i in numeric_columns]
    # df[log_cols] = np.log(df[numeric_columns].values)
    # TRANSFORM_COLUMNS.extend(log_cols)

    scaler = StandardScaler()
    columns_to_scale = [*FEATURE_COLUMNS, *TRANSFORM_COLUMNS]
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    # sq_cols = [f"{i}_sq" for i in numeric_columns]
    # df[sq_cols] = np.square(df[numeric_columns].values)
    # TRANSFORM_COLUMNS.extend(sq_cols)

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

    columns_to_average = [f'{OUTPUT_COLUMN}_{i}' for i in range(1, SEQUENCE_LENGTH + 1)]

    df['average'] = df[columns_to_average].mean(axis=1)

    FEATURE_COLUMNS.append('average')

    corr_matrix = df.corr()
    print(corr_matrix)

    df_train = df[df.index < '2019-02-01']
    df_test = df[(df.index >= '2019-04-15') & (df.index <= '2020-03-11')]

    return df_train, df_test


class WaterDataset(Dataset):
    def __init__(self, df, use_seq):
        self.use_seq = use_seq
        self.df = df

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
