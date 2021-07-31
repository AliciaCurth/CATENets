""".
Reference: Jinsung Yoon, James Jordon, Mihaela van der Schaar,
"GANITE: Estimation of Individualized Treatment Effects using Generative Adversarial Nets",
International Conference on Learning Representations (ICLR), 2018.

Paper link: https://openreview.net/forum?id=ByKWUeWA-

Last updated Date: April 25th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

data_loading.py

Note: Load real-world individualized treatment effects estimation datasets

(1) data_loading_twin: Load twins data.
  - Reference: http://data.nber.org/data/linked-birth-infant-death-data-vital-statistics-data.html
"""
# stdlib
import random
from pathlib import Path
from typing import Tuple

# third party
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import catenets.logger as log

from .network import download_if_needed

np.random.seed(0)
random.seed(0)

DATASET = "Twin_Data.csv.gz"
URL = "https://bitbucket.org/mvdschaar/mlforhealthlabpub/raw/0b0190bcd38a76c405c805f1ca774971fcd85233/data/twins/Twin_Data.csv.gz"  # noqa: E501


def preprocess(
    fn_csv: Path,
    train_ratio: float = 0.8,
    treatment_type: str = "rand",
    seed: int = 42,
    treat_prop: float = 0.5,
) -> Tuple:
    """Load twins data.

    Args:
      - fn_csv: dataset csv
      - train_ratio: the ratio of training data
      - treatment_type: the treatment selection strategy
      - seed: random seed

    Returns:
      - train_x: features in training data
      - train_t: treatments in training data
      - train_y: observed outcomes in training data
      - train_potential_y: potential outcomes in training data
      - test_x: features in testing data
      - test_potential_y: potential outcomes in testing data
    """

    # Load original data (11400 patients, 30 features, 2 dimensional potential outcomes)
    df = pd.read_csv(fn_csv)

    cleaned_columns = []
    for col in df.columns:
        cleaned_columns.append(col.replace("'", "").replace("’", ""))
    df.columns = cleaned_columns

    feat_list = list(df)

    # 8: factor not on certificate, 9: factor not classifiable --> np.nan --> mode imputation
    medrisk_list = [
        "anemia",
        "cardiac",
        "lung",
        "diabetes",
        "herpes",
        "hydra",
        "hemo",
        "chyper",
        "phyper",
        "eclamp",
        "incervix",
        "pre4000",
        "dtotord",
        "preterm",
        "renal",
        "rh",
        "uterine",
        "othermr",
    ]
    # 99: missing
    other_list = ["cigar", "drink", "wtgain", "gestat", "dmeduc", "nprevist"]

    other_list2 = ["pldel", "resstatb"]  # but no samples are missing..

    bin_list = ["dmar"] + medrisk_list
    con_list = ["dmage", "mpcb"] + other_list
    cat_list = ["adequacy"] + other_list2

    for feat in medrisk_list:
        df[feat] = df[feat].apply(lambda x: df[feat].mode()[0] if x in [8, 9] else x)

    for feat in other_list:
        df.loc[df[feat] == 99, feat] = df.loc[df[feat] != 99, feat].mean()

    df_features = df[con_list + bin_list]

    for feat in cat_list:
        df_features = pd.concat(
            [df_features, pd.get_dummies(df[feat], prefix=feat)], axis=1
        )

    # Define features
    feat_list = [
        "dmage",
        "mpcb",
        "cigar",
        "drink",
        "wtgain",
        "gestat",
        "dmeduc",
        "nprevist",
        "dmar",
        "anemia",
        "cardiac",
        "lung",
        "diabetes",
        "herpes",
        "hydra",
        "hemo",
        "chyper",
        "phyper",
        "eclamp",
        "incervix",
        "pre4000",
        "dtotord",
        "preterm",
        "renal",
        "rh",
        "uterine",
        "othermr",
        "adequacy_1",
        "adequacy_2",
        "adequacy_3",
        "pldel_1",
        "pldel_2",
        "pldel_3",
        "pldel_4",
        "pldel_5",
        "resstatb_1",
        "resstatb_2",
        "resstatb_3",
        "resstatb_4",
    ]

    x = np.asarray(df_features[feat_list])
    y0 = np.asarray(df[["outcome(t=0)"]]).reshape((-1,))
    y1 = np.asarray(df[["outcome(t=1)"]]).reshape((-1,))

    # Preprocessing
    scaler = MinMaxScaler()
    scaler.fit(x)
    x = scaler.transform(x)

    no, dim = x.shape

    if treatment_type == "rand":
        # assign with p=0.5
        prob = np.ones(x.shape[0]) * treat_prop
    elif treatment_type == "logistic":
        # assign with logistic prob
        coef = np.random.uniform(-0.1, 0.1, size=[np.shape(x)[1], 1])
        prob = 1 / (1 + np.exp(-np.matmul(x, coef)))

    w = np.random.binomial(1, prob)
    y = y1 * w + y0 * (1 - w)

    potential_y = np.vstack((y0, y1)).T

    # Train/test division
    idx = np.random.permutation(no)
    train_idx = idx[: int(train_ratio * no)]
    test_idx = idx[int(train_ratio * no) :]

    train_x = x[train_idx, :]
    train_w = w[train_idx]
    train_y = y[train_idx]
    train_potential_y = potential_y[train_idx, :]

    test_x = x[test_idx, :]
    test_potential_y = potential_y[test_idx, :]

    return train_x, train_w, train_y, train_potential_y, test_x, test_potential_y


def load(
    data_path: Path,
    train_ratio: float = 0.8,
    treatment_type: str = "rand",
    seed: int = 42,
    treat_prop: float = 0.5,
) -> Tuple:
    """
    Download the dataset if needed.
    Load the dataset.
    Preprocess the data.
    Return train/test split.
    """
    csv = data_path / DATASET

    download_if_needed(csv, http_url=URL)

    log.debug(f"load dataset {csv}")

    return preprocess(
        csv,
        train_ratio=train_ratio,
        treatment_type=treatment_type,
        seed=seed,
        treat_prop=treat_prop,
    )
