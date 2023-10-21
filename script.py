# %%
from sklearnex import patch_sklearn

patch_sklearn()

import os
from tensorflow import keras
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import (
    RobustScaler
)
from sklearn import svm
import xgboost as xgb
import numpy as np
from concurrent.futures import ProcessPoolExecutor


zip_file = keras.utils.get_file(
    fname="cora.tgz",
    origin="https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
    extract=True,
)
data_dir = os.path.join(os.path.dirname(zip_file), "cora")


# %%
citations = pd.read_csv(
    os.path.join(data_dir, "cora.cites"),
    sep="\t",
    header=None,
    names=["target", "source"],
)
print("Citations shape:", citations.shape)

# %%
column_names = ["paper_id"] + [f"term_{idx}" for idx in range(1433)] + ["subject"]
papers = pd.read_csv(
    os.path.join(data_dir, "cora.content"),
    sep="\t",
    header=None,
    names=column_names,
)
print("Papers shape:", papers.shape)

# %%
class_values = sorted(papers["subject"].unique())
class_idx = {name: id for id, name in enumerate(class_values)}
paper_idx = {name: idx for idx, name in enumerate(sorted(papers["paper_id"].unique()))}

papers["paper_id"] = papers["paper_id"].apply(lambda name: paper_idx[name])
citations["source"] = citations["source"].apply(lambda name: paper_idx[name])
citations["target"] = citations["target"].apply(lambda name: paper_idx[name])
papers["subject"] = papers["subject"].apply(lambda value: class_idx[value])

# %%
papers.set_index("paper_id", inplace=True)
papers.head()

# %%
base_papers = papers.copy(deep=True)
target_cit = papers.copy(deep=True)
src_cit = papers.copy(deep=True)

base_papers.iloc[:, 1:-1] *= 100
target_cit.iloc[:, 1:-1] = 0
src_cit.iloc[:, 1:-1] = 0

for _, row in citations.iterrows():
    target_cit.iloc[row["source"], 1:-1] = papers.iloc[row["target"], 1:-1]
    src_cit.iloc[row["target"], 1:-1] = papers.iloc[row["source"], 1:-1]

# %%
def train_with_algo(X: pd.DataFrame, y: pd.DataFrame, test_size: float, method: str):
    scaler = RobustScaler()
    scaler.fit(X)

    X = scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=99)

    if method == "svm":
        clf = svm.LinearSVC(dual="auto", max_iter=10000000, random_state=99)
    elif method == "decision_tree":
        clf = DecisionTreeClassifier(criterion="log_loss", random_state=99)
    elif method == "xgboost":
        clf = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=len(np.unique(y_train)),
            learning_rate=0.1,  # Adjust the learning rate
            n_estimators=1000,  # Number of boosting rounds (trees)
            subsample=0.8,  # Fraction of samples used for training each tree
            colsample_bytree=0.8,  # Fraction of features used for training each tree
            random_state=99,  # Set a random seed for reproducibility
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=1000,
            criterion="log_loss",
            class_weight="balanced_subsample",
            random_state=99,
        )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return (
        accuracy_score(y_true=y_test, y_pred=y_pred),
        f1_score(y_true=y_test, y_pred=y_pred, average="macro"),
        precision_score(y_true=y_test, y_pred=y_pred, average="macro"),
        recall_score(y_true=y_test, y_pred=y_pred, average="macro"),
    )


# %%
def train(tgt_msg: int, src_msg: int, test_size: float, method: str):
    pm = base_papers.copy(deep=True)
    pm.iloc[:, 1:-1] -= tgt_msg * target_cit.iloc[:, 1:-1]
    pm.iloc[:, 1:-1] += src_msg * src_cit.iloc[:, 1:-1]

    pm.reset_index(inplace=True)
    X = pm.iloc[:, 1:-1]
    y = pm.iloc[:, -1]

    return train_with_algo(X, y, test_size=test_size, method=method)


# %%
def train_func(algo: str):
    with open(f"out-{algo}.csv", "+a") as f, open(f"out-{algo}-log.csv", "+a") as log:
        f.write("split,tgt_msg,src_msg,accuracy,f1_score,precision,recall\n")
        log.write("algo,split,tgt_msg,src_msg,accuracy,f1_score,precision,recall\n")

        for split in range(10, 50, 10):
            for tgt_msg in range(0, 10, 1):
                for src_msg in range(0, 10, 1):
                    a, f1, p, re = train(tgt_msg, src_msg, split / 100.0, algo)

                    print(algo, split, tgt_msg, src_msg, a, f1, p, re)

                    f.write(
                        f"{split},{tgt_msg},{src_msg},{a:.3},{f1:.3},{p:.3},{re:.3}\n"
                    )

                    log.write(
                        f"{algo},{split},{tgt_msg},{src_msg},{a:.3},{f1:.3},{p:.3},{re:.3}\n"
                    )


# %%
algorithms = ["svm", "decision_tree", "xgboost", "random_forest"]

with ProcessPoolExecutor() as executor:
    executor.map(train_func, algorithms)

# algorithms = ["xgboost"]

# for i in algorithms:
#     train_func(i)



