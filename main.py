import numpy as np
import pandas as pd
import json
import math
from typing import Dict, List


def read_weights(fname: str) -> Dict[str, float]:
    with open(fname, "r") as file:
        weights = json.load(file)
    w_sum = sum(weights.values())
    assert math.isclose(
        w_sum, 1.0, rel_tol=1e-9
    ), f"Weights should sum up to 1.0, not {w_sum}"
    return weights


def clean_data(data: List[Dict]):
    for trip in data:
        # TODO manage return trips
        if "desc" in trip.keys():
            num_changes = len(trip["desc"].split("dep"))
            assert num_changes == len(
                trip["desc"].split("arr")
            ), f"Inconsistent description for trip {trip[id]}: number of arrivals and departures differs"
            # if there was no dep&arr hints in the description, assume no changes
            trip["num_changes"] = max(0, num_changes - 2)
        else:
            trip["num_changes"] = 0


def check_data_fields(df: pd.DataFrame, weights: Dict[str, float]):
    for data_field in weights.keys():
        assert data_field in df.columns


def normalize(df: pd.DataFrame):
    """Apply min-max normalization to input data
        for minimization problem

    Args:
        df (pd.DataFrame): dataframe with input data
    """
    
    columns = ["co2_kg", "price_eur", "duration_out_sec", "num_changes"]
    for col in columns:
        if df[col].max() != df[col].min():
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        else:
            df[col] = 0
    # normalize work time for minimization
    col = "workTime_sec"
    if df[col].max() != df[col].min():
        df[col] = (df[col].max() - df[col]) / (df[col].max() - df[col].min())
    else:
        df[col] = 0


# TODO choose best
def preference_func(diff: float) -> float:
    return max(0, diff)


def compute_preference_matrix(df: pd.DataFrame, weights: Dict[str, float]) -> np.ndarray:
    num_trips = df.shape[0]
    columns = weights.keys()
    pref_mtx = np.zeros((num_trips, num_trips))
    for i in range(num_trips):
        for j in range(num_trips):
            if i == j:
                continue
            pref_sum = 0
            for col in columns:
                diff = df[col][i] - df[col][j]
                pref_sum += weights[col] * preference_func(diff)
            pref_mtx[i, j] = pref_sum
    return pref_mtx


def compute_score(df: pd.DataFrame, weights: Dict[str, float]):
    pref_mtx = compute_preference_matrix(df, weights)
    pos_flow = pref_mtx.sum(axis=1) / (df.shape[0] - 1)
    neg_flow = pref_mtx.sum(axis=0) / (df.shape[0] - 1)
    net_flow = pos_flow - neg_flow

    df["score"] = net_flow


def rank_trips(fname: str, weights_fname: str, out_name: str):
    weights = read_weights(weights_fname)
    with open(fname, "r") as file:
        data = json.load(file)
    clean_data(data)

    df = pd.DataFrame(data)
    check_data_fields(df, weights)
    normalize(df)

    compute_score(df, weights)

    df["rank"] = df["score"].rank(method="max", ascending=False)
    # if want to save sorted data
    # df = df.sort_values(by="rank")
    df.to_json(out_name, orient="records")


def main():
    fname = "data/ex3-data.json"
    out_name = "data/sorted.json"
    weights_fname = "data/weights.json"
    rank_trips(fname, weights_fname, out_name)


if __name__ == "__main__":
    main()
