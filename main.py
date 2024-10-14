import argparse
import copy
import numpy as np
import pandas as pd
import json
import math
from typing import Dict, List


def read_input_data(fname: str) -> List[Dict]:
    """Read json file specified by the name and process errors

    Args:
        fname (str): relative path to the input file

    Returns:
        List[Dict]: the data from the file
    """
    data = []
    try:
        with open(fname, "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: The file '{fname}' does not exist.")
    except json.JSONDecodeError:
        print(f"Error: The file '{fname}' is not a valid JSON file.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return data


def read_weights(fname: str) -> Dict[str, float]:
    """Read json file with weights for trip criteria
        and checks that they sum up to 1.

    Args:
        fname (str): path to the input json file

    Returns:
        Dict[str, float]: dictionary with criteria weights as values
    """

    weights = read_input_data(fname)
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
    """Check that weights keys correspond to data field names

    Args:
        df (pd.DataFrame): dataframe with trips data
        weights (Dict[str, float]): weights for criteria
    """
    for data_field in weights.keys():
        assert (
            data_field in df.columns
        ), f"Error: criterion {data_field} from weights is not in the data fields"


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


def preference_func(alt_a: float, alt_b: float) -> float:
    """Preference function that reflects how much
        preference one alternative has over another

    Args:
        alt_a (float): first alternative
        alt_b (float): second alternative

    Returns:
        float: preference value
    """
    return max(0, alt_a - alt_b)


def compute_preference_matrix(
    df: pd.DataFrame, weights: Dict[str, float]
) -> np.ndarray:
    """Compute a preference index for each criterion
        for each pair of alternatives, reflecting the
        extent to which one alternative is preferred
        over the other based on the preference function

    Args:
        df (pd.DataFrame): input data
        weights (Dict[str, float]): weights for criteria

    Returns:
        np.ndarray: preference matrix with the size of
        (num trips, num trips)
    """
    num_trips = df.shape[0]
    columns = weights.keys()
    pref_mtx = np.zeros((num_trips, num_trips))
    for i in range(num_trips):
        for j in range(num_trips):
            if i == j:
                continue
            pref_sum = 0
            for col in columns:
                pref_sum += weights[col] * preference_func(df[col][i], df[col][j])
            pref_mtx[i, j] = pref_sum
    return pref_mtx


def compute_score(df: pd.DataFrame, weights: Dict[str, float]):
    """Computes and adds a score column to the input data
        using provided weights for criteria that should be
        used for scoring. Score computation is based on the
        PROMETHEE (Preference Ranking Organization Method
        for Enrichment Evaluations)

    Args:
        df (pd.DataFrame): input data to be scored
        weights (Dict[str, float]): weights for criteria
    """
    pref_mtx = compute_preference_matrix(df, weights)
    if df.shape[0] == 1:
        df["score"] = 0.0
        return
    # how much the alternative is preferred over the others
    pos_flow = pref_mtx.sum(axis=1) / (df.shape[0] - 1)
    # how much the alternative is outranked by all others
    neg_flow = pref_mtx.sum(axis=0) / (df.shape[0] - 1)
    net_flow = pos_flow - neg_flow

    df["score"] = net_flow


def rank_trips(fname: str, weights_fname: str, out_name: str):
    """Reads input file with trip info and saves to json file
        output data with rank and score information added

    Args:
        fname (str): path to the input json file with data to be ranked
        weights_fname (str): path to file with weights for input data
        out_name (str): file path where to save output json data
    """
    weights = read_weights(weights_fname)
    data = read_input_data(fname)
    assert len(data) > 0, f"The input data is empty \n"
    data_original = copy.deepcopy(data)
    clean_data(data)

    df = pd.DataFrame(data)
    check_data_fields(df, weights)
    normalize(df)

    compute_score(df, weights)
    df_original = pd.DataFrame(data_original)
    df_original["score"] = df["score"]
    df_original["rank"] = df_original["score"].rank(method="max", ascending=False)
    # if want to save sorted data
    # df_original = df_original.sort_values(by="rank")
    df_original.to_json(out_name, orient="records")


def main():
    parser = argparse.ArgumentParser(
        description="Read a file with trip descriptions and rank teh trips"
    )
    parser.add_argument("filename", help="The relative path to the input json file")
    args = parser.parse_args()
    in_name = args.filename
    out_name = "data/sorted.json"
    weights_fname = "data/weights.json"
    rank_trips(in_name, weights_fname, out_name)


if __name__ == "__main__":
    main()
