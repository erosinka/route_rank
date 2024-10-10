import numpy as np
import pandas as pd
import json


# TODO: either user input or read from file
# assert that weigts sum to 1
weights = {'co2_kg': 0.1, 'price_eur': 0.4, 'workTime_sec': 0.2,
           'duration_out_sec': 0.2, 'num_changes': 0.1} 

def clean_data(data):
    for trip in data:
        # TODO manage these fields in df
        del trip['boo_return']
        if 'desc' in trip.keys():
            trip['num_changes'] = len(trip['desc'].split('dep')) - 2
            del trip['desc']
        else:
            # decide if desc is necessary field
            trip['num_changes'] = 0

def normalize(df):
    columns=['co2_kg', 'price_eur', 'duration_out_sec']
    for col in columns:
        df[col] = (df[col] - df[col].min())/(df[col].max() - df[col].min())
    # normalize work time for minimization
    col = 'workTime_sec'
    df[col] = (df[col].max() - df[col])/(df[col].max() - df[col].min())

# TODO choose best
def preference_func(diff):
    return max(0, diff)

def compute_preference_matrix(df, weights):
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

def compute_ranking(df, weights):
    pref_mtx = compute_preference_matrix(df, weights)
    pos_flow = pref_mtx.sum(axis=1) / (df.shape[0] - 1)
    neg_flow = pref_mtx.sum(axis=0) / (df.shape[0] - 1)
    net_flow = pos_flow - neg_flow

    df['rank'] = net_flow


def rank_trips(fname):
    with open(fname, 'r') as file:
        data = json.load(file)
    clean_data(data)

    df = pd.DataFrame(data)
    normalize(df)

    compute_ranking(df, weights)
 
    sorted_ids = df.sort_values(by='rank')['id']
    order_map = {value: index for index, value in enumerate(sorted_ids)}
    with open(fname, 'r') as file:
        data = json.load(file)
    sorted_data = sorted(data, key=lambda x: order_map[x['id']])
    with open('data/sorted.json', 'w') as file:
        json.dump(sorted_data, file, indent=4)

def main():
    fname = 'data/ex3-data.json'
    rank_trips(fname)

if __name__ == "__main__":
    main()
