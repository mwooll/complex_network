import networkx as nx
import pandas as pd
import numpy as np

from signed_network import SignedNetwork
from visualizations import plot_errors, histogram

def read_data_to_frame(data_file):
    df = pd.read_csv(data_file,
                     delimiter = ",",
                     names = ["source", "target", "weight", "time"],
                     dtype = {"source": int, "target": int})
    return df

def rename_column_entries(df, columns: list):
    unique_values = pd.unique(df[columns].values.ravel("C"))
    rename_dict = {old: index for index, old in enumerate(unique_values)}
    different_values = len(unique_values)
    
    df = df.replace({col: rename_dict for col in columns})
    return df, different_values

def get_mean_stdev(data):
    return np.mean(data), np.std(data)

def make_experiment(df, short_name, epsilon=1):
    max_weight = 10

    network = SignedNetwork(df,
                            create_using=nx.DiGraph,
                            empty=0)

    results = {}
    functions = [predict_hit_or_miss]
    keys = ["miss"]
    for func, key in zip(functions, keys):
        results[key] = {}
        errors = network.make_predictions(epsilon, func, max_weight, "stale")[1]
        results[key] = get_mean_stdev(errors)

    # stale_errors = network.make_predictions(epsilon, func, max_weight, "stale")[1]
    # results["stale"] = get_mean_stdev(stale_errors)

    # return results
    with open(f"keep/{short_name}.txt", 'w') as f:
        f.write(f"{short_name} (mean, stdev)\n\n")
        for key, value in results.items():
                f.write(f"\t{key}: {value}\n")

def predict_mean_fair_good(u, v, df, fair, good, new_u, new_v):
    if fair.size == 0 or good.size == 0:
        return 1

    fair_factor = np.mean(fair) if new_u else fair[u]
    good_factor = np.mean(good) if new_v else good[v]
    return fair_factor*good_factor

def predict_mean_weight(u, v, df, fair, good, new_u, new_v):
    if df.size == 0 :
        return 1

    fair_factor = 1 if new_u else fair[u]
    good_factor = np.mean(df["weight"]) if new_v else good[v]
    return fair_factor*good_factor

def predict_default(u, v, df, fair, good, new_u, new_v):
    fair_factor = 1 if new_u else fair[u]
    good_factor = 1 if new_v else good[v]
    return fair_factor*good_factor

def predict_hit_or_miss(u, v, df, fair, good, new_u, new_v):
    fair_factor = 1 if new_u else fair[u]
    good_factor = 1 if new_v else good[v]
    return -1 if fair_factor*good_factor < 0 else 1


if __name__ == "__main__":
    data = [("data/soc-sign-bitcoinotc.csv", "otc"),
            ("data/soc-sign-bitcoinalpha.csv", "alpha")]

    statistics_files = []

    epsilon = 1
    edges = 4000
    for data_set_file, short_hand in data:
        df = read_data_to_frame(data_set_file)
        df = df.sort_values("time")

        short = df[:edges]
        short_renamed, short_count = rename_column_entries(short, ["source", "target"])
        short_renamed.loc[short_renamed["weight"] > 0, "weight"] =  1
        short_renamed.loc[short_renamed["weight"] < 0, "weight"] = -1
        
        name = f"miss_{short_hand}_untouched"
        make_experiment(short_renamed, name, epsilon)

        statistics_files.append(name)
        print(f"done with {name}")

        extreme = df[df["weight"] != 1][:edges]
        extreme_renamed, extreme_count = rename_column_entries(extreme, ["source", "target"])
        extreme_renamed.loc[extreme_renamed["weight"] > 0, "weight"] =  1
        extreme_renamed.loc[extreme_renamed["weight"] < 0, "weight"] = -1
        name = f"miss_{short_hand}_filtered"
        make_experiment(extreme_renamed, name, epsilon)

        statistics_files.append(name)
        print(f"done with {name}")

    with open(f"keep/missed_results_{edges}_{epsilon}.txt", "w") as master:
        for file_name in statistics_files:
            with open(f"keep/{file_name}.txt", "r") as f:
                text = f.read()
            master.write(text)
            master.write("\n")

    print("done")