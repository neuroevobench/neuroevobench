import os
import yaml
import numpy as np
import pandas as pd


def yaml_log_to_pd(fname: str):
    """Load a YAML log file and convert it to a Pandas DataFrame."""
    # Load the YAML file
    with open(fname, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    # Convert the YAML data to a list of dictionaries
    all_log = []
    for s_id in data.keys():
        all_log.append(data[s_id])

    # Convert the YAML data list to a Pandas DataFrame
    df = pd.DataFrame(all_log)
    return df


def filter_log(
    hyper_log,
    meta_log,
    strategy_name: str,
    metric: str,
    num_search_iters: int = 50,
):
    """Filter log for best hyperparameters and corresponding learning curve."""
    run_id = hyper_log.filter({"strategy_name": strategy_name}).run_id.iloc[0]
    search_iters = meta_log[run_id].time.iter_id
    log = meta_log[run_id].stats[metric].mean

    # Get all yaml files in the experiment directory
    for file in os.listdir(
        os.path.join("../", meta_log[run_id].meta.experiment_dir)
    ):
        if file.endswith(".yaml"):
            # Check if "search" is in file name
            if "search" in file:
                search_fname = os.path.join(
                    "../", meta_log[run_id].meta.experiment_dir, file
                )

    # Load hyperparam search log
    search_log = yaml_log_to_pd(search_fname)
    # Loop over all evals and find the best one
    best_score, max_scores = -10e10, []
    for i in range(num_search_iters):
        try:
            run_ids = search_iters == i
            sub_log = log[run_ids]
            time = meta_log[run_id].time.num_gens[run_ids]
            max_score_run = sub_log.max()
            if max_score_run > best_score:
                best_score = sub_log.max()
                best_s_id = i
                best_lcurve = sub_log
                best_time = time
            max_scores.append(max_score_run)
        except Exception as e:
            print(f"{i} failed: {e}")

    # Get the best hyperparameters
    best_hypers = search_log[search_log["eval_id"] == best_s_id]["params"].iloc[
        0
    ]
    return {
        "perf": best_score,
        "s_id": best_s_id,
        "lcurve": best_lcurve,
        "hypers": best_hypers,
        "max_scores": np.array(max_scores),
        "time": best_time,
    }
