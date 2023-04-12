from typing import Dict, Any, Optional
import sys
import pandas as pd
from pathlib import Path

if sys.version_info < (3, 8):
    # Load with pickle5 for python version compatibility
    import pickle5 as pickle
else:
    import pickle


def save_pkl_object(obj: Any, filename: str) -> None:
    """Store objects as pickle files.

    Args:
        obj (Any): Object to pickle.
        filename (str): File path to store object in.
    """
    with open(filename, "wb") as output:
        # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


class CSV_Logger(object):
    def __init__(self, log_dir: str = "temp/"):
        self.log_dir = log_dir
        self.update_counter = 0

    def update(
        self,
        time_tic: Dict[str, float],
        stats_tic: Dict[str, float],
        model: Optional[Any] = None,
        save: bool = True,
        verbose: bool = True,
    ):
        """Store data in csv format / pandas dataframe."""
        combine_dict = {**time_tic, **stats_tic}
        if self.update_counter == 0:
            self.data_df = pd.DataFrame([combine_dict])
        else:
            self.data_df = self.data_df.append(combine_dict, ignore_index=True)
        self.update_counter += 1

        if verbose:
            print(combine_dict)
        if save:
            filepath = Path(self.log_dir + "log.csv")
            filepath.parent.mkdir(parents=True, exist_ok=True)
            self.data_df.to_csv(filepath)
            print(f"Stored data at: {self.log_dir + 'log.csv'}")
