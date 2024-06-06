import pandas as pd
import os
from datetime import date

class Results:
    """
    Results class for managing the storing and processing of dataframes
    """

    def __init__(self) -> None:
        pass
    
    def create_new_directory(self, run_type: str = "results", verbose: bool = False) -> None:
        """
        Creates a new results directory for the model instance files

        Args:
            run_type: type of simulation ('results' for normal runs, 'sensitivity' for sensitivity analysis)
        """
        today = date.today()
        sim_run_counter = 1
        directory = f"{run_type}/{today}_0{sim_run_counter}"
        try:
            base_path = os.path.dirname(__file__)
        except NameError:
            base_path = os.getcwd()
        self.path = os.path.join(base_path, directory)
        # Create new directory if one or more models have been instantiated on the same day
        while os.path.exists(self.path):
            sim_run_counter += 1
            directory = f"{run_type}/{today}_{sim_run_counter}" if sim_run_counter >= 10 else f"{run_type}/{today}_0{sim_run_counter}"
            self.path = os.path.join(base_path, directory)
        os.mkdir(self.path)
        if verbose:
            print(f"Directory created: {directory.split('/')[-1]}")

    def store(self, prefix: str, data: list[tuple[str, pd.DataFrame]], verbose: bool = False) -> list[str]:
        """
        Store pandas dataframes as csv files in path of class directory. Returns list with filepaths

        Args:
            prefix: prefix for all csv files
            data: list of tuples containing the name of the df to save as first element 
                and the actual df as second element
        """
        filepaths = []
        for d in data:
            sim_run_counter = 1
            filepath = f"{prefix}_{d[0]}_{sim_run_counter}.csv"
            path = os.path.join(self.path, filepath)
            # Create new filename for each new run
            while os.path.exists(path):
                sim_run_counter += 1
                filepath = f"{prefix}_{d[0]}_{sim_run_counter}.csv"
                path = os.path.join(self.path, filepath)
            d[1].to_csv(path)
            filepaths.append(path)
            if verbose:
                print(f"\ndf {d[0]} stored")
        return filepaths
