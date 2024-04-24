import pandas as pd
import os
from datetime import date
from sklearn.metrics.pairwise import cosine_similarity
from utils import string_to_array

class Results:
    """
    Results class for managing the storing and processing of dataframes
    """

    def __init__(self) -> None:
        pass
    
    def create_new_directory(self) -> None:
        """
        Creates a new results directory for the model instance files
        """
        today = date.today()
        sim_run_counter = 1
        directory = f"results/{today}_{sim_run_counter}"
        try:
            base_path = os.path.dirname(__file__)
        except NameError:
            base_path = os.getcwd()
        self.path = os.path.join(base_path, directory)
        # Create new directory if one or more models have been instantiated on the same day
        while os.path.exists(self.path):
            sim_run_counter += 1
            directory = f"results/{today}_{sim_run_counter}"
            self.path = os.path.join(base_path, directory)
        os.mkdir(self.path)
        print(f"Results directory created: {directory.split('/')[-1]}")

    def store(self, prefix: str, data: list[tuple[str, pd.DataFrame]]) -> list[str]:
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
            print(f"\ndf {d[0]} stored")
        return filepaths
    
    def load(self, filename: str) -> pd.DataFrame:
        """
        Helper method to load a dataframe in order to avoid setting large dataframes
        as part of the model class attributes

        Args:
            filename: path of CSV file
        """
        df = pd.read_csv(filename)
        return df

    def get_vector_diff_df(self, filename: str) -> pd.DataFrame:
        """
        Get vector differences dataframeas cosine similarity between first 
        and last vector of each agent

        Args:
            filename: path of CSV file
        """
        df = self.load(filename)
        filtered_df = df[df["agent_type"] == "UserAgent"][["AgentID", "Step", "vector"]]
        result_data = []
        for agent_id, group in filtered_df.groupby("AgentID"):
            sorted_group = group.sort_values("Step")
            if len(sorted_group) > 1:
                first_vector = string_to_array(sorted_group.iloc[0]["vector"])
                last_vector = string_to_array(sorted_group.iloc[-1]["vector"])
                diff = cosine_similarity(first_vector, last_vector)[0][0]
                result_data.append({"AgentID": agent_id, "vector_diff": diff})
        return pd.DataFrame(result_data)
