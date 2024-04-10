import mesa
import pandas as pd
import numpy as np
from model.agents import ItemAgent, UserAgent
from data.data_preparation import get_model_df, get_users_df, get_items_df, get_categories


def get_vector(agent: mesa.Agent) -> np.array:
    """
    Helper method to get agent vector for data collection
    """
    if isinstance(agent, UserAgent):
        return agent.vector.copy()
    return None

class RecommenderSystemModel(mesa.Model):
    """
    Recommender system model
    """
    
    def __init__(
        self, 
        n_users: int,
        steps: int = 10,
        priority: str | None = None,
        dummy: bool = False
    ):
        """
        Create a new recommender system model instance

        Args:
            n_users: number of users to simulate as agents
            steps: number of steps per simulation run
            priority: item priority for hidden agenda
            dummy: use of pre-loaded data for faster data loading
        """
        
        # Model initialization
        print("Initializing model...\n")
        super().__init__()
        self.num_users = n_users
        self.schedule = mesa.time.RandomActivation(self)
        self.steps = steps
        self.priority = priority
        
        # Model dataframe extraction
        df = get_model_df(sample_users=n_users, dummy=dummy)
        len_df = len(df)
        print(f"Model dataframe ready. Interactions: {len_df}")
        
        # Items dataframe extraction
        df_items = get_items_df(df, self.priority)
        len_df_items = len(df_items)
        print(f"Items dataframe ready. Items: {len_df_items}")
        
        # Users dataframe extraction
        df_users = get_users_df(df, df_items)
        len_df_users = len(df_users)
        print(f"Users dataframe ready. Users: {len_df_users}")
        
        # User agents creation
        print("Creating user agents...")
        df_users["unique_id"] = range(1, len_df_users + 1)
        user_agents = df_users.apply(self.create_user, axis=1)
        for a in user_agents:
            self.schedule.add(a)
        print("Users added.")
        
        # Item agents creation
        print("Creating item agents...")
        df_items["unique_id"] = range(len_df_users + 1, len_df_users + 1 + len_df_items)
        item_agents = df_items.apply(self.create_item, axis=1)
        for i in item_agents:
            self.schedule.add(i)
        print("Items added.")
        print("Finished model initialization.")

        # Data collection setup
        self.datacollector = mesa.DataCollector(
            agent_reporters={"vector": lambda a: get_vector(a)}
        )

    def step(self) -> None:
        """
        Advance model by one step
        """
        self.schedule.step()
        self.datacollector.collect(self)

    def create_user(self, user_row: pd.Series) -> UserAgent:
        """
        Create user agent
        """
        return UserAgent(user_row, self)

    def create_item(self, item_row: pd.Series) -> ItemAgent:
        """
        Create item agent
        """
        return ItemAgent(item_row, self)

    def run_model(self) -> None:
        """
        Run model simulation
        """
        for i in range(self.steps):
            self.step()
            print(f"Step {i + 1} executed.")

    def get_processed_df(self) -> pd.DataFrame:
        """
        Get processed dataframe with results
        """
        df_vectors = self.datacollector.get_agent_vars_dataframe()
        df_vectors.dropna(inplace=True)
        
        # Convert vector to single dimension pandas series
        df_vectors["vector"] = df_vectors["vector"].apply(lambda x: x[0])
        df_vectors_wide = df_vectors["vector"].apply(pd.Series)
        df_vectors_wide.columns = get_categories()
        return df_vectors_wide
