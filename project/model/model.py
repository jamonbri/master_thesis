import mesa
import pandas as pd
import numpy as np
from model.agents import ItemAgent, UserAgent
from data.data_preparation import get_model_df, get_users_df, get_items_df, get_categories


def get_vector(agent: mesa.Agent) -> np.array:
    if isinstance(agent, UserAgent):
        return agent.vector.copy()
    return None

class RecommenderSystemModel(mesa.Model):
    
    def __init__(
        self, 
        n_users: int,
        steps: int = 10,
        priority: str | None = None,
        dummy: bool = False
    ):
        
        print("Initializing model...\n")
        super().__init__()
        self.num_users = n_users
        self.schedule = mesa.time.RandomActivation(self)
        self.steps = steps
        
        df = get_model_df(sample_users=n_users, dummy=dummy)
        len_df = len(df)
        print(f"Model dataframe ready. Interactions: {len_df}")
        
        df_items = get_items_df(df)
        len_df_items = len(df_items)
        print(f"Items dataframe ready. Items: {len_df_items}")
        
        df_users = get_users_df(df, df_items)
        len_df_users = len(df_users)
        print(f"Users dataframe ready. Users: {len_df_users}")
        
        print("Creating user agents...")
        df_users["unique_id"] = range(1, len_df_users + 1)
        user_agents = df_users.apply(self.create_user, axis=1)
        for a in user_agents:
            self.schedule.add(a)
        print("Users added.")
        
        print("Creating item agents...")
        df_items["unique_id"] = range(len_df_users + 1, len_df_users + 1 + len_df_items)
        item_agents = df_items.apply(self.create_item, axis=1)
        for i in item_agents:
            self.schedule.add(i)
        print("Items added.")
        print("Finished model initialization.")

        self.datacollector = mesa.DataCollector(
            agent_reporters={"vector": lambda a: get_vector(a)}
        )

    def step(self) -> None:
        self.schedule.step()
        self.datacollector.collect(self)

    def create_user(self, user_row: pd.Series) -> UserAgent:
        return UserAgent(user_row, self)

    def create_item(self, item_row: pd.Series) -> ItemAgent:
        return ItemAgent(item_row, self)

    def run_model(self) -> None:
        for i in range(self.steps):
            self.step()
            print(f"Step {i + 1} executed.")

    def get_processed_df(self) -> pd.DataFrame:
        df_vectors = self.datacollector.get_agent_vars_dataframe()
        df_vectors.dropna(inplace=True)
        df_vectors["vector"] = df_vectors["vector"].apply(lambda x: x[0])
        df_vectors_wide = df_vectors["vector"].apply(pd.Series)
        df_vectors_wide.columns = get_categories()
        return df_vectors_wide
