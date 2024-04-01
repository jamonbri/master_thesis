import mesa
import pandas as pd
from model.agents import ItemAgent, UserAgent
from data.data_preparation import get_model_df, get_users_df, get_items_df


class RecommenderSystemModel(mesa.Model):
    
    def __init__(
        self, 
        n_users: int,
        priority: str | None = None,
        dummy: bool = False
    ):
        
        print("Initializing model...\n")
        super().__init__()
        self.num_users = n_users
        self.schedule = mesa.time.RandomActivation(self)
        
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

    def step(self) -> None:
        self.schedule.step()

    def create_user(self, user_row: pd.Series) -> UserAgent:
        return UserAgent(user_row, self)

    def create_item(self, item_row: pd.Series) -> ItemAgent:
        return ItemAgent(item_row, self)

    def run_model(self, n: int = 10) -> None:
        for i in range(n):
            self.step()
            print(f"Step {i + 1} executed.")
