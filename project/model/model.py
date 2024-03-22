import mesa
import pandas as pd
import numpy as np
from model.agents import ItemAgent, UserAgent
from data.data_preparation import get_model_df, get_users_df, get_items_df
from sklearn.metrics.pairwise import cosine_similarity


class RecommenderSystemModel(mesa.Model):
    
    def __init__(
        self, 
        n_users: int,
        priority: str | None = None,
        dummy: bool = False
    ):
        super().__init__()
        self.num_users = n_users
        self.schedule = mesa.time.RandomActivation(self)
        df = get_model_df(sample_users=n_users, dummy=dummy)
        df_users = get_users_df(df)
        df_items = get_items_df(df)
        user_agents = df_users.apply(self.create_user, axis=1)
        for a in user_agents:
            self.schedule.add(a)
        item_agents = df_items.apply(self.create_item, axis=1)
        for i in item_agents:
            self.schedule.add(i)

    def step(self) -> None:
        self.schedule.step()

    def create_user(self, user_row: pd.Series) -> UserAgent:
        return UserAgent(user_row, self)

    def create_item(self, item_row: pd.Series) -> ItemAgent:
        return ItemAgent(item_row, self)

    def calculate_cosine_similarity(self, agent_a: UserAgent, agent_b: UserAgent) -> np.ndarray:
        X = agent_a.get_vector_from_attr()
        Y = agent_b.get_vector_from_attr()
        return cosine_similarity(X, Y)

    def find_most_similar_agent(self, agent: UserAgent) -> UserAgent | None:
        max_similarity = -1
        most_similar_agent = None
        for other_agent in self.get_agents_of_type(UserAgent):
            if other_agent != agent:
                similarity = self.calculate_cosine_similarity(agent, other_agent)
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_agent = other_agent
        return most_similar_agent

    