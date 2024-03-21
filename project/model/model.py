import mesa
from model.agents import create_item, create_user
from data.data_preparation import get_model_df, get_users_df, get_items_df


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
        user_agents = df_users.apply(create_user, args=(self,), axis=1)
        for a in user_agents:
            self.schedule.add(a)
        item_agents = df_items.apply(create_item, args=(self,), axis=1)
        for i in item_agents:
            self.schedule.add(i)

    def step(self) -> None:
        self.schedule.step()

    