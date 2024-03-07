import mesa
from agents import ItemAgent, UserAgent
from data.data_preparation import get_model_dataframe


class RecommenderSystemModel(mesa.Model):
    
    def __init__(
        self, 
        n_users: int,
        dataset: str
    ):
        super().__init__()
        self.num_users = n_users
        self.schedule = mesa.time.RandomActivation(self)
        df = get_model_dataframe(dataset, n_users)
        for i in range(self.num_agents):
            a = ItemAgent(i, self)
            self.schedule.add(a)

    def step(self) -> None:
        self.schedule.step()

    