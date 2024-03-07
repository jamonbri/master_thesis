import mesa
from agents import ItemAgent, UserAgent


class RecommenderSystemModel(mesa.Model):
    
    def __init__(self, n_items, n_users):
        super().__init__()
        self.num_items = n_items
        self.num_users = n_users
        self.schedule = mesa.time.RandomActivation(self)
        for i in range(self.num_agents):
            a = ItemAgent(i, self)
            self.schedule.add(a)

    def step(self) -> None:
        self.schedule.step()

    