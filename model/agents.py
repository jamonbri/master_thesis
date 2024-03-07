import mesa

class ItemAgent(mesa.Agent):
    
    def __init__(
        self, 
        unique_id,
        model,
        category: str, 
        rating: float, 
        priority: float
    ) -> None:
        super().__init__(unique_id, model)
        self.category = category
        self.rating = rating
        self.priority = priority
        self.consumption = 0.0

    def step(self) -> None:
        print(f"Taking a step")


class UserAgent(mesa.Agent):
    def __init__(
        self, 
        unique_id,
        model,
        items: list, 
        freq_probability: float, 
        feedback_likelihood: float, 
        item_likelihood: float
    ) -> None:
        super().__init__(unique_id, model)
        self.items = items
        self.freq_probability = freq_probability
        self.feedback_likelihood = feedback_likelihood
        self.item_likelihood = item_likelihood

    def step(self) -> None:
        print(f"Taking a step")
    