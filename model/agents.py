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
    