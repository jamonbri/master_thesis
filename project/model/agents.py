import mesa
import pandas as pd
import numpy as np
from data.data_preparation import get_categories

class ItemAgent(mesa.Agent):
    
    def __init__(
        self, 
        item_row: pd.Series,
        model: mesa.Model
    ) -> None:
        super().__init__(item_row.index[0], model)
        self.n_read = item_row["is_read"]
        self.mean_rating = item_row["rating"]
        self.n_reviews = item_row["is_reviewed"]
        self.priority = item_row["priority"]
        self.fantasy = item_row["fantasy"]
        self.non_fiction = item_row["non_fiction"]
        self.mystery = item_row["mystery"]
        self.young_adult = item_row["young_adult"]
        self.graphic = item_row["graphic"]
        self.thriller = item_row["thriller"]
        self.paranormal = item_row["paranormal"]
        self.romance = item_row["romance"]
        self.history = item_row["history"]
        self.biography = item_row["biography"]
        self.historical_fiction = item_row["historical_fiction"]
        self.comics = item_row["comics"]
        self.poetry = item_row["poetry"]
        self.crime = item_row["crime"]
        self.children = item_row["children"]
        self.fiction = item_row["fiction"]

    def step(self) -> None:
        print(f"Taking a step")


class UserAgent(mesa.Agent):
    def __init__(
        self, 
        user_row: pd.DataFrame,
        model: mesa.Model
    ) -> None:
        super().__init__(user_row.index[0], model)
        self.books = user_row["book_id"]
        self.review_probability = user_row["is_reviewed"]
        self.mean_rating = user_row["rating"]
        self.n_books = user_row["is_read"]
        self.fantasy = user_row["fantasy"]
        self.non_fiction = user_row["non_fiction"]
        self.mystery = user_row["mystery"]
        self.young_adult = user_row["young_adult"]
        self.graphic = user_row["graphic"]
        self.thriller = user_row["thriller"]
        self.paranormal = user_row["paranormal"]
        self.romance = user_row["romance"]
        self.history = user_row["history"]
        self.biography = user_row["biography"]
        self.historical_fiction = user_row["historical_fiction"]
        self.comics = user_row["comics"]
        self.poetry = user_row["poetry"]
        self.crime = user_row["crime"]
        self.children = user_row["children"]
        self.fiction = user_row["fiction"]

    def get_vector_from_attr(self) -> np.array:
        cat_cols = get_categories()
        vector = []
        for col in cat_cols:
            vector.append(getattr(self, col))
        array = np.array(vector)
        return array.reshape(1, -1)
    
    def step(self) -> None:
        print(f"Taking a step")
    