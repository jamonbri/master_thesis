from dataclasses import dataclass, fields
from typing import List
import pandas as pd
from itertools import islice
from operator import attrgetter

@dataclass
class CategoryVector:
    fantasy: float = 0
    non_fiction: float = 0
    mystery: float = 0
    young_adult: float = 0
    graphic: float = 0
    thriller: float = 0
    paranormal: float = 0
    romance: float = 0
    history: float = 0
    biography: float = 0
    historical_fiction: float = 0
    comics: float = 0
    poetry: float = 0
    crime: float = 0
    children: float = 0
    fiction: float = 0
    
    def vector_sum(self, other):
        summed_fields = {field.name: getattr(self, field.name) + getattr(other, field.name) for field in fields(self)}
        return type(self)(**summed_fields)

    def constant_multiply(self, constant):
        multiplied_fields = {field.name: getattr(self, field.name) * constant for field in fields(self)}
        return type(self)(**multiplied_fields)

def normalize_vector(vector: CategoryVector) -> CategoryVector:
    total = sum(getattr(vector, field.name) for field in fields(vector))
    if total == 0:
        return vector
    normalized_fields = {field.name: getattr(vector, field.name) / total for field in fields(vector)}
    return type(vector)(**normalized_fields)

def calculate_weighted_mean(vectors: List[CategoryVector], ratings: List[float]) -> CategoryVector:
    total_weight = sum(ratings)
    if total_weight == 0:
        total_weight = len(ratings)
        ratings = [1] * total_weight
    weighted_sum_vector = CategoryVector()
    
    for vector, rating in zip(vectors, ratings):
        weighted_vector = vector.constant_multiply(rating)
        weighted_sum_vector = weighted_sum_vector.vector_sum(weighted_vector)

    mean_vector = weighted_sum_vector.constant_multiply(1 / total_weight)
    normalized_mean_vector = normalize_vector(mean_vector)   
    return normalized_mean_vector

def vector_to_dict(vector: CategoryVector) -> dict:
    return {field.name: getattr(vector, field.name) for field in fields(vector)}

def get_top_books_from_category_vector(
    df: pd.DataFrame, 
    vector: CategoryVector, 
    n_top_categories: int = 3, 
    n_books_per_category: int = 3
) -> dict:
    vector_dict = vector_to_dict(vector)
    sorted_vector_dict = dict(sorted(vector_dict.items(), key=lambda item: item[1], reverse=True))
    sliced_vector_dict = dict(islice(sorted_vector_dict.items(), n_top_categories))
    top_books_dict = {k: [] for k in sliced_vector_dict.keys()}
    for category in sliced_vector_dict.keys():
        tmp_df = df.sort_values(by=lambda x: x["genres"].get(category), ascending=False)
        books = tmp_df.head(3)["book_id"].tolist()
        top_books_dict[category] = books
    return top_books_dict
