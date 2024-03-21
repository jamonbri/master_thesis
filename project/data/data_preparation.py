import pandas as pd
import requests
import numpy as np
from bs4 import BeautifulSoup
import re
from model.category_vector import CategoryVector, calculate_weighted_mean, get_top_books_from_category_vector
from dataclasses import asdict
import ast

def get_categories() -> list[str]:
    return [
        "fantasy", 
        "non_fiction", 
        "mystery", 
        "young_adult",
        "graphic",
        "thriller",
        "paranormal",
        "romance",
        "history",
        "biography",
        "historical_fiction",
        "comics",
        "poetry",
        "crime",
        "children",
        "fiction"
    ]

def load_data(
    file_name: str, 
    head: int | None = 500
) -> pd.DataFrame:
    """Loads data from CSV and JSON files"""

    file_path = f"data/datasets/{file_name}"
    file_extension = file_name.split(".")[-1]
    if file_name.split(".")[-1] == "csv":
        df = pd.read_csv(file_path, nrows=head)
    elif file_extension == "json":
        df = pd.read_json(file_path, lines=True, orient='records', nrows=head)
    else:
        raise Exception(f"Format of dataset {file_name} not loadable.")
    return df

def get_model_df(n_users: int | None = None, sample_users: int = 100, dummy: bool = False) -> pd.DataFrame:
    """Gets general model df with interactions between users and items"""
    
    if dummy:  # Preload for testing
        return pd.read_csv("data/datasets/goodreads/goodreads_interactions_sample.csv", index_col="index")
    
    # Load all users data
    
    df_users_raw = load_data("goodreads/goodreads_interactions.csv", n_users)
    user_ids_sample = df_users_raw["user_id"].sample(sample_users)
    df_users_filtered = df_users_raw.loc[df_users_raw["user_id"].isin(user_ids_sample)]

    # Normalize rating

    df_users_filtered.loc[:, "rating"] = df_users_filtered["rating"].astype("float")
    df_users_filtered.loc[:, "rating"] = df_users_filtered["rating"] / 5.0
    
    # Load all items data

    df_items_raw = load_data("goodreads/goodreads_book_genres_initial.json", None)
    unique_item_ids = df_users_filtered["book_id"].unique().tolist()
    df_items_filtered = df_items_raw.loc[df_items_raw["book_id"].isin(unique_item_ids)]
    df_items_filtered = df_items_filtered[df_items_filtered["genres"].apply(lambda x: bool(x))]
    
    # Get categories into columns
    
    df_items_filtered.loc[:, "genres"] = df_items_filtered["genres"].apply(reformat_dict)
    df_items_filtered_normalized = pd.json_normalize(df_items_filtered["genres"])
    df_items_result = pd.concat([df_items_filtered.reset_index(drop=True), df_items_filtered_normalized.reset_index(drop=True)], axis=1)
    
    # Combine dfs and return
    
    df_combined = pd.merge(df_users_filtered, df_items_result, how="inner", on="book_id")
    return df_combined.drop("genres", axis=1)

def reformat_dict(d: dict) -> dict:
    """Reformats dict from JSON as columns for df"""
    genres = {}
    for genre, value in d.items():
        for g in genre.split(","):
            g = g.strip().replace(" ", "_").replace("-", "_")
            genres[g] = genres.get(g, 0) + value if value > 0 else 0  # Some genres had a -1, so they were removed
    genres.update({k: 0 for k in get_categories() if k not in genres})
    return genres

def get_items_df(df: pd.DataFrame, priority: str | None = None) -> pd.DataFrame:
    """Get aggregated items df"""
    aggregations = {
        "is_read": "sum",
        "rating": "mean",
        "is_reviewed": "sum"
    }
    aggregations.update({k: "mean" for k in get_categories()})
    tmp_df = df.copy().drop("user_id", axis=1)
    items_df = tmp_df.groupby(by=["book_id"]).agg(aggregations)
    if not priority:
        items_df["priority"] = 0
    elif priority == "random":
        items_df["priority"] = np.random.rand(len(items_df))
    return items_df

def get_users_df(df: pd.DataFrame, based_on: str = "all") -> pd.DataFrame:
    """Get aggregated users df"""
    tmp_df = df.copy()
    cat_cols = get_categories()
    tmp_df_normalized = normalize_category_df(tmp_df)
    aggregations = {
        "is_reviewed": "mean",
        "is_read": "sum",
        "rating": "mean",
        "book_id": lambda x: list(x)
    }
    users_df = tmp_df_normalized.groupby(by=["user_id"])[tmp_df.columns.difference(cat_cols)].agg(aggregations)
    aggregations = {k: "mean" for k in cat_cols}
    if based_on == "all":
        users_cat_df = tmp_df_normalized.groupby(by=["user_id"])[cat_cols].agg(aggregations)
        users_cat_df_normalized = normalize_category_df(users_cat_df)
    elif based_on == "is_read":
        condition = tmp_df_normalized["is_read"] == 1
        users_cat_df = tmp_df_normalized[condition].groupby(by=["user_id"])[cat_cols].agg(aggregations)
        users_cat_df_normalized = normalize_category_df(users_cat_df)
    elif based_on == "is_reviewed":
        condition = tmp_df_normalized["is_reviewed"] == 1
        users_cat_df = tmp_df_normalized[condition].groupby(by=["user_id"])[cat_cols].agg(aggregations)
        users_cat_df_normalized = normalize_category_df(users_cat_df)
    elif based_on == "rating":
        condition = tmp_df_normalized["rating"] > 0
        users_cat_df = tmp_df_normalized[condition].groupby(by=["user_id"])[cat_cols].agg(aggregations)
        users_cat_df_normalized = normalize_category_df(users_cat_df, penalize_col="rating")
    users_df_normalized = users_df.join(users_cat_df_normalized, how="left")
    return users_df_normalized

def normalize_category_df(df: pd.DataFrame, penalize_col: str | None = None) -> pd.DataFrame:
    tmp_df = df.copy()
    cat_cols = get_categories()
    if penalize_col:
        for col in cat_cols:
            tmp_df[col] *= tmp_df[penalize_col]
    df_normalized = tmp_df[cat_cols].div(tmp_df[cat_cols].sum(axis=1), axis=0)
    for col in tmp_df.columns.difference(cat_cols):
        df_normalized[col] = tmp_df[col]
    return df_normalized

def get_users_vectors(df: pd.DataFrame) -> pd.DataFrame:
    users = df["user_id"].unique().tolist()
    results = pd.DataFrame(columns=["category_vector", "review_probability", "books_read"])
    results.index.name = "user_id"
    for user in users:
        tmp_df = df.loc[df["user_id"] == user]
        category_vector = calculate_weighted_mean(
            vectors=tmp_df["category_vector"].tolist(), 
            ratings=tmp_df["rating"].tolist()
        )
        top_books_read = get_top_books_from_category_vector(tmp_df, category_vector)
        user_vector = {
            "category_vector": category_vector,
            "review_probability": tmp_df["is_reviewed"].mean(),
            "books_read": tmp_df["book_id"].unique().tolist()
        }
        results.loc[user] = user_vector
    return results.fillna(0)

