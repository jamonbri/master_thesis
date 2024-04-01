import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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
    
    print("Loading data...")
    
    if dummy:  # Preload for testing
        print("Dummy data read")
        return pd.read_csv("data/datasets/goodreads/goodreads_interactions_sample.csv", index_col="index")
    
    # Load all users data
    
    df_users_raw = load_data("goodreads/goodreads_interactions.csv", n_users)
    user_ids_sample = df_users_raw["user_id"].sample(sample_users)
    df_users_filtered = df_users_raw.loc[df_users_raw["user_id"].isin(user_ids_sample)]
    print("    - Users loaded")

    # Normalize rating

    df_users_filtered.loc[:, "rating"] = df_users_filtered["rating"].astype("float")
    df_users_filtered.loc[:, "rating"] = df_users_filtered["rating"] / 5.0
    
    # Load all items data

    df_items_raw = load_data("goodreads/goodreads_book_genres_initial.json", None)
    unique_item_ids = df_users_filtered["book_id"].unique().tolist()
    df_items_filtered = df_items_raw.loc[df_items_raw["book_id"].isin(unique_item_ids)]
    df_items_filtered = df_items_filtered[df_items_filtered["genres"].apply(lambda x: bool(x))]
    print("    - Items loaded")
    
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
    
    print("Getting items dataframe...")
    cat_cols = get_categories()
    aggregations = {
        "is_read": "sum",
        "rating": "mean",
        "is_reviewed": "sum"
    }
    aggregations.update({k: "mean" for k in cat_cols})
    tmp_df = df.copy().drop("user_id", axis=1)
    items_df = tmp_df.groupby(by=["book_id"]).agg(aggregations)
    items_df["vector"] = items_df.apply(lambda row: np.array(row[cat_cols]).reshape(1, -1), axis=1)
    items_df = items_df.drop(cat_cols, axis=1)
    if not priority:
        items_df["priority"] = 0
    elif priority == "random":
        items_df["priority"] = np.random.rand(len(items_df))
    return items_df

def get_users_df(df: pd.DataFrame, df_items: pd.DataFrame, based_on: str = "all") -> pd.DataFrame:
    """Get aggregated users df"""
    
    print("Getting users dataframe...")
    tmp_df = df.copy()
    cat_cols = get_categories()
    for col in cat_cols:
        tmp_df[col] = tmp_df[col].apply(lambda x: 1 if x > 0 else 0)
    aggregations = {
        "is_reviewed": "mean",
        "is_read": "sum",
        "rating": "mean",
        "book_id": lambda x: list(x)
    }
    aggregations.update({k: "sum" for k in cat_cols})
    users_df = tmp_df.groupby(by=["user_id"]).agg(aggregations)
    users_df["vector"] = users_df.apply(lambda row: np.array(row[cat_cols]).reshape(1, -1), axis=1)
    users_df = users_df.drop(cat_cols, axis=1)
    users_df["book_id"] = users_df.apply(calculate_book_score, args=(df_items,), axis=1)
    return users_df

def calculate_book_score(row: pd.Series, df_items: pd.DataFrame) -> dict:
    """Calulate item scores based on cosine similarity with users"""

    books = {}
    normalized_user_vector = normalize_vector(row["vector"])
    for book_id in row["book_id"]:
        item_vector = df_items[df_items.index == book_id]["vector"].item()
        normalized_item_vector = normalize_vector(item_vector)
        similarity = cosine_similarity(normalized_user_vector, normalized_item_vector)
        books.update({book_id: similarity[0][0]})
    return books
    
def normalize_vector(v: np.array) -> np.array:
    if not np.count_nonzero(v):
        return v
    return (v - v.min()) / (v.max() - v.min())
