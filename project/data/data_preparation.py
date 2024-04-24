import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

def get_categories() -> list[str]:
    """
    Return list of categories
    """
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
    file_path: str, 
    head: int | None = 500
) -> pd.DataFrame:
    """
    Loads data from CSV and JSON files
    
    Args:
        file_path: file path 
        head: number of rows to load
    """

    file_extension = file_path.split(".")[-1]
    if file_extension == "csv":
        df = pd.read_csv(file_path, nrows=head)
    elif file_extension == "json":
        df = pd.read_json(file_path, lines=True, orient='records', nrows=head)
    return df

def get_model_df(
    n_users: int | None = None, 
    sample_users: int = 100, 
    dummy: bool = False, 
    seed: int | None = None
) -> pd.DataFrame:
    """
    Gets general model df with interactions between users and items

    Args:
        n_users: number of users to extract from CSV
        sample_users: number of users to sample (i.e. agents)
        dummy: load a pre-saved dummy dataset
        seed: random state seed for user sampling
    """
    
    print("Loading data...")

    try:
        base_path = os.path.dirname(__file__)
    except NameError:
        base_path = os.getcwd()

    file_path = os.path.join(base_path, "datasets/goodreads")
    
    if dummy:  # Preload for testing
        print("Dummy data read")
        return pd.read_csv(f"{file_path}/goodreads_interactions_sample.csv", index_col="index")
    
    # Load all users data
    
    df_users_raw = load_data(f"{file_path}/goodreads_interactions.csv", n_users)
    user_ids_sample = df_users_raw["user_id"].sample(n=sample_users, random_state=seed)
    df_users_filtered = df_users_raw.loc[df_users_raw["user_id"].isin(user_ids_sample)]
    print("    - Users loaded")

    # Normalize rating

    df_users_filtered.loc[:, "rating"] = df_users_filtered["rating"].astype("float")
    df_users_filtered.loc[:, "rating"] = df_users_filtered["rating"] / 5.0
    
    # Load all items data

    df_items_raw = load_data(f"{file_path}/goodreads_book_genres_initial.json", None)
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
    print(f"    - Model dataframe ready. Interactions: {len(df_combined)}")
    return df_combined.drop("genres", axis=1)

def reformat_dict(d: dict) -> dict:
    """
    Reformats dict from JSON as columns for df

    Args: 
        d: dictionary containing categories and their count
    """
    
    genres = {}
    for genre, value in d.items():
        for g in genre.split(","):
            g = g.strip().replace(" ", "_").replace("-", "_")
            genres[g] = genres.get(g, 0) + value if value > 0 else 0  # Some genres had a -1, so they were removed
    genres.update({k: 0 for k in get_categories() if k not in genres})
    return genres

def get_items_df(df: pd.DataFrame, priority: str | None = None) -> pd.DataFrame:
    """
    Get aggregated items df
    
    Args:
        df: interactions dataframe
        priority: item priority strategy
    """
    
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
    items_df["priority"] = items_df.apply(calculate_priority, args=(priority,), axis=1)
    print(f"    - Items dataframe ready. Items: {len(items_df)}")
    return items_df

def calculate_priority(row: pd.Series, priority: str | float | None = None) -> float:
    """
    Calculate priority column for items
    
    Args:
        row: items dataframe row
        priority: priority strategy
    """

    categories = get_categories()
    if not priority:
        return 0
    elif isinstance(priority, float):
        return float(np.random.random() < priority)
    elif priority in categories:
        cat_index = categories.index(priority)
        max_value = np.max(row["vector"])
        max_indices = np.where(row["vector"] == max_value)[1]
        return float(cat_index in max_indices and not np.all(row["vector"] == 0))

def get_users_df(df: pd.DataFrame, df_items: pd.DataFrame) -> pd.DataFrame:
    """
    Get aggregated users df
    
    Args: 
        df: interactions dataframe
        df_items: items dataframe
    """
    
    print("Getting users dataframe...")
    tmp_df = df.copy()
    cat_cols = get_categories()
    for col in cat_cols:
        tmp_df[col] = tmp_df[col].apply(lambda x: 1 if x > 0 else 0)
    aggregations = {
        "is_reviewed": "sum",
        "is_read": "sum",
        "rating": "mean",
        "book_id": lambda x: list(x)
    }
    aggregations.update({k: "sum" for k in cat_cols})
    users_df = tmp_df.groupby(by=["user_id"]).agg(aggregations)
    users_df["vector"] = users_df.apply(lambda row: np.array(row[cat_cols]).reshape(1, -1), axis=1)
    users_df = users_df.drop(cat_cols, axis=1)
    users_df["book_id"] = users_df.apply(calculate_book_score, args=(df_items,), axis=1)
    users_df["book_id_length"] = users_df["book_id"].apply(len)
    max_book_list = users_df["book_id_length"].max()
    users_df["rec_proba"] = users_df["book_id_length"] / max_book_list
    print(f"    - Users dataframe ready. Users: {len(users_df)}")
    return users_df.drop("book_id_length", axis=1)

def calculate_book_score(row: pd.Series, df_items: pd.DataFrame) -> dict:
    """
    Calculate item scores based on cosine similarity with users
    
    Args:
        row: users dataframe row
        df_items: items dataframe
    """

    books = {}
    user_vector = row["vector"]
    for book_id in row["book_id"]:
        item = df_items[df_items.index == book_id]
        if item["priority"].item() > 0:
            books.update({book_id: item["priority"].item()})
        else:
            item_vector = item["vector"].item()
            similarity = cosine_similarity(user_vector, item_vector)
            books.update({book_id: similarity[0][0]})
    return books
