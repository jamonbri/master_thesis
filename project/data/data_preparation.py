import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from utils import divide_into_three, get_categories


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
    load_users: int | None = None, 
    n_users: int = 100, 
    thresholds: tuple[int, int, int] = [5, 20, 50],
    dummy: bool = False, 
    seed: int | None = None,
    ignorant_proportion: float = 1.0
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
    
    # Load all items data

    df_items_raw = load_data(f"{file_path}/goodreads_book_genres_initial.json", None)
    df_items_non_empty = df_items_raw[df_items_raw["genres"].apply(lambda x: bool(x))]
    books = df_items_non_empty["book_id"]
    
    # Load all users data
    
    df_users_raw = load_data(f"{file_path}/goodreads_interactions.csv", load_users)
    df_users_with_books = df_users_raw[df_users_raw["book_id"].isin(books)]
    df_users_filtered = process_df_users_raw(
        df=df_users_with_books, n_users=n_users, seed=seed, thresholds=thresholds, ignorant_proportion=ignorant_proportion
    )
    print("    - Users loaded")

    # Normalize rating

    df_users_filtered.loc[:, "rating"] = df_users_filtered["rating"].astype("float")
    df_users_filtered.loc[:, "rating"] = df_users_filtered["rating"] / 5.0
    
    # Filter items data

    unique_item_ids = df_users_filtered["book_id"].unique().tolist()
    df_items_filtered = df_items_non_empty.loc[df_items_non_empty["book_id"].isin(unique_item_ids)]
    print("    - Items loaded")
    
    # Get categories into columns
    
    df_items_filtered.loc[:, "genres"] = df_items_filtered["genres"].apply(reformat_dict)
    df_items_filtered_normalized = pd.json_normalize(df_items_filtered["genres"])
    df_items_result = pd.concat([df_items_filtered.reset_index(drop=True), df_items_filtered_normalized.reset_index(drop=True)], axis=1)
    
    # Combine dfs and return
    
    df_combined = pd.merge(df_users_filtered, df_items_result, how="inner", on="book_id")
    print(f"    - Model dataframe ready. Interactions: {len(df_combined)}")
    return df_combined.drop("genres", axis=1)

def process_df_users_raw(
    df: pd.DataFrame, 
    n_users: int, 
    seed: int | None, 
    thresholds: tuple[int, int, int],
    ignorant_proportion: float
) -> pd.DataFrame:
    # Filter to read-only entries
    read_only_df = df[df["is_read"] == 1]

    # Count books per user and filter users with up to top threshold books
    tmp_df = read_only_df.groupby("user_id")["book_id"].count().reset_index().rename(columns={"book_id": "book_count"})
    user_ids = tmp_df[tmp_df["book_count"] <= thresholds[2]]["user_id"].tolist()
    filtered_df = df[df["user_id"].isin(user_ids)]
    filtered_df = filtered_df.merge(tmp_df, on="user_id", how="left")

    # Divide users into three groups
    divisions = divide_into_three(n_users)

    # Sample from each user group
    low_df_user_ids = filtered_df[filtered_df["book_count"] <= thresholds[0]]
    low_user_ids = low_df_user_ids["user_id"].drop_duplicates().sample(n=divisions[0], random_state=seed)
    mid_df_user_ids = filtered_df[(filtered_df["book_count"] <= thresholds[1]) & (filtered_df["book_count"] > thresholds[0])]
    mid_user_ids = mid_df_user_ids["user_id"].drop_duplicates().sample(n=divisions[1], random_state=seed)
    high_df_user_ids = filtered_df[filtered_df["book_count"] > thresholds[1]]
    high_user_ids = high_df_user_ids["user_id"].drop_duplicates().sample(n=divisions[2], random_state=seed)

    # Concatenate samples into one DataFrame
    low_df = filtered_df[filtered_df["user_id"].isin(low_user_ids)]
    low_df["persona"] = "low"
    mid_df = filtered_df[filtered_df["user_id"].isin(mid_user_ids)]
    mid_df["persona"] = "mid"
    high_df = filtered_df[filtered_df["user_id"].isin(high_user_ids)]
    high_df["persona"] = "high"

    # Add ignorance
    if ignorant_proportion == 1.0:
        for sub_df in [low_df, mid_df, high_df]:
            sub_df['ignorant'] = True
    elif ignorant_proportion > 0:
        for sub_df in [low_df, mid_df, high_df]:
            sub_df['ignorant'] = get_naiveness(sub_df, ignorant_proportion, seed)
    else:
        for sub_df in [low_df, mid_df, high_df]:
            sub_df["ignorant"] = False

    return pd.concat([low_df, mid_df, high_df])

def get_naiveness(df: pd.DataFrame, ignorant_proportion: float, seed: int | None) -> pd.Series:
    """
    Get users that are naive

    Args:
        df: users dataframe
        ignorant_proportion: proportion of naive users
        seed: random state
    """
    if "user_id" in df.columns:
        unique_users = df['user_id'].drop_duplicates()
        user_id_col = "user_id"
    else:
        unique_users = pd.Series(df.index.drop_duplicates())
        user_id_col = None
    shuffled_users = unique_users.sample(frac=1, random_state=seed)
    half_point = round(len(shuffled_users) * ignorant_proportion)
    naiveness_map = {user_id: True for user_id in shuffled_users[:half_point]}
    naiveness_map.update({user_id: False for user_id in shuffled_users[half_point:]})
    if user_id_col:
        return df[user_id_col].map(naiveness_map)
    else:
        return df.index.map(naiveness_map)

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

def get_items_df(df: pd.DataFrame, priority: str | None = None, verbose: bool = False) -> pd.DataFrame:
    """
    Get aggregated items df
    
    Args:
        df: interactions dataframe
        priority: item priority strategy
    """
    
    if verbose:
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
    
    if verbose:
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

def get_users_df(
    df: pd.DataFrame, 
    df_items: pd.DataFrame, 
    thresholds: tuple[int, int, int],
    n_recs: int,
    social_influence: bool,
    ignorant_proportion: float,
    seed: int | None,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Get aggregated users df
    
    Args: 
        df: interactions dataframe
        df_items: items dataframe
        thresholds: books per year limit for low-mid and mid-high reader personas 
        n_recs: number of recommendations
        social_influence: boolean to toggle on social influence
        ignorant_proportion: optional calculation of naiveness 
        seed: random state
    """
    if verbose:
        print("Getting users dataframe...")
    tmp_df = df.copy()
    cat_cols = get_categories()
    for col in cat_cols:
        tmp_df[col] = tmp_df[col].apply(lambda x: 1 if x > 0 else 0)
    aggregations = {
        "is_reviewed": "sum",
        "is_read": "sum",
        "rating": "mean",
        "book_id": lambda x: list(x),
        "ignorant": "first",
        "persona": "first"
    }
    aggregations.update({k: "sum" for k in cat_cols})
    
    # Calculate probabilities of reading for each user

    low_readers_average = thresholds[0] / 2
    mid_readers_average = (thresholds[1] - thresholds[0]) / 2 + thresholds[0]
    high_readers_average = (thresholds[2] - thresholds[1]) / 2 + thresholds[1]
    low_readers_proba = round(low_readers_average / 360, 4)
    mid_readers_proba = round(mid_readers_average / 360, 4)
    high_readers_proba = round(high_readers_average / 360, 4)

    users_df = tmp_df.groupby(by=["user_id"]).agg(aggregations)
    users_df["vector"] = users_df.apply(lambda row: np.array(row[cat_cols]).reshape(1, -1), axis=1)
    users_df = users_df.drop(cat_cols, axis=1)
    users_df["book_id"] = users_df.apply(calculate_book_score, args=(df_items,), axis=1)
    users_df["read_proba"] = np.where(
        users_df["is_read"] <= thresholds[0], 
        low_readers_proba, 
        np.where(
            users_df["is_read"] <= thresholds[1], 
            mid_readers_proba, 
            high_readers_proba
        )
    )

    if social_influence:
        users_df = get_social_influences(users_df)
    else:
        users_df["following"] = None

    if ignorant_proportion == 1:
        users_df["ignorant"] == True
    elif ignorant_proportion > 0:
        low_df = users_df[users_df["persona"] == "low"]
        low_df["ignorant"] = get_naiveness(low_df, ignorant_proportion, seed)
        mid_df = users_df[users_df["persona"] == "mid"]
        mid_df["ignorant"] = get_naiveness(mid_df, ignorant_proportion, seed)
        high_df = users_df[users_df["persona"] == "high"]
        high_df["ignorant"] = get_naiveness(high_df, ignorant_proportion, seed)
        users_df = pd.concat([low_df, mid_df, high_df])

    if n_recs:
        users_df = matrix_cosine_similarity(users_df, df_items, n_recs)
    else:
        users_df["similarities"] = None
        
    if verbose:
        print(f"    - Users dataframe ready. Users: {len(users_df)}")
    return users_df

def matrix_cosine_similarity(df_reference: pd.DataFrame, df_compare: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    """Calculates cosine similarity between each vector of df_reference and all
    vectors of df_compare via matrix multiplication. Simulates a 'cache' to avoid over-computations
    
    Args:
        df_reference: reference df
        df_compare: reference df
        n: number of books with scores sorted by similarity
    """
    # Calculate matrixes and similarites

    tmp_reference = df_reference["vector"].apply(lambda x: x.reshape(16).astype(float))
    tmp_compare = df_compare["vector"].apply(lambda x: x.reshape(16).astype(float))
    matrix_reference = np.stack(tmp_reference.values)
    matrix_compare = np.stack(tmp_compare.values)
    norms_reference = np.linalg.norm(matrix_reference, axis=1)
    norms_compare = np.linalg.norm(matrix_compare, axis=1)
    matrix_reference_normalized = matrix_reference / norms_reference[:, np.newaxis]
    matrix_compare_normalized = matrix_compare / norms_compare[:, np.newaxis]
    similarities = np.dot(matrix_reference_normalized, matrix_compare_normalized.T)

    # Override with priority

    priority_indices = np.where(df_compare["priority"] == 1)[0]
    for idx in priority_indices:
        ref_indices = np.where(df_reference['ignorant'] == False)[0]
        non_ignorant_indices = set(range(len(df_reference))) - set(ref_indices)
        similarities[list(non_ignorant_indices), idx] = 1.0

    
    # Add similarities as column to reference df
    
    df_reference["similarities"] = [
        dict(sorted({df_compare.index[j]: round(similarities[i, j], 4) for j in range(len(df_compare))}.items(),
            key=lambda item: item[1], reverse=True)[:n])
        for i in range(len(df_reference))
    ]
    return df_reference

def get_social_influences(df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """Get users to follow based on cosine similarity between them
    
    Args:
        df: users df
        top_k: number of users to return as top    
    """
    vectors = np.stack(df["vector"].apply(lambda x: np.reshape(x, (1, -1))).values)
    vectors = vectors.reshape(150, 16)
    similarity_matrix = cosine_similarity(vectors)
    sim_df = pd.DataFrame(similarity_matrix, index=df.index, columns=df.index)

    def find_top_5_similar(user_id: int) -> list[int]:
        return sim_df.loc[user_id].drop(user_id).nlargest(top_k).index.tolist()

    df["following"] = df.index.map(find_top_5_similar)
    return df

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
            books.update({book_id: round(similarity[0][0], 4)})
    return books
