import pandas as pd
import requests
import numpy as np
from bs4 import BeautifulSoup
from collections import defaultdict
import gzip
import json
import re

def load_data(
    file_name: str, 
    head: int | None = 500
) -> pd.DataFrame:
    file_path = f"datasets/{file_name}"
    file_extension = file_name.split(".")[-1]
    if file_name.split(".")[-1] == "csv":
        df = pd.read_csv(file_path, nrows=head)
    elif file_extension == "json":
        try:
            df = pd.read_json(file_path, lines=True, orient='records', nrows=head)
        except:
            df = pd.DataFrame(fix_json(file_path, head))
    else:
        raise Exception(f"Format of dataset {file_name} not loadable.")
    return df

def fix_json(file_path: str, head: int | None) -> list:
    count = 0
    results = []
    with open(file_path, "r") as file:
        for line in file:
            modified_line = extract_specific_values(line)
            results.append(modified_line)
            count += 1
            if head is not None and count > head:
                break
    return results 

def extract_specific_values(s: str) -> dict:
    extracted_values = {}
    patterns = {
        'userId': r"'userId':\s*'([^']*)'",
        'stars': r"'stars':\s*([\d.]+)",
        'itemId': r"'itemId':\s*'([^']*)'",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, s)
        if match:
            extracted_values[key] = match.group(1)  
    return extracted_values

def get_genreline_divs(itemId: str) -> list:
    url = f"https://www.librarything.com/work/{itemId}"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        genreline_divs = soup.find_all("div", class_="genreline")       
        return [div.text for div in genreline_divs]
    else:
        raise ConnectionError(f"Satus code: {response.status_code}, message: {response.text}")

def get_model_dataframe(n_users: int | None, sample_users: int) -> pd.DataFrame:
    df_users_raw = load_data("goodreads/goodreads_interactions.csv", n_users)
    user_ids_sample = df_users_raw["user_id"].sample(sample_users)
    df_users_filtered = df_users_raw.loc[df_users_raw["user_id"].isin(user_ids_sample)]
    df_items_raw = load_data("goodreads/goodreads_book_genres_initial.json", None)
    unique_item_ids = df_users_filtered["book_id"].unique().tolist()
    df_items_filtered = df_items_raw.loc[df_items_raw["book_id"].isin(unique_item_ids)]
    df_items_filtered["new_genres"] = df_items_filtered.loc[:, "genres"].apply(get_genres_from_dict)
    df_combined = pd.merge(df_users_filtered, df_items_filtered, how="left", on="book_id")
    return df_combined.dropna(subset=["new_genres"]).loc[df_combined["is_read"] == 1]

def get_genres_from_dict(d: dict) -> dict:
    genres = {}
    all_votes = 0
    for genre, value in d.items():
        for g in genre.split(","):
            genres.update({g.strip(): value})
            all_votes += value
    if genres and all_votes:
        return {k: v / all_votes for k, v in genres.items()}
    return {}

def get_category_vector(df: pd.DataFrame) -> list:
    unique_genres = set()
    for _, row in df.iterrows():
        if isinstance(row["new_genres"], dict):
            unique_genres.update(row["new_genres"].keys())
    return list(unique_genres)

def get_users_vectors(df: pd.DataFrame, vector: list) -> pd.DataFrame:
    users = df["user_id"].unique().tolist()
    results = pd.DataFrame(columns=vector)
    results.index.name = "user_id"
    for user in users:
        weighted_sums = defaultdict(float)
        counts = defaultdict(int)
        tmp_df = df.loc[df["user_id"] == user]
        for _, row in tmp_df.iterrows():
            for k, v in row["new_genres"].items():
                weighted_sums[k] += v * row["rating"] / 5
                counts[k] += 1
        user_vector = {key: weighted_sums[key] / counts[key] for key in weighted_sums}
        user_vector_total = sum(user_vector.values())
        if user_vector_total:
            user_vector = {key: user_vector[key] / user_vector_total for key in user_vector}
        results.loc[user] = user_vector
    return results.fillna(0)

