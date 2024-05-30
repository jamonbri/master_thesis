import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Any
import ast
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

def divide_into_three(n: int) -> tuple[int, int, int]:
    part = n // 3
    remainder = n % 3
    if remainder == 0:
        return (part, part, part)
    elif remainder == 1:
        return (part + 1, part, part)
    else:  
        return (part + 1, part + 1, part)

def string_to_array(s):
    s = s.strip("[]")
    s = s.split()
    return np.array([float(x) for x in s]).reshape(1, -1)

def string_to_dict(s):
    return ast.literal_eval(s)

def plot_agent_vector(df: pd.DataFrame, agent_id: int) -> None:
    filtered_df = df.xs(agent_id, level=1)
    for col in filtered_df.columns:
        plt.figure(figsize=(5, 3))
        plt.plot(filtered_df.index, filtered_df[col], linestyle="-", label=col)
        plt.title(f"Line chart of column {col}")
        plt.xlabel("Time")
        plt.ylabel(col)
        plt.legend()
        plt.grid(False)
        plt.show()

def plot_vector_diffs(df: pd.DataFrame, model: str) -> None:
    ax = df["vector_diff"].hist(
        bins=20, 
        grid=False, 
        edgecolor="black"
    )
    median_value = df["vector_diff"].median()
    plt.axvline(median_value, color="r", linestyle="dotted", linewidth=2, label=f"Median: {median_value:.6f}")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of cosine similarity between first and last vector per user ({model})")
    plt.legend()
    plt.show()

def list_file_paths(directory: str) -> list:
    file_paths = []
    for root, _, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)
    return file_paths

def get_value_from_results(df: pd.DataFrame, id: int, col_name: str, step: int | None = None) -> Any:
    if "AgentID" not in df.columns:
        filtered_df = df[df["unique_id"] == id]
        return filtered_df[col_name].iloc[0]
    filtered_df = df[df["AgentID"] == id]
    if step == 0:
        row = filtered_df[filtered_df["Step"] == filtered_df["Step"].min()]
    elif step == -1:
        row = filtered_df[filtered_df["Step"] == filtered_df["Step"].max()]
    else:
        row = filtered_df[filtered_df["Step"] == step]
    return row[col_name].iloc[0]

def normalize_vector(vector: np.ndarray | str, as_percentage: bool = False) -> np.ndarray:
    if isinstance(vector, str):
        vector = string_to_array(vector)
    total_sum = vector.sum()
    percentage = 100 if as_percentage else 1
    result = vector * percentage / total_sum if total_sum > 0 else 0
    return result

def unit_normalize_vector(v: np.ndarray) -> np.array:
    """
    Unit vector normalization
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def plot_book_distribution_by_genre(df: pd.DataFrame) -> None:
    df["normalized_vector"] = df["vector"].apply(normalize_vector)
    df["max_position"] = df["normalized_vector"].apply(lambda x: np.argmax(x))
    max_values = df["max_position"].value_counts().reindex(range(0, 15), fill_value=0)
    plt.figure(figsize=(10, 6))
    plt.bar(get_categories(), max_values, color="blue")
    plt.xlabel("Genres")
    plt.ylabel("Count of Books")
    plt.xticks(rotation=45)
    plt.show()

def get_vector_diff_df(filename: str) -> pd.DataFrame:
    """
    Get vector differences dataframeas cosine similarity between first 
    and last vector of each agent

    Args:
        filename: path of CSV file
    """
    df = pd.read_csv(filename)
    filtered_df = df[df["agent_type"] == "UserAgent"][["AgentID", "Step", "vector"]]
    result_data = []
    for agent_id, group in filtered_df.groupby("AgentID"):
        sorted_group = group.sort_values("Step")
        if len(sorted_group) > 1:
            first_vector = string_to_array(sorted_group.iloc[0]["vector"])
            last_vector = string_to_array(sorted_group.iloc[-1]["vector"])
            diff = cosine_similarity(first_vector, last_vector)[0][0]
            result_data.append({"AgentID": agent_id, "vector_diff": diff})
    return pd.DataFrame(result_data)