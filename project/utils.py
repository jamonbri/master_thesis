import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Any


def string_to_array(s):
    s = s.strip("[]")
    s = s.split()
    return np.array([float(x) for x in s]).reshape(1, -1)

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