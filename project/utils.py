import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Any
import ast
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

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

def get_filtered_categories() -> list[str]:
    """
    Return list of categories without general categories (fiction, non_fiction and historical_fiction)
    """
    categories = get_categories()
    categories.remove("non_fiction")
    categories.remove("fiction")
    return categories

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
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(data=df, x="vector_diff", bins=20, alpha=0.5)
    median_value = df["vector_diff"].median()
    plt.axvline(median_value, color="r", linestyle="dotted", linewidth=2, label=f"Median: {median_value:.6f}")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Count")
    plt.title(f"Histogram of cosine similarity between first and last vector per user ({model})")
    plt.legend()
    plt.xlim(0.5, 1)
    plt.show()

def plot_vector_diffs_by_persona(df: pd.DataFrame, model: str) -> None:
    plt.figure(figsize=(10, 6))
    def update_df(row):
        if row["persona"] == "low":
            return "Casual"
        elif row["persona"] == "mid":
            return "Selective"
        else:
            return "Avid"
    df["persona"] = df.apply(update_df, axis=1)
    ax = sns.histplot(data=df, x="vector_diff", hue="persona", bins=20, alpha=0.5, element="poly", hue_order=["Avid", "Selective", "Casual"])
    sns.move_legend(ax, "upper left")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Count")
    plt.title(f"Histogram of cosine similarity between first and last vector per user ({model})")
    plt.xlim(0.5, 1)
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

def plot_book_distribution_by_genre(df: pd.DataFrame, stats: str = "max", filtered: bool = False) -> None:
    """
    Distribution of books by genre with stacked bars for the top three genre positions per book.
    """
    tmp_df = df.copy()
    # Apply normalization
    tmp_df["vector"] = tmp_df["vector"].apply(string_to_array)
    tmp_df["normalized_vector"] = tmp_df["vector"].apply(lambda x: normalize_vector(x.flatten()))
    tmp_df = tmp_df[tmp_df["normalized_vector"].apply(lambda x: isinstance(x, np.ndarray))]

    if filtered:
        categories = get_filtered_categories()
        tmp_df["normalized_vector"] = tmp_df["normalized_vector"].apply(lambda x: np.delete(x, [-1, 1]))
        title = "(without fiction and non_fiction)"
    else:
        categories = get_categories()
        title = ""

    # Extract the top 3 indices for each vector
    def top_indices(x):
        unique_values = np.unique(x)[::-1]
        top_indices = []
        for value in unique_values[:3]: 
            indices = np.where(x == value)
            top_indices.append(list(indices[0]))

        while len(top_indices) < 3:
            top_indices.append(None)
        
        return top_indices

    tmp_df["top_indices"] = tmp_df["normalized_vector"].apply(top_indices)
    tmp_df["max_position"] = tmp_df["top_indices"].apply(lambda x: x[0])
    tmp_df["second_max_position"] = tmp_df["top_indices"].apply(lambda x: x[1])
    tmp_df["third_max_position"] = tmp_df["top_indices"].apply(lambda x: x[2])

    # Calculate counts and reindex to include all categories
    max_values = tmp_df.explode("max_position")["max_position"].value_counts().reindex(range(len(categories)), fill_value=0)
    second_max_values = tmp_df.explode("second_max_position")["second_max_position"].value_counts().reindex(range(len(categories)), fill_value=0)
    third_max_values = tmp_df.explode("third_max_position")["third_max_position"].value_counts().reindex(range(len(categories)), fill_value=0)

    # Print stats
    total_counts = max_values + second_max_values + third_max_values

    if stats == "total":
        uniformity_test = get_stats(total_counts)
        sorted_indices = total_counts.sort_values(ascending=False).index
    elif stats == "max":
        uniformity_test = get_stats(max_values)
        sorted_indices = max_values.sort_values(ascending=False).index
    elif stats == "second_max":
        uniformity_test = get_stats(second_max_values)
        sorted_indices = second_max_values.sort_values(ascending=False).index
    elif stats == "third_max":
        uniformity_test = get_stats(third_max_values)
        sorted_indices = third_max_values.sort_values(ascending=False).index
    print(f"Standard deviation for {stats}: {uniformity_test[1]}")
    print(f"Coefficient of variation for {stats}: {uniformity_test[0]}")
    
    # Plot
    max_values = max_values.iloc[sorted_indices]
    second_max_values = second_max_values.iloc[sorted_indices]
    third_max_values = third_max_values.iloc[sorted_indices]
    sorted_categories = [categories[i] for i in sorted_indices]
    plt.figure(figsize=(12, 8))
    plt.bar(sorted_categories, max_values, color="blue", label="Max")
    plt.bar(sorted_categories, second_max_values, bottom=max_values, color="green", label="Second Max")
    plt.bar(sorted_categories, third_max_values, bottom=max_values + second_max_values, color="red", label="Third Max")

    plt.xlabel("Genres")
    plt.ylabel("Count of Books")
    plt.title(f"Distribution of book genres by most common categories in vectors {title}")
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()

def get_stats(series: pd.Series) -> tuple[float, float]:
    avg = series.mean()
    std_dev = series.std()
    return std_dev / avg, std_dev

def get_vector_diff_df(filename: str | pd.DataFrame) -> pd.DataFrame:
    """
    Get vector differences dataframeas cosine similarity between first 
    and last vector of each agent

    Args:
        filename: path of CSV file or dataframe
    """
    if isinstance(filename, str):
        df = pd.read_csv(filename)
    else:
        df = filename
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

def load_results_dfs(path: str, i: int, model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load dataframes from results

    Args:
        path: path to simulation folder
        i: number of simulation run
        model: simulation model run
    """
    results_path = os.path.join(path, "run_raw_1.csv")
    users_path = os.path.join(path, "initial_users_1.csv")
    df_results = get_vector_diff_df(results_path)
    df_users = pd.read_csv(users_path)
    df_results["run"] = i
    df_users["run"] = i
    return df_results, df_users

def get_books_read(df: pd.DataFrame) -> pd.DataFrame:
    df_filtered = df[df["agent_type"] == "UserAgent"]
    df_filtered["books_read"] = df_filtered["user_books_consumed"].apply(ast.literal_eval).apply(len)
    df_filtered = df_filtered.sort_values(by=["AgentID", "Step"])
    df_grouped = df_filtered.groupby("AgentID").last()["books_read"].reset_index()
    return df_grouped

def plot_books_consumed(model: str) -> None:
    results = []
    base_path = "data/results"
    runs = sorted(os.listdir(base_path))
    runs.pop(0)
    if model == "benchmark":
        start = 0
        stop = 20
    elif model == "covert":
        start = 20
        stop = 40
    elif model == "overt":
        start = 40
        stop = 60
    elif model == "overt_w_si":
        start = 60
        stop = 80
    for i in range(start, stop):
        full_path = os.path.join(base_path, runs[i], "run_raw_1.csv")
        df = pd.read_csv(full_path)
        results.append(get_books_read(df))
        print(f"{i + 1}/{stop}", end="\r")
    result = pd.concat(results)
    result = result.groupby("AgentID")["books_read"].mean().reset_index()
    ax = sns.histplot(data=result, x="books_read", alpha=0.5, color="purple")
    plt.xlabel("Books consumed")
    plt.ylabel("Count")
    plt.title(f"Histogram of average books consumed per user ({model})")
    plt.show()