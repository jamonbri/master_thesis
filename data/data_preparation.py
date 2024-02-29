import pandas as pd
import requests
import os


def download_by_name(
    project: str, 
    fname: str, 
    base_url: str = "https://datarepo.eng.ucsd.edu/mcauley_group/gdrive"
) -> None:
    """Downloads a specific dataset from a project by filename"""
    file_path = f"datasets/{project}/{fname}"
    if os.path.exists(file_path):
        print(f"Dataset {fname} already exists")
        return
    url = f"{base_url}/{project}/{fname}"
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception as e:
        print(f"There was a problem downloading the dataset {fname}: {e}")
        return
    print(f"Dataset {fname} has been downloaded.")
