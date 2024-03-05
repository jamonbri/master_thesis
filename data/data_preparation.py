import pandas as pd
import requests
from bs4 import BeautifulSoup
import gzip
import json
import re

def load_data(
    file_name: str, 
    head: int = 500
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
