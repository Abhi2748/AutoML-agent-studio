import pandas as pd
from fastapi import UploadFile
from typing import Tuple

def save_and_load_csv(file: UploadFile) -> Tuple[pd.DataFrame, str]:
    try:
        file_location = f"app/data/{file.filename}"
        with open(file_location, "wb+") as f:
            f.write(file.file.read())
        df = pd.read_csv(file_location)
        return df, ""
    except Exception as e:
        return None, str(e)
