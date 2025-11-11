import pandas as pd
import numpy as np


def iml_to_df(path):
    """
    Read a .iml file and return a pandas DataFrame.
    """
    # Read the file
    with open(path, "rb") as f:
        datas = (
            f.read()
            .decode("utf-8", errors="ignore")
            .replace("Temp", "/nTemp")
            .split("/n")[1:]
        )

    # columns
    columns = ["Temp", "Dist", "Strain", "Stress", "Force"]

    # Remove strings
    datas = [
        data.replace(" ", "")
        .replace("Temp", "")
        .replace("Dist", "")
        .replace("Strain", "")
        .replace("Stress", "")
        .replace("Force", "")
        .replace("Rate", "")
        .replace("Limit", "")
        for data in datas
    ]
    datas = [
        data.replace("`C", "")
        .replace("mm", "")
        .replace("%", "")
        .replace("N", "")
        .replace("/", "")
        .replace("min", "")
        .replace(";", ",")
        for data in datas
    ]
    datas = [data.split(",")[0:5] for data in datas]

    # convert to float
    for i in range(len(datas)):
        for j in range(len(datas[i])):
            datas[i][j] = float(datas[i][j])

    # Create DataFrame
    df = pd.DataFrame(datas, columns=columns)
    df["Time"] = np.arange(0, len(datas)) * 0.29772

    return df


if __name__ == "__main__":
    # iml to dataframe to csv
    from tkinter import filedialog
    import os
    import shutil

    paths = filedialog.askopenfilenames(
        title="Select .iml files", filetypes=[("IML files", "*.iml")]
    )
    date = input("Enter date (YYYYMMDD): ")
    if not date:
        raise ValueError("Date cannot be empty")
    
    for path in paths:
        df = iml_to_df(path)
        # Set the name of the file
        name_iml = os.path.basename(path)
        name_csv = name_iml.replace(".iml", ".csv")
        # Save to csv
        save_path_csv = os.path.join("app/datas/Et", date, "linkam/csv", name_csv)
        os.makedirs(os.path.dirname(save_path_csv), exist_ok=True)
        df.to_csv(save_path_csv, index=False)
        # Move the original file to the same folder
        save_path_iml = os.path.join("app/datas/Et", date, "linkam/iml", name_iml)
        os.makedirs(os.path.dirname(save_path_iml), exist_ok=True)
        shutil.copy(path, save_path_iml)
        # Print the path of the saved file
        print(f"Saved {save_path_csv}")
