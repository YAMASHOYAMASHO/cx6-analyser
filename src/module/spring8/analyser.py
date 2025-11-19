from matplotlib import pyplot as plt
import numpy as np
from tkinter import filedialog
from tqdm import tqdm
from traitlets import default
import os
import json
import os
import pandas as pd
from pathlib import Path
from util import select_multiple_subdirectories  # 同じパッケージ内の util.py
from sp8_class import SPring8_single

plt.rcParams["figure.figsize"] = [5, 4]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

# 測定条件について
TEMP_INITIAL = 25  # 初期温度
TEMP_TOP = 250  # 最高温度
TEMP_FINAL = 100  # 最終温度
TEMP_SPEED = 5  # 昇温速度 (℃/分)
PICTURE_SPEED = 0.5  # 何分毎に写真を撮るか

class Config:
    temp_initial = TEMP_INITIAL
    temp_top = TEMP_TOP
    temp_final = TEMP_FINAL
    temp_speed = TEMP_SPEED
    picture_speed = PICTURE_SPEED

PARENT_PATH = str((Path(__file__).resolve().parents[3] / "datas" / "SPring8").resolve())

class Path:
    parent = PARENT_PATH
    data = select_multiple_subdirectories(parent_dir=parent)

for sample in tqdm(Path.data, desc="Processing samples"):
    config_path = os.path.join(sample, "config.json")
    # load existing config.json if present, otherwise start with empty dict
    try:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        else:
            cfg = {}
    except json.JSONDecodeError:
        cfg = {}

    # update config with current settings
    cfg["temp_initial"] = Config.temp_initial
    cfg["temp_top"] = Config.temp_top
    cfg["temp_final"] = Config.temp_final
    cfg["temp_speed"] = Config.temp_speed
    cfg["picture_speed"] = Config.picture_speed

    # write back config.json (pretty-printed, preserving unicode)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    paths = os.listdir(sample)
    for path in tqdm(paths, desc=f"Processing files in {sample}", leave=False):
        # skip config.json
        if path == "config.json":
            continue

        # calculate temperature
        num = int(path)
        time = num * Config.picture_speed
        temp = Config.temp_initial + Config.temp_speed * time
        if temp > Config.temp_top:
            temp = Config.temp_top - (temp - Config.temp_top)

        # ==== temporary code ====
        if os.path.exists(os.path.join(sample, path, ".json")):
            os.remove(os.path.join(sample, path, ".json"))
        # ==== end of temporary code ====
        data = SPring8_single(os.path.join(sample, path), time=time, temp=temp, alpha=2)
        with open("datas/status/PEEK 110.json", 'r') as f:
            status = json.load(f)
        data.set_status(status, logging=lambda msg: None)
        data.fitting(logging=lambda msg: None)

        name = sample.split("/")[-1] + "_" + path
        data.save_datas(name=name, logging=lambda msg: None)            
print("All processing complete.")