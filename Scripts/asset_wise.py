#!/usr/bin/env python3
import os
import re
import argparse
import glob
import pandas as pd

def main(event_info_path: str, datasets_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    assets_out_dir = os.path.join(out_dir, "assets")
    os.makedirs(assets_out_dir, exist_ok=True)

    # Load global event metadata
    event_info = pd.read_csv(event_info_path, sep=";")
    if event_info["event_id"].dtype != int:
        event_info["event_id"] = event_info["event_id"].astype(int)

    asset_to_event_ids = {}

    # Process all event dataset files
    csv_paths = sorted(glob.glob(os.path.join(datasets_dir, "*.csv")))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {datasets_dir}")

    for path in csv_paths:
        filename = os.path.basename(path)
        match = re.match(r"(\d+)\.csv$", filename)
        event_id = int(match.group(1)) if match else None

        try:
            df = pd.read_csv(path, sep=";")
        except Exception as e:
            print(f"Skipping {filename}: read error ({e})")
            continue

        if event_id is not None:
            df["event_id"] = event_id

        # Merge metadata if we have event_id
        if event_id is not None and "event_id" in df.columns:
            df = df.merge(event_info, on="event_id", how="left")
        else:
            for col in event_info.columns:
                if col not in df.columns:
                    df[col] = pd.NA

        if "asset_id" not in df.columns:
            print(f"Skipping {filename}: no asset_id column")
            continue

        # Append per asset
        for asset_id, group in df.groupby("asset_id"):
            if event_id is not None:
                asset_to_event_ids.setdefault(int(asset_id), set()).add(event_id)

            asset_file = os.path.join(assets_out_dir, f"{int(asset_id)}.csv")
            write_header = not os.path.exists(asset_file)
            group.to_csv(asset_file, sep=";", index=False, mode="a", header=write_header)

    # Write per-asset event metadata
    for asset_id, event_ids in asset_to_event_ids.items():
        subset = event_info[event_info["event_id"].isin(sorted(event_ids))].copy()
        out_path = os.path.join(assets_out_dir, f"{asset_id}_events.csv")
        subset.to_csv(out_path, sep=";", index=False)

    print(f"Completed. Output written to: {assets_out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Group event datasets by asset, merging metadata."
    )
    parser.add_argument("--event-info", default="event_info.csv")
    parser.add_argument("--datasets-dir", default="datasets")
    parser.add_argument("--out-dir", default="out")
    args = parser.parse_args()

    main(args.event_info, args.datasets_dir, args.out_dir)
