import pandas as pd

# Input file names
path = "CARE_To_Compare"
files = ["%s/Wind Farm A/event_info.csv" % path,"%s/Wind Farm B/event_info.csv" % path,"%s/Wind Farm C/event_info.csv" % path]

# Read and concatenate
dfs = []
for file in files:
    df = pd.read_csv(file, sep=";")
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)

# Remove duplicate header rows if they existed inside the files
combined = combined[combined["event_id"] != "event_id"]

# Convert event_id to int (optional cleanup)
combined["event_id"] = combined["event_id"].astype(int)

# Save final CSV
combined.to_csv("events_merged.csv", sep=";", index=False)

print("Merged into events_merged.csv")
