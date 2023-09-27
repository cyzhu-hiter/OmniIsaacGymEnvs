import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

log_dir = 'runs/set2'  # The path to your TensorBoard log directory

def extract_scalars_from_event_file(event_file_path):
    rows = []
    try:
        ea = event_accumulator.EventAccumulator(event_file_path)
        ea.Reload()  # loads events from the event file
        
        # Extract and accumulate scalar data from the event file
        for tag in ea.Tags()['scalars']:
            for event in ea.Scalars(tag):
                rows.append({"wall_time": event.wall_time, "step": event.step, "tag": tag, "value": event.value})
    except Exception as e:
        print(f"Error reading {event_file_path}: {str(e)}")
    return rows

all_rows = []

# Traverse through the log directory and read scalar data from all event files
for foldername, subfolders, filenames in os.walk(log_dir):
    for filename in filenames:
        if filename.startswith('events.out.tfevents'):
            event_file_path = os.path.join(foldername, filename)
            rows = extract_scalars_from_event_file(event_file_path)
            all_rows.extend(rows)

# Convert the list of rows to a Pandas DataFrame
df = pd.DataFrame(all_rows)

# Now, df is a DataFrame containing the scalar data from the TensorBoard logs.
# You can analyze or post-process this DataFrame as needed.

print(df)