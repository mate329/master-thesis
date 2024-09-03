import os
import numpy as np
import pandas as pd
from collections import defaultdict

def process_data_for_user(user_dirs):
    results = {
        'distance': [], 
        'time_diff': [], 
        'speed_pixels_per_ms': [],
        'swipe_angle': []
    }

    for directory in user_dirs:
        print(f"Processing directory: {directory}")
        for root, dirs, files in os.walk(directory):
            for file in files:
                if "speed_angle_orientation" in file and file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    df = pd.read_csv(file_path, index_col=False)

                    # Calculate averages and standard deviations for the required fields
                    for field in results.keys():
                        results[field].append((df[field].mean(), df[field].std()))

    # Calculate overall averages and standard deviations
    final_results = []
    for field in results.keys():
        all_avgs = [result[0] for result in results[field] if not np.isnan(result[0])]
        all_stds = [result[1] for result in results[field] if not np.isnan(result[1])]
        overall_avg = np.mean(all_avgs)
        overall_std = np.mean(all_stds)
        final_results.extend([overall_avg, overall_std])

    return final_results

# The parent directory containing all recordings
parent_directory = "../../bankAppData"

# Identify unique users and their directories
user_dirs = defaultdict(list)
for root, dirs, files in os.walk(parent_directory):
    for dir in dirs:
        username = '_'.join(dir.split('_')[:2])
        user_dirs[username].append(os.path.join(root, dir))

# Process data for each user and compile into a single DataFrame
all_user_results = []
for user, dirs in user_dirs.items():
    print(f"Processing data for user: {user}")
    user_results = process_data_for_user(dirs)
    all_user_results.append([user] + user_results)

# Create a DataFrame from the results
final_df = pd.DataFrame(all_user_results, columns=['username', 
                                                   'onScroll_distance_avg', 'onScroll_distance_std',
                                                   'onScroll_time_diff_avg', 'onScroll_time_diff_std',
                                                   'onScroll_speed_pixels_per_ms_avg', 'onScroll_speed_pixels_per_ms_std',
                                                   'onScroll_swipe_angle_avg', 'onScroll_swipe_angle_std'])

# Saving the final summary DataFrame to a CSV file
output_file = os.path.join(parent_directory, 'onScroll_results.csv')
final_df.to_csv(output_file, index=False)

print(f"Overall summary results saved to {output_file}")
