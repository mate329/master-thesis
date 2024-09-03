import pandas as pd

# Paths to your CSV files
csv_file_paths = [
    '../bankAppData/enterPINactivity_results.csv',
    '../bankAppData/entryActivity_results.csv',
    '../bankAppData/entryPINxyClick_results.csv',
    '../bankAppData/onScroll_results.csv'
]

# Read the first CSV file
merged_df = pd.read_csv(csv_file_paths[0])

# Merge the remaining CSV files
for file_path in csv_file_paths[1:]:
    df = pd.read_csv(file_path)
    merged_df = pd.merge(merged_df, df, on='username', how='outer')

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('GOLD_CSV.csv', index=False)

print("CSV files have been merged.")
