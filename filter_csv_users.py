import pandas as pd

# Define the list of top 15 user_ids
# top_15_users = [22, 18, 35, 36, 5, 39, 34, 28, 8, 14, 13, 30, 23, 21, 29]
custom = [22, 18, 35, 36, 5, 39, 34, 28, 8, 14]

# Load the CSV file
file_path = './ai/entry_activity_image_csv_80_10_10_confirmed.csv'  # Update with your CSV file path
df = pd.read_csv(file_path)

# Filter the DataFrame for the top 15 users
filtered_df = df[df['user_id'].isin(custom)]

# Save the filtered DataFrame to a new CSV file
output_file_path = './ai/filtered.csv'  # Update with the desired output file path
filtered_df.to_csv(output_file_path, index=False)

print(f"Filtered data for the top 15 users has been saved to {output_file_path}")
