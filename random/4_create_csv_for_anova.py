import pandas as pd
import os

def prepare_data_for_anova(file_path, output_file_path, seed=2743, samples=1000):
    try:
        # Load the dataset
        df = pd.read_csv(file_path)

        # Check if there are at least two days of data
        days_present = df['Day'].unique()
        axes_present = df['Axis'].unique()
        if len(days_present) < 2:
            print(f"Skipping {file_path} as it has less than two days of data.")
            return

        # Initialize a DataFrame for parsed data
        parsed_df = pd.DataFrame()

        # Sample and add data for each day and axis
        for day in days_present:
            for axis in axes_present:
                # Create a unique column name for each day-axis combination
                column_name = f"{day.capitalize()}_{axis}"
                # Query the DataFrame for the specific day and axis
                query = f'Day == "{day}" and Axis == "{axis}"'
                # Check if there are enough samples for this day-axis combination
                if len(df.query(query)) >= samples:
                    # Sample the data and add to the parsed DataFrame
                    parsed_df[column_name] = df.query(query)['Measurement'].sample(n=samples, random_state=seed).to_list()
                else:
                    print(f"Skipping {file_path} for {day}, {axis} as it has less than {samples} samples.")

        # Save the parsed data to a new CSV file, only if there is data to save
        if not parsed_df.empty:
            parsed_df.to_csv(output_file_path, index=False)
            print(f"Created ANOVA-ready file: {output_file_path}")
        else:
            print(f"Skipping {file_path} as no suitable data was found for ANOVA.")

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

def process_directory_for_anova(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if "sorted_processed" in file:
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(root, 'anova_ready_' + file)
                prepare_data_for_anova(input_file_path, output_file_path)

# Specify your directory
directory = 'bankAppData/all_data/entryActivity'  # Replace with your directory path
process_directory_for_anova(directory)
