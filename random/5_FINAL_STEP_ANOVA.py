import pandas as pd
import os
from statsmodels.stats.anova import AnovaRM
import numpy as np

def perform_anova_for_axis_with_sampling(df, axis, output_file_path, num_samples=100, num_iterations=10):
    try:
        # Columns for the specific axis
        axis_columns = [col for col in df.columns if axis in col]

        # Check if at least two days are present for this axis
        if len(axis_columns) < 2:  # Need at least two days
            return False

        # Writing results to a text file
        with open(output_file_path, 'w') as file:
            file.write(f"ANOVA RM results for axis {axis}:\n")
            
            for i in range(num_iterations):
                # Set a new random state for each iteration
                np.random.seed(None)  # Reseeding
                random_state = np.random.randint(0, 10000)

                # Sample data
                df_sampled = df[axis_columns].sample(n=num_samples, random_state=random_state)

                # Extract and rename columns for ANOVA
                df_sampled.columns = [f'Day{i}' for i in range(1, len(axis_columns) + 1)]

                # Transforming data to long format for ANOVA RM
                long_format = pd.melt(df_sampled.reset_index(), id_vars=['index'], value_vars=df_sampled.columns, 
                                      var_name='Day', value_name='Measurement')

                # Conducting repeated measures ANOVA
                anova_results = AnovaRM(data=long_format, depvar='Measurement', subject='index', within=['Day']).fit()

                # Append iteration results to the file
                file.write(f"Iteration {i+1}:\n")
                file.write(anova_results.summary().as_text())
                file.write("\n\n")

        print(f"ANOVA RM results saved for axis {axis}: {output_file_path}")
        return True

    except Exception as e:
        print(f"Error processing ANOVA for axis {axis} in file {output_file_path}: {e}")
        return False

def process_directory_for_anova_results(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if "anova_ready" in file:
                input_file_path = os.path.join(root, file)
                df = pd.read_csv(input_file_path)

                # Perform ANOVA for each axis with sampling and save results
                for axis in ['angularSpeedX', 'angularSpeedY', 'angularSpeedZ']:
                    output_file_path = os.path.join(root, f'10x_ANOVA_RM_RESULTS_{axis}_{file}.txt')
                    perform_anova_for_axis_with_sampling(df, axis, output_file_path)

# Specify your directory
directory = 'bankAppData/all_data/entryActivity'  # Replace with your directory path
process_directory_for_anova_results(directory)
