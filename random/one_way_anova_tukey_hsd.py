import os
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from statsmodels.formula.api import ols

def perform_anova_and_tukey_hsd_on_csv_files(parent_directory):
    user_relevance = {}
    
    for root, _, files in os.walk(parent_directory):
        for file in files:
            if file.endswith('.csv') and ('gyroscope' in file or 'accelerometer' in file):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)

                # Reshape to long format
                df_long = pd.melt(df, value_vars=['angularSpeedX', 'angularSpeedY', 'angularSpeedZ'],
                                  var_name='Treatment', value_name='Value')

                # Perform the one-way ANOVA
                model = ols('Value ~ C(Treatment)', data=df_long).fit()
                anova_results = sm.stats.anova_lm(model, typ=2)
                
                # Determine relevance based on p-value
                p_value = anova_results['PR(>F)'][0]
                relevance = "relevant" if p_value < 0.05 else "not relevant"
                user_relevance[file] = relevance

                # Prepare the results text
                anova_text = f"{anova_results}\nStatistical Relevance: {relevance}\n"

                # If the ANOVA is relevant, perform Tukey's HSD test
                if relevance == "relevant":
                    tukey = pairwise_tukeyhsd(endog=df_long['Value'], groups=df_long['Treatment'], alpha=0.05)
                    tukey_text = tukey.summary().as_text()
                else:
                    tukey_text = "Tukey's HSD test was not performed as ANOVA is not significant.\n"

                # Define the output file path for ANOVA results
                output_file_path_anova = os.path.join(root, "anova_results.txt")
                # Write the ANOVA results to a text file
                with open(output_file_path_anova, 'a') as f:
                    f.write(f"ANOVA results for {file}:\n")
                    f.write(anova_text)
                    f.write("\n")

                # Define the output file path for Tukey's HSD results
                output_file_path_tukey = os.path.join(root, "tukey_hsd_results.txt")
                # Write the Tukey's HSD results to a text file
                with open(output_file_path_tukey, 'a') as f:
                    f.write(f"Tukey's HSD results for {file}:\n")
                    f.write(tukey_text)
                    f.write("\n\n")

                print(f"ANOVA results for {file} saved to {output_file_path_anova}")
                print(f"Tukey's HSD results for {file} saved to {output_file_path_tukey}")

    # Print out relevance for each user at the end of the script's run
    for user, relevance in user_relevance.items():
        print(f"Data for {user} is {relevance}")

# The parent directory containing the CSV files
parent_directory = "bankAppData/all_data/entryActivity"  # Replace with your actual directory path

# Run the function
perform_anova_and_tukey_hsd_on_csv_files(parent_directory)
