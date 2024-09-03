import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM

# Load the dataset
file_path = '/Users/matiarasetina/devv/znanstveni/angularSpeedX_data.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Perform repeated measures ANOVA for each axis
axes = data['Axis'].unique()
anova_results = {}

for axis in axes:
    # Filter data for the current axis
    axis_data = data[data['Axis'] == axis]

    # Prepare data for ANOVA (ensure 'Day' is categorical)
    axis_data['Day'] = axis_data['Day'].astype('category')

    # Ensure the data is suitable for RM ANOVA
    if axis_data.groupby('Day')['Measurement'].nunique().min() <= 1:
        print(f"No sufficient variability for axis {axis}.")
        continue

    # Perform the ANOVA
    model = AnovaRM(axis_data.dropna(), 'Measurement', 'Day', within=['Axis'], aggregate_func='mean')
    results = model.fit()
    anova_results[axis] = results.summary()

# Output the results
for axis, result in anova_results.items():
    print(f"Results for Axis {axis}:\n{result}\n")
