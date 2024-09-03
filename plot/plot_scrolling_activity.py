import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

# Function to find and read the screen resolution from the CSV with "deviceModel" in its filename
def find_screen_resolution(directory):
    for file in os.listdir(directory):
        if "deviceModel" in file and file.endswith('.csv'):
            df = pd.read_csv(os.path.join(directory, file))
            if 'screenResolutionX' in df.columns and 'screenResolutionY' in df.columns:
                # Assuming there's only one set of screen resolutions in this file
                return df['screenResolutionX'].iloc[0], df['screenResolutionY'].iloc[0]
    # Default resolution in case no file is found
    return 600, 1200

def simplify_swipes(df):
    # Convert the coordinate columns to numeric, to handle '-' subtraction operation
    df['eventX'] = pd.to_numeric(df['eventX'], errors='coerce')
    df['eventY'] = pd.to_numeric(df['eventY'], errors='coerce')
    df['eventRawX'] = pd.to_numeric(df['eventRawX'], errors='coerce')
    df['eventRawY'] = pd.to_numeric(df['eventRawY'], errors='coerce')

    # Define a tolerance for grouping swipes
    tolerance = 200

    # Initialize an empty list to store the representative swipes
    simplified_swipes = []

    # Group by swipes that start within ±tolerance units of each other
    while not df.empty:
        # Take the first swipe as the reference
        ref_swipe = df.iloc[0]
        within_tolerance = df.apply(lambda swipe: 
                                    abs(swipe.eventX - ref_swipe.eventX) <= tolerance and
                                    abs(swipe.eventY - ref_swipe.eventY) <= tolerance, axis=1)
        # Group swipes within the tolerance
        group = df[within_tolerance]

        # Choose a representative swipe for the group (e.g., the first swipe)
        representative_swipe = group.iloc[0]

        # Append the representative swipe to the list
        simplified_swipes.append(representative_swipe)

        # Remove the grouped swipes from the original DataFrame
        df = df[~within_tolerance]

    # Convert the list of swipes to a DataFrame
    simplified_df = pd.DataFrame(simplified_swipes)
    simplified_df.reset_index(drop=True, inplace=True)

    return simplified_df

# Function to plot swipes (updated to use screen resolution)
def plot_swipes(file_path, screen_resolution):
    try:
        df = pd.read_csv(file_path)
        df_simplified = simplify_swipes(df)

        if all(col in df_simplified.columns for col in ['eventX', 'eventY', 'eventRawX', 'eventRawY']):
            fig = go.Figure()

            # Add traces for the swipes
            for i, row in df_simplified.iterrows():
                color = px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                fig.add_trace(go.Scatter(
                    x=[row['eventX'], row['eventRawX']], # x represents horizontal axis
                    y=[row['eventY'], row['eventRawY']], # y represents vertical axis
                    mode='markers+lines+text',
                    marker=dict(size=10, color=color),
                    line=dict(color=color),
                    text=['Start', 'End'],
                    textposition='top center',
                    name=f'Swipe {i+1}'
                ))

            # Set the axes range based on the data points with a margin
            margin = 50  # Adjust if necessary
            x_range = [min(df_simplified['eventX'].min(), df_simplified['eventRawX'].min()) - margin, 
                    max(df_simplified['eventX'].max(), df_simplified['eventRawX'].max()) + margin]
            y_range = [min(df_simplified['eventY'].min(), df_simplified['eventRawY'].min()) - margin, 
                    max(df_simplified['eventY'].max(), df_simplified['eventRawY'].max()) + margin]

            fig.update_layout(
                title=f"Swiping Activity in {os.path.basename(file_path)}",
                xaxis_title="X Coordinate",
                yaxis_title="Y Coordinate",
                xaxis=dict(range=x_range),
                yaxis=dict(range=y_range),
                showlegend=True,
                width=screen_resolution[0], # Width of the plot
                height=screen_resolution[1] # Height of the plot, should be greater than width for vertical look
            )

            # Save plot as PNG
            output_file = os.path.splitext(file_path)[0] + '_scroll_plot.png'
            pio.write_image(fig, output_file)
            print(f"Plot saved as {output_file}")
    except Exception as e:
        print(f"Error reading file {file_path} error: {e}")

def find_and_plot_files(start_path):
    screen_resolution = find_screen_resolution(start_path)
    for root, dirs, files in os.walk(start_path):
        for file in files:
            if 'scrollingActivity' in file and file.endswith('.csv'):
                file_path = os.path.join(root, file)
                plot_swipes(file_path, screen_resolution)

# Starting directory
start_directory = '../bankAppData/'
find_and_plot_files(start_directory)