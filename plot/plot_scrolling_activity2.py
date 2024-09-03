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
    # Convert the 'date' column to a datetime object, if it's not already
    df['date'] = pd.to_datetime(df['date'])

    # Extract the second component of the datetime
    df['second'] = df['date'].dt.floor('S')  # Rounds down to the nearest second

    # Initialize a list to store the simplified swipes
    simplified_swipes = []

    # Group by the second and simplify each group
    for _, group in df.groupby('second'):
        # Here you can further simplify each group if needed
        # For now, I'm taking the first swipe of each group as a representative
        simplified_swipes.append(group.iloc[0])

    # Convert the list of swipes to a DataFrame using pd.concat
    simplified_df = pd.concat(simplified_swipes, axis=1).transpose()
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
            output_file = os.path.splitext(file_path)[0] + '_scroll_plot_2.png'
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