import pandas as pd
import plotly.graph_objects as go
import os

def find_csv_files(directory, pattern):
    """ Recursively find all CSV files in 'directory' that contain 'pattern' in their filename. """
    matching_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if pattern in file and file.endswith('.csv'):
                matching_file = os.path.join(root, file)
                matching_files.append(matching_file)
                print(f"Found file: {matching_file}")
    return matching_files

def find_device_model_file(directory):
    """ Find a 'deviceModel' file in the specified directory. """
    for file in os.listdir(directory):
        if 'deviceModel' in file and file.endswith('.csv'):
            return os.path.join(directory, file)
    return None

def process_and_save_plot(pin_click_file, device_model_file):
    """ Read the CSV files, extract data, create a plot with large annotated dots, and save it as PNG. """
    try:
        print(f"Processing: {pin_click_file} with {device_model_file}")
        pin_click_data = pd.read_csv(pin_click_file)
        device_model_data = pd.read_csv(device_model_file)

        # Extract screen resolution
        screen_width = device_model_data['screenResolutionX'].iloc[0]
        screen_height = device_model_data['screenResolutionY'].iloc[0]

        # Create a scatter plot with large dots
        fig = go.Figure()

        # Add large dots with numbers
        for index, row in pin_click_data.iterrows():
            fig.add_trace(go.Scatter(x=[row['eventRawX']], y=[row['eventRawY']], 
                                     mode='markers+text', 
                                     text=[str(index + 1)], 
                                     textposition="bottom center",
                                     marker=dict(size=100)))  # Increased size of dots

        # Update layout
        fig.update_layout(
            title=f"Taps on Mobile Screen - {os.path.basename(pin_click_file)}",
            xaxis=dict(title='X Coordinate', range=[0, screen_width]),
            yaxis=dict(title='Y Coordinate', range=[0, screen_height], autorange="reversed"),
            width=screen_width,
            height=screen_height
        )

        # Save the plot as PNG
        png_filename = os.path.splitext(pin_click_file)[0] + '.png'
        fig.write_image(png_filename)
        print(f"Saved plot as: {png_filename}")

    except Exception as e:
        print(f"Error processing file {pin_click_file}: {e}")

# Set your base directory here
base_directory = '../bankAppData'

# Find all "enterPINxyClick" files
print("Searching for 'enterPINxyClick' files...")
pin_files = find_csv_files(base_directory, 'enterPINxyClick')

if not pin_files:
    print("No 'enterPINxyClick' files found. Please check the directory and file patterns.")

# Process each 'enterPINxyClick' file
for pin_file in pin_files:
    directory = os.path.dirname(pin_file)
    device_model_file = find_device_model_file(directory)
    
    if device_model_file:
        process_and_save_plot(pin_file, device_model_file)
    else:
        print(f"No 'deviceModel' file found in the directory of {pin_file}")

# Note: This script must be executed in a local environment where Plotly and its dependencies are installed.
# The path to the files must be correctly set for the script to work.