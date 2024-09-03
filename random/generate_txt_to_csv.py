import csv
import os

# Define the headers for each activity type
headers_mapping = {
    'enterPINactivity': [
        "sensorName", "sensorType", "angularSpeedX", "angularSpeedY", "angularSpeedZ", 
        "timestamp", "date", "elapsedTimeNano", "username"
    ],
    'enterPINbuttonsOrder': [
        "buttonOrder", "date", "timestamp", "elapsedTimeNano", "username"
    ],
    'enterPINxyClick': [
        "btnText", "viewX", "viewY", "viewGetLeft", "viewGetTop", 
        "viewGetRight", "viewGetBottom", "eventX", "eventY", 
        "eventRawX", "eventRawY", "eventTime", "dateFormat", 
        "elapsedTimeNanos", "username"
    ],
    'entryActivity': [
        "sensorName", "sensorType", "angularSpeedX", "angularSpeedY", "angularSpeedZ", 
        "timestamp", "date", "elapsedTimeNano", "username"
    ],
    'onFling': [
        "velocityX", "velocityY", "event1RawX", "event1RawY", "event2RawX", "event2RawY", "event1Size", "event2Size", 
        "date", "elapsedTimeNano", "username"
    ],
    'onScroll': [
        "distanceX", "distanceY", 
        "e1X", "e1Y", "e2X", "e2Y", 
        "e1RawX", "e1RawY", "e2RawX", "e2RawY", 
        "e1Size", "e2Size", 
        "e1EventTime", "e2EventTime", 
        "e1DownTime", "e2DownTime", 
        "e1Action", "e2Action", 
        "e1Orientation", "e2Orientation", 
        "date", "elapsedTimeNano", 
        "username"
    ],
    'onTouch': [
        "action", "eventX", "eventY", "eventRawX", "eventRawY", "eventSize",
        "eventOrientation", "date", "elapsedTimeNano", "username"
    ],
    'scrollingActivity': [
        "actionMove", "eventX", "eventY", "eventRawX", "eventRawY", "eventSize", 
        "eventOrientation", "eventPointerCount", "eventTouchMajor", "eventToolMajor", "eventToolMinor", "eventTouchMinor", 
        "eventDownTime", "eventEventTime", "xVelocity", "yVelocity", "date", "elapsedTimeNano", "username"
    ]
}

# Function to process each file
def process_file(directory, filename):
    # Extract the activity type from the filename
    activity_match = filename.split('_')[1].split('-')[0]
    if activity_match:
        activity_type = activity_match
        headers = headers_mapping.get(activity_type, None)

        if headers:
            input_file = os.path.join(directory, filename)
            output_file = os.path.join(directory, filename.replace('.txt', '.csv'))

            with open(input_file, 'r') as txt_file, open(output_file, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(headers)  # Write the headers

                for line in txt_file:
                    data = line.strip().split()
                    writer.writerow(data)

            print(f"Processed {filename} for activity {activity_type}")
        else:
            print(f"Activity type not found in headers_mapping: {activity_type}")
            print(f"No headers defined for activity type in {filename}")
    else:
        print(f"Activity type not found in filename: {filename}")

# Path to the directory containing the .txt files
directory_path = './bankAppData'

# Check if directory exists
if not os.path.exists(directory_path):
    print(f"Directory does not exist: {directory_path}")
else:
    print(f"Processing files in directory: {directory_path}")

    # Walk through all subdirectories
    for dirpath, dirnames, filenames in os.walk(directory_path):
        for filename in filenames:
            if filename.endswith('.txt'):
                print(f"Found file: {filename} in directory: {dirpath}")
                process_file(dirpath, filename)
            else:
                print(f"Skipping non-txt file: {filename} in directory: {dirpath}")

print("All files have been processed.")
