import pandas as pd

def extract_distinct_event_names(input_file):
    """Extract unique eventName values from a BETH CSV file."""
    df = pd.read_csv(input_file, usecols=['eventName'])
    
    # Convert eventName to string to handle any non-string values
    df['eventName'] = df['eventName'].astype(str)
    
    # Get unique eventName values
    unique_event_names = sorted(df['eventName'].unique())
    
    # Print unique event names
    print("Distinct Event Names:")
    for event_name in unique_event_names:
        print(event_name)
    
    # Save to a file for reference
    with open("./Preprocessing/distinct_validation_event_names.txt", "w") as f:
        f.write("Distinct Event Names:\n")
        for event_name in unique_event_names:
            f.write(f"{event_name}\n")
    
    return unique_event_names

# Example usage
if __name__ == "__main__":
    input_file = "./Preprocessing/archive/labelled_validation_data.csv"  # Update with your CSV file path
    unique_events = extract_distinct_event_names(input_file)
    print(f"\nTotal unique event names: {len(unique_events)}")