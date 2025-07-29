import pandas as pd
import json
import ast
from datetime import datetime
import warnings

def parse_beth_csv_row(row):
    """Parse a single BETH CSV row into standardized JSON format."""
    # Comprehensive mapping for all distinct eventName values
    threat_type_mapping = {
        "accept": "Network Activity",
        "accept4": "Network Activity",
        "access": "File Access",
        "bind": "Network Activity",
        "cap_capable": "Privilege Operation",
        "clone": "Process Creation",
        "close": "File Operation",
        "connect": "Network Connection",
        "dup": "File Descriptor Operation",
        "dup2": "File Descriptor Operation",
        "dup3": "File Descriptor Operation",
        "execve": "Command Execution",
        "fchmod": "File Permission Change",
        "fstat": "File Metadata Access",
        "getdents64": "Directory Listing",
        "getsockname": "Network Activity",
        "kill": "Process Termination",
        "lstat": "File Metadata Access",
        "openat": "File Access",
        "prctl": "Process Control",
        "sched_process_exit": "Process Termination",
        "security_bprm_check": "Security Check",
        "security_file_open": "File Access",
        "security_inode_unlink": "File Deletion",
        "setgid": "Privilege Operation",
        "setuid": "Privilege Operation",
        "socket": "Network Activity",
        "stat": "File Metadata Access",
        "symlink": "File Operation",
        "umount": "Filesystem Operation",
        "unlink": "File Deletion",
        "unlinkat": "File Deletion"
    }
    
    # Parse args JSON string
    args_raw = row.get("args", "[]")
    try:
        args_data = ast.literal_eval(args_raw)
        args_details = args_data
    except (ValueError, SyntaxError):
        args_details = {}
        warnings.warn(f"Malformed args literal in row: {args_raw}")

    # Parse StackAddress (stringified list of integers)
    stack_address_raw = row.get("StackAddress", "[]")
    #Handling non-list inputs
    if isinstance(stack_address_raw, str) and stack_address_raw.startswith("["):
        try:
            stack_address_data = ast.literal_eval(stack_address_raw)
            stack_address_details = stack_address_data if isinstance(stack_address_data, list) else []
        except (ValueError, SyntaxError):
            stack_address_details = []
            warnings.warn(f"Malformed StackAddress literal in row: {stack_address_raw}")
    else:
        #if not a proper list-like string, just wrap it in a list if it's numeric
        try: 
            val = int(float(stack_address_raw))
            stack_address_details = [val]
        except Exception:
            stack_address_details = []
            warnings.warn(f"Invalid StackAddress value: {stack_address_raw}, setting to empty list")
    

    # Infer source_ip from hostName
    host_name = row.get("hostName", "unknown")
    if "ip-" in host_name:
        # Extract IP from hostName like 'ip-10-100-1-26'
        source_ip = host_name.split("ip-")[-1].replace("-", ".")
    else:
        # Handle non-IP hostNames like 'ubuntu'
        source_ip = "unknown"
        warnings.warn(f"Non-IP hostName detected: {host_name}, setting source_ip to 'unknown'")

    return {
        "timestamp": row.get("timestamp", datetime.now().isoformat()),
        "source_ip": source_ip,
        "threat_type": threat_type_mapping.get(row.get("eventName", "unknown"), "Unknown"),
        "details": {
            "processId": row.get("processId", ""),
            "parentProcessId": row.get("parentProcessId", ""),
            "userId": row.get("userId", ""),
            "mountNameSpace": str(row.get("mountNameSpace", "")).strip(),
            "processName": row.get("processName", ""),
            "hostName": row.get("hostName", ""),
            "eventId": row.get("eventId", ""),
            "eventName": row.get("eventName", ""),
            "StackAddress": stack_address_details,
            "argsNum": row.get("argsNum", ""),
            "returnValue": row.get("returnValue", ""),
            "args": args_details,
            "sus": row.get("sus", "0") == "1",
            "evil": row.get("evil", "0") == "1"
        }
    }

def preprocess_beth_csv(input_file, output_file, chunksize=None):
    """Process BETH CSV logs and save standardized JSON."""
    standardized_data = []
    
    # Handle gzipped or regular CSV file
    if input_file.endswith('.gz'):
        if chunksize:
            for chunk in pd.read_csv(input_file, compression='gzip', chunksize=chunksize):
                chunk = chunk.astype(str)
                for _, row in chunk.iterrows():
                    standardized_entry = parse_beth_csv_row(row.to_dict())
                    standardized_data.append(standardized_entry)
        else:
            df = pd.read_csv(input_file, compression='gzip')
            df = df.astype(str)
            for _, row in df.iterrows():
                standardized_entry = parse_beth_csv_row(row.to_dict())
                standardized_data.append(standardized_entry)
    else:
        if chunksize:
            for chunk in pd.read_csv(input_file, chunksize=chunksize):
                chunk = chunk.astype(str)
                for _, row in chunk.iterrows():
                    standardized_entry = parse_beth_csv_row(row.to_dict())
                    standardized_data.append(standardized_entry)
        else:
            df = pd.read_csv(input_file)
            df = df.astype(str)
            for _, row in df.iterrows():
                standardized_entry = parse_beth_csv_row(row.to_dict())
                standardized_data.append(standardized_entry)
    
    # Save to output file
    with open(output_file, "w") as f:
        json.dump(standardized_data, f, indent=2)
    
    return standardized_data

# Example usage
if __name__ == "__main__":
    input_file = "./archive/labelled_training_data.csv"  # Update with your CSV file path
    output_file = "standardized_beth_data.json"
    standardized_data = preprocess_beth_csv(input_file, output_file, chunksize=10000)
    print(f"Processed {len(standardized_data)} log entries into {output_file}")