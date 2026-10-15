import os
from datetime import datetime
import shutil

def create_archive_directory(timestamp=None, output_dir="output"):
    """Create timestamped archive directory and return path and timestamp"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
    
    archive_dir = os.path.join(output_dir, "archived_runs", timestamp)
    os.makedirs(archive_dir, exist_ok=True)
    return archive_dir, timestamp

def archive_file(source_path, archive_dir, new_name=None):
    """Copy file to archive directory"""
    if os.path.exists(source_path):
        filename = new_name or os.path.basename(source_path)
        destination = os.path.join(archive_dir, filename)
        shutil.copy2(source_path, destination)
        print(f"✓ Archived: {filename}")
        return True
    else:
        print(f"✗ File not found for archiving: {source_path}")
        return False

def save_timestamp(timestamp, output_dir="output"):
    """Save timestamp for use by subsequent scripts"""
    timestamp_file = os.path.join(output_dir, ".last_run_timestamp")
    os.makedirs(output_dir, exist_ok=True)
    with open(timestamp_file, 'w') as f:
        f.write(timestamp)

def get_last_timestamp():
    """Get timestamp from last run"""
    timestamp_file = "output/.last_run_timestamp"
    if os.path.exists(timestamp_file):
        with open(timestamp_file, 'r') as f:
            return f.read().strip()
    return None