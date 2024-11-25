# reassemble_large_files.py to reassemble the split files
import os
import re
import shutil

def reassemble_files(directory, min_size=3 * 1024 * 1024):
    """
    Reassembles split files in the given directory if their combined size is greater than min_size,
    irrespective of the file extension.
    """
    part_pattern = re.compile(r'^(.*)\.part(\d+)$')
    for root, _, files in os.walk(directory):
        # Map base filenames to their parts
        parts_dict = {}
        for file in files:
            match = part_pattern.match(file)
            if match:
                base_name = match.group(1)
                part_number = int(match.group(2))
                parts_dict.setdefault(base_name, []).append((part_number, file))

        for base_name, part_files in parts_dict.items():
            # Sort parts by part number
            part_files.sort(key=lambda x: x[0])
            part_paths = [os.path.join(root, f[1]) for f in part_files]
            # Calculate total size
            total_size = sum(os.path.getsize(p) for p in part_paths)
            if total_size >= min_size:
                assembled_file_path = os.path.join(root, base_name)
                with open(assembled_file_path, 'wb') as outfile:
                    for _, part in part_files:
                        part_path = os.path.join(root, part)
                        with open(part_path, 'rb') as infile:
                            outfile.write(infile.read())
                print(f"Reassembled {assembled_file_path} from {len(part_files)} parts.")
            else:
                print(f"Skipped {base_name}, total size less than {min_size} bytes.")

if __name__ == "__main__":
    directories = ['./fine_tuned_model', './results/checkpoint-4957']
    for directory in directories:
        if os.path.exists(directory):
            reassemble_files(directory)
        else:
            print(f"Directory not found: {directory}")