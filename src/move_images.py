import os
import shutil

def copy_directory(src_dir, dst_dir):
  """
  Recursively copies all files from source directory to destination directory.

  Args:
    src_dir: Path to the source directory.
    dst_dir: Path to the destination directory.
  """
  for root, dirs, files in os.walk(src_dir):
    # Copy files
    for file in files:
      src_file = os.path.join(root, file)
      dst_file = os.path.join(dst_dir, file)
      shutil.copy2(src_file, dst_file)  # Preserves file metadata

# Example usage
source_dir = "data/Validation Data"
destination_dir = "collapsed_data/test"

# copy_directory(source_dir, destination_dir)
# print(f"All files copied from {source_dir} to {destination_dir}")
files = os.listdir(destination_dir)
print(len(files))