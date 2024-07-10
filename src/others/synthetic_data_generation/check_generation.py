import os
import sys 

'''
Example usages:
python check_generation.py "$DATAPATH/dutch_real/raw_data/" csv "$DATAPATH/synthetic/raw_data/" txt
python check_generation.py "$DATAPATH/dutch_real/data/" csv "$DATAPATH/synthetic/data/" txt
'''


def find_files(base_dir, extension):
    """ Walk through the directory recursively and find all files with the given extension. """
    for dirpath, dirnames, filenames in os.walk(base_dir):
        for filename in filenames:
            if filename.endswith(extension):
                # Yield relative path and base name without extension
                yield os.path.relpath(dirpath, base_dir), filename[:-len(extension)]

if __name__ == '__main__':
  source_dir, source_extension = sys.argv[1], sys.argv[2]
  target_dir, target_extension = sys.argv[3], sys.argv[4]
  # Set of (subdirectory relative path, filename without extension) for each source and target file
  source_files = set(find_files(source_dir, source_extension))
  target_files = set(find_files(target_dir, target_extension))

  # Check for corresponding target files for each source file
  unmatched = 0

  for rel_path, base_name in source_files:
      if (rel_path, base_name) not in target_files:
          print(f"No match found for {os.path.join(source_dir, rel_path, base_name)}{source_extension} in {target_dir}")
          unmatched += 1

  # Check for extra target files
  for rel_path, base_name in target_files:
      if (rel_path, base_name) not in source_files:
          print(f"Extra file {os.path.join(target_dir, rel_path, base_name)}{target_extension} found in {source_dir}")
          unmatched += 1

  if unmatched == 0:
      print(
        f"All {len(source_files)} files at {source_dir} match perfectly with " 
        f"all {len(target_files)} files at {target_dir}."
      )
  else:
      print(
        f"Mismatch in files. {unmatched} issues found.\n"
        f"# of files at {source_dir} = {len(source_files)}\n"
        f"# of files at {target_dir} = {len(target_files)}\n"
      )

