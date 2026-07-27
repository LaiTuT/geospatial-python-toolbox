import os


def delete_files_by_name(folder_path: str, name_substring: str) -> int:
	if not os.path.isdir(folder_path):
		raise ValueError(f"Folder not found: {folder_path}")

	deleted = 0
	for entry in os.listdir(folder_path):
		full_path = os.path.join(folder_path, entry)
		if os.path.isfile(full_path) and name_substring in entry:
			os.remove(full_path)
			deleted += 1

	return deleted


if __name__ == "__main__":
	# Configure these two variables as needed.
	folder_path = r"D:\Dataproces\10m\rep\mou_washinton_reprojected\TFM_shp"
	name_substring = "_TFM_50"

	removed_count = delete_files_by_name(folder_path, name_substring)
	print(f"Deleted {removed_count} file(s) from {folder_path}.")
