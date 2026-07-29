import os


REQUIRED_SHP_EXTS = {
	".cpg",
	".dbf",
	".prj",
	".sbn",
	".sbx",
	".shp",
	".shp.xml",
	".shx",
}


def _list_files(folder_path: str) -> list[str]:
	if not os.path.isdir(folder_path):
		raise ValueError(f"Folder not found: {folder_path}")
	return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]


def _split_base_ext(filename: str) -> tuple[str, str]:
	lower_name = filename.lower()
	if lower_name.endswith(".shp.xml"):
		return filename[: -len(".shp.xml")], ".shp.xml"
	base, ext = os.path.splitext(filename)
	return base, ext.lower()


def check_tif_shp_pairs(tif_folder: str, shp_folder: str) -> dict:
	tif_files = _list_files(tif_folder)
	shp_files = _list_files(shp_folder)

	tif_basenames = {
		os.path.splitext(name)[0]
		for name in tif_files
		if name.lower().endswith(".tif")
	}

	shp_map: dict[str, set[str]] = {}
	for name in shp_files:
		base, ext = _split_base_ext(name)
		if ext in REQUIRED_SHP_EXTS:
			shp_map.setdefault(base, set()).add(ext)

	missing_shp_for_tif = []
	incomplete_shp = []

	for tif_base in sorted(tif_basenames):
		exts = shp_map.get(tif_base)
		if not exts:
			missing_shp_for_tif.append(tif_base)
			continue
		missing_exts = sorted(REQUIRED_SHP_EXTS - exts)
		if missing_exts:
			incomplete_shp.append((tif_base, missing_exts))

	shp_without_tif = sorted(base for base in shp_map.keys() if base not in tif_basenames)

	return {
		"missing_shp_for_tif": missing_shp_for_tif,
		"incomplete_shp": incomplete_shp,
		"shp_without_tif": shp_without_tif,
	}


if __name__ == "__main__":
	# Configure these two variables as needed.
	tif_folder = r"F:\tfm_dsm_all"
	shp_folder = r"F:\tfm_shp_all"

	result = check_tif_shp_pairs(tif_folder, shp_folder)

	print("Missing shapefile bundle for TIF:")
	for name in result["missing_shp_for_tif"]:
		print(f"  {name}")

	print("\nIncomplete shapefile bundle:")
	for name, missing_exts in result["incomplete_shp"]:
		print(f"  {name} -> missing: {', '.join(missing_exts)}")

	print("\nShapefile bundle without TIF:")
	for name in result["shp_without_tif"]:
		print(f"  {name}")
