import os
from typing import List, Tuple


def _get_shp_schema(shp_path: str) -> List[Tuple[str, str, int, int]]:
	try:
		from osgeo import ogr
	except ImportError as exc:
		raise RuntimeError("GDAL/OGR is required. Install osgeo/gdal to use this script.") from exc

	datasource = ogr.Open(shp_path)
	if datasource is None:
		raise RuntimeError(f"Failed to open shapefile: {shp_path}")

	layer = datasource.GetLayer(0)
	layer_defn = layer.GetLayerDefn()

	schema = []
	for i in range(layer_defn.GetFieldCount()):
		field_defn = layer_defn.GetFieldDefn(i)
		schema.append(
			(
				field_defn.GetName(),
				field_defn.GetFieldTypeName(field_defn.GetType()),
				field_defn.GetWidth(),
				field_defn.GetPrecision(),
			)
		)

	return schema


def check_shp_attributes(folder_path: str) -> dict:
	if not os.path.isdir(folder_path):
		raise ValueError(f"Folder not found: {folder_path}")

	shp_files = [
		os.path.join(folder_path, f)
		for f in os.listdir(folder_path)
		if f.lower().endswith(".shp")
	]

	if not shp_files:
		return {"reference": None, "mismatches": []}

	reference_path = shp_files[0]
	reference_schema = _get_shp_schema(reference_path)

	mismatches = []
	for shp_path in shp_files[1:]:
		schema = _get_shp_schema(shp_path)
		if schema != reference_schema:
			mismatches.append({
				"file": shp_path,
				"schema": schema,
			})

	return {
		"reference": {
			"file": reference_path,
			"schema": reference_schema,
		},
		"mismatches": mismatches,
	}


if __name__ == "__main__":
	# Configure this variable as needed.
	folder_path = r"D:\path\to\shp_folder"

	result = check_shp_attributes(folder_path)

	if result["reference"] is None:
		print("No .shp files found.")
	else:
		print("Reference shapefile:")
		print(f"  {result['reference']['file']}")

		if not result["mismatches"]:
			print("All shapefiles have the same attribute schema.")
		else:
			print("Shapefiles with different attribute schema:")
			for item in result["mismatches"]:
				print(f"  {item['file']}")
