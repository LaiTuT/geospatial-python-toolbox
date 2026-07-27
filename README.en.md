# Geospatial Python Tools

A collection of standalone Python utilities for GeoTIFF, Shapefile, DEM/DSM, LiDAR EPT, and geospatial YOLO workflows. Scripts are grouped by domain and consistently named with lowercase `snake_case` filenames.

The repository has been normalized and all Python/JSON files pass static syntax checks. Some legacy scripts still use configuration constants near the top of the file; review them before running. Test destructive tools on copied data first.

## Setup

Python 3.10+ is recommended. Install GDAL through Conda, especially on Windows:

```bash
conda create -n geospatial-tools python=3.11
conda activate geospatial-tools
conda install -c conda-forge gdal
pip install -r requirements.txt
```

ArcPy is only required by `tools/vector/raster_mask_to_shapefile_arcpy.py` and must be run in an ArcGIS Pro Python environment. Machine-learning tools may additionally require PyTorch, Ultralytics, or ONNX Runtime.

See [README.md](README.md) for the complete tool catalog and [docs/renaming.md](docs/renaming.md) for the old-to-new filename map.

Run the dependency-free repository checks with:

```bash
python scripts/check_repository.py
```

Licensed under the [MIT License](LICENSE).
