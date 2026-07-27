# Name: RasterToPolyline_Ex_02.py
# Description: Converts a raster dataset to polyline features.
# Requirements: None

# Import system modules
import os
import arcpy
from arcpy import env

# Set environment settings
input_folder = r"D:\Dataproces\10m\rep\moutain_reprojected\TFM"
output_folder = r"D:\Dataproces\10m\rep\moutain_reprojected\TFM_shp"
env.workspace = input_folder

# Set local variables
backgrVal = "ZERO"
dangleTolerance = 100
field = "VALUE"

# Execute RasterToPolyline for all TIF files
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)
tif_list = arcpy.ListRasters("*", "TIF")
if not tif_list:
    raise RuntimeError("No TIF files found in input folder.")

for tif in tif_list:
    base_name = os.path.splitext(os.path.basename(tif))[0]
    outLines = os.path.join(output_folder, base_name + ".shp")
    arcpy.RasterToPolyline_conversion(
        tif,
        outLines,
        backgrVal,
        dangleTolerance,
        "SIMPLIFY",
        field,
    )
