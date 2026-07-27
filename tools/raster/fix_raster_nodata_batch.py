import os
import subprocess
from pathlib import Path

IN_DIR  = r"D:\Dataproces\DSM\USGS_NED_DSM_AK_IFSAR_LowerSE_L1_C351_2014_TIFF_2016"
OUT_DIR = r"D:\Dataproces\DSM\USGS_NED_DSM_AK_IFSAR_LowerSE_L1_C351_2014_TIFF_2016\fixed"   # 建议不要放在 IN_DIR 里面

THRESHOLD = -5000
NODATA = -9999

def run_cmd(args):
    p = subprocess.run(args, capture_output=True, text=True)
    if p.returncode != 0:
        print("命令失败：")
        print(" ".join(args))
        print(p.stdout)
        print(p.stderr)
    return p.returncode

def delete_sidecars(tif: Path):
    # 删除 ArcGIS / GDAL 可能产生的旧统计缓存
    candidates = [
        tif.with_suffix(tif.suffix + ".aux.xml"),
        tif.with_suffix(".aux.xml"),
        tif.with_suffix(tif.suffix + ".ovr"),
        tif.with_suffix(".ovr"),
    ]
    for f in candidates:
        if f.exists():
            try:
                f.unlink()
                print(f"  删除缓存: {f}")
            except Exception as e:
                print(f"  删除失败: {f}  {e}")

def run_fix(in_path: Path, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        out_path.unlink()

    # 把 -9999、极低值、NaN、Inf 都写成 -9999，并设置 NoData=-9999
    calc_expr = (
        f"where((A=={NODATA}) | (A<={THRESHOLD}) | isnan(A) | isinf(A), "
        f"{NODATA}, A)"
    )

    cmd = [
        "python", "-m", "osgeo_utils.gdal_calc",
        "-A", str(in_path),
        "--outfile", str(out_path),
        "--calc", calc_expr,
        "--NoDataValue", str(NODATA),
        "--type", "Float32",
        "--co", "COMPRESS=LZW",
        "--co", "TILED=YES",
        "--co", "BIGTIFF=IF_SAFER",
        "--overwrite"
    ]

    rc = run_cmd(cmd)
    if rc != 0:
        return rc

    # 再强制写一次 NoData，保险
    rc = run_cmd([
        "python", "-m", "osgeo_utils.gdal_edit",
        "-a_nodata", str(NODATA),
        str(out_path)
    ])
    if rc != 0:
        return rc

    # 清掉旧统计
    run_cmd([
        "python", "-m", "osgeo_utils.gdal_edit",
        "-unsetstats",
        str(out_path)
    ])

    # 删除可能的 sidecar 统计缓存
    delete_sidecars(out_path)

    # 重新计算统计，NoData=-9999 应该会被排除
    run_cmd([
        "gdalinfo",
        "-stats",
        str(out_path)
    ])

    return 0

def main():
    in_dir = Path(IN_DIR).resolve()
    out_dir = Path(OUT_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tifs = []
    for root, _, files in os.walk(in_dir):
        root_path = Path(root).resolve()

        # 避免 OUT_DIR 在 IN_DIR 里时被重复扫描
        try:
            root_path.relative_to(out_dir)
            continue
        except ValueError:
            pass

        for f in files:
            if f.lower().endswith((".tif", ".tiff")):
                tifs.append(root_path / f)

    if not tifs:
        print("未找到 tif 文件。")
        return

    total = len(tifs)
    fixed = failed = 0

    for i, tif in enumerate(tifs, 1):
        rel = tif.relative_to(in_dir)
        out_path = out_dir / rel

        print(f"[{i}/{total}] FIX  {tif}")
        rc = run_fix(tif, out_path)

        if rc == 0:
            fixed += 1
        else:
            failed += 1
            print(f"  [FAILED] returncode={rc}")

    print("\n====== 完成 ======")
    print(f"总数: {total}  修复: {fixed}  失败: {failed}")
    print(f"输出目录: {out_dir}")

if __name__ == "__main__":
    main()
