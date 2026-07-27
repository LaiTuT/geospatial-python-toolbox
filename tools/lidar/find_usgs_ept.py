import json
import os
import subprocess
import time
import urllib.request
from urllib.error import HTTPError, URLError
from concurrent.futures import ThreadPoolExecutor, as_completed

from osgeo import gdal, osr

# =========================
# 需要你改的配置
# =========================
DEM_PATH = r"D:\Dataproces\1m\AL_25Co_B1_2017\TIFF\USGS_1m_x38y372_AL_25Co_B1_2017.tif"

# 只扫某些前缀（建议先用 "AL_"，不然目录很多）
PREFIX_STARTS_WITH = "A"   # 全扫就改成 None

AWS_REGION = "us-west-2"
S3_BUCKET = "usgs-lidar-public"

# 并发下载 ept.json 的线程数
WORKERS = 16

# 输出文件
OUT_MATCH_TXT = r"D:\Dataproces\Gdal_warp\ept_matches.txt"
OUT_JSONL     = r"D:\Dataproces\Gdal_warp\ept_scan_results.jsonl"
CACHE_DIR     = r"D:\Dataproces\Gdal_warp\ept_cache"
# =========================

BASE_HTTPS = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com"
TIMEOUT_SEC = 20

gdal.UseExceptions()
os.makedirs(CACHE_DIR, exist_ok=True)


def run_cmd(cmd: list[str]) -> tuple[int, str, str]:
    """运行命令并返回 (returncode, stdout, stderr)"""
    p = subprocess.run(cmd, text=True, capture_output=True)
    return p.returncode, p.stdout, p.stderr


def list_top_prefixes_via_awscli() -> list[str]:
    """
    用 aws s3 ls 列出桶根目录下的 PRE 前缀。
    输出示例行： '                           PRE AL_SWCentral_1_B22/'
    """
    cmd = [
        "aws", "s3", "ls", f"s3://{S3_BUCKET}/",
        "--no-sign-request",
        "--region", AWS_REGION
    ]
    rc, out, err = run_cmd(cmd)
    if rc != 0:
        raise RuntimeError(f"aws s3 ls 失败 (rc={rc})\nSTDERR:\n{err}")

    prefixes = []
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith("PRE "):
            continue
        p = line.replace("PRE ", "").strip()
        p = p.rstrip("/")  # 去掉末尾 /
        if PREFIX_STARTS_WITH and not p.startswith(PREFIX_STARTS_WITH):
            continue
        prefixes.append(p)

    return prefixes


def dem_bbox_in_3857(dem_path: str) -> tuple[float, float, float, float]:
    """读取 DEM bbox，并投到 EPSG:3857 (xmin,ymin,xmax,ymax)"""
    ds = gdal.Open(dem_path)
    if ds is None:
        raise RuntimeError(f"无法打开 DEM: {dem_path}")

    gt = ds.GetGeoTransform()
    w = ds.RasterXSize
    h = ds.RasterYSize
    proj = ds.GetProjection()

    x0, px_w, _, y0, _, px_h = gt
    x1 = x0 + w * px_w
    y1 = y0 + h * px_h
    xmin, xmax = (x0, x1) if x0 < x1 else (x1, x0)
    ymin, ymax = (y1, y0) if y1 < y0 else (y0, y1)

    srs_dem = osr.SpatialReference()
    srs_dem.ImportFromWkt(proj)
    srs_3857 = osr.SpatialReference()
    srs_3857.ImportFromEPSG(3857)
    tx = osr.CoordinateTransformation(srs_dem, srs_3857)

    corners = [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]
    xs, ys = [], []
    for x, y in corners:
        X, Y, _ = tx.TransformPoint(x, y)
        xs.append(X)
        ys.append(Y)

    ds = None
    return min(xs), min(ys), max(xs), max(ys)


def bbox_intersects(a, b) -> bool:
    """a,b: (xmin,ymin,xmax,ymax)"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def ept_bbox_3857(ept: dict) -> tuple[float, float, float, float]:
    """
    EPT boundsConforming 顺序通常是 [xmin,ymin,zmin,xmax,ymax,zmax]
    """
    b = ept.get("boundsConforming") or ept.get("bounds")
    if not b or len(b) < 6:
        raise ValueError("ept.json 缺少 boundsConforming/bounds 或格式不对")
    xmin = float(b[0]); ymin = float(b[1]); xmax = float(b[3]); ymax = float(b[4])
    return xmin, ymin, xmax, ymax


def fetch_ept_json(prefix: str):
    """
    下载 ept.json（带缓存）
    返回: (prefix, ok, match, url, ept_bbox, err)
    """
    url = f"{BASE_HTTPS}/{prefix}/ept.json"
    cache_path = os.path.join(CACHE_DIR, f"{prefix.replace('/', '_')}_ept.json")

    # cache 读取
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                ept = json.load(f)
            return prefix, True, ept, url, None
        except Exception:
            pass

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
            data = resp.read()
        ept = json.loads(data)

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(ept, f)

        return prefix, True, ept, url, None
    except (HTTPError, URLError, TimeoutError) as e:
        return prefix, False, None, url, str(e)
    except Exception as e:
        return prefix, False, None, url, str(e)


def main():
    dem_bb = dem_bbox_in_3857(DEM_PATH)
    print("DEM bbox (EPSG:3857):", dem_bb)

    prefixes = list_top_prefixes_via_awscli()
    print(f"Found {len(prefixes)} prefixes to scan (filter={PREFIX_STARTS_WITH!r}).")

    # 清空旧输出
    for p in [OUT_MATCH_TXT, OUT_JSONL]:
        if os.path.exists(p):
            os.remove(p)

    matches = []
    t0 = time.time()

    with open(OUT_JSONL, "a", encoding="utf-8") as jout:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = [ex.submit(fetch_ept_json, p) for p in prefixes]

            for i, fu in enumerate(as_completed(futs), 1):
                prefix, ok, ept, url, err = fu.result()
                rec = {"prefix": prefix, "url": url, "ok": ok, "error": err}

                if ok and ept is not None:
                    try:
                        eb = ept_bbox_3857(ept)
                        hit = bbox_intersects(dem_bb, eb)
                        rec["ept_bbox_3857"] = eb
                        rec["match"] = hit
                        if hit:
                            matches.append(prefix)
                    except Exception as e:
                        rec["ok"] = False
                        rec["error"] = f"parse bounds failed: {e}"

                jout.write(json.dumps(rec, ensure_ascii=False) + "\n")

                if i % 50 == 0 or i == len(prefixes):
                    dt = time.time() - t0
                    print(f"[{i}/{len(prefixes)}] matches={len(matches)} elapsed={dt:.1f}s")

    matches = sorted(set(matches))
    with open(OUT_MATCH_TXT, "w", encoding="utf-8") as f:
        for p in matches:
            f.write(p + "\n")

    print("Done.")
    print("Matches saved to:", OUT_MATCH_TXT)
    print("Scan log saved to:", OUT_JSONL)
    if matches:
        print("Top matches:")
        for p in matches[:30]:
            print("  ", p)


if __name__ == "__main__":
    main()
