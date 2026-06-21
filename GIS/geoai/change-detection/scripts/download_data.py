#!/usr/bin/env python3
"""下载双时相 Sentinel-2 影像对"""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
console = Console()

def main():
    parser = argparse.ArgumentParser(description="下载双时相 Sentinel-2 影像")
    parser.add_argument("--region", default="beijing", help="预定义区域")
    parser.add_argument("--bbox", nargs=4, type=float, help="自定义 bbox")
    parser.add_argument("--date-a", default="2022-06-01", help="时相 A 日期")
    parser.add_argument("--date-b", default="2023-06-01", help="时相 B 日期")
    parser.add_argument("--cloud", type=float, default=20.0, help="最大云量")
    parser.add_argument("--name", default=None, help="输出名称前缀")
    args = parser.parse_args()

    from config import RAW_DIR, PRESET_REGIONS

    bbox = args.bbox or PRESET_REGIONS.get(args.region, {}).get("bbox")
    if not bbox:
        console.print(f"[red]请指定有效区域或 bbox[/red]"); return

    console.print(f"下载: A={args.date_a} B={args.date_b}, bbox={bbox}")

    from cdd.downloader import BiTemporalDownloader
    dl = BiTemporalDownloader(RAW_DIR)
    result = dl.download_pair(
        bbox=bbox, date_a=args.date_a, date_b=args.date_b,
        max_cloud_cover=args.cloud, out_name=args.name,
    )

    console.print(f"[green]时相 A: {result['time_a']}[/green]")
    console.print(f"[green]时相 B: {result['time_b']}[/green]")

if __name__ == "__main__":
    main()
