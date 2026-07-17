from __future__ import annotations
import argparse,json
from pathlib import Path
from core import inspect,preview
def main():
    parser=argparse.ArgumentParser(description='检查栅格或矢量元数据');parser.add_argument('path');parser.add_argument('--json');parser.add_argument('--preview');args=parser.parse_args();path=Path(args.path).expanduser().resolve();result=inspect(path)
    if args.preview:result['preview']=str(preview(path,Path(args.preview)))
    payload=json.dumps(result,ensure_ascii=False,indent=2);print(payload)
    if args.json:Path(args.json).write_text(payload,encoding='utf-8')
if __name__=='__main__':main()
