"""定位本项目，以及只读引用的 CTA 项目。"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "cache"
OUTPUT_ROOT = ROOT / "output"


def cta_root():
    env = os.environ.get("CTA_ROOT")
    if env:
        return Path(env)
    candidates = [
        Path("/Users/yanyunfeng/文稿/cta"),
        Path("/workspace/cta"),
    ]
    for path in candidates:
        if (path / "cta" / "config.py").is_file():
            return path
    raise FileNotFoundError(
        "找不到现有 CTA 项目。请设置环境变量 CTA_ROOT，指向本地 cta 目录。"
    )


def add_cta_to_path():
    path = cta_root()
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)
    return path
