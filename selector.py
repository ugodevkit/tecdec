# Copyright (C) 2026  Falconio Ugo
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org>.

"""
LLM-jp Corpus Data Selector
===========================

This utility manages the downloading and sampling of text data from the LLM-jp Corpus v4 repository.
It is designed to handle Git LFS (Large File Storage) efficiently by:
1. Initializing the repository with `GIT_LFS_SKIP_SMUDGE=1` to avoid downloading terabytes of data.
2. Restoring pointer files if the repository is in an inconsistent state.
3. Providing two modes of operation:
   - Conservative: Uses only files that have already been fully downloaded.
   - Active: Forces the download of a specific random file (approx. 1GB) if needed.

Dependencies:
- git (must be installed and in PATH)
- git-lfs (must be installed)
"""

import os
import json
import gzip
import random
import subprocess
from pathlib import Path

REPO_URL = "https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v4.git"
DATA_DIR = Path("llm_jp_data").resolve()
REPO_DIR = DATA_DIR / "llm-jp-corpus-v4"

def ensure_repo():
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True)
    
    if not REPO_DIR.exists():
        print(f"Cloning {REPO_URL}...")
        # GIT_LFS_SKIP_SMUDGE=1 ensures we only download pointers, not huge files initially
        env = os.environ.copy()
        env["GIT_LFS_SKIP_SMUDGE"] = "1"
        subprocess.run(["git", "clone", REPO_URL], cwd=str(DATA_DIR), check=True, env=env)
        # Disable automatic smudge to prevent future accidental downloads
        subprocess.run(["git", "config", "lfs.fetchexclude", "*"], cwd=str(REPO_DIR), check=False)
    else:
        # Repo exists, check if we need to restore pointers if directory is empty
        ja_dir = REPO_DIR / "ja"
        if not ja_dir.exists() or not list(ja_dir.rglob("*.jsonl.gz")):
             print("Repo structure exists but pointers missing. Restoring...")
             subprocess.run(["git", "restore", "."], cwd=str(REPO_DIR), check=False)
             subprocess.run(["git", "checkout", "."], cwd=str(REPO_DIR), check=False)

def is_lfs_pointer(file_path):
    if not file_path.exists(): return False
    if file_path.stat().st_size > 2000:
        return False
    try:
        with open(file_path, 'rb') as f:
            header = f.read(100)
            return header.startswith(b"version https://git-lfs.github.com/spec/v1")
    except:
        return False

def fetch_lfs_file(file_path):
    # Ensure we get the path relative to the repo root, just like GitProcesses.java
    try:
        # Resolve both paths to absolute to ensure relative_to works correctly
        rel_path = file_path.resolve().relative_to(REPO_DIR.resolve())
    except ValueError:
        # If standard relative_to fails, try string manipulation as backup
        # This handles cases where symlinks might confuse pathlib
        f_str = str(file_path.resolve())
        r_str = str(REPO_DIR.resolve())
        if f_str.startswith(r_str):
            rel_path = f_str[len(r_str):].lstrip(os.sep)
        else:
            rel_path = file_path.name

    print(f"Fetching LFS content for {rel_path}...")
    # Explicitly pull only this file
    # On Windows, path separators might need to be forward slashes for git include patterns
    include_pattern = str(rel_path).replace(os.sep, "/")
    
    # Temporarily allow fetching this specific file if global exclude is on
    # We use -I (include) with git lfs pull, but sometimes local config interferes.
    # Let's ensure we try to checkout explicitly.
    subprocess.run(["git", "config", "--unset", "lfs.fetchexclude"], cwd=str(REPO_DIR), check=False)
    subprocess.run(["git", "lfs", "pull", "--include", include_pattern], cwd=str(REPO_DIR), check=True)
    # Restore exclude to prevent accidental huge downloads later
    subprocess.run(["git", "config", "lfs.fetchexclude", "*"], cwd=str(REPO_DIR), check=False)

def extract_random_line(file_path):
    try:
        lines = []
        # Read first 2000 lines to pick from
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for _ in range(2000):
                line = f.readline()
                if not line: break
                lines.append(line)
        
        if lines:
            line = random.choice(lines)
            data = json.loads(line)
            if "text" in data and len(data["text"]) > 50:
                return data["text"]
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return None

def get_random_sample(active_mode=True):
    ensure_repo()
    
    ja_dir = REPO_DIR / "ja"
    if not ja_dir.exists():
        return f"Error: 'ja' directory not found at {ja_dir.absolute()}."
        
    files = list(ja_dir.rglob("*.jsonl.gz"))
    if not files:
        return f"Error: No .jsonl.gz files found in {ja_dir.absolute()}."
    
    # 1. Conservative strategy: Check if we have any already-downloaded files
    downloaded_files = [f for f in files if not is_lfs_pointer(f)]
    if downloaded_files:
        print(f"Found {len(downloaded_files)} locally available files. Using one.")
        random.shuffle(downloaded_files)
        for f in downloaded_files[:5]:
            res = extract_random_line(f)
            if res: return res

    # 2. Active strategy: If allowed, download a new one
    if active_mode:
        print("No suitable local data found. Selecting random file to download...")
        random.shuffle(files)
        for f in files[:3]: # Try a few candidates
            # If it's a pointer, or we just want to ensure we have it, fetch it.
            # Even if we think we have it (conservative failed above), try fetching.
            try:
                fetch_lfs_file(f)
                res = extract_random_line(f)
                if res: return res
            except Exception as e:
                print(f"Failed to download/read {f}: {e}")

    return "Error: Could not extract valid sample text. Try checking internet connection or disk space."

if __name__ == "__main__":
    print(get_random_sample(active_mode=True))
