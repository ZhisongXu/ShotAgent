"""
face_restoration_tool_export.py

Usage:
    python face_restoration_tool_export.py /path/to/result/folder -o face_tools.txt

Function:
- Iterate over each first-level subdirectory of the given root directory (e.g. 00000_00-250506_175126)
- Search the subtree of each subdirectory for a file named result_scores_faces.txt
  (typically under .../img_tree/.../subtask-face restoration/...)
- Parse lines in the format "name, score" and select the entry with the highest score
- Write results into an output text file with format: KEY: short_name
  where KEY is the first-level folder name truncated at the first '-' (e.g. "00000_00-250506_175126" -> "00000_00")
  and short_name is derived from the best entry by splitting on '-' and taking the last token
  (e.g. "tool-face-0-gfpgan" -> "gfpgan")
"""

import os
import argparse
import re
from typing import Optional, Tuple

# Regex to find a floating/integer number in a string
FLOAT_RE = re.compile(r'[-+]?\d*\.\d+|[-+]?\d+')

def find_result_file(root_dir: str, filename: str = "result_scores_faces.txt") -> Optional[str]:
    """
    Walk the directory tree under root_dir and return the first path
    where 'filename' is found. If not found, return None.
    """
    for dirpath, _, files in os.walk(root_dir):
        if filename in files:
            return os.path.join(dirpath, filename)
    return None

def parse_scores_file(path: str) -> Optional[Tuple[str, float]]:
    """
    Parse a result_scores_faces.txt file where each meaningful line contains:
        name, score
    Return a tuple (best_name, best_score) for the highest score found.
    If no valid entries are found or the file cannot be parsed, return None.
    """
    best_name = None
    best_score = None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue

                # Prefer splitting by the first comma: "name, score"
                parts = line.split(',', 1)
                if len(parts) == 2:
                    name = parts[0].strip()
                    score_part = parts[1].strip()
                else:
                    # Fall back: whitespace-separated, last token is score
                    tokens = line.split()
                    if len(tokens) < 2:
                        continue
                    name = ' '.join(tokens[:-1])
                    score_part = tokens[-1]

                # Extract the first numeric substring from score_part
                m = FLOAT_RE.search(score_part)
                if not m:
                    continue
                try:
                    score = float(m.group(0))
                except ValueError:
                    continue

                if (best_score is None) or (score > best_score):
                    best_score = score
                    best_name = name
    except Exception:
        return None

    if best_name is None:
        return None
    return best_name, best_score

def short_name_from_full(name: str) -> str:
    """
    Convert a full tool name into a short name by taking the substring after the last '-'.
    Examples:
        "tool-face-0-gfpgan" -> "gfpgan"
        "0-img" -> "img"
    If no '-' exists, returns the original trimmed name.
    """
    name = name.strip()
    if '-' in name:
        return name.split('-')[-1]
    return name

def key_from_dirname(dirname: str) -> str:
    """
    Create the key used in output by taking the part of the top-level directory
    name before the first '-'.
    Example:
        "00000_00-250506_175126" -> "00000_00"
    If no '-' exists, return the original directory name.
    """
    if '-' in dirname:
        return dirname.split('-', 1)[0]
    return dirname

def main(root: str, out_file: str):
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        print(f"Error: {root} is not a directory.")
        return

    results = []
    # Iterate over first-level entries in sorted order for stable output
    for name in sorted(os.listdir(root)):
        subpath = os.path.join(root, name)
        if not os.path.isdir(subpath):
            continue
        key = key_from_dirname(name)
        result_path = find_result_file(subpath, filename="result_scores_faces.txt")
        if not result_path:
            # Mark missing result file
            results.append((key, None, "MISSING"))
            continue
        parsed = parse_scores_file(result_path)
        if parsed is None:
            # File present but no valid scores parsed
            results.append((key, None, "NO_VALID_SCORES"))
            continue
        best_name, best_score = parsed
        short = short_name_from_full(best_name)
        results.append((key, short, best_score))

    # Write final output
    try:
        with open(out_file, 'w', encoding='utf-8') as fout:
            for key, short, score in results:
                if short is None:
                    fout.write(f"{key}: {score}\n")
                else:
                    # By default write only short name (matches your example).
                    fout.write(f"{key}: {short}\n")
        print(f"Done. Processed {len(results)} subfolders. Results written to: {out_file}")
    except Exception as e:
        print(f"Failed to write output file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find best face restoration tool per folder")
    parser.add_argument("root", help="Root directory containing results subfolders")
    parser.add_argument("-o", "--out", default="face_tools.txt", help="Output file path (default: face_tools.txt)")
    args = parser.parse_args()
    main(args.root, args.out)
