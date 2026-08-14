"""
toolchain_merge.py

Merge face-restoration results (best_face.txt) into toolchain records
(WebPhoto-Test_toolchain_results.txt).

Usage:
    python toolchain_merge.py best_face.txt Results_toolchain.txt -o merged_toolchain.txt

Behavior:
- Parse "key: value" lines from both files.
- For each key present in the toolchain file:
    - if best_face has a method for the key and method != "img":
        - if toolchain already contains "face-restoration@..." -> replace it with new method
        - else insert "(face-restoration@METHOD)" into the first task position as described above
    - otherwise leave the toolchain value unchanged
- Write output preserving one line per key: "key: merged_value"
"""

import argparse
import re
from typing import Dict

# regex to find existing face-restoration@<method>
FACE_RE = re.compile(r'(face-restoration@)([A-Za-z0-9_\-]+)', flags=re.IGNORECASE)


def load_mapping(path: str) -> Dict[str, str]:
    """Load file with lines like 'key: value' into a dict. Ignores blank lines."""
    mapping = {}
    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.rstrip('\n')
            if not line.strip():
                continue
            if ':' not in line:
                continue
            key, val = line.split(':', 1)
            mapping[key.strip()] = val.lstrip()
    return mapping


def replace_existing_face(chain: str, method: str) -> str:
    """Replace existing face-restoration@... occurrences with the provided method."""
    return FACE_RE.sub(r'\1' + method, chain)


def find_super_resolution_index(chain: str) -> int:
    """
    Return the index of the substring 'super-resolution@' in chain in a case-insensitive way.
    Returns -1 if not found.
    """
    return chain.lower().find('super-resolution@')


def insert_after_super_resolution(chain: str, method: str) -> str:
    """
    Insert "(face-restoration@{method})" immediately after the full super-resolution@<method>
    token (i.e., after the method name, before following '-' / whitespace / punctuation / end).
    If no 'super-resolution@' found, return the original chain unchanged.
    """
    idx = find_super_resolution_index(chain)
    if idx == -1:
        return chain

    snippet = f"(face-restoration@{method})"
    start_after_at = idx + len('super-resolution@')
    # scan from start_after_at to find the end of the method name
    n = len(chain)
    delimiters = set(['-', ' ', '\t', '\n', '.', ',', ';', ':', '!', '?', ')', '('])
    end = start_after_at
    while end < n and chain[end] not in delimiters:
        end += 1
    # insert snippet at 'end' (before the delimiter or at end of string)
    return chain[:end] + snippet + chain[end:]


def insert_fallback(chain: str, method: str) -> str:
    """
    Fallback insertion when super-resolution is not present:
    - If there's a '-' before the first space, insert before that '-'
    - Else if there's a space, insert before the first space
    - Else insert before trailing punctuation (.,!?), or append at end
    """
    snippet = f"(face-restoration@{method})"
    if FACE_RE.search(chain):
        return replace_existing_face(chain, method)

    hyphen_idx = chain.find('-')
    space_idx = chain.find(' ')
    if hyphen_idx != -1 and (space_idx == -1 or hyphen_idx < space_idx):
        return chain[:hyphen_idx] + snippet + chain[hyphen_idx:]
    if space_idx != -1:
        return chain[:space_idx] + snippet + chain[space_idx:]
    m = re.search(r'([.!?])\s*$', chain)
    if m:
        idx = m.start(1)
        return chain[:idx] + snippet + chain[idx:]
    return chain + ' ' + snippet


def merge_mappings(best_map: Dict[str, str], toolchain_map: Dict[str, str]) -> Dict[str, str]:
    """
    For each key in toolchain_map, merge face method from best_map if applicable.
    """
    out = {}
    for key, chain in toolchain_map.items():
        method = best_map.get(key)
        if method is None:
            out[key] = chain
            continue
        method_clean = method.strip()
        if method_clean.lower() == 'img' or method_clean == '':
            out[key] = chain
            continue

        # If there's an existing face-restoration, replace it
        if FACE_RE.search(chain):
            merged = replace_existing_face(chain, method_clean)
            out[key] = merged
            continue

        # Try to insert after super-resolution if present
        merged = insert_after_super_resolution(chain, method_clean)
        if merged != chain:
            out[key] = merged
            continue

        # Fallback insertion
        merged = insert_fallback(chain, method_clean)
        out[key] = merged

    return out


def write_mapping(path: str, mapping: Dict[str, str]):
    """Write mapping to file preserving iteration order of mapping."""
    with open(path, 'w', encoding='utf-8') as f:
        for key, val in mapping.items():
            f.write(f"{key}: {val}\n")


def main():
    parser = argparse.ArgumentParser(description="Merge best_face into toolchain results (insert after super-resolution)")
    parser.add_argument("best_face", help="Path to face_tools.txt")
    parser.add_argument("toolchain", help="Path to Results_toolchain.txt")
    parser.add_argument("-o", "--out", default="merged_toolchain.txt", help="Output file path")
    args = parser.parse_args()

    best_map = load_mapping(args.best_face)
    toolchain_map = load_mapping(args.toolchain)

    merged = merge_mappings(best_map, toolchain_map)
    write_mapping(args.out, merged)
    print(f"Merged {len(merged)} entries. Output written to: {args.out}")


if __name__ == "__main__":
    main()
