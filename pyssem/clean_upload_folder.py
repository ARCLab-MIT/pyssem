#!/usr/bin/env python3
"""
Remove collision, heatmap, and launch files from grid_search_upload.
"""
from pathlib import Path

# Configuration
TARGET_DIR = Path("grid_search_upload")

# Patterns to match files/directories to remove
REMOVE_PATTERNS = [
    "*collision*",
    "*heatmap*",
    "*launch*",
    "heatmaps",  # directory name
]


def remove_files_by_pattern(directory: Path, patterns: list):
    """
    Remove files and directories matching the given patterns.
    """
    if not directory.exists():
        print(f"ERROR: Directory not found: {directory}")
        return
    
    files_removed = 0
    dirs_removed = 0
    
    # First, remove directories
    for pattern in patterns:
        for path in directory.rglob(pattern):
            if path.is_dir():
                try:
                    import shutil
                    shutil.rmtree(path)
                    dirs_removed += 1
                    print(f"Removed directory: {path.relative_to(directory)}")
                except Exception as e:
                    print(f"WARNING: Failed to remove directory {path}: {e}")
    
    # Then, remove files
    for pattern in patterns:
        for path in directory.rglob(pattern):
            if path.is_file():
                try:
                    path.unlink()
                    files_removed += 1
                    if files_removed % 50 == 0:
                        print(f"Removed {files_removed} files...")
                except Exception as e:
                    print(f"WARNING: Failed to remove file {path}: {e}")
    
    print(f"\n✅ Completed!")
    print(f"   Directories removed: {dirs_removed}")
    print(f"   Files removed: {files_removed}")


def main():
    """Main function."""
    print(f"\n=== Cleaning {TARGET_DIR} ===")
    print(f"Removing files/directories matching:")
    for pattern in REMOVE_PATTERNS:
        print(f"  - {pattern}")
    print("=" * 60 + "\n")
    
    remove_files_by_pattern(TARGET_DIR, REMOVE_PATTERNS)
    print("\nDone!")


if __name__ == "__main__":
    main()
