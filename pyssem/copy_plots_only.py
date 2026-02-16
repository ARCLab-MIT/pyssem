#!/usr/bin/env python3
"""
Copy only plot files from grid_search to grid_search_upload, preserving folder structure.
"""
import shutil
from pathlib import Path

# Configuration
SOURCE_DIR = Path("grid_search")
TARGET_DIR = Path("grid_search_upload")

# File extensions to copy (plot/image files)
PLOT_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.pdf', '.svg', '.gif', '.bmp', '.tiff', '.tif'}


def copy_plots_only(source: Path, target: Path):
    """
    Recursively copy only plot files from source to target, preserving directory structure.
    """
    if not source.exists():
        print(f"ERROR: Source directory not found: {source}")
        return
    
    # Create target directory
    target.mkdir(parents=True, exist_ok=True)
    
    # Counters
    files_copied = 0
    dirs_created = 0
    
    # Walk through source directory
    for source_path in source.rglob('*'):
        # Skip directories
        if source_path.is_dir():
            continue
        
        # Check if it's a plot file
        if source_path.suffix.lower() in PLOT_EXTENSIONS:
            # Get relative path from source
            rel_path = source_path.relative_to(source)
            target_path = target / rel_path
            
            # Create parent directories if needed
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if not target_path.parent.exists():
                dirs_created += 1
            
            # Copy the file
            try:
                shutil.copy2(source_path, target_path)
                files_copied += 1
                if files_copied % 10 == 0:
                    print(f"Copied {files_copied} files...")
            except Exception as e:
                print(f"WARNING: Failed to copy {source_path}: {e}")
    
    print(f"\n✅ Completed!")
    print(f"   Directories created: {dirs_created}")
    print(f"   Plot files copied: {files_copied}")
    print(f"   Target directory: {target}")


def main():
    """Main function."""
    print(f"\n=== Copying plots from {SOURCE_DIR} to {TARGET_DIR} ===")
    print(f"Source: {SOURCE_DIR.absolute()}")
    print(f"Target: {TARGET_DIR.absolute()}")
    print(f"Extensions: {', '.join(sorted(PLOT_EXTENSIONS))}")
    print("=" * 60 + "\n")
    
    copy_plots_only(SOURCE_DIR, TARGET_DIR)
    print("\nDone!")


if __name__ == "__main__":
    main()
