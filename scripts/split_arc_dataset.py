#!/usr/bin/env python3
"""
Split ARC AGI training dataset into ARC-1 and ARC-2 subsets.

ARC-1: Original 400 training tasks from 2019
ARC-2: Additional 600 training tasks added in 2024

Usage:
    python scripts/split_arc_dataset.py

Outputs:
    data/arc1_training.json - 400 ARC-1 tasks
    data/arc2_training.json - 600 ARC-2 tasks
"""

import json
import urllib.request
from pathlib import Path


def fetch_arc1_task_ids() -> set[str]:
    """
    Fetch the list of original ARC-1 training task IDs from GitHub.

    Returns:
        Set of 400 task IDs (e.g., {'007bbfb7', '00d62c1b', ...})
    """
    url = "https://api.github.com/repos/fchollet/ARC-AGI/contents/data/training"

    print(f"Fetching ARC-1 task IDs from GitHub...")
    with urllib.request.urlopen(url) as response:
        files = json.loads(response.read())

    # Extract task IDs (filenames without .json extension)
    task_ids = {
        f['name'].replace('.json', '')
        for f in files
        if f['name'].endswith('.json')
    }

    print(f"Found {len(task_ids)} ARC-1 task IDs")
    return task_ids


def split_dataset(input_path: str, arc1_ids: set[str]) -> tuple[dict, dict]:
    """
    Split training dataset into ARC-1 and ARC-2 subsets.

    Args:
        input_path: Path to full training dataset JSON
        arc1_ids: Set of ARC-1 task IDs

    Returns:
        Tuple of (arc1_dataset, arc2_dataset) dicts
    """
    print(f"\nLoading full training dataset from {input_path}...")
    with open(input_path, 'r') as f:
        full_dataset = json.load(f)

    print(f"Total tasks in dataset: {len(full_dataset)}")

    # Split into two datasets
    arc1_dataset = {}
    arc2_dataset = {}

    for task_id, task_data in full_dataset.items():
        if task_id in arc1_ids:
            arc1_dataset[task_id] = task_data
        else:
            arc2_dataset[task_id] = task_data

    print(f"ARC-1 tasks: {len(arc1_dataset)}")
    print(f"ARC-2 tasks: {len(arc2_dataset)}")

    return arc1_dataset, arc2_dataset


def save_datasets(arc1_data: dict, arc2_data: dict, output_dir: str = "data"):
    """
    Save ARC-1 and ARC-2 datasets to separate JSON files.

    Args:
        arc1_data: ARC-1 task dataset
        arc2_data: ARC-2 task dataset
        output_dir: Output directory (default: "data")
    """
    output_path = Path(output_dir)

    arc1_path = output_path / "arc1_training.json"
    arc2_path = output_path / "arc2_training.json"

    print(f"\nSaving datasets...")
    print(f"  ARC-1 → {arc1_path}")
    with open(arc1_path, 'w') as f:
        json.dump(arc1_data, f, indent=2)

    print(f"  ARC-2 → {arc2_path}")
    with open(arc2_path, 'w') as f:
        json.dump(arc2_data, f, indent=2)

    print("\n✓ Dataset split complete!")


def main():
    """Main entry point."""
    # Fetch ARC-1 task IDs from GitHub
    arc1_ids = fetch_arc1_task_ids()

    # Split the training dataset
    arc1_data, arc2_data = split_dataset(
        "data/arc-agi_training_challenges.json",
        arc1_ids
    )

    # Verify the split
    expected_arc1 = 400
    expected_arc2 = 600

    if len(arc1_data) != expected_arc1:
        print(f"\n⚠️  WARNING: Expected {expected_arc1} ARC-1 tasks, got {len(arc1_data)}")

    if len(arc2_data) != expected_arc2:
        print(f"⚠️  WARNING: Expected {expected_arc2} ARC-2 tasks, got {len(arc2_data)}")

    # Save to files
    save_datasets(arc1_data, arc2_data)


if __name__ == "__main__":
    main()
