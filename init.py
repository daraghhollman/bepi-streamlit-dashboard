"""
A script to run to prepare the repository for usage.
"""

import subprocess


def run(cmd):
    print(f"\nRunning: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


print("\nDownloading large files...")

run("uv run python ./bepi-region-prediction/src/setup/init.py")

print("\nRunning setup scripts")

run("uv run python ./bepi-region-prediction/src/determine_messenger_regions.py")
run("uv run python ./bepi-region-prediction/src/create_probability_maps.py")

print("\nSetup complete")
