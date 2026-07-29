"""
A script to run to prepare the repository for usage.
"""

import subprocess

print("\nDownloading large files...")

subprocess.call("uv run python ./bepi-region-prediction/src/setup/init.py", shell=True)

print("\nRunnning setup scripts")

subprocess.call("uv run python ./bepi-region-prediction/src/determine_messenger_regions.py", shell=True)
subprocess.call("uv run python ./bepi-region-prediction/src/create_probability_maps.py", shell=True)
