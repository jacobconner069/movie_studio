import os

# Must run before `core` is imported anywhere: synthetic data, no network needed.
os.environ["MOVIES_OFFLINE"] = "1"
