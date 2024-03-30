import os

home_dir = os.path.expanduser("~")
cache_dir = os.path.join(home_dir, ".trill_cache")
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
