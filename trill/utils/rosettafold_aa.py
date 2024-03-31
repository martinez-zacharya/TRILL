import os

import requests


def rfaa_setup(args, cache_dir):
    base_url = "http://files.ipd.uw.edu/pub/RF-All-Atom/weights/RFAA_paper_weights.pt"
    weights_path = f'{cache_dir}/RFAA/RFAA_paper_weights.pt'
    if not os.path.exists(weights_path):
        r = requests.get(base_url)
        print('Downloading RFAA weights...')
        with open(weights_path, "wb") as file:
            file.write(r.content)
        print('Finished downloading RFAA weights!')
