import requests
import os
import subprocess
from loguru import logger

def download_file(url, download_path):
    """
    Download a file from a URL to a given path, chunk by chunk.
    """
    with requests.get(url, stream=True) as r:
        r.raise_for_status()  # Raises a HTTPError if the response status code is 4XX/5XX
        with open(download_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):  # 8KB chunks
                f.write(chunk)

def extract_tar_gz(tar_path, extract_to):
    """
    Extracts a .tar.gz file to a specified directory using the command-line tar utility.
    """
    # Ensure the target directory exists
    os.makedirs(extract_to, exist_ok=True)
    
    # Build the tar extraction command
    cmd = ["tar", "-xzf", tar_path, "-C", extract_to]
    
    # Execute the command
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Extracted {tar_path} successfully to {extract_to}.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error extracting {tar_path}: {e}")

def download_and_extract(download_url, extract_to):
    """
    Downloads a .tar.gz file from the given URL and extracts it to the specified path.
    """
    # Ensure the extract_to directory exists
    os.makedirs(extract_to, exist_ok=True)
    
    # Get the file name from the download_url
    file_name = download_url.split('/')[-1]
    download_path = os.path.join(extract_to, file_name)
    
    # Download the file
    logger.info(f"Downloading {file_name}...")
    download_file(download_url, download_path)
    logger.info(f"Downloaded {file_name} successfully.")
    
    # Extract the .tar.gz file
    logger.info(f"Extracting {file_name}...")
    extract_tar_gz(download_path, extract_to)
    logger.info(f"Extracted {file_name} successfully.")



