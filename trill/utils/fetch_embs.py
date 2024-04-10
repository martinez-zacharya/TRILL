import os

import h5py
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from loguru import logger

dataset_ids = {
    'UniProtKB': 'uniprot_sprot',
    'A.thaliana': 'UP000006548_3702',
    'C.elegans': 'UP000001940_6239',
    'E.coli': 'UP000000625_83333',
    'H.sapiens': 'UP000005640_9606',
    'M.musculus': 'UP000000589_10090',
    'R.norvegicus': 'UP000002494_10116',
    'SARS-CoV-2': 'UP000464024_2697049'
}

def convert_embeddings_to_csv(h5_file_path, csv_file_path):
    # Open the HDF5 file
    with h5py.File(h5_file_path, 'r') as h5_file:
        emb_dict = {}
        data_list = []
        for prot in h5_file.keys():
            h5_data = h5_file[prot]
            np_data = np.array(h5_data)
            emb_dict[prot] = np_data
        for label, array in emb_dict.items():
            item_dict = {str(i): value for i, value in enumerate(array)}
            item_dict['Label'] = label
            data_list.append(item_dict)


    df = pd.DataFrame(data_list)

    df = df[[col for col in df if col != 'Label'] + ['Label']]
    df.to_csv(csv_file_path, index=False)
    logger.info(f'CSV file saved to {csv_file_path}')

def download_embeddings(args):
    dataset_id = dataset_ids[args.uniprotDB]
    if args.rep == 'per_AA':
        save_path = os.path.join(args.outdir, f'{args.uniprotDB}_ProtT5-XL_perAA.h5')
        base_url = "https://ftp.ebi.ac.uk/pub/contrib/UniProt/embeddings/current_release/"
        file_url = f"{base_url}{dataset_id}/per-residue.h5"
    elif args.rep == 'avg':
        base_url = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/embeddings/"
        save_path = os.path.join(args.outdir, f'{args.uniprotDB}_ProtT5-XL_AVG.h5')
        file_url = f"{base_url}{dataset_id}/per-protein.h5"
    
    response = requests.get(file_url, stream=True)
    response.raise_for_status()  # Check if the request was successful
    
    # Get the total file size
    file_size = int(response.headers.get('Content-Length', 0))
    
    # Set up the progress bar
    progress_bar = tqdm(total=file_size, unit='B', unit_scale=True, unit_divisor=1024)
    
    with open(save_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
            # Update the progress bar
            progress_bar.update(len(chunk))
    
    # Close the progress bar
    progress_bar.close()
    
    if file_size != 0 and progress_bar.n != file_size:
        logger.error("ERROR, something went wrong")
    
    logger.info(f'Embeddings file downloaded to {save_path}')
    return save_path

