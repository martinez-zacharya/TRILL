import argparse
import os
import pandas as pd

def generate_class_key_csv(args):
    all_headers = []
    all_labels = []
    
    # If directory is provided
    if args.dir:
        for filename in os.listdir(args.dir):
            if filename.endswith('.fasta'):
                class_label = os.path.splitext(filename)[0]
                
                with open(os.path.join(args.dir, filename), 'r') as fasta_file:
                    for line in fasta_file:
                        line = line.strip()
                        if line.startswith('>'):
                            all_headers.append(line[1:])
                            all_labels.append(class_label)
    
    # If text file with paths is provided
    elif args.fasta_paths_txt:
        with open(args.fasta_paths_txt, 'r') as txt_file:
            for path in txt_file:
                path = path.strip()
                if not path:  # Skip empty or whitespace-only lines
                    continue
                
                class_label = os.path.splitext(os.path.basename(path))[0]
                
                if not os.path.exists(path):
                    print(f"File {path} does not exist.")
                    continue
                
                with open(path, 'r') as fasta_file:
                    for line in fasta_file:
                        line = line.strip()
                        if line.startswith('>'):
                            all_headers.append(line[1:])
                            all_labels.append(class_label)
    else:
        print('prepare_class_key requires either a path to a directory of fastas or a text file of fasta paths!')
        raise RuntimeError
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame({
        'Label': all_headers,
        'Class': all_labels
    })
    outpath = os.path.join(args.outdir, f'{args.name}_class_key.csv')
    df.to_csv(outpath, index=False)
    print(f"Class key CSV generated and saved as '{outpath}'.")