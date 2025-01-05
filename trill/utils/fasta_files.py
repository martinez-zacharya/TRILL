import os
import subprocess

def remove_invalid_seqs_aa(query):
    # Extract base directory and filename
    base_dir = os.path.dirname(query)
    base_filename = os.path.basename(query)
    output_file = os.path.join(base_dir, f"cleaned_{base_filename}")
    
    # Run the seqkit command
    seqkit_cmd = [
        "seqkit", "grep",
        "-s", "-v", "-r",
        "-p", "[BJOXUZ]",
        query,
        "-o", output_file,
    ]
    result = subprocess.run(seqkit_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
def truncate_seqs(query, trunc_len):
    # Extract base directory and filename
    base_dir = os.path.dirname(query)
    base_filename = os.path.basename(query)
    output_file = os.path.join(base_dir, f"truncated_{base_filename}")
    
    # Run the seqkit truncate command
    seqkit_truncate_cmd = f'seqkit subseq --update-faidx -r 1:{trunc_len} {query} -o {output_file}'.split(' ')
    subprocess.run(seqkit_truncate_cmd)
