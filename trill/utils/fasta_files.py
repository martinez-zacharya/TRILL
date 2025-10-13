import os
import subprocess
# from trill.utils.externals import ensure_bin

# SEQKIT  = ensure_bin("seqkit")

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
    result = subprocess.run(seqkit_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
    
def truncate_seqs(query, trunc_len):
    # Extract base directory and filename
    base_dir = os.path.dirname(query)
    base_filename = os.path.basename(query)
    output_file = os.path.join(base_dir, f"truncated_{base_filename}")
    
    # Run the seqkit truncate command
    seqkit_truncate_cmd = f'seqkit subseq -r 1:{trunc_len} {query} -o {output_file}'.split(' ')
    subprocess.run(seqkit_truncate_cmd, check=True)
