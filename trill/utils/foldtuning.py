import subprocess
from loguru import logger
import pandas as pd
import io
import os
import numpy as np
import shutil
from trill.utils.fasta_files import remove_invalid_seqs_aa, truncate_seqs
from trill.utils.foldseek_utils import run_foldseek_databases


def foldtune(args):
    for i in range(1, int(args.foldtune_rounds) + 1):
        logger.info(f'Foldtuning round {i}:')
        abspath = os.path.abspath(args.outdir)
        if args.fast_folding and i == 1:
            # Finding ProstT5 weights
            logger.info('Finding ProstT5 weights and downloading if missing from trill cache')
            prostt5_weights_path = run_foldseek_databases(args)


        if i == 1:
            logger.info('Embedding input sequences with ESM2-650M')
            embed_cmd = f'trill {args.name}_foldtune_input {args.GPUs} --RNG_seed {args.RNG_seed} --outdir {args.outdir} embed esm2_t33_650M {args.query} --avg'.split(' ')
            subprocess.run(embed_cmd)
            if not args.fast_folding:
                logger.info('Folding input sequences with ESMFold')
                fold_cmd = f'trill {args.name}_foldtune_input {args.GPUs} --RNG_seed {args.RNG_seed} --outdir {abspath}/{args.name}_foldtune_input_structs fold ESMFold {abspath}/{args.query} --batch_size {args.fold_batch_size}'.split(' ')
                subprocess.run(fold_cmd)

        if i == 1:
            logger.info('Finetuning ProtGPT2 for 1 epoch')
            if args.finetune_strategy:
                finetune_cmd = f'trill {args.name}_round{i} {args.GPUs} --RNG_seed {args.RNG_seed} --outdir {abspath} finetune ProtGPT2 {args.query} --epochs 1 --batch_size {args.finetune_batch_size} --strategy {args.finetune_strategy}'.split(' ')
            else:
                finetune_cmd = f'trill {args.name}_round{i} {args.GPUs} --RNG_seed {args.RNG_seed} --outdir {abspath} finetune ProtGPT2 {args.query} --epochs 1 --batch_size {args.finetune_batch_size}'.split(' ')
            subprocess.run(finetune_cmd)
            seqkit_stats_cmd = f'seqkit stats -a -T {args.query}'.split(' ')
            result = subprocess.run(seqkit_stats_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            output = result.stdout


            df = pd.read_csv(io.StringIO(output), sep='\t')
            num_seqs = df.num_seqs.values
            median = df.Q2.values
        else:
            logger.info('Finetuning ProtGPT2 for 1 epoch')
            if args.finetune_strategy:
                finetune_cmd = f'trill {args.name}_round{i} {args.GPUs} --RNG_seed {args.RNG_seed} --outdir {abspath} finetune ProtGPT2 {output_fasta} --epochs 1 --batch_size {args.finetune_batch_size} --finetuned {abspath}/{args.name}_round{i-1}_ProtGPT2_1_fp32.pt --strategy {args.finetune_strategy}'.split(' ')
            else:
                finetune_cmd = f'trill {args.name}_round{i} {args.GPUs} --RNG_seed {args.RNG_seed} --outdir {abspath} finetune ProtGPT2 {output_fasta} --epochs 1 --batch_size {args.finetune_batch_size} --finetuned {abspath}/{args.name}_round{i-1}_ProtGPT2_1.pt'.split(' ')
            subprocess.run(finetune_cmd)



        logger.info('Generating sequences with finetuned ProtGPT2')

        if args.finetune_strategy:
            curr_ckpt = os.path.join(abspath, f'{args.name}_round{i}_ProtGPT2_1_fp32.pt')
        else:
            curr_ckpt = os.path.join(abspath, f'{args.name}_round{i}_ProtGPT2_1.pt')
        logger.info(f"Generating with checkpoint: {curr_ckpt}")

        # Construct the generation command
        gen_cmd = [
            'trill',
            f"{args.name}_round{i}",
            args.GPUs,
            '--outdir', abspath,
            '--RNG_seed', args.RNG_seed,
            'lang_gen', 'ProtGPT2',
            '--finetuned', curr_ckpt,
            '--batch_size', str(args.lang_gen_batch_size),
            '--max_length', str(int(median[0])),
            '--temp', '1',
            '--seed_seq', '',
            '--num_return_sequences', '1000'
        ]
        subprocess.run(gen_cmd)
        
        gen_fasta_files = []

        # Loop through all files in the output directory
        for file in os.listdir(abspath):
            # Check if the file matches the naming pattern
            if file.startswith(f"{args.name}_round{i}") and file.endswith(".fasta"):
                gen_fasta_files.append(file)

        with open(f'{abspath}/{args.name}_foldtune_generated_sequences_round{i}.fasta', 'w+') as outfile:
            for fasta_file in gen_fasta_files:
                with open(fasta_file, 'r') as infile:
                    for line in infile:
                        # Write non-empty lines to the output file
                        if line.strip():  # Avoid empty lines
                            outfile.write(line)
                    # Add a newline at the end of each file to separate sequences properly
                    outfile.write('\n')


        truncate_seqs(f'{abspath}/{args.name}_foldtune_generated_sequences_round{i}.fasta', int(median[0]))


        remove_invalid_seqs_aa(f'{abspath}/truncated_{args.name}_foldtune_generated_sequences_round{i}.fasta')

        logger.info('Embedding generated sequences with ESM2-650M')
        embed_cmd = f'trill {args.name}_round{i} {args.GPUs} --RNG_seed {args.RNG_seed} --outdir {args.outdir} embed esm2_t33_650M {abspath}/cleaned_truncated_{args.name}_foldtune_generated_sequences_round{i}.fasta --avg'.split(' ')
        subprocess.run(embed_cmd)

        if i == 1:
            input_embs = os.path.join(abspath, f'{args.name}_foldtune_input_esm2_t33_650M_AVG.csv')
        else:
            input_embs = os.path.join(abspath, f'{args.name}_foldtune_most-distant_round{i-1}_esm2_t33_650M_AVG.csv')

        generated_embs = os.path.join(abspath, f'{args.name}_round{i}_esm2_t33_650M_AVG.csv')

        input_df = pd.read_csv(input_embs)
        test_df = pd.read_csv(generated_embs)

        if not args.fast_folding:
            logger.info('Folding generated sequences with ESMFold')
            fold_cmd = f'trill {args.name}_round{i} {args.GPUs} --RNG_seed {args.RNG_seed} --outdir {abspath}/{args.name}_foldtune_generated_structs_round{i} fold ESMFold {abspath}/cleaned_truncated_{args.name}_foldtune_generated_sequences_round{i}.fasta --batch_size {args.fold_batch_size}'.split(' ')
            subprocess.run(fold_cmd)

        # Ranking generated structures by STRUCTURAL similarity to inputs
        logger.info('Assessing structural similarity of generated structures to inputs with foldseek and TM-Align')
        if args.fast_folding:
            foldseek_cmd = f'foldseek createdb {abspath}/cleaned_truncated_{args.name}_foldtune_generated_sequences_round{i}.fasta {abspath}/{args.name}_foldtune_generated_sequences_round{i}_db --prostt5-model {prostt5_weights_path}'
            if int(args.GPUs) != 0:
                foldseek_cmd += ' --gpu 1'
            subprocess.run(foldseek_cmd, shell=True, check=True)

            if i == 1:
                foldseek_cmd = f'foldseek createdb {abspath}/{args.query} {abspath}/{args.name}_foldtune_input_db --prostt5-model {prostt5_weights_path}'
                if int(args.GPUs) != 0:
                    foldseek_cmd += ' --gpu 1'
                subprocess.run(foldseek_cmd, shell=True, check=True)
            
            foldseek_cmd = (
            f'foldseek easy-search {abspath}/{args.name}_foldtune_generated_sequences_round{i}_db '
            f'{abspath}/{args.name}_foldtune_input_db '
            f'{abspath}/{args.name}_foldtune_foldseek_round{i}.tsv tmp_round{i} '
            )
        else:
            foldseek_cmd = (
                f'foldseek easy-search {abspath}/{args.name}_foldtune_generated_structs_round{i}/ '
                f'{abspath}/{args.name}_foldtune_input_structs/ '
                f'{abspath}/{args.name}_foldtune_foldseek_round{i}.tsv tmp_round{i} '
                '--alignment-type 1 '
                '--format-output "query,target,fident,bits,alntmscore" '
            )

        subprocess.run(foldseek_cmd, shell=True, check=True)

        score_df = highest_avg_score_by_query(f'{abspath}/{args.name}_foldtune_foldseek_round{i}.tsv', f'{abspath}/cleaned_truncated_{args.name}_foldtune_generated_sequences_round{i}.fasta', args)

        # Extract embeddings (exclude the label column)--alignment-type 1
        test_embeddings = test_df.iloc[:, :-1].to_numpy()
        input_embeddings = input_df.iloc[:, :-1].to_numpy()
        labels = test_df.iloc[:, -1].to_numpy()  # Extract labels (last column)

        # Compute the 100 most distant indices on average
        most_distant_indices = compute_average_rank(input_embeddings, test_embeddings, score_df)

        # Extract the most distant embeddings and labels
        most_distant_embeddings = test_df.iloc[most_distant_indices]
        most_distant_labels = [labels[idx] for idx in most_distant_indices]

        # Save the most distant embeddings as a new CSV
        most_distant_embeddings.to_csv(os.path.join(abspath, f'{args.name}_foldtune_most-distant_round{i}_esm2_t33_650M_AVG.csv'), index=False)

        # Save the labels to a text file for SeqKit
        labels_file = os.path.join(abspath, f"{args.name}_foldtune_most-distant_round{i}_labels.txt")
        with open(labels_file, "w+") as f:
            f.write("\n".join(most_distant_labels))

        seqkit_command = [
            "seqkit", "grep",
            "--pattern-file", labels_file,
            f"{abspath}/cleaned_truncated_{args.name}_foldtune_generated_sequences_round{i}.fasta"
        ]

        # Define the output file path
        output_fasta = os.path.join(abspath, f"{args.name}_foldtune_most-distant_round{i}.fasta")

        # Run the command and redirect the output to the desired file
        with open(output_fasta, "w+") as output_file:
            subprocess.run(seqkit_command, stdout=output_file, check=True)

        if args.finetune_strategy:
            if os.path.isdir(f'{abspath}/{args.name}_round{i}_ProtGPT2_1.pt'):
                shutil.rmtree(f'{abspath}/{args.name}_round{i}_ProtGPT2_1.pt')


def compute_average_rank(input_embeddings, test_embeddings, alntmscore_df):
    # Initialize an array to store ranks
    if test_embeddings.shape[0] != len(alntmscore_df):
        raise ValueError("Mismatch between test_embeddings rows and DataFrame entries")

    rank_sums = np.zeros(test_embeddings.shape[0])

    for input_vec in input_embeddings:
        distances = np.sum(np.abs(test_embeddings - input_vec), axis=1)
        ranks = np.argsort(np.argsort(-distances))  # descending order
        rank_sums += ranks

    avg_ranks = rank_sums / len(input_embeddings)

    # Add avg_rank to DataFrame
    alntmscore_df = alntmscore_df.copy()
    alntmscore_df['avg_rank'] = avg_ranks
    # Filter by score threshold
    filtered_df = alntmscore_df[alntmscore_df['avg_score'] > 0]

    # Select top N by avg_rank
    top_indices = filtered_df.sort_values(by='avg_rank', ascending=False).head(100).index.to_numpy()
    return top_indices

def get_fasta_headers(fasta_path):
    headers = set()
    with open(fasta_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                headers.add(line[1:].strip().split()[0])
    return headers

def highest_avg_score_by_query(tsv_path, fasta_path, args):
    # Determine column names based on mode
    if args.fast_folding:
        col_names = ['query','target','fident','alnlen','mismatch','gapopen','qstart','qend','tstart','tend','evalue','bits']
        score_column = 'bits'
    else:
        col_names = ['query', 'target', 'fident', 'bits', 'alntmscore']
        score_column = 'alntmscore'

    # Load Foldseek TSV
    df = pd.read_csv(tsv_path, sep='\t', names=col_names)
    # Compute average score for each query
    avg_scores_df = df.groupby('query')[score_column].mean().reset_index()
    avg_scores_df.rename(columns={score_column: f'avg_score'}, inplace=True)

    # Get all headers from the FASTA file
    fasta_headers = get_fasta_headers(fasta_path)

    # Identify headers missing from Foldseek results
    found_headers = set(avg_scores_df['query'])
    missing_headers = fasta_headers - found_headers

    # Create DataFrame for missing headers with score 0.0
    missing_df = pd.DataFrame({
        'query': list(missing_headers),
        f'avg_score': 0.0
    })

    # Combine and sort the results
    combined_df = pd.concat([avg_scores_df, missing_df], ignore_index=True)
    return combined_df.sort_values(by='avg_score').reset_index(drop=True)

