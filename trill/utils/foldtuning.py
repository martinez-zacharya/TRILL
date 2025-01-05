import subprocess
from loguru import logger
import pandas as pd
import io
import os
import numpy as np
from trill.utils.fasta_files import remove_invalid_seqs_aa, truncate_seqs


def foldtune(args):

    for i in range(1, int(args.foldtune_rounds) + 1):
        logger.info(f'Foldtuning round {i}:')
        abspath = os.path.abspath(args.outdir)

        if i == 1:
            logger.info('Embedding input sequences with ESM2-650M')
            embed_cmd = f'trill {args.name}_foldtune_input {args.GPUs} --outdir {args.outdir} embed esm2_t33_650M {args.query} --avg'.split(' ')
            subprocess.run(embed_cmd)

        if i == 1:
            logger.info('Finetuning ProtGPT2 for 1 epoch')
            finetune_cmd = f'trill {args.name}_round{i} {args.GPUs} --outdir {abspath} finetune ProtGPT2 {args.query} --epochs 1 --batch_size {args.finetune_batch_size}'.split(' ')
            subprocess.run(finetune_cmd)
            seqkit_stats_cmd = f'seqkit stats -a -T {args.query}'.split(' ')
            result = subprocess.run(seqkit_stats_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            output = result.stdout


            df = pd.read_csv(io.StringIO(output), sep='\t')
            num_seqs = df.num_seqs.values
            median = df.Q2.values
        else:
            logger.info('Finetuning ProtGPT2 for 1 epoch')
            finetune_cmd = f'trill {args.name}_round{i} {args.GPUs} --outdir {abspath} finetune ProtGPT2 {output_fasta} --epochs 1 --batch_size {args.finetune_batch_size} --finetuned {abspath}/{args.name}_round{i-1}_ProtGPT2_1.pt'.split(' ')
            subprocess.run(finetune_cmd)



        logger.info('Generating sequences with finetuned ProtGPT2')

        curr_ckpt = os.path.join(abspath, f'{args.name}_round{i}_ProtGPT2_1.pt')
        logger.info(f"Generating with checkpoint: {curr_ckpt}")

        # Construct the generation command
        gen_cmd = [
            'trill',
            f"{args.name}_round{i}",
            args.GPUs,
            '--outdir', abspath,
            'lang_gen', 'ProtGPT2',
            '--finetuned', curr_ckpt,
            '--batch_size', args.lang_gen_batch_size,
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
        embed_cmd = f'trill {args.name}_round{i} {args.GPUs} --outdir {args.outdir} embed esm2_t33_650M {abspath}/cleaned_truncated_{args.name}_foldtune_generated_sequences_round{i}.fasta --avg'.split(' ')
        subprocess.run(embed_cmd)

        if i == 1:
            input_embs = os.path.join(abspath, f'{args.name}_foldtune_input_esm2_t33_650M_AVG.csv')
        else:
            input_embs = os.path.join(abspath, f'{args.name}_foldtune_most-distant_round{i-1}_esm2_t33_650M_AVG.csv')

        generated_embs = os.path.join(abspath, f'{args.name}_round{i}_esm2_t33_650M_AVG.csv')

        input_df = pd.read_csv(input_embs)
        test_df = pd.read_csv(generated_embs)

        # Extract embeddings (exclude the label column)
        test_embeddings = test_df.iloc[:, :-1].to_numpy()
        input_embeddings = input_df.iloc[:, :-1].to_numpy()
        labels = test_df.iloc[:, -1].to_numpy()  # Extract labels (last column)

        # Compute the 100 most distant indices on average
        most_distant_indices = compute_average_rank(input_embeddings, test_embeddings)

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

        logger.info('Folding generated sequences with ESMFold')
        fold_cmd = f'trill {args.name}_round{i} {args.GPUs} --outdir {abspath}/{args.name}_foldtune_generated_structs fold ESMFold {abspath}/{output_fasta} --batch_size {args.fold_batch_size}'.split(' ')
        subprocess.run(fold_cmd)

def compute_average_rank(input_embeddings, test_embeddings):
    # Initialize an array to store ranks
    rank_sums = np.zeros(test_embeddings.shape[0])

    for input_vec in input_embeddings:
        # Compute L1 distances for the current input vector
        distances = np.sum(np.abs(test_embeddings - input_vec), axis=1)
        # Rank the distances (smallest to largest)
        ranks = np.argsort(np.argsort(-distances))  # Use negative for descending order
        # Add ranks to rank_sums
        rank_sums += ranks

    # Compute average rank
    avg_ranks = rank_sums / len(input_embeddings)
    # Select the indices of the 100 highest average ranks
    most_distant_indices = np.argsort(avg_ranks)[-100:][::-1]
    return most_distant_indices



