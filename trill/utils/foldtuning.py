import subprocess
from loguru import logger
import pandas as pd
import io
import os
import numpy as np
import shutil
from trill.utils.fasta_files import remove_invalid_seqs_aa, truncate_seqs
from trill.utils.foldseek_utils import run_foldseek_databases
from trill.utils.ensure_foldseek import gpu_build_installed, should_use_gpu_build


BROKEN_PROGEN2_MODELS = ('progen2-large', 'progen2-BFD90')


def resolve_foldtune_generator(args):
    generator = getattr(args, 'foldtune_generator', 'protgpt2')
    if generator == 'progen2':
        gen_model = args.progen2_model
        if gen_model in BROKEN_PROGEN2_MODELS:
            raise ValueError(
                f'{gen_model} has a broken HuggingFace config. Use '
                'progen2-small, progen2-medium, progen2-oas, or progen2-xlarge.')
        return gen_model, True, '', False
    if generator == 'zymctrl':
        if not getattr(args, 'ctrl_tag', None):
            raise ValueError(
                'Foldtuning with --foldtune_generator Zymctrl requires --ctrl_tag '
                '(an EC number, e.g. 4.2.1.1) matching every enzyme in the input fasta.')
        return 'ZymCTRL', False, f' --ctrl_tag {args.ctrl_tag}', True
    return 'ProtGPT2', False, '', True


def foldtune(args):
    if "COLAB_GPU" in os.environ or "COLAB_JUPYTER_TOKEN" in os.environ:
        pixi_path = os.path.expanduser("~/.pixi/bin/pixi")
        ensure_foldseek_cmd = [pixi_path, "run", "ensure-foldseek"]
    else:
        ensure_foldseek_cmd = f'pixi run ensure-foldseek'.split(' ')
    ensure_env = os.environ.copy()
    if int(args.GPUs) == 0:
        ensure_env["TRILL_FOLDSEEK_GPU"] = "0"
    subprocess.run(ensure_foldseek_cmd, env=ensure_env)

    gen_model, is_progen2, ctrl_flag, uses_fp32_ckpt = resolve_foldtune_generator(args)

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
            subprocess.run(embed_cmd, check=True)
            if not args.fast_folding:
                logger.info('Folding input sequences with ESMFold')
                if not os.path.isabs(args.query):
                    fold_cmd = f'trill {args.name}_foldtune_input {args.GPUs} --RNG_seed {args.RNG_seed} --outdir {abspath}/{args.name}_foldtune_input_structs fold ESMFold {abspath}/{args.query} --batch_size {args.fold_batch_size}'.split(' ')
                else:
                    fold_cmd = f'trill {args.name}_foldtune_input {args.GPUs} --RNG_seed {args.RNG_seed} --outdir {abspath}/{args.name}_foldtune_input_structs fold ESMFold {args.query} --batch_size {args.fold_batch_size}'.split(' ')
                subprocess.run(fold_cmd, check=True)

        if i == 1:
            logger.info(f'Finetuning {gen_model} for 1 epoch')
            if args.finetune_strategy:
                finetune_cmd = f'trill {args.name}_round{i} {args.GPUs} --RNG_seed {args.RNG_seed} --outdir {abspath} finetune {gen_model} {args.query} --epochs 1 --batch_size {args.finetune_batch_size}{ctrl_flag} --strategy {args.finetune_strategy}'.split(' ')
            else:
                finetune_cmd = f'trill {args.name}_round{i} {args.GPUs} --RNG_seed {args.RNG_seed} --outdir {abspath} finetune {gen_model} {args.query} --epochs 1 --batch_size {args.finetune_batch_size}{ctrl_flag}'.split(' ')
            subprocess.run(finetune_cmd, check=True)
            seqkit_stats_cmd = f'seqkit stats -a -T {args.query}'.split(' ')
            result = subprocess.run(seqkit_stats_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            output = result.stdout


            df = pd.read_csv(io.StringIO(output), sep='\t')
            num_seqs = df.num_seqs.values
            median = df.Q2.values
        else:
            logger.info(f'Finetuning {gen_model} for 1 epoch')
            if args.finetune_strategy and uses_fp32_ckpt:
                prev_ckpt = f'{abspath}/{args.name}_round{i-1}_{gen_model}_1_fp32.pt'
                finetune_cmd = f'trill {args.name}_round{i} {args.GPUs} --RNG_seed {args.RNG_seed} --outdir {abspath} finetune {gen_model} {output_fasta} --epochs 1 --batch_size {args.finetune_batch_size} --finetuned {prev_ckpt}{ctrl_flag} --strategy {args.finetune_strategy}'.split(' ')
            else:
                prev_ckpt = f'{abspath}/{args.name}_round{i-1}_{gen_model}_1.pt'
                strat = f' --strategy {args.finetune_strategy}' if args.finetune_strategy else ''
                finetune_cmd = f'trill {args.name}_round{i} {args.GPUs} --RNG_seed {args.RNG_seed} --outdir {abspath} finetune {gen_model} {output_fasta} --epochs 1 --batch_size {args.finetune_batch_size} --finetuned {prev_ckpt}{ctrl_flag}{strat}'.split(' ')
            subprocess.run(finetune_cmd, check=True)



        logger.info(f'Generating sequences with finetuned {gen_model}')

        if args.finetune_strategy and uses_fp32_ckpt:
            curr_ckpt = os.path.join(abspath, f'{args.name}_round{i}_{gen_model}_1_fp32.pt')
        else:
            curr_ckpt = os.path.join(abspath, f'{args.name}_round{i}_{gen_model}_1.pt')
        logger.info(f"Generating with checkpoint: {curr_ckpt}")

        # Construct the generation command
        gen_cmd = [
            'trill',
            f"{args.name}_round{i}",
            args.GPUs,
            '--outdir', abspath,
            '--RNG_seed', str(args.RNG_seed),
            'lang_gen', gen_model,
            '--finetuned', curr_ckpt,
            '--batch_size', str(args.lang_gen_batch_size),
            '--max_length', str(int(median[0])),
            '--temp', '1',
            '--seed_seq', '',
            '--num_return_sequences', f'{int(args.num_to_generate_per_round)}'
            ]

        if args.foldtune_generator == 'zymctrl':
            gen_cmd += ['--ctrl_tag', str(args.ctrl_tag)]
        subprocess.run(gen_cmd, check=True)
        
        gen_fasta_files = []

        for file in os.listdir(abspath):
            if file.startswith(f"{args.name}_round{i}") and file.endswith(".fasta"):
                gen_fasta_files.append(file)

        with open(f'{abspath}/{args.name}_foldtune_generated_sequences_round{i}.fasta', 'w+') as outfile:
            for fasta_file in gen_fasta_files:
                with open(os.path.join(abspath, fasta_file), 'r') as infile:
                    for line in infile:
                        if line.strip():
                            outfile.write(line)
                    outfile.write('\n')


        truncate_seqs(f'{abspath}/{args.name}_foldtune_generated_sequences_round{i}.fasta', int(median[0]))


        remove_invalid_seqs_aa(f'{abspath}/truncated_{args.name}_foldtune_generated_sequences_round{i}.fasta')

        logger.info('Embedding generated sequences with ESM2-650M')
        embed_cmd = f'trill {args.name}_round{i} {args.GPUs} --RNG_seed {args.RNG_seed} --outdir {args.outdir} embed esm2_t33_650M {abspath}/cleaned_truncated_{args.name}_foldtune_generated_sequences_round{i}.fasta --avg'.split(' ')
        subprocess.run(embed_cmd, check=True)

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
            subprocess.run(fold_cmd, check=True)

        # Ranking generated structures by STRUCTURAL similarity to inputs
        logger.info('Assessing structural similarity of generated structures to inputs with foldseek and TM-Align')
        if args.fast_folding:
            gpu_flag = ' --gpu 1' if (int(args.GPUs) != 0 and gpu_build_installed()
                                      and should_use_gpu_build()) else ''
            foldseek_cmd = f'foldseek createdb {abspath}/cleaned_truncated_{args.name}_foldtune_generated_sequences_round{i}.fasta {abspath}/{args.name}_foldtune_generated_sequences_round{i}_db --prostt5-model {prostt5_weights_path}{gpu_flag}'
            subprocess.run(foldseek_cmd.split(' '), check=True)

            if i == 1:
                query_path = args.query if os.path.isabs(args.query) else f'{abspath}/{args.query}'
                foldseek_cmd = f'foldseek createdb {query_path} {abspath}/{args.name}_foldtune_input_db --prostt5-model {prostt5_weights_path}{gpu_flag}'
                subprocess.run(foldseek_cmd.split(' '), check=True)
            
            foldseek_cmd = f'foldseek easy-search {abspath}/{args.name}_foldtune_generated_sequences_round{i}_db {abspath}/{args.name}_foldtune_input_db {abspath}/{args.name}_foldtune_foldseek_round{i}.tsv tmp_round{i}'.split(' ')
        else:
            foldseek_cmd = [
                'foldseek', 'easy-search',
                f'{abspath}/{args.name}_foldtune_generated_structs_round{i}/',
                f'{abspath}/{args.name}_foldtune_input_structs/',
                f'{abspath}/{args.name}_foldtune_foldseek_round{i}.tsv',
                f'tmp_round{i}',
                '--alignment-type', '1',
                '--format-output', 'query,target,qtmscore,ttmscore,alntmscore',
            ]

        subprocess.run(foldseek_cmd, check=True)

        # Extract embeddings (exclude the label column)
        test_embeddings = test_df.iloc[:, :-1].to_numpy()
        input_embeddings = input_df.iloc[:, :-1].to_numpy()
        labels = test_df.iloc[:, -1].to_numpy()  # generated-seq labels (last column)


        foldseek_tsv = f'{abspath}/{args.name}_foldtune_foldseek_round{i}.tsv'
        most_distant_indices = select_most_distant(input_embeddings, test_embeddings, labels, foldseek_tsv, args)

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
            'seqkit', "grep",
            "--pattern-file", labels_file,
            f"{abspath}/cleaned_truncated_{args.name}_foldtune_generated_sequences_round{i}.fasta"
        ]

        output_fasta = os.path.join(abspath, f"{args.name}_foldtune_most-distant_round{i}.fasta")

        with open(output_fasta, "w+") as output_file:
            subprocess.run(seqkit_command, stdout=output_file, check=True)


        if args.finetune_strategy and uses_fp32_ckpt:
            if os.path.isdir(f'{abspath}/{args.name}_round{i}_{gen_model}_1.pt'):
                shutil.rmtree(f'{abspath}/{args.name}_round{i}_{gen_model}_1.pt')


def structurally_valid_queries(tsv_path, args):
    if not os.path.exists(tsv_path) or os.path.getsize(tsv_path) == 0:
        return set()
    if args.fast_folding:
        col_names = ['query', 'target', 'fident', 'alnlen', 'mismatch', 'gapopen',
                     'qstart', 'qend', 'tstart', 'tend', 'evalue', 'bits']
        df = pd.read_csv(tsv_path, sep='\t', names=col_names, dtype={'query': str, 'target': str})
        best = df.groupby('query')['evalue'].min()
        keep = best[best <= float(args.fast_fold_evalue)]
    else:
        col_names = ['query', 'target', 'qtmscore', 'ttmscore', 'alntmscore']
        df = pd.read_csv(tsv_path, sep='\t', names=col_names, dtype={'query': str, 'target': str})
        df['global_tm'] = df[['qtmscore', 'ttmscore']].min(axis=1)
        best = df.groupby('query')['global_tm'].max()
        keep = best[best >= float(args.fold_tmscore_threshold)]
    return {str(q).strip() for q in keep.index}


def select_most_distant(input_embeddings, test_embeddings, test_labels, tsv_path, args):
    """Up to 100 structurally-valid generated seqs, most distant from the training set.

    Distance of a generated seqs to the natural training set in ESM2-650M embedding space is set by
    ``args.selection_distance``:
      - ``'min'`` (default): nearest-neighbor L1 distance.
      - ``'max'``: L1 distance to the farthest training point.
    Survivors are ranked by that distance in decreasing order (ties keep input order).
    Raises ``RuntimeError`` if no generated seq passes the structural filter.
    """
    n_test = test_embeddings.shape[0]
    if args.selection_distance == 'max':
        dist = np.zeros(n_test)
        for input_vec in input_embeddings:
            dist = np.maximum(dist, np.sum(np.abs(test_embeddings - input_vec), axis=1))
    else:
        dist = np.full(n_test, np.inf)
        for input_vec in input_embeddings:
            dist = np.minimum(dist, np.sum(np.abs(test_embeddings - input_vec), axis=1))

    passing = structurally_valid_queries(tsv_path, args)
    candidates = [i for i, lab in enumerate(test_labels) if str(lab).strip() in passing]
    if not candidates:
        raise RuntimeError(
            'Foldtuning stopped: no generated sequence passed the structural fold '
            'filter this round (regular mode requires min(qtmscore, ttmscore) >= '
            f'{getattr(args, "fold_tmscore_threshold", 0.5)} to an input; fast mode '
            f'requires a foldseek e-value <= {getattr(args, "fast_fold_evalue", 1e-3)}). '
            'Potentially fix by changing input sequences, generating more sequences per round, '
            'lowering the structural threshold, or potentially using a different protein language model')

    candidates.sort(key=lambda i: dist[i], reverse=True)
    return candidates[:100]



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

