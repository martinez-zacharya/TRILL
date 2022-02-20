import pandas as pd
import os
from Bio import SeqIO


def FineTuneQueryValidation(fastaInput):
    if not fastaInput.endswith('.fasta'):
        raise ValueError('Input needs to be in .fasta format.')
    if os.stat(fastaInput).st_size == 0:
        raise ValueError('Input file is empty')
    temp = open(fastaInput, 'r').read()
    if " " in temp:
        raise ValueError("Invalid space in either header or sequence")

    fastas = SeqIO.parse(fastaInput, 'fasta')
    output_df = open('query_df.csv', 'w')
    output_df.write('Gene,Sequence\n')
    
    for fasta in fastas:
        name, seq = fasta.id, str(fasta.seq)
        if ',' in name or ',' in seq:
            raise ValueError("Invalid character in file - ','")
        output_df.write(name + ',' + seq + '\n')
    output_df.close()

    query = pd.read_csv('query_df.csv')
    query['Label'] = 1
    query.to_csv('query_df_labeled.csv')

    return True

def FineTuneDatabaseValidation(fastaInput):
    if not fastaInput.endswith('.fasta'):
        raise ValueError('Input needs to be in .fasta format.')
    if os.stat(fastaInput).st_size == 0:
        raise ValueError('Input file is empty')
    temp = open(fastaInput, 'r').read()
    if " " in temp:
        raise ValueError("Invalid space in either header or sequence")

    fastas = SeqIO.parse(fastaInput, 'fasta')
    output_df = open('database_df.csv', 'w')
    output_df.write('Gene,Sequence\n')
    
    for fasta in fastas:
        name, seq = fasta.id, str(fasta.seq)
        if ',' in name or ',' in seq:
            raise ValueError("Invalid character in file - ','")
        output_df.write(name + ',' + seq + '\n')
    output_df.close()

    database = pd.read_csv('database_df.csv')
    database['Label'] = 0
    database.to_csv('database_df_labeled.csv')

    return True





