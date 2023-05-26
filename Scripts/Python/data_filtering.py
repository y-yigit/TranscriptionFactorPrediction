#!usr/bin/env python3

"""
data_filtering.py

Filters FIMO output based on regulon genes and stores the resulting output in a fasta file
"""

__author__ = "Yaprak Yigit"
__version__ = "0.1"

import os
import argparse
import pandas as pd

# Parse command line arguments
parser = argparse.ArgumentParser(description='Combine FIMO motif files and output as a fasta file')
parser.add_argument('fimo_dir', type=str, help='Path to the FIMO directory')
parser.add_argument('output_file', type=str, help='Output file in fasta format')
args = parser.parse_args()

# Read in data
motifs = []
for root, dirs, files in os.walk(args.fimo_dir):
    for file in files:
        if file.endswith(".tsv"):
            motifs.append(pd.read_csv(os.path.join(root, file), sep="\t", header=0))

# Get overlapping sequence names
overlap = set(motifs[0]['sequence_name'])
for i in range(1, len(motifs)):
    overlap &= set(motifs[i]['sequence_name'])

# Combine all motifs and remove overlapping motifs
all_motifs = pd.concat(motifs, ignore_index=True)
all_motifs = all_motifs[~all_motifs['sequence_name'].isin(overlap)]

# Write the remaining motifs to a fasta file
with open(args.output_file, 'w') as open_file:
    for index, row in all_motifs.iterrows():
        open_file.write(f">{row['sequence_name']}\n{row['matched_sequence']}\n")