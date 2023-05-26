#!usr/bin/env python3

"""
clean_non_motifs.py

Removes cre motifs in a fasta file from a different fasta file. The output is stored in a new fasta file
"""

__author__ = "Yaprak Yigit"
__version__ = "0.1"

import argparse
from Bio import SeqIO

# Create an argument parser
parser = argparse.ArgumentParser(description='Remove sequences in one fasta file that are present in another fasta file')
parser.add_argument('non_motif_file', type=str, help='the input fasta file to filter')
parser.add_argument('motif_file', type=str, help='the fasta file containing sequences to remove')
parser.add_argument('output_file', type=str, help='the output fasta file')
args = parser.parse_args()

non_motif_file = SeqIO.parse(args.non_motif_file, "fasta")
motif_file = SeqIO.parse(args.motif_file, "fasta")

# Get the set of sequence IDs in the motif file
seq_ids_to_remove = set(seq.id for seq in motif_file)

# Open the output fasta file for writing
with open(args.output_file, "w") as output_file:
    # Loop over the sequences in fnon_motif_file and write those that are not in seq_ids_to_remove
    for seq in SeqIO.parse(args.non_motif_file, "fasta"):
        if seq.id not in seq_ids_to_remove:
            SeqIO.write(seq, output_file, "fasta")