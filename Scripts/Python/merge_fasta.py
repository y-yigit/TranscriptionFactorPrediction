#!/usr/bin/env python3

"""
merge_fasta.py

Merge two fasta files
"""

__author__ = "Yaprak Yigit"
__version__ = "0.1"


import argparse
import re
from Bio import SeqIO

def merge_fasta_files(input_file1, input_files2, output_file, duplicate_species):
    """ Merge two fasta files

    :param input_file1: Fasta file 1
    :param input_files2: Fasta file 2
    :param output_file: Name of the output file
    :param duplicate_species: Species to skip from file 2
    """

    sequences1 = list(SeqIO.parse(input_file1, 'fasta'))
    sequences2 = []
    for input_file2 in input_files2:
        sequences2.extend(list(SeqIO.parse(input_file2, 'fasta')))

    species_reformatted = [species.lower() for species in duplicate_species] + [species.replace("_", " ").lower() for species in duplicate_species]
    # Filter the duplicate species
    sequences2_filtered = [seq for seq in sequences2 if not any(species in seq.description.lower() for species in species_reformatted)]

    # Merge the sequences from both files
    raw_sequences = sequences1 + sequences2_filtered
    merged_sequences = []
    for item in raw_sequences:
        if "[" in item.description:
            pattern = r"\[(.*?)\]"
            match = re.search(pattern, item.description)
            if match:
                temp_specie = "_".join(match.group(1).split(" ")[0:2])
                item.description = temp_specie
                item.id = temp_specie
                item.name = temp_specie
                merged_sequences.append(item)
        else:
            merged_sequences.append(item)

    # Write the merged sequences to the output file
    with open(output_file, 'w') as output_handle:
        SeqIO.write(merged_sequences, output_handle, 'fasta')
    print(f"Merged sequences from '{input_file1}' and '{', '.join(input_files2)}', and saved to '{output_file}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge two fasta files, skipping sequences with specified species in the header of the second file(s).")
    parser.add_argument("input_file1", help="Path to the first input fasta file.")
    parser.add_argument("--additional_files", nargs='+',
                        help="Paths to the second input fasta file(s). Use space to separate multiple files.")
    parser.add_argument("output_file", help="Path to the output fasta file.")
    parser.add_argument("--duplicate_species", nargs='+', default=[],
                        help="Species to skip from the second file(s) if found in the header.")
    args = parser.parse_args()

    print(args.additional_files, "\n", args.output_file, "\n", args.duplicate_species, "\n")
    merge_fasta_files(args.input_file1, args.additional_files, args.output_file, args.duplicate_species)
