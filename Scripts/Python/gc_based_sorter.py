#!usr/bin/env python3
""" gc_based_sorter.py
This script sorts DNA sequences by the GC percentage of the matching species

Variables:
    max_length (int):
        The length of the features
"""

import os
import random
import re
import argparse
import numpy as np
from Bio import SeqIO
import pandas as pd
from dna import DNA

max_length = 50
# Formats numerical ranges into a readable string
list_gc = lambda start, end: "|".join([str(i) for i in range(start, end)])

def calculate_gc_percentage(input_dir):
    """ Reads fasta files and creates groups based on the GC percentages

    :param input_dir: Path of input files
    """
    gc_groups = {list_gc(20, 30): [], list_gc(30, 40): [], list_gc(40, 50): [], list_gc(50,60): [],
                 list_gc(60, 70): []}
    gc_counts, species = [],[]

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            dna_obj = DNA(root + "/" + file)
            nucleotide_percentages = dna_obj.calculate_nucleotide_percentages()
            gc_content = nucleotide_percentages["C"] + nucleotide_percentages["G"]

            id = "_".join(file.split("_")[0:2])
            for key in gc_groups:
                if str(gc_content) in key:
                    gc_groups[key].append(id)
                    gc_counts.append(gc_content)
                    species.append(id)

    for key in gc_groups:
        amount_of_val_ids = round(len(gc_groups[key]) * 0.2)
        val_ids = random.sample(gc_groups[key], amount_of_val_ids)
        train_ids = [item for item in gc_groups[key] if item not in val_ids]
        gc_groups[key] = [train_ids, val_ids]
    return gc_groups, gc_counts, species

def create_negative_labels(fasta_file, output_dir, group_name, train_val_ids):
    """ Reads a fasta file and extracts sequences with a length of max_length

    The selected sequences are one-hot encoded before they are returned

    :param fasta_file: Path to fasta file of non-motifs
    :param output_dir: Output path
    :param group_name: Group name based on the GC percentage created by list_gc
    :param train_val_ids: Randomly selected training and validation data

    """
    train_sequences = []
    val_sequences = []

    gc_list = group_name.split("|")
    output_file_train = f"{output_dir}/CcpA_negatives_train{gc_list[0]}_{gc_list[-1]}.npy"
    output_file_val = f"{output_dir}/CcpA_negatives_val{gc_list[0]}_{gc_list[-1]}.npy"
    pattern = r"\[(.*?)\]"
    for record in SeqIO.parse(fasta_file, "fasta"):
        if not "[" in record.id:
            specie = record.id.split("|")[0]
        else:
            match = re.search(pattern, record.id)
            if match:
                specie = "_".join(match.group(1).split(" ")[0:2])

        if record.seq != "" and len(record.seq) >= max_length:
            chunks = [record.seq[i:i + int(max_length)] for i in range(0, len(record.seq), int(max_length))]
            for non_motif in chunks:
                if len(non_motif) == int(max_length) and specie in train_val_ids[0]:
                    train_sequences.extend([non_motif, non_motif[::-1]])
                elif len(non_motif) == int(max_length) and specie in train_val_ids[1]:
                    val_sequences.extend([non_motif, non_motif[::-1]])

    random.shuffle(train_sequences)
    random.shuffle(val_sequences)

    if len(train_sequences) >= 20000:
        train_sequences = train_sequences[0:20000]
    if len(val_sequences) >= 4000:
        val_sequences = val_sequences[0:4000]

    # Save the datasets
    train_sequences = DNA.one_hot_encoder(train_sequences)
    val_sequences = DNA.one_hot_encoder(val_sequences)
    np.save(output_file_train, np.array(train_sequences, dtype=object), allow_pickle=True)
    np.save(output_file_val, np.array(val_sequences, dtype=object), allow_pickle=True)

def save_gc(species, gc_content, data_file):
    """ Updates an R file containing information about the bacterial species with GC percentages

    :param species: List of bacterial species
    :param gc_content: List of GC percentages in the same order as the species
    """
    df = pd.read_csv(data_file)

    for index, row in df.iterrows():
        if row["Species"] in species:
            species_index = species.index(row["Species"])
            df.at[index, "GC content"] = gc_content[species_index]
    df.to_csv(data_file, index=False)

def main():
    parser = argparse.ArgumentParser(description='Process paths for the pipeline.')
    parser.add_argument('--intergenic_dir', type=str, help='Path to a folder containing fasta files with intergenic sequences')
    parser.add_argument('--data_dir', type=str, help='Path to a directory where the CNN data will be stored')
    parser.add_argument('--non_motifs', type=str, help='A file containing DNA sequences that are not CRE motifs')
    parser.add_argument('--data_file', type=str, help='Specify the file for the GC report(optional)')

    args = parser.parse_args()

    intergenic_dir = args.intergenic_dir
    data_dir = args.data_dir
    non_motifs = args.non_motifs

    gc_groups, gc_counts, species = calculate_gc_percentage(intergenic_dir)

    if args.data_file is not None:
        save_gc(species, gc_counts, args.data_file)

    for gc_group in gc_groups:
        create_negative_labels(non_motifs, data_dir, gc_group, gc_groups[gc_group])

if __name__ == "__main__":
    main()
