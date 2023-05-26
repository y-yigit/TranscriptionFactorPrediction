#!usr/bin/env python3

"""
extract_fimo_motifs.py

This script processes multiple FIMO tsv files and combines the found motifs.
It also generates random sequences for each organism.

:fimo_folder: A folder containing multiple FIMO results folder
:fasta_folder: A folder with fasta files for whole genomes. The files should be named after the FIMO folders,
               only the file extension can diverge.

__author__ = "Yaprak Yigit"
__version__ = "0.1"
"""

import os
import pandas as pd
from dna import DNA

fimo_folder = "/home/ubuntu/Yaprak/Data/FIMO_results"
fasta_folder = "/home/ubuntu/Yaprak/Data/Genomes"


final_dataframe = pd.DataFrame(columns=["sequence", "species", "transcription_factor"])

for root, dirs, files in os.walk(fimo_folder):
    for file in files:
        if file.endswith(".tsv"):
            fimo_dataframe = pd.read_csv(root + "/" + file, engine="python", sep="\\t", lineterminator='\r')
            sequence_names = fimo_dataframe["sequence_name"].str.split("|")[0]
            sequence_list = fimo_dataframe["matched_sequence"].tolist()
            # The FIMO files contain comments, these result in NAs
            sequence_list = list(filter(None, sequence_list))
            sequence_names = list(filter(None, sequence_names))
            dna_obj = DNA(sequence_list)
            # Use the GC content of the whole genome
            nucleotide_percentages = DNA(fasta_folder + "/" + root.split("/")[-1] + ".ffn").calculate_nucleotide_percentages()
            dna_obj.change_sequence_lengths(len(sequence_list[0])+40, nucleotide_percentages)
            modified_sequences = dna_obj.dna_sequences
            # Extract the file name from the path and remove the strain name
            species = "_".join(root.split("/")[-1].split("_")[0:2]) * len(modified_sequences)
            feature_dataframe = pd.DataFrame(columns=["sequence", "species", "transcription_factor"])
            feature_dataframe["sequence"] = modified_sequences
            feature_dataframe["species"] = species
            feature_dataframe["transcription_factor"] = species
            random_negatives = [DNA.generate_random_dna(len(modified_sequences[0]), nucleotide_percentages) for base in
                                range(0, len(sequence_list)*1000)]
            additional_rows = [[negative, species+"_random_yy", "NA"] for negative in random_negatives]
            feature_dataframe = feature_dataframe.append(pd.DataFrame(additional_rows, columns=feature_dataframe.columns))
            final_dataframe = final_dataframe.append(feature_dataframe)
# Save the results
final_dataframe.to_csv("example.csv", sep='\t')