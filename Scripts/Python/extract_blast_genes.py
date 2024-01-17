#!usr/bin/env python3

"""
extract_blast_genes.py

Script for processing and storing BLAST results

.. note:: This scripts expects data to be downloaded from CollecTF and Genome2D
"""

__author__ = "Yaprak Yigit"
__version__ = "0.1"

import csv
import os
import pandas as pd
from regulon import Regulon
import argparse
import sys

def extract_ccpa_genes_collectf(data_dir) -> list:
    """ Extracts CcpA genes from multiple collectf files

    :param data_dir: Data directory with regulon files from CollecTF

    :return:
    """
    ccpa_genes = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".tsv"):
            tsv_file = os.path.join(data_dir, filename)
            ccpa_regulon = Regulon(tsv_file, "\\t").regulon_dataframe
            ccpa_genes += ccpa_regulon["regulated genes (locus_tags)"].str.replace("_", "").tolist()
    return ccpa_genes

def retrieve_blast_genes(gene_list, blast_results) -> list:
    """ Collects proteinortho output

    :param gene_list: List of gene names
    :param blast_results: BLAST output file

    :return:
    .. note:: Proteinortho removed the underscores for bacillus subtilis
    """
    homologs = pd.read_csv(blast_results, engine="python", sep="\\t", lineterminator='\r')

    columns_to_check = ["Bacillus_subtilis_subsp_subtilis_str_168_ASM904v1_genomic.g2d.faa",
                        "Clostridioides_difficile_630_ASM920v1_genomic.g2d.faa",
                        "Lactococcus_lactis_subsp_cremoris_MG1363_ASM942v1_genomic.g2d.faa",
                        "Streptococcus_pneumoniae_D39_ASM1436v2_genomic.g2d.faa",
                        "Streptococcus_suis_P1_7_ASM9190v1_genomic.g2d.faa"]
    filtered_df = pd.DataFrame()
    # Loop through the DataFrame rows
    for index, row in homologs.iterrows():
        # Check if any of the specified columns have a value in gene_list
        if any(row[col].replace("_", "") in gene_list for col in columns_to_check):
            filtered_df = filtered_df.append(row)

    # Reset the index of the filtered DataFrame
    filtered_df.reset_index(drop=True, inplace=True)


    #homologs = homologs[homologs["Bacillus_subtilis_subsp_subtilis_str_168_ASM904v1_genomic.g2d.faa"].isin(gene_list)]
    homologs.drop(["# Species", "Genes", "Alg.-Conn."], axis=1, inplace=True)
    # Get all values that are not empty
    all_ccpa_genes = [str(value).replace("_", "") for value in filtered_df.values.flatten() if value != '*']
    return all_ccpa_genes

def save_sequences(locus_list, out_file, data_dir):
    """ Writes regulon gene sequences to an output fasta file

    :param locus_list: List of gene loci
    :param out_file: Output directory
    :param data_dir: Data directory with intergenic fasta files downloaded from Genome2D

    """
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".ffn"):
                ccpa_regulon = Regulon()
                ccpa_regulon.extract_dna_sequences(root + "/" + file)
                species = "_".join(file.replace("_genomic.g2d.intergenic.ffn", "").split("_")[0:2])
                rows = ccpa_regulon.match_intergenic_regulations2(locus_list, species)
                for row in rows:
                    with open(out_file, "a") as open_file:
                        open_file.write(f">{row[1]}|{row[2]}\n{row[0]}\n")

def arg_parser():
    """ The arguments in the main function are processed with argparse
    """
    parser = argparse.ArgumentParser(description='Argument parser that specifies paths')
    parser.add_argument('output_file', type=str, help='The name for the fasta output file')
    parser.add_argument('intergenic_dir', type=str, help='Path to the folder containing intergenic sequences')
    parser.add_argument('blast_file', type=str, help='The name of the BLAST output file')
    parser.add_argument('regulon_dir', type=str, help='The path to the regulon directory')
    args = parser.parse_args()
    main(args)

def main(args):
    ccpa_loci = extract_ccpa_genes_collectf(args.regulon_dir)
    locus_list = retrieve_blast_genes(ccpa_loci, args.blast_file)
    save_sequences(locus_list, args.output_file, args.intergenic_dir)

if __name__ == '__main__':
    sys.exit(arg_parser())
