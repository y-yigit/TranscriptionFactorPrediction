#!usr/bin/env python3

"""
extract_blast_genes.py

.. note:: The file Bacillus_subtilis_subsp_subtilis_str_168_ASM904v1_genomic.g2d.faa is required
"""

__author__ = "Yaprak Yigit"
__version__ = "0.1"

import os
import pandas as pd
from dna import DNA
from regulon import Regulon
import argparse
import sys

def extract_ccpa_genes(action, data_dir, regulon_file) -> list:
    """ Either rops or selects ccpA genes from Bacillus Subtilis

    :param action:
    :param data_dir:
    :param regulon_file:

    :return:
    """
    ccpa_regulon = Regulon(regulon_file, "\\t")
    ccpa_regulon.extract_dna_sequences(data_dir+"/Bacillus_subtilis_subsp_subtilis_str_168_ASM904v1_genomic.g2d.intergenic.ffn")
    regulon_dataframe_with_sequences = ccpa_regulon.match_intergenic_regulations("gene locus")
    ccpa_regulators = ccpa_regulon.filter_on_regulator(regulon_dataframe_with_sequences, "regulator name",
                                                       "ccpA", action)
    ccpa_regulators = ccpa_regulon.filter_on_mode(ccpa_regulators, "mode", ["activation", "repression"])
    return ccpa_regulators["gene locus"].str.replace("_", "").tolist()

def retrieve_blast_genes(gene_list, blast_results) -> list:
    """ Collects proteinortho output

    :param gene_list:
    :param blast_results:

    :return:
    .. note:: Proteinortho removed the underscores for bacillus subtilis
    """
    homologs = pd.read_csv(blast_results, engine="python", sep="\\t", lineterminator='\r')
    homologs = homologs[homologs["Bacillus_subtilis_subsp_subtilis_str_168_ASM904v1_genomic.g2d.faa"].isin(gene_list)]
    homologs.drop(["# Species", "Genes", "Alg.-Conn."], axis=1, inplace=True)
    # Get all values that are not empty
    all_ccpa_genes = [value.replace("_", "") for value in homologs.values.flatten() if value != '*']
    return all_ccpa_genes

def save_sequences(locus_list, out_file, data_dir):
    """ Writes regulon gene sequences to an output fasta file

    :param out_file:
    :param data_dir:

    """
    final_dataframe = pd.DataFrame(columns=["sequence", "species", "transcription_factor"])
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".ffn"):
                ccpa_regulon = Regulon()
                ccpa_regulon.extract_dna_sequences(root + "/" + file)
                species = "_".join(file.replace("_genomic.g2d.intergenic.ffn", "").split("_")[0:2])
                # It is possible that the dataframe_rows is None if the species does not have the ccpa regulon
                rows = ccpa_regulon.match_intergenic_regulations2(locus_list, species)
                for row in rows:
                    with open(out_file, "a") as open_file:
                        open_file.write(f">{row[1]}|{row[2]}\n{row[0]}\n")

def arg_parser():
    """ The arguments in the main function are processed with argparse
    """
    parser = argparse.ArgumentParser(description='Argument parser that sets paths and specifies which action to perform')
    parser.add_argument('action', type=str, help='Whether to drop or select the ccpA genes')
    parser.add_argument('output_file', type=str, help='The name for the fasta output file')
    parser.add_argument('intergenic_dir', type=str, help='Path to the folder containing intergenic sequences')
    parser.add_argument('blast_file', type=str, help='The name of the BLAST output file')
    parser.add_argument('regulon_file', type=str, help='The path to the regulon file for Bacillus Subtilis')
    args = parser.parse_args()
    main(args)

def main(args):
    ccpa_genes = extract_ccpa_genes(args.action, args.intergenic_dir, args.regulon_file)
    locus_list = retrieve_blast_genes(ccpa_genes, args.blast_file)
    save_sequences(locus_list, args.output_file, args.intergenic_dir)

if __name__ == '__main__':
    sys.exit(arg_parser())