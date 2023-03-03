#!usr/bin/env python3

"""
regulon.py

This module processes regulons

Notes:
- Load Biopython with "$ module load Biopython/1.79-foss-2021a" before executing
  if the script is run on the RUG cluster

TODO: Specify file format
"""

__author__ = "Yaprak Yigit"
__version__ = "0.1"

import argparse
import os
import sys
import pandas
from Bio import SeqIO

class Regulon():
    """ Class for matching intergenic sequences to a regulatory genes

    :param fasta_file: A file in fasta format
    :param regulon_dataframe: A pandas dataframe of self.regulon_file
    :param regulon_dataframe: A copy of self.regulon_dataframe
    :param regulon_file: A file with columns and rows, including a header row,
                         one column should contain a locus tag
    :param separator: The delimiter of the regulon file
    """

    def __init__(self, regulon_file, separator):
        self.sequence_dict = None
        self.regulon_file = regulon_file
        self.separator = separator
        self.regulon_dataframe = None
        self.validate_files()

    def validate_files(self):
        """ Checks the regulon file and tries to return self.regulon_file

        :raises FileNotFoundError: If the regulon_file cannot be found.
        """
        if not os.path.exists(self.regulon_file):
            print("The regulon file does not exist")
        # Try to open the regulon file
        try:
            # Specify the engine for correct delimiter interpretation
            self.regulon_dataframe = pandas.read_csv(self.regulon_file, engine="python",
                                                     sep=self.separator, lineterminator='\r')
        except FileExistsError:
            sys.exit("Invalid file")

    def extract_dna_sequences(self, fasta_file):
        """ Reads a fasta file and returns the sequences

        :return: sequence_dict: A dictionary with locus tags as keys and the matching sequences
                                as values.

        Notes:
        - This method assumes that the file is in fasta format
        """
        sequence_dict = {}
        try:
            for record in SeqIO.parse(fasta_file, "fasta"):
                record_elements = record.id.split("|")
                locus_tag = record_elements[0]
                sequence_dict[locus_tag] = record.seq
        except:
            print("Invalid file")
        self.sequence_dict = sequence_dict

    def match_intergenic_regulations(self, locus_col):
        """ Matches DNA sequences from the fasta file to the regulon file by locus tag

        :param locus_col: A string specifying the name of the column containing regulator locus ids in the regulon file
        :return: regulon_dataframe_with_sequences: A shallow copy of the self.regulon_dataframe with an additional
                 column that contains the intergenic sequences
        """
        sequence_col = []
        for locus_tag in self.regulon_dataframe[locus_col]:
            if locus_tag in self.sequence_dict:
                # Conversion to string is required, since the sequence_dict contains a Bio object
                sequence_col.append(str(self.sequence_dict[locus_tag]))
            else:
                sequence_col.append("")
        # Shallow copy, dependent on the original dataframe
        regulon_dataframe_with_sequences = self.regulon_dataframe.copy(deep=False)
        regulon_dataframe_with_sequences["sequence"] = sequence_col
        return regulon_dataframe_with_sequences

    def get_non_regulators(self, regulator_col):
        """ Finds the intergenic regions that are not involved in regulating genes
        :param regulator_col: A string specifying the name of the column containing genes
        :return: non_regulators: A dictionary with sequence id as keys and non-regulatory intergenic sequences as values
        """
        non_regulators = {}
        for sequence_id in self.sequence_dict:
            if sequence_id not in self.regulon_dataframe[regulator_col]:
                non_regulators[sequence_id] = self.sequence_dict[sequence_id]
        return non_regulators

    def filter_on_mode(self, regulon_dataframe, mode_col, modes_list):
        """ Filters the mode column in the regulon data frame for specified modes

        :param regulon_dataframe: A pandas dataframe containing a regulon
        :param mode_col: A string specifying the name of the column containing modes
        :param modes_list: A list that contains strings which specify modes

        :return: filtered_regulon: The input regulon dataframe filtered on the mode column

        Notes:
        - Filtering is case-sensitive
        """
        filtered_regulon = regulon_dataframe[regulon_dataframe[mode_col].isin(modes_list)]
        return filtered_regulon

    def filter_on_regulator(self, regulon_dataframe, regulator_col, regulator, option="select"):
        """ Filters the gene column in the regulon data frame for a specified regulator

        :param regulon_dataframe: A pandas dataframe containing a regulon
        :param regulator_col: A string specifying the name of the column containing genes
        :param regulator: A string specifying one gene

        :return: filtered_regulon: The input regulon dataframe filtered on the regulator column

        Notes:
        - Filtering is case-sensitive
        """
        if option == "select":
            filtered_regulon = regulon_dataframe[regulon_dataframe[regulator_col] == regulator]
        elif option == "drop":
            filtered_regulon = regulon_dataframe[regulon_dataframe[regulator_col] != regulator]
        else:
            raise ValueError("Incorrect value for parameter option, use select or drop")
        return filtered_regulon

    def output_to_csv(self, output_file):
        """
        the path should exist
        :param output_file:
        TODO: Reconsider this method
        """
        self.regulon_dataframe_copy.to_csv(output_file)
