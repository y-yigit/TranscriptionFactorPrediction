#!usr/bin/env python3

"""
sequence_matcher.py

This program demonstrates how to match regulon genes with fasta sequences

Usage:
  Run this program from the command line following the example:
      $ python3 sequence_matcher.py <fasta_file> <regulon_file> <separator>

  All arguments are described in function arge_parser

Notes:
- Load Biopython with "$ module load Biopython/1.79-foss-2021a" before executing
  if the script is run on the RUG cluster

TODO: Add parameters for filter conditions
"""

__author__ = "Yaprak Yigit"
__version__ = "0.1"

import argparse
import os
import sys
import pandas
from Bio import SeqIO

class LocusMatcher():
    """ Class for matching intergenic sequences to a regulatory genes

    :param fasta_file: A file in fasta format
    :param regulon_dataframe: A pandas dataframe of self.regulon_file
    :param regulon_dataframe: A copy of self.regulon_dataframe
    :param regulon_file: A file with columns and rows, including a header row,
                         one column should contain a locus tag
    :param separator: The delimiter of the regulon file
    """

    def __init__(self, fasta_file, regulon_file, separator):
        self.fasta_file = fasta_file
        self.regulon_dataframe = None
        self.regulon_dataframe_copy = None
        self.regulon_file = regulon_file
        self.separator = separator
        self.validate_files()

    def validate_files(self):
        """ Checks the file locations and tries to return self.regulon_file

        :raises FileNotFoundError: If either the fasta_file or regulon_file cannot be found.
        """
        if not os.path.exists(self.fasta_file):
            print("The fasta file does not exist")
        elif not os.path.exists(self.regulon_file):
            print("The regulon file does not exist")
        # Try to open the regulon file
        try:
            # Specify the engine for correct separator interpretation
            self.regulon_dataframe = pandas.read_csv(self.regulon_file, engine="python",
                                                     sep=self.separator, lineterminator='\r')
        except FileExistsError:
            sys.exit("There was a problem reading the file")

    def extract_sequences(self):
        """ Reads a fasta file and returns the sequences

        :return: sequence_dict: A dictionary with locus tags as keys and the matching sequences
                                as values.

        Notes:
        - This method assumes that the fasta file is valid.
        """
        sequence_dict = {}
        for record in SeqIO.parse(self.fasta_file, "fasta"):
            record_elements = record.id.split("|")
            locus_tag = record_elements[0]
            sequence_dict[locus_tag] = record.seq
        return sequence_dict

    def add_sequence_column(self, locus_col, sequence_dict):
        """ Matches DNA sequences from the fasta file to the regulon file by locus tag

        :param locus_col: A string specifying the name of the column containing locus ids
        :param sequence_dict: A dictionary with locus tags as keys and the matching sequences
                              as values.
        """
        # Shallow copy, only the sequence column differs
        self.regulon_dataframe_copy = self.regulon_dataframe.copy(deep=False)
        sequence_col = []
        for locus_tag in self.regulon_dataframe[locus_col]:
            if locus_tag in sequence_dict:
                # Conversion to string is required, since the sequence_dict contains Bio object
                sequence_col.append(str(sequence_dict[locus_tag]))
            else:
                sequence_col.append("")
        self.regulon_dataframe_copy["sequence"] = sequence_col

    def filter_on_mode(self, mode_col, modes_list):
        """ Filters the mode column in the regulon data frame for specified modes

        :param mode_col: A string specifying the name of the column containing modes
        :param modes_list: A list that contains strings which specify modes

        Notes:
        - Filtering is case-sensitive
        """
        self.regulon_dataframe_copy = self.regulon_dataframe_copy[
            self.regulon_dataframe_copy[mode_col].isin(modes_list)]

    def filter_on_gene(self, gene_col, gene):
        """ Filters the gene column in the regulon data frame for a specified gene

        :param gene_col: A string specifying the name of the column containing genes
        :param gene: A string specifying one gene

        Notes:
        - Filtering is case-sensitive
        """
        self.regulon_dataframe_copy = self.regulon_dataframe_copy[
            self.regulon_dataframe_copy[gene_col] == gene]

    def modified_file_to_csv(self, output_file):
        """
        the path should exist
        :param output_file:
        """
        self.regulon_dataframe_copy.to_csv(output_file)

def arg_parser():
    """ Retrieves command line arguments
    """
    parser = argparse.ArgumentParser(description="Submit a fasta and regulations file. "
                                                 "Both files should contain gene identifiers "
                                                 "of the same database format. \n"
                                                 "Usage: python3 sequence_matcher.py <fasta_file> "
                                                 "<regulon_file> <separator>")
    parser.add_argument('fasta_file', type=str, help='A file in fasta format')
    parser.add_argument('regulon_file', type=str, help='A file with columns and rows, '
                                                       'including a header row, '
                                                       'one column should contain a locus tag.')
    parser.add_argument('separator', type=str, help='The delimiter of the regulon file, '
                                                    'it requires characters escaping'
                                                    'if it contains special characters')
    args = parser.parse_args()
    main(args)


def main(args):
    """ Demonstrates the usage of class LocusMatcher"""
    locus_obj = LocusMatcher(args.fasta_file, args.regulon_file, args.separator)
    sequence_dict = locus_obj.extract_sequences()
    locus_obj.add_sequence_column("gene locus", sequence_dict)
    locus_obj.filter_on_mode("mode", ["activation", "repression"])
    locus_obj.filter_on_gene("regulator name", "ccpA")
    locus_obj.modified_file_to_csv("test.csv")

if __name__ == '__main__':
    sys.exit(arg_parser())
