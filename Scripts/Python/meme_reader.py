#!usr/bin/env python3

"""
meme_reader.py

Parses a MEME output file and stores the found motifs in a fasta file
"""

__author__ = "Yaprak Yigit"
__version__ = "0.1"

import re
import argparse
import pandas as pd
import sys

class MEMEReader():
    """
    Class for processing MEME output
    """

    def __init__(self, meme_file):
        """
        :param meme_file: A string specifying a path to a MEME output file
        """
        self.meme_file = meme_file

    def find_motifs(self):
        """ Parses a MEME output file and saves the header, p-value, and sequence
        """
        motifs = False
        rows = []
        with open(self.meme_file, "r") as open_file:
            line = open_file.readline()
            while line:
                line = open_file.readline()
                if "MEME-1 sites sorted by position p-value" in line:
                    # Skip the next two lines and read the third
                    next(open_file)
                    next(open_file)
                    next(open_file)
                    line = open_file.readline()
                    motifs = True
                if "-"*10 in line:
                    motifs = False
                elif motifs == True:
                    line_elements = line.split()
                    rows.append([line_elements[0], float(line_elements[2]), line_elements[4]])
        return rows

def main():
    """ Runs the whole program using command line arguments
    """
    parser = argparse.ArgumentParser(description='Parse a MEME output file and save the results')
    parser.add_argument('meme_file', type=str, help='MEME output')
    parser.add_argument('output_file', type=str, help='Output file for MEME motifs')
    args = parser.parse_args()
    meme_reader = MEMEReader(args.meme_file)
    list_of_rows = meme_reader.find_motifs()
    with open(args.output_file, 'w') as out_file:
        for row in list_of_rows:
            out_file.write(">{}\n{}\n".format(row[0], row[2]))

if __name__ == "__main__":
    sys.exit(main())
