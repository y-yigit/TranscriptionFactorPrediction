#!usr/bin/env python3

"""
meme_reader.py

Parses a MEME output file and stores the found motifs in a fasta file
"""

__author__ = "Yaprak Yigit"
__version__ = "0.1"

import csv
import argparse
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
                    # Skip the next three lines and read the third
                    next(open_file)
                    next(open_file)
                    next(open_file)
                    line = open_file.readline()
                    motifs = True
                if "-"*10 in line:
                    motifs = False
                elif motifs == True:
                    line_elements = line.split()
                    motif = line_elements[4]
                    if len(line_elements) == 6 and line_elements[3] != ".":
                        motif = line_elements[3][-1] + line_elements[4] + line_elements[5][0]
                    rows.append([line_elements[0], motif])
        return rows

def main():
    """ Receive command line arguments and read and store the MEME file"""
    parser = argparse.ArgumentParser(description='Parse a MEME output file and save the results')
    parser.add_argument('meme_file', type=str, help='MEME output')
    parser.add_argument('output_file', type=str, help='Output file for MEME motifs without file extension')
    args = parser.parse_args()
    meme_reader = MEMEReader(args.meme_file)
    list_of_rows = meme_reader.find_motifs()

    with open(args.output_file, 'w') as fasta_file, open(args.output_file+".csv", 'w', newline='') as csv_file:
        fasta_writer = fasta_file.write
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Species", "Sequence"])
        for row in list_of_rows:
            fasta_writer(">{}\n{}\n".format(row[0], row[1]))
            csv_writer.writerow(row)

if __name__ == "__main__":
    sys.exit(main())
