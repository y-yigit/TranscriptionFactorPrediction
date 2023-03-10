#!usr/bin/env python3

"""
DNA.py

.. note:: SeqIO sequence object cannot be used as a string directly,
          they need to be converted to a string first

..todo:: Additional sequence information such as gene identifiers are lost, return more information
..todo:: Check if the promotor length is required and how to find it
..todo:: Encoding is slow, the class would benefit from a performance update.
"""

__author__ = "Yaprak Yigit"
__version__ = "0.1"

import re
import random
from typing import Union
import numpy as np
from Bio import SeqIO

class DNA():
    """ Class for modifying DNA sequences """

    def __init__(self, sequences: Union[str, list], promotor_length=100):
        """
        :param sequence: A path to a fasta file or a list containing DNA strings
        :type sequence: str, list
        :param promotor_length: The length of a promotor
        :type promotor_length: int
        """
        self.promotor_length = promotor_length
        self.dna_sequences = []
        self.__clean_sequence__(sequences)

    def __clean_sequence__(self, sequences: Union[str, list]):
        """ Reads DNA sequences and substitutes all non bases in the sequence with the base guanine

        If the parameter sequences is of the type string the method will interpret it as a path
        and open the fasta file.

        :param sequence: A path to a fasta file or a list containing DNA strings
        :type sequence: str, list

        :raises: FileNotFoundError: The path in the sequences parameter is incorrect
        :raises: OSError: This error is thrown by the module Bio, it occurs when a path is too long
        :raises: TypeError: The parameter sequences is of an incorrect type
        """
        if isinstance(sequences, str):
            try:
                self.dna_sequences = [re.sub("[^AGTC]", "G", str(record.seq).upper())
                                      for record in SeqIO.parse(sequences, "fasta") if record.seq != ""]
            except FileNotFoundError:
                print("Incorrect file")
        elif isinstance(sequences, list):
            self.dna_sequences = [re.sub("[^ATGC]", "G", sequence.upper())
                                  for sequence in sequences if sequence != ""]
        else:
            raise TypeError("The parameter sequences is not a list or string")

    def change_sequence_lengths(self, max_length: int, nucleotide_percentages: dict):
        """ Modifies all sequences in self.dna_sequences to share the same length
        """
        modified_sequences = []
        for sequence in self.dna_sequences:
            number_of_extra_bases = max_length - len(sequence)
            random_sequence = self.generate_random_dna(number_of_extra_bases, nucleotide_percentages)
            # This is not the exact middle base, but it is very close to it
            middle_point = round(len(random_sequence) / 2)
            modified_sequence = random_sequence[:middle_point] + sequence + random_sequence[middle_point:]
            modified_sequences.append(modified_sequence)
        # Overwrite the class variable
        self.dna_sequences = modified_sequences

    def calculate_nucleotide_percentages(self) -> dict:
        """ calculates the nucleotides percentages of all the bases in the class parameter sequences

        :return: A dictionary with gene identifiers as keys and DNA strings are values
        :rtype: dict
        """
        dna_string = "".join(self.dna_sequences)
        return {base: round((dna_string.count(base) / len(dna_string)) * 100)
                for base in ["A", "C", "G", "T"]}

    @staticmethod
    def one_hot_encoder(dna_sequences) -> list:
        """ Static method that encodes all DNA strings in the self.dna_sequences list

        The DNA strings are encoded to numpy arrays. Each array has 4 rows, one for each base.

        :return: A list of numpy arrays
        :rtype: list
        """
        mapping = dict(zip("AGCT", range(4)))
        # Encode every base in every sequence with np.eye
        return [np.eye(4)[[mapping[base] for base in sequence]] for sequence in dna_sequences]


    def generate_random_dna(self, count: int, nucleotide_percentages: dict) -> list:
        """ Generates a sequence of bases with the GC content of the input fasta file

        :param count: Integer specifying the total number of bases
        :type count: int
        :param nucleotide_percentages: A dictionary with bases as keys and the percentage
            of their occurrence in a sequence as float
        :type nucleotide_percentages: dict
        :return: A list of random DNA strings
        :rtype: list

        .. note:: String concatenation might be slow
        """
        dna_set = "".join([base*nucleotide_percentages[base] for base in nucleotide_percentages])

        random_sequence = ""
        for index in range(0, count):
            random_index = random.randint(0, len(dna_set) - 1)
            random_sequence += dna_set[random_index]
        return random_sequence
