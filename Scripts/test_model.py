#!usr/bin/env python3

"""
test_model.py

This script processes input data and classifies it with a CNN model

__author__ = "Yaprak Yigit"
__version__ = "0.1"
"""

from regulon import Regulon
from dna import DNA
import numpy as np
from CNN import CNN
import tensorflow as tf

# Change these paths
cnn_datasets = "/home/ubuntu/Yaprak/Data/CNN_model"
regulon_file = "/home/ubuntu/Yaprak/Data/Regulons/Regulations_Bacillus_subtilis.txt"
fasta_file = "/home/ubuntu/Yaprak/Data/Intergenic/"\
             "Bacillus_subtilis_subsp_subtilis_str_168_ASM904v1_genomic.g2d.intergenic.ffn"

def prepare_input_data() -> tuple:
    """ Processes gff and fasta files to create labels for a CNN model

    This function links gff and fasta files in order to match promotor genes with DNA sequences.
    Matches are filtered for specific regulons and or modes. The resulting genes can be used as true positives. The
    remaining fasta sequences that did not match can be used as true negatives.

    :return: Two dictionaries containing DNA sequences
    :rtype: tuple
    .. todo:: Add parameters and change the dictionary type since the sequence identifiers are not used
    """
    ccpa_regulon = Regulon(regulon_file, "\\t")
    ccpa_regulon.extract_dna_sequences(fasta_file)
    regulon_dataframe_with_sequences = ccpa_regulon.match_intergenic_regulations("gene locus")
    ccpa_regulators = ccpa_regulon.filter_on_regulator(regulon_dataframe_with_sequences, "regulator name", "ccpA", "select")
    ccpa_regulators = ccpa_regulon.filter_on_mode(ccpa_regulators, "mode", ["activation", "repression"])

    true_negatives = ccpa_regulon.get_non_regulators("regulator locus")
    true_positives = ccpa_regulon.pandas_to_dict(ccpa_regulators, "gene locus", "sequence")
    return true_negatives, true_positives

def find_longest_sequence(dna_dictionaries:list) -> int:
    """ Finds the longest value out of all dictionaries in a list

    :param dna_dictionaries: List of dictionaries that contain DNA sequences as values
    :type dna_dictionaries: list

    :return: max_sequence_length: The length of the longest value in all dictionaries
    :rtype: int
    .. note:: The longest sequence will be used as the standard sequence length, so maybe outliers should be removed
    """
    max_sequence_length = 0
    # The built-in max function on list(dictionary.values()) did not give the expected outcome
    for dictionary in dna_dictionaries:
        for key in dictionary:
            if len(dictionary[key]) > max_sequence_length:
                max_sequence_length = len(dictionary[key])
    return max_sequence_length

def encode_many_sequences(dna_dictionaries, nucleotide_percentages, max_length) -> list:
    """ Encodes values of all dictionaries in a list

    :param dna_dictionaries: List of dictionaries that contain DNA sequences as values
    :type dna_dictionaries: list
    :param nucleotide_percentages: A dictionary containing the keys "A", "G", "C", and "T" with floats as values
    :type nucleotide_percentages: dict
    :param max_length: The maximum length of a sequence found in the file
    :type max_length: int

    :return: A list containing lists of encoded sequences
    :rtype: list
    """
    nested_lists = []
    max_sequence_length = 0
    for dna_dictionary in dna_dictionaries:
        dna_obj = DNA(list(dna_dictionary.values()))
        dna_obj.change_sequence_lengths(max_length, nucleotide_percentages)
        encoded_sequences = dna_obj.one_hot_encoder(dna_obj.dna_sequences)
        nested_lists.append(encoded_sequences)
    return nested_lists

def get_random_negatives(count: int, fasta_file: str) -> list:
    """ Creates sequences for true negative labels based on the nucleotide percentage of a fasta file

    :param count: Integer specifying the total number of bases
    :type count: int
    :param fasta_file: A path to a fasta file
    :type fasta_file: str
    
    :return: A list of random DNA strings
    :rtype: list

    .. todo:: Test this method with the CNN model
    """
    dna_obj = DNA(fasta_file)
    nucleotide_percentage = dna_obj.calculate_nucleotide_percentages()
    dna_sequences = dna_obj.generate_random_dna(count, nucleotide_percentage)
    return dna_sequences

def create_labels(positive_data, negative_data):
    """ Create the labels for the CNN model

    :param positive_data: Positive sequences
    :type positive_data: list
    :param negative_data: Negatives sequences
    :type negative_data: list
    :return: A tensor object and numpy array
    :rtype: tuple
    """
    # Create labels for your data
    positive_labels = np.ones(len(positive_data))
    negative_labels = np.zeros(len(negative_data))

    data = tf.stack(positive_data + negative_data)
    labels = np.concatenate((positive_labels, negative_labels))
    return data, labels

def test_model(data, test_set_fraction, labels, sequence_length):
    """ Test a CNN model
    :param data: A tensor object of the input data
    :type data: object
    :param test_set_fraction: A fraction below 1
    :type test_set_fraction: float
    :param labels: A numpy array of labels
    :type labels: object
    :param sequence_length: The length of the longest sequence
    :type sequence_length: int
    """
    # The values for input_shape might not be correct
    my_model = CNN((sequence_length, 4,))
    my_model.divide_labels(data, labels, test_set_fraction)
    my_model.compile_model()
    my_model.train_model()

# Get the nucleotide percentages using the whole file (GC content)
nucleotide_percentages = DNA(fasta_file).calculate_nucleotide_percentages()

# Extract the necessary data
dna_dictionaries = prepare_input_data()
sequence_length = find_longest_sequence(dna_dictionaries)
intergenic_true_negatives, true_positives = encode_many_sequences(dna_dictionaries, nucleotide_percentages,
                                                                  sequence_length)
# Create labels and modify the data for tensorflow
random_true_negatives = get_random_negatives(sequence_length, fasta_file)
data, labels = create_labels(true_positives, intergenic_true_negatives)
test_model(data, 0.8, labels, sequence_length)