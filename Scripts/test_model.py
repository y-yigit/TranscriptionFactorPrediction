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

    The function will also add additional bases if the length of the sequence is smaller than max_length

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
    for dna_dictionary in dna_dictionaries:
        dna_obj = DNA(list(dna_dictionary.values()))
        dna_obj.change_sequence_lengths(max_length, nucleotide_percentages)
        encoded_sequences = dna_obj.one_hot_encoder(dna_obj.dna_sequences)
        nested_lists.append(encoded_sequences)
    return nested_lists

def get_random_negatives(count: int, nucleotide_percentages: dict, number_of_labels: int) -> list:
    """ Creates sequences for true negative labels based on the nucleotide percentage of a fasta file

    :param count: Integer specifying the total number of bases
    :type count: int
    :param nucleotide_percentages: A dictionary containing the keys "A", "G", "C", and "T" with floats as values
    :type nucleotide_percentages: dict
    :param number_of_labels: The number of labels to create
    :type nucleotide_percentages: int


    :return: A list of numpy arrays
    :rtype: list
    .. todo:: Test this method with the CNN model
    """
    random_sequences = [DNA.generate_random_dna(count, nucleotide_percentages) for label in range(0, number_of_labels)]
    encoded_sequences = DNA.one_hot_encoder(random_sequences)
    return encoded_sequences

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

    # Combine data and labels into two separate arrays
    data = np.concatenate((positive_data, negative_data))
    labels = np.concatenate((positive_labels, negative_labels))

    # Shuffle data and labels
    permutation = np.random.permutation(len(data))
    shuffled_data = data[permutation]
    shuffled_labels = labels[permutation]
    return shuffled_data, shuffled_labels

def test_model(data, test_set_fraction, labels, sequence_length, model_name):
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
    my_model = CNN((sequence_length, 4,), model_name)
    my_model.divide_labels(data, labels, test_set_fraction)
    my_model.compile_model()
    my_model.report_summary()
    my_model.train_model()

def get_fimo_data(directory) -> tuple:
    positives = []
    negatives = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".tsv"):
                fimo_dataframe = pandas.read_csv(root + "/" + file, engine="python", sep="\\t", lineterminator="\r")
                fimo_dataframe.dropna(subset=["matched_sequence"], inplace=True)
                positive_sequences = fimo_dataframe["matched_sequence"].tolist()
                sequence_length = len(positive_sequences[0])
                dna_obj = DNA(positive_sequences)
                nucleotide_percentages = DNA(fasta_file).calculate_nucleotide_percentages()
                # Add additional bases to align motifs
                dna_obj.change_sequence_lengths(sequence_length+40, nucleotide_percentages)
                encoded_sequences = dna_obj.one_hot_encoder(dna_obj.dna_sequences)
                positives += encoded_sequences
                # Change the fasta file for each loop once the data is ready
                negatives += get_random_negatives(sequence_length+40, nucleotide_percentages, len(positive_sequences))

    data, labels = create_labels(positives, negatives)
    return data, labels, sequence_length+40#1,2,3#shuffled_data, shuffled_labels, sequence_length

# Get the nucleotide percentages using the whole file (GC content)
nucleotide_percentages = DNA(fasta_file).calculate_nucleotide_percentages()

# Extract the necessary data
dna_dictionaries = prepare_input_data()
whole_sequence_length = find_longest_sequence(dna_dictionaries)
intergenic_true_negatives, true_positives = encode_many_sequences(dna_dictionaries, nucleotide_percentages,
                                                                  whole_sequence_length)
# Create labels and modify the data for tensorflow
random_true_negatives = get_random_negatives(whole_sequence_length, nucleotide_percentages, len(true_positives))

# Test with intergenic true negatives
# There are not enough intergenic true negative labels yet, so this is commented out
#intergenic_data, intergenic_labels = create_labels(true_positives, intergenic_true_negatives)
#test_model(intergenic_data, 0.8, intergenic_labels, whole_sequence_length, "cnn_lstm_4")

# Test with true negatives from random sequences
random_data, random_labels = create_labels(true_positives, random_true_negatives)
test_model(random_data, 0.8, random_labels, whole_sequence_length, "cnn_lstm_4")
# Random true negative performs worse than intergenic true negatives, probably because there are less intergenic labels

# Use motifs found by FIMO instead of whole sequences
fimo_data, fimo_labels, fimo_sequence_length = get_fimo_data(fimo_folder)
test_model(fimo_data, 0.8, fimo_labels, fimo_sequence_length, "cnn_lstm_4")

# Testing different models, the main goal is to find a model that can also classify the unprocessed sequences well
#test_model(random_data, 0.8, random_labels, whole_sequence_length, "cnn_blstm_2")
# Higher accuracy but more loss
#test_model(fimo_data, 0.8, fimo_labels, fimo_sequence_length, "cnn_blstm_2")

# Less loss and less accuracy
#test_model(random_data, 0.8, random_labels, whole_sequence_length, "cnn_lstm_small_7_1_1_5_new")
#test_model(fimo_data, 0.8, fimo_labels, fimo_sequence_length, "cnn_lstm_small_7_1_1_5_new")

# Less loss and less accuracy
#test_model(random_data, 0.8, random_labels, whole_sequence_length, "cnn_lstm_small_7_1_1_4_new")
#test_model(fimo_data, 0.8, fimo_labels, fimo_sequence_length, "cnn_lstm_small_7_1_1_4_new")
