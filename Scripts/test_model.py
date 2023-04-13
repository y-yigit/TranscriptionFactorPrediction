#!usr/bin/env python3

"""
__author__ = "Yaprak Yigit"
__version__ = "0.1"
"""

from regulon import Regulon
from dna import DNA
import numpy as np
from CNN import CNN
import tensorflow as tf
import os
import pandas
from Bio import SeqIO

# import screed # a library for reading in FASTA/FASTQ

# Enable logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

sequence_file = "/home1/p312436/features_yy.csv"
cnn_dataset = "/home1/p312436/features_yy_final.csv"
fasta_folder = "/home1/p312436/Data/Intergenic_ccpA"
test_file = "/home1/p312436/Data/Genomes/Neisseria_gonorrhoeae_FA_1090_ASM684v1_genomic.fna"

max_length = 51


def read_kmers_from_file(filename: str, ksize: int) -> list:
    """ Screen a fasta file and create all possible sequence of a certain length

    :param filename: A path to a fasta file
    :param ksize: Size of the k-mers
    :return: A list of k-mers

    .. note:: This is Jelmers function
    """
    all_kmers = []
    with open(filename) as open_file:
        for record in SeqIO.parse(open_file, "fasta"):
            sequence = str(record.seq)
            kmers = []
            n_kmers = len(sequence) - ksize + 1
            for i in range(n_kmers):
                kmers.append(sequence[i:i + ksize])
            kmers = DNA.one_hot_encoder(kmers)
            all_kmers += kmers
    return all_kmers


def create_labels(positive_data: list, negative_data: list) -> tuple:
    """ Create the labels for the CNN model

    :param positive_data: Positive sequences
    :param negative_data: Negatives sequences
    :return: A tensor object and numpy array
    """
    # Create labels for the data
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


def save_labels(input_file: str):
    """ Extend DNA sequences from a dataset and save the resulting dataset

    :param input_file: A path to a csv file
    """
    label_dataframe = pandas.read_csv(input_file, engine="python", sep=",")
    final_dataframe = pandas.DataFrame(columns=["sequence", "species", "transcription_factor", "modified_sequence"])

    for specie in pandas.unique(label_dataframe["species"]):
        sequences = label_dataframe[label_dataframe["species"] == specie]["sequence"].tolist()
        dna_obj = DNA(sequences)
        # Search for files in the directory that contain the substring specie
        standard_file = "Bacillus_subtilis_subsp_subtilis_str_168_ASM904v1_genomic.g2d.intergenic.ffn"
        matches = [os.path.join(fasta_folder, file) if specie in file else os.path.join(fasta_folder, standard_file) for
                   file in os.listdir(fasta_folder)]

        nucleotide_percentages = DNA(matches[0]).calculate_nucleotide_percentages()
        dna_obj.change_sequence_lengths(max_length, nucleotide_percentages)
        positive_sequences = dna_obj.dna_sequences
        negative_sequences = [dna_obj.generate_random_dna(max_length, nucleotide_percentages) for i in range(0, 1000)]
        feature_dataframe = pandas.DataFrame(columns=["sequence", "species", "transcription_factor"])
        feature_dataframe["modified_sequence"] = positive_sequences + negative_sequences
        feature_dataframe["sequence"] = sequences + ["random"] * len(negative_sequences)
        feature_dataframe["species"] = [specie] * (len(positive_sequences)) + ["random"] * len(negative_sequences)
        feature_dataframe["transcription_factor"] = ["ccpA"] * len(positive_sequences) + ["random"] * len(
            negative_sequences)
        final_dataframe = final_dataframe.append(feature_dataframe)
    final_dataframe.to_csv("features_yy_final.csv", sep='\t', index=False)


def prepare_input(input_file: str) -> tuple:
    """ Reads labels from a csv file and encodes them
    :param filename: A path to a csv file
    """
    label_dataframe = pandas.read_csv(input_file, engine="python", sep="\\t", lineterminator="\r")
    positives = label_dataframe[label_dataframe["species"] != ("random")]["modified_sequence"].tolist()
    negatives = label_dataframe[label_dataframe["species"] == ("random")]["modified_sequence"].tolist()

    encoded_positives = DNA.one_hot_encoder(positives)
    encoded_negatives = DNA.one_hot_encoder(negatives)
    return len(positives[0]), encoded_positives, encoded_negatives


def test_model(data: object, test_set_fraction: float, labels: list, sequence_length, model_name: str):
    """ Test a CNN model
    :param data: A tensor object of the input data
    :param test_set_fraction: A fraction below 1
    :param labels: A numpy array of labels
    :param sequence_length: The length of the longest sequence
    """
    # The values for input_shape might not be correct
    my_model = CNN((sequence_length, 4,), model_name)
    my_model.divide_labels(data, labels, test_set_fraction)
    my_model.compile_model()
    my_model.report_summary()
    my_model.train_model(30, 32)


# Save the input data with extended sequences
save_labels(sequence_file)

# Dataset for evaluating the model
test = read_kmers_from_file(test_file, max_length)

sequence_length, encoded_positives, encoded_negatives = prepare_input(cnn_dataset)

data, labels = create_labels(encoded_positives, encoded_negatives)
test_model(data, 0.8, labels, sequence_length, "cnn_lstm_4")
my_model = CNN((sequence_length, 4,), "cnn_lstm_4")
my_model.evaluate_model(test, data, labels)
