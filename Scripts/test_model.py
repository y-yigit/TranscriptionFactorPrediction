#!usr/bin/env python3

"""
__author__ = "Yaprak Yigit"
__version__ = "0.1"
"""

from dna import DNA
import numpy as np
from CNN import CNN
import tensorflow as tf
import os
import pandas
from Bio import SeqIO
from sklearn.preprocessing import OneHotEncoder

# import screed # a library for reading in FASTA/FASTQ

# Enable logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

train_sequences = "/home/ubuntu/Yaprak/train_sequences.csv"
test_sequences = "/home/ubuntu/Yaprak/test_sequences.csv"

train_dataset = "/home/ubuntu/Yaprak/train_dataset.csv"
test_dataset = "/home/ubuntu/Yaprak/test_dataset.csv"
fasta_folder = "/home/ubuntu/Yaprak/Data/Intergenic_ccpA"
intergenic_file = "/home/ubuntu/Yaprak/Data/Intergenic/Bacillus_amyloliquefaciens_L-S60_ASM97348v1_genomic.g2d.intergenic.ffn"

max_length = 51


def read_kmers_from_file(filename, ksize):
    all_kmers = []
    with open(filename) as open_file:
        for record in SeqIO.parse(open_file, "fasta"):
            sequence = str(record.seq)

            kmers = []
            n_kmers = len(sequence) - ksize + 1

            for i in range(n_kmers):
                kmer = sequence[i:i + ksize]
                kmers.append(kmer)

            kmers = DNA.one_hot_encoder(kmers)
            all_kmers += kmers

    return all_kmers


def create_labels(positive_data, negative_data):
    """ Create the labels for the CNN model

    :param positive_data: Positive sequences
    :type positive_data: list
    :param negative_data: Negatives sequences
    :type negative_data: list
    :return: A tensor object and numpy array
    :rtype: tuple
    """
    one_hot_encoder = OneHotEncoder(categories='auto')

    positive_labels = [1 for label in range(0, len(positive_data))]
    negative_labels = [0 for label in range(0, len(negative_data))]

    # Combine data and labels
    data = np.concatenate((positive_data, negative_data))
    labels = np.array(positive_labels + negative_labels).reshape(-1, 1)
    # Shuffle data and labels
    permutation = np.random.permutation(len(data))
    shuffled_data = data[permutation]
    # Extra transformation for labels
    shuffled_labels = one_hot_encoder.fit_transform(labels[permutation]).toarray()
    return shuffled_data, shuffled_labels


def save_labels(input_file, output_file):
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
    final_dataframe.to_csv(output_file, sep='\t', index=False)


def prepare_input(input_file):
    label_dataframe = pandas.read_csv(input_file, engine="python", sep="\\t", lineterminator="\r")
    positives = label_dataframe[label_dataframe["species"] != ("random")]["modified_sequence"].tolist()
    negatives = label_dataframe[label_dataframe["species"] == ("random")]["modified_sequence"].tolist()

    encoded_positives = DNA.one_hot_encoder(positives)
    encoded_negatives = DNA.one_hot_encoder(negatives)

    return encoded_positives, encoded_negatives


def test_model(train_data, test_data, train_labels, test_labels, model_name):
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
    print("Training the model")
    my_model = CNN((max_length, 4,), model_name)
    my_model.set_input(train_data, test_data, train_labels, test_labels)
    my_model.compile_model()
    my_model.report_summary()
    my_model.train_model(30, 32)
    # my_model.evaluate_model(test_data)


# Save the input data
save_labels(train_sequences, "/home/ubuntu/Yaprak/train_dataset.csv")
save_labels(test_sequences, "/home/ubuntu/Yaprak/test_dataset.csv")

intergenic_data = read_kmers_from_file(intergenic_file, max_length)
# Generate train data
encoded_positives, encoded_negatives = prepare_input(train_dataset)
train_data, train_labels = create_labels(encoded_positives, encoded_negatives)
# Generate test data

encoded_positives, encoded_negatives = prepare_input(test_dataset)
test_data, test_labels = create_labels(encoded_positives, encoded_negatives)
# identifies multiple classes
# test_model(train_data, test_data, train_labels, test_labels, "cnn_lstm_4")
test_model(train_data, test_data, train_labels, test_labels, "cnn_lstm_4")

