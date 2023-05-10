#!usr/bin/env python3

"""
__author__ = "Yaprak Yigit"
__version__ = "0.1"
"""

from dna import DNA
import numpy as np
from CNN import CNN
import itertools
import random
import os
import pandas
from Bio import SeqIO
from sklearn.preprocessing import OneHotEncoder
import sys
#import screed # a library for reading in FASTA/FASTQ

# Enable logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

# Paden in config file

motif_dataset = "/home/ubuntu/Yaprak/final_motif_dataset.fasta"
non_motifs = "/home/ubuntu/Yaprak/non_motifs_filtered.fasta"
# deze gebruiken voor parameters
train_dataset = "/home/ubuntu/Yaprak/Data/CNN_datasets/train_dataset.csv"
test_dataset = "/home/ubuntu/Yaprak/Data/CNN_datasets/test_dataset.csv"
intergenic_folder = "/home/ubuntu/Yaprak/Data/Intergenic"

max_length = 54

def extract_non_motifs(input_file):
    dna_obj = DNA(input_file)
    non_motifs = dna_obj.dna_sequences
    negative_labels = []
    for non_motif in non_motifs:
        kmers = []
        for i in range(0, len(non_motif)):
            chunk = non_motif[i:i+54]
            if len(chunk) == 54:
                negative_labels.append(chunk)
                # Add the reverse strand as well
                negative_labels.append(chunk[::-1])
        negative_labels += kmers
    encoded_negative_labels = DNA.one_hot_encoder(negative_labels)
    return encoded_negative_labels

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

def process_motifs(label_dataframe, augment):
    final_dataframe = pandas.DataFrame(columns=["sequence", "species", "transcription_factor", "modified_sequence"])
    positive_labels = []

    for specie in pandas.unique(label_dataframe["species"]):
        motifs = label_dataframe[label_dataframe["species"] == specie]["sequence"].tolist()


        specie_name_formatted = "_".join(specie.split()[0:2])
        # Search for files in the directory that contain the substring specie
        matches = [os.path.join(intergenic_folder, file) for file in os.listdir(intergenic_folder) if specie_name_formatted in file]
        # Sometimes the full scientific name does not have matches
        try:
            test = matches[0]
        except IndexError as er:
            split_name  = specie.split()[0]
            matches = [os.path.join(intergenic_folder, file) for file in os.listdir(intergenic_folder) if split_name[0] in file]
            if matches == None:
                matches = [os.path.join(intergenic_folder, file) for file in os.listdir(intergenic_folder) if split_name[1] in file]
        matched_sequences = DNA(matches[0])
        sequences = matched_sequences.dna_sequences

        padded_sequences = find_motifs(sequences, motifs, augment)
        encoded_sequences = DNA.one_hot_encoder(padded_sequences)
        positive_labels += encoded_sequences
    return positive_labels

def find_motifs(sequences, motifs, augment):
    padded_motifs = []
    for sequence in sequences:
        for motif in motifs:
            if motif in sequence:
                motif_index = sequence.find(motif)
                starting_point = motif_index - 40
                intergenic_noise = sequence[starting_point:motif_index]
                if starting_point < 0:
                    padded_motifs.append("0" * (40 - len(intergenic_noise)) + intergenic_noise + motif)
                    padded_motifs.append("0" * (40 - len(intergenic_noise)) + intergenic_noise + motif[::-1])
                elif starting_point > 0:
                    if augment == True:
                        for index in range(0, len(intergenic_noise)):
                            middle_point = round(len(intergenic_noise) / 2)
                            augmented_sequence = intergenic_noise[:middle_point] + motif + intergenic_noise[middle_point:]
                            padded_motif = augmented_sequence[index:] + augmented_sequence[0:index]
                            padded_motifs.append(padded_motif)
                            padded_motifs.append(padded_motif[::-1])
                    else:
                        padded_motif = intergenic_noise + motif
                        padded_motifs.append(padded_motif)
                        padded_motifs.append(padded_motif[::-1])
                else:
                    padded_motifs.append("0" * 40 + motif)
                    padded_motifs.append("0" * 40 + motif[::-1])
    return padded_motifs

def prepare_input(input_file):
    """ Reads an input file to a pandas dataframe and ecnodes the sequences

    :param input_file: A string specifying the path to a csv file
    :return: A tuple containing encoded positive and negative labels
    """
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
    my_model = CNN((max_length, 30,), model_name)
    my_model.set_input(train_data, test_data, train_labels, test_labels)
    my_model.compile_model()
   # my_model.report_summary()
    my_model.train_model(100, 32)
    my_model.predict_labels()
    #my_model.evaluate_model(test_data)

def main():

    #! afhankelijk van input parameter
    
    species = []
    sequences = []

    # parse the fasta file and append headers and sequences to the lists
    for record in SeqIO.parse(motif_dataset, "fasta"):
        species.append(record.id.split("|")[0])
        sequences.append(str(record.seq))

    # create a pandas dataframe with two columns
    label_dataframe = pandas.DataFrame({'species': species, 'sequence': sequences})
    train_sequences = label_dataframe[label_dataframe["species"].str.contains("Streptococcus_pyogenes") == False]
    test_sequences = label_dataframe[label_dataframe["species"].str.contains("Streptococcus_pyogenes")]

    # 679928 in total
    encoded_negatives = extract_non_motifs(non_motifs)
    encoded_positives_train = process_motifs(train_sequences, True)
    encoded_positives_test = process_motifs(test_sequences, False)
    #np.save("train_positives.npy", np.array(encoded_positives_train, dtype=object), allow_pickle=True)
  #  np.load("train_positives.npy", allow_pickle=True)
    #np.save("test_positives.npy", np.array(encoded_positives_test, dtype=object), allow_pickle=True)
   # np.save("train_negatives.npy", np.array(encoded_negatives[0:100000], dtype=object), allow_pickle=True)
    #np.save("test_negatives.npy", np.array(encoded_negatives[100000:200000], dtype=object), allow_pickle=True)
    
    train_data, train_labels = create_labels(encoded_positives_train, encoded_negatives[0:100000])
    test_data, test_labels = create_labels(encoded_positives_test, encoded_negatives[100000:200000])
    
    #!

    try:
        test_model(train_data, test_data, train_labels, test_labels, "model_sigme54_weights")
    except Exception as er:
        print(er)

if __name__ == "__main__":
    sys.exit(main())

