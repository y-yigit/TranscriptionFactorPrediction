#!usr/bin/env python3

"""
This script processes DNA motifs and classifies them with a CNN model

Variables:
    max_length (int):
        The length for features

    window_size (int):
        The length of the cropped intergenic sequences
        The combined sizes of the window_size and motif length (14) are equal to the variable max_length

.. todo:: Store paths in a config or bash file
.. todo:: Test hypertuning
.. todo:: Save model + datasets
"""
__author__ = "Yaprak Yigit"
__version__ = "0.1"

from sklearn.model_selection import train_test_split

from dna import DNA
import numpy as np
from CNN import CNN
from regulon import Regulon
import itertools
import csv
import random
import os
import pandas
from Bio import SeqIO
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
import sys
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

# Enable logging
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

regulon_file = "/home1/p312436/Regulations_Bacillus_subtilis.txt"
#motif_dataset = "/home1/p312436/final_motifs_filtered.fasta"
motif_dataset = "/home1/p312436/experimental_motifs.fasta"
#motif_dataset = "/home1/p312436/motifs.fasta"
non_motifs = "/home1/p312436//non_motifs_filtered.fasta"
trained_model = "/home1/p312436/Streptococcus_suis_model.h5"
intergenic_folder = "/home1/p312436/Data/Intergenic"
train_file = "/home1/p312436/Data/CNN_datasets/CcpA_negatives_train30_34.npy"
val_file = "/home1/p312436/Data/CNN_datasets/CcpA_negatives_val30_34.npy"
test_set = "/home1/p312436/Streptococcus_pyogenes_A20_ASM30753v1_genomic.g2d.intergenic.ffn"
max_length = 54
window_size = 40
# Enable logging
import re
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

one_hot_encoder = OneHotEncoder(categories='auto')

def extract_intergenic_sequences(fasta_file):
    """ Reads a fasta file and extracts sequences with a length of 64

    The selected sequences are one-hot encoded before they are returned

    :param input_file: A path to a fasta file containing intergenic sequences
    :return: A list of one-hot encoded sequences
    """
    dna_obj = DNA(fasta_file)
    non_motifs = dna_obj.dna_sequences
    negative_data = []
    ids = []
    for non_motif, id in zip(non_motifs, dna_obj.ids):
        for i in range(0, len(non_motif), 3):
            chunk = non_motif[i:i+max_length]
            if len(chunk) == max_length:
                    negative_data.append(chunk)
                    ids.append(id)
                    ids.append(id)
                    # Add the reverse strand as well
                    negative_data.append(chunk[::-1])

    shuffled_indexes = [i for i in range(0, len(negative_data))]
    negative_data = [negative_data[index] for index in shuffled_indexes]
    ids = [ids[index] for index in shuffled_indexes]
    encoded_negatives = DNA.one_hot_encoder(list(set(negative_data)))
    #print(encoded_negatives)
    return (encoded_negatives, ids)

def create_negative_labels(fasta_file, motifs, test_specie):
    """ Reads a fasta file and extracts sequences with a length of 64

    The selected sequences are one-hot encoded before they are returned

    :param input_file: A path to a fasta file containing intergenic sequences
    :return: A list of one-hot encoded sequences
    """
    non_motifs_train = []
    ids = []
    non_motifs_test = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        specie = record.id.split("|")[0]
        # I checked with meme so this is unnecessary
        if record.seq != "" and len(record.seq) >= max_length and any(motif in record.seq for motif in motifs) == False:
            for i in range(len(record.seq) - max_length + 1):
                non_motif = record.seq[i: i + max_length]
                if specie != test_specie:
                    non_motifs_train.extend([non_motif, non_motif[::-1]])
                if specie == test_specie:
                    non_motifs_test.extend([non_motif, non_motif[::-1]])

    shuffle_indexes = lambda shuffle_list: [i for i in range(len(shuffle_list))]
    non_motifs_train = [non_motifs_train[index] for index in shuffle_indexes(non_motifs_train)]
    non_motifs_test = [non_motifs_test[index] for index in shuffle_indexes(non_motifs_test)]
    return DNA.one_hot_encoder(list(set(non_motifs_train))), DNA.one_hot_encoder(list(set(non_motifs_test)))

def create_labels(positive_data, negative_data, ids=None):
    """ Create the labels for the CNN model

    :param positive_data: One hot encoded positive sequences
    :type positive_data: list
    :param negative_data: One hot encoded negatives sequences
    :type negative_data: list
    :return: A tensor object and numpy array
    :rtype: tuple
    """


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
    if ids != None:
        id_arrays = np.concatenate((ids[0], ids[1]))
        shuffled_ids = id_arrays[permutation]
        return shuffled_data, shuffled_labels, shuffled_ids.tolist()
    else:
        return shuffled_data, shuffled_labels

def process_motifs(label_dataframe: object, augment: bool) -> list:
    """ Reads a dataframe and processes all the sequences

    This function searches for related files for each species and the processes the motifs in the sequences
    with the DNA module and the function find_motifs. This function returns data that can be used in a CNN model.

    :param label_dataframe: A pandas dataframe with the columns species and sequence
    :param augment: Whether or not to augment the data
    :return: A list of numpy arrays that are one-hot encoded DNA motifs
    ..note:: The intergenic folder contains relevant data for all species, but the name does not always fully match
    """
    positive_labels = []

    # Process each species separately in order to preserve their unique GC content
    for specie in pandas.unique(label_dataframe["species"]):
        motifs = label_dataframe[label_dataframe["species"] == specie]["sequence"].tolist()

        specie_name_formatted = "_".join(specie.split()[0:2])
        # Search for files in the directory that contain the substring specie
        matches = [os.path.join(intergenic_folder, file) for file in os.listdir(intergenic_folder) if specie_name_formatted in file]
        # Sometimes the full scientific name does not have matches, in that case check for the genus or the species
        try:
            test = matches[0]
        except IndexError as er:
            split_name = specie.split()[0]
            matches = [os.path.join(intergenic_folder, file) for file in os.listdir(intergenic_folder) if split_name[0] in file]
            if matches == None:
                matches = [os.path.join(intergenic_folder, file) for file in os.listdir(intergenic_folder) if split_name[1] in file]
        matched_sequences = DNA(matches[0])
        nucleotide_percentages = matched_sequences.calculate_nucleotide_percentages()
        sequences = matched_sequences.dna_sequences

        padded_sequences = find_motifs(sequences, motifs, nucleotide_percentages, augment)

        random.shuffle(padded_sequences)
        #print(padded_sequences)
        encoded_sequences = DNA.one_hot_encoder(padded_sequences)
        positive_labels += encoded_sequences
    return positive_labels

def process_test_motifs(motifs):
    dna_obj = DNA(test_set)
    gc_content = dna_obj.calculate_nucleotide_percentages()
    new_motifs = []
    non_motif_sequences = DNA(non_motifs).dna_sequences
    passed_motifs = []
    for motif in motifs:
        for sequence in dna_obj.dna_sequences:
            motif_matches = re.finditer(motif, sequence)

            for match in motif_matches:
                start_index = match.start()
                end_index = match.end()
                base_size = (54 - len(motif)) // 2
                remainder = (54 - len(motif)) % 2
                parts = [base_size + 1] * remainder + [base_size] * (2 - remainder)

                if start_index and len(sequence[start_index:end_index]) - len(motif) >= max_length:
                    new_motif = sequence[start_index-parts[0]:start_index] + motif + sequence[end_index:end_index+parts[1]]
                    new_motifs.append(new_motif)
                else:
                    new_motif = sequence[start_index - parts[0]:start_index] + motif + sequence[end_index:end_index + parts[1]]
                    random_dna = dna_obj.generate_random_dna(max_length - len(new_motif), gc_content)
                    index = random.randint(0, len(random_dna))
                    new_motifs.append(random_dna[:index] + new_motif + random_dna[index:])
            if [match for match in motif_matches] == [] and motif not in passed_motifs:
                passed_motifs.append(motif)
                random_dna = dna_obj.generate_random_dna(max_length-len(motif), gc_content)
                index = random.randint(0, len(random_dna))
                new_motifs.append(random_dna[:index] + motif + random_dna[index:])
    return DNA.one_hot_encoder(new_motifs)

def find_motifs(sequences: list, motifs: list, nucleotide_percentages, augment: bool) -> list:
    """ Tries to find motifs in a list of sequences

    The function will search for the motif in each sequence, the sequences are already expected to contain the motif
    If the motif is found then it will be extended with the upstream and downstream sequences.
    It will be extended with zeros or a combination of upstream sequences, downstream sequences, and zeros
    if there are less than 50 surrounding nucleotides on both sides.

    The motif will be shifted on both sides if the augment parameter is True and if the motif is found in the sequence

    :param sequences: A list of DNA sequences containing intergenic regions from the ccpA regulon
    :param motifs: A list of cre motifs
    :param augment: Whether or not to augment the data
    :return: A list with the padded motifs
    """
    padded_motifs = []
    base_percentage = lambda seq: {base: round((seq.count(base) / len(seq)) * 100) for base in ["A", "C", "G", "T"] if seq.count(base) != 0}
    for motif in motifs:
        amount_of_padding = max_length - len(motif)
        for sequence in sequences:
            motif_matches = re.finditer(motif, sequence)

            padded_motif = "0" * amount_of_padding + motif
            padded_motifs.extend([padded_motif, padded_motif[::-1]])

            for match in motif_matches:
                start_index = match.start()
                end_index = match.end()
                if start_index and len(sequence) + len(motif) >= max_length:
                    for i in range(0, 50):
                        position = random.randint(0, amount_of_padding)
                        head, tail = (sequence[start_index - position:start_index], sequence[end_index: end_index + amount_of_padding - position])

                        gc_content_head, gc_content_tail = base_percentage(head), base_percentage(tail)
                        new_head = replace_string_characters(head, DNA.generate_random_dna(len(head), gc_content_head), 70)
                        new_tail = replace_string_characters(tail, DNA.generate_random_dna(len(tail), gc_content_tail), 70)
                        padded_motif = new_head + motif + new_tail
                        # Moat recent change i made, hyperpar optimization next
                     #   padded_motif = head + motif + tail
                        if len(padded_motif) == max_length:
                            padded_motifs.extend([padded_motif, padded_motif[::-1]])
                        else:
                            padding = DNA.generate_random_dna(max_length-len(padded_motif), nucleotide_percentages)
               #             padded_motif = padding[:position] + padded_motif + padding[position:]
                            padded_motif = "0" * (max_length-len(padded_motif)) + padded_motif
                            padded_motifs.extend([padded_motif, padded_motif[::-1]])
              #  else: # dit is ook nieuw
              #      for i in range(0, 50):
              #          padding = DNA.generate_random_dna(max_length - len(motif), nucleotide_percentages)
             #           padded_motif = padding[:position] + motif + padding[position:]
            #            padded_motifs.extend([padded_motif, padded_motif[::-1]])
    return list(set(padded_motifs))
def replace_string_characters(original_string, replacement_string, percent):
    num_chars = int(len(original_string) * percent / 100)
    chars_to_replace = random.sample(range(len(original_string)), num_chars)
    modified_string = ''.join(random.choice(replacement_string) if i in chars_to_replace else char for i, char in enumerate(original_string))
    return modified_string

def find_conserved_positions(fasta_file: str) -> tuple:
    """ Reads a fasta file containing motifs and calculates the most conserved nucleotide for each position

    Only positions that have a nucleotide frequency of 75% or higher are considered conserved
    :param fasta_file: A path to a fasta file
    :return: mutations, a list of tuples with a nucleotide postion, the nucleotide, and the frequency
             motifs, a list of the motifs in the fasta file
             dna_set, a string of DNA based on the GC percentage in the fasta file
    """
    conserved = 75
    mutations = []
    dna_obj = DNA(fasta_file)
    motifs = dna_obj.dna_sequences
    nucleotide_percentages = dna_obj.calculate_nucleotide_percentages()
    dna_set = "".join([base * nucleotide_percentages[base] for base in nucleotide_percentages])

    # First count each nucleotide at each position
    nucleotide_frequencies = nucleotide_frequencies = {index:{"A":0,"G":0,"T":0,"C":0} for index in range(0,14)}
    for motif in motifs:
        for index, nucleotide in enumerate(motif):
            nucleotide_frequencies[index][nucleotide] += 1

    # Then calculate the nucleotide percentages
    for nucleotide_position in nucleotide_frequencies:
        for nucleotide in nucleotide_frequencies[nucleotide_position]:
            frequency = 100/len(motifs)*nucleotide_frequencies[nucleotide_position][nucleotide]
            if frequency >= conserved:
                mutations.append((nucleotide_position, nucleotide))
    return mutations, motifs, dna_set

def prepare_test_set():
    ccpa_regulon = Regulon(regulon_file, "\\t")
    ccpa_regulon = ccpa_regulon.filter_on_regulator(ccpa_regulon.regulon_dataframe, "regulator name", "ccpA", option="select")
    cre_genes = ccpa_regulon["gene locus"].tolist()
    sequences = []
    labels = []
    ids = []
    with open(test_set, "r") as open_file:
        for record in SeqIO.parse(open_file, "fasta"):
            sequence = str(record.seq)
            id = record.id.split("|")[0]
            for i in range(0, len(sequence), 3):
                chunk = sequence[i:i + max_length]
                if len(chunk) == max_length:
                    sequences.extend([chunk, chunk[::-1]])
                    ids.extend([id]*2)
                    if id in cre_genes:
                        labels.extend([1]*2)
                    else:
                        labels.extend([0]*2)
    return DNA.one_hot_encoder(sequences), tf.data.Dataset.from_tensor_slices(np.array(labels)), ids

def hyper_tune(train_data, validation_data, test_data, train_labels, validation_labels, test_labels, test_ids = None):
    try:
        hypermodel = tf.keras.models.load_model(trained_model+"htjddhgdr")
    except:
        my_model = CNN((max_length, 30,))
        my_model.set_model((max_length, 30), "model_lstm_bi")
        my_model.set_input(train_data, validation_data, test_data, train_labels, validation_labels)
        #my_model.compile_model()

        hypermodel, history, best_epoch = my_model.create_hyper_model()
    #    hypermodel.save("validation_model23222.h5")

    pred_probs = hypermodel.predict(test_data)
  #  pred_probs = my_model.train_model(200, 32)

    for prob in [0.5,0.6,0.7,0.8,0.9]:
        pred_probs = (pred_probs > prob).astype(int)
        # Get the predicted label index for each input data point
        pred_labels_idx = np.argmax(pred_probs, axis=1)

        # Map the predicted label
        label_mapping = {0: "Non-motif", 1: "Motif"}
        pred_labels = [label_mapping[idx] for idx in pred_labels_idx]

        true_positives, true_negatives, false_positives, false_negatives = (0,0,0,0)
        cm = confusion_matrix(np.argmax(test_labels, axis=1), pred_labels_idx)
        TP = cm[1][1]
        FP = cm[0][1]
        FN = cm[1][0]
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * (precision * recall) / (precision + recall)

        print(f"F1 score: {f1} \nPrecision: {precision} \nRecall: {recall} \nConfusion Matrix :\n{cm}")

def main():
    """ The main function prepares the data and then calls the CNN model

    It leaves every organism out of the training and validation dataset once, so that it can be used in the test data.
    The end results such as the F1 and precision score are written to a file.
    """
    species = []
    sequences = []

    # Extract the species and sequences from a fasta file
    for record in SeqIO.parse(motif_dataset, "fasta"):
        headerless = record.id.split("|")[0]
        # Select only the scientific name
        temp_specie = "_".join(headerless.split("_")[0:2])
        species.append(temp_specie)
        sequences.append(str(record.seq))

    label_dataframe = pandas.DataFrame({'species': species, 'sequence': sequences})

    # The cre motifs validated on, around 20% of the total motifs
    validation_species_motifs = ["Streptococcus_pneumoniae", "Clostridium_difficile"]

    train_motifs = label_dataframe.loc[~label_dataframe["species"].isin(validation_species_motifs + ["Streptococcus_pyogenes"])]
    val_motifs = label_dataframe.loc[label_dataframe["species"].isin(validation_species_motifs)]
    # Misses 40 instances, check
    test_motifs = label_dataframe.loc[label_dataframe["species"] == "Streptococcus_pyogenes", "sequence"].tolist()




    #####################################################################################################################
    # Wat ik hier moet doen, pipeline voor genereren van training/validatie sets aanpassen
    # Meer species, want ik wil van elke gc_groep 100 000 sequenties, zo liefst gevarieerd mogelijk
    # Voor elke groep moeten ook een aantal organismen er uit, zodat het 20% validatie data heeft
    #
    #
    # Deze regel wordt overschreven
    update_this, encoded_negatives_test = create_negative_labels(non_motifs, train_motifs, "Streptococcus_pyogenes")
    #encoded_negatives_train = DNA.one_hot_encoder(lines)
    #####################################################################################################################
    # Verbetering
    # Load the NumPy array from file
    train_negatives = np.load(train_file, allow_pickle=True)
    val_negatives = np.load(val_file, allow_pickle=True)


    # Validation
    train_motifs = process_motifs(train_motifs, True)
    val_motifs = process_motifs(val_motifs, True)
    test_motifs = process_test_motifs(test_motifs)
    print(len(train_motifs), len(val_motifs), len(test_motifs))

    train_data, train_labels = create_labels(train_motifs, train_negatives.tolist())
    validation_data, validation_labels = create_labels(val_motifs, val_negatives.tolist())

    # This is fine
    test_data, test_labels = create_labels(test_motifs, encoded_negatives_test)
    print(len(train_data), len(validation_data), len(test_data))

    results = hyper_tune(train_data, validation_data, tf.convert_to_tensor(test_data, dtype=tf.float32), train_labels, validation_labels, test_labels)


if __name__ == "__main__":
    sys.exit(main())

