#!usr/bin/env python3

"""
This script processes DNA motifs and classifies them with a CNN model

Variables:
    max_length (int):
        The length of the features
"""

__author__ = "Yaprak Yigit"
__version__ = "0.1"

import os
import random
import numpy as np
import pandas
import sys
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import argparse
from keras.utils import plot_model
from dna import DNA
from CNN import CNN

max_length = 50


import re
# Enable logging
# #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
one_hot_encoder = OneHotEncoder(categories='auto')

def my_roc(test_labels, pred_labels, test_specie):
    """ Create a ROC curve after classifying
    """
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(test_labels, pred_labels)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {test_specie.lower().replace("_", " ")}')
    plt.legend(loc='lower right')
    plt.savefig(f"Output/Images/roc_{test_specie.lower()}.png")
    plt.clf()


def plot_confusion_matrix(tp, fp, fn, tn, test_specie):
    """ Create a confusion matrix after classifying
    """
    matrix = np.array([[tp, fp], [fn, tn]])

    # Plotting the confusion matrix
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {test_specie.lower().replace("_", " ")}')
    plt.colorbar()

    classes = ['Positive', 'Negative']
    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, str(matrix[i, j]), ha='center', va='center', color='white' if matrix[i, j] > matrix.max() / 2 else 'black')
    plt.savefig(f"Output/Images/cm_{test_specie.lower()}.png")
    plt.clf()

def get_regulon_genes(directory_path):
    """ Extracts regulon genes from a file

    :param directory_path: Path to a directory containing regulon files
    :return: List of accession numbers from regulon genes
    """
    gene_list = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            df = pandas.read_csv(file_path, sep='\t', header=None, names=['col1', 'col2', 'col3'])
            selected_values = df['col2'].tolist() + [''.join(str(value).split('_')) for value in df['col2'].tolist() if "_" in str(value)]
        if selected_values:
            gene_list.extend(selected_values)
    return gene_list
def create_labels(positive_data, negative_data, ids=None):
    """ Create labels for the CNN model

    :param positive_data: A list of one hot encoded motif sequences
    :param negative_data: A list of one hot encoded non-motif sequences

    :return: A tensor object and numpy array
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

def process_motifs(label_dataframe: object, regulon_genes, intergenic_folder) -> list:
    """ Reads a dataframe and processes all the sequences

    This function searches for related files for each species and the processes the motifs in the sequences
    with the DNA module and the function find_motifs. This function returns data that can be used in a CNN model.

    :param label_dataframe: A pandas dataframe with the columns species and sequence
    :param regulon_genes: A list of genes regulated by the CcpA regulon

    :return: A list of numpy arrays that are one-hot encoded DNA motifs
    ..note:: The intergenic folder contains relevant data for all species, but the name does not always fully match
    """
    positive_sequences = []

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

        matched_sequences = []
        with open(matches[0], "r") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                if any(str(item) in record.id for item in regulon_genes):
                    matched_sequences.append(str(record.seq))
                if any(str(item) in "".join(record.id.split("_")) for item in regulon_genes):
                    matched_sequences.append(str(record.seq))
        padded_sequences = find_motifs(matched_sequences, motifs)

        random.shuffle(padded_sequences)

        encoded_sequences = DNA.one_hot_encoder(padded_sequences)
        positive_sequences += encoded_sequences
    return positive_sequences

def find_motifs(sequences: list, motifs: list) -> list:
    """ Tries to find motifs in a list of sequences

    The function will search for the motif in each sequence, the sequences are already expected to contain the motif
    If the motif is found then it will be extended with the upstream and downstream sequences.
    It will be extended with zeros or a combination of upstream sequences, downstream sequences, and zeros
    if there are less than 50 surrounding nucleotides on both sides.

    The motif will be shifted on both sides if the augment parameter is True and if the motif is found in the sequence

    :param sequences: A list of DNA sequences containing intergenic regions from the CcpA regulon
    :param motifs: A list of CRE motifs
    :return: A list with padded motifs
    """
    base_percentage = lambda seq: {base: round((seq.count(base) / len(seq)) * 100) for base in ["A", "C", "G", "T"] if
                                   seq.count(base) != 0}
    padded_motifs = []
    for motif in motifs:
        amount_of_padding = max_length - len(motif)
        padded_motif = "0" * amount_of_padding + motif
        for sequence in sequences:
            motif_matches = re.finditer(motif, sequence)

            padded_motifs.extend([padded_motif])

            for match in motif_matches:
                start_index = match.start()
                end_index = match.end()
                if start_index and len(sequence) + len(motif) >= max_length:
                    for i in range(0, 80):
                        position = random.randint(0, amount_of_padding)
                        head, tail = (sequence[start_index - position:start_index],
                                      sequence[end_index: end_index + amount_of_padding - position])

                        gc_content_head, gc_content_tail = base_percentage(head), base_percentage(tail)
                        new_head = replace_string_characters(head, DNA.generate_random_dna(len(head), gc_content_head),
                                                             70)
                        new_tail = replace_string_characters(tail, DNA.generate_random_dna(len(tail), gc_content_tail),
                                                             70)
                        padded_motif = new_head + motif + new_tail
                        if len(padded_motif) == max_length:
                            padded_motifs.extend([padded_motif])
                        else:
                            padded_motif = "0" * (max_length - len(padded_motif)) + padded_motif
                            padded_motifs.extend([padded_motif])


    print(len(padded_motifs), len(list(set(padded_motifs))))
    return list(set(padded_motifs))

def replace_string_characters(original_string, replacement_string, percentage):
    """ Randomly replaces string characters based on a percentage

    :param original_string: String that will undergo modifications
    :param replacement_string: String from which the replacement characters are taken
    :param percentage: A number preferable between 0 and 100
    :return: Newly generated string
    """
    num_chars = int(len(original_string) * percentage / 100)
    chars_to_replace = random.sample(range(len(original_string)), num_chars)
    modified_string = ''.join(random.choice(replacement_string) if i in chars_to_replace else char for i, char in enumerate(original_string))
    return modified_string

def create_test_set(regulon_file, test_set):
    """ Creates a test set for the CNN model

    :param regulon_file: TSV file with motif binding sites downloaded from CollecTF
    :param test_set: Fasta file containing sequences of the same species and strain as the TSV file
    :return: - sequences, a list of DNA sequences
             - labels, a list of binary labels
             - ids, a list of sequence identifiers
    """
    ccpa_regulon = pandas.read_csv(regulon_file,sep='\\t', engine='python')
    cre_genes = ccpa_regulon["regulated genes (locus_tags)"].tolist()
    cre_genes = [gene.replace("_", "") for gene in cre_genes if gene !=  None]
    sequences = []
    labels = []
    ids = []
    with open(test_set, "r") as open_file:
        for record in SeqIO.parse(open_file, "fasta"):
            sequence = str(record.seq)
            id = record.id.split("|")[0]
            id = id.replace("_", "")
            for i in range(0, len(sequence), 3):
                chunk = sequence[i:i + max_length]
                if len(chunk) == max_length:
                    sequences.extend([chunk])
                    ids.extend([id]*1)
                    if id in cre_genes:
                        labels.extend([1]*1)
                    else:
                        labels.extend([0]*1)
    return sequences, labels, ids

def run_classification(train_data, validation_data, test_data, train_labels, validation_labels, test_labels, model_name, ids, test_specie):
    """ Executes the CNN model and reports the results

    The data should be TensorFlow tensors, and the labels should undergo transformation using a one_hot_encoder
    Both examples are in the main function.
    """
    # Set the input shape
    my_model = CNN((max_length))
    my_model.set_model((max_length,), model_name)
    my_model.set_input(train_data, validation_data, test_data, train_labels, validation_labels)
    pred_probs = my_model.run_model(test_specie)
    pred_probs = pred_probs.tolist()

    # The test label format is different
    test_labels = one_hot_encoder.inverse_transform(test_labels).flatten()

    # Change the threshold
    for prob in [0.5, 0.6, 0.7, 0.8, 0.9]:
        pred_labels_idx = [1 if predictions[1] >= prob else 0 for predictions in pred_probs]
        true_positives, true_negatives, false_positives, false_negatives = (0, 0, 0, 0)

        incorrect_labels = []
        found_motifs = []
        corrected_pred_labels = []
        # Calculate TP, FP, TN, FN
        for index, (prediction, true_value) in enumerate(zip(pred_labels_idx, test_labels)):
            current_id = ids[index]
            if prediction == 1 and prediction == true_value:
                true_positives += 1
                corrected_pred_labels.append(1)
                found_motifs.append(current_id)
            elif prediction == 0 and prediction == true_value:
                corrected_pred_labels.append(0)
                true_negatives += 1
            elif prediction == 1 and prediction != true_value:
                false_positives += 1
                corrected_pred_labels.append(0)
            elif prediction == 0 and prediction != true_value:
                corrected_pred_labels.append(2)
                incorrect_labels.append(current_id)

        # Correction for the file reading function
        for idx, label in enumerate(incorrect_labels):
            # When a motif is classified as a non motif and there are other ids of that motif are classified correctly
            # This chunk might not contain the motif by chance
            if label in found_motifs:
                corrected_pred_labels[idx] = 1
            # When a motif is classified as a non motif and there are no other ids of that motif classified correctly
            else:
                corrected_pred_labels[idx] = 0
                false_negatives += 1

        # Show metrics and visualizations
        try:
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
            f1 = 2 * (precision * recall) / (precision + recall)
            accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
            plot_confusion_matrix(true_positives, false_positives, false_negatives, true_negatives, test_specie)
            plot_model(my_model.model, to_file='model_visualization.png', show_shapes=True)
            my_roc(test_labels, pred_labels_idx, test_specie)
            print(f"F1 score: {f1} \n Accuracy {accuracy} \nPrecision: {precision} \nRecall: {recall} \n"
                  f"Confusion Matrix: \n {true_negatives} \t {false_positives}\n{false_negatives} \t {true_positives}")
        except Exception as error:
            print(f"\nConfusion Matrix :\n {true_negatives} \t {false_positives}\n{false_negatives} \t {true_positives}"
                  f"\n Something might be wrong")

def parse_arguments():
    """ Argument parser

    The train_file and val_file should be numpy files
    :return: args
    """
    parser = argparse.ArgumentParser(description="Your script description here.")
    parser.add_argument('--motif_dataset', type=str,
                        help="Path to the motif dataset.")
    parser.add_argument('--intergenic_folder', type=str,
                        help="Path to the intergenic folder.")
    parser.add_argument('--train_file', type=str,
                        help="Path to the training file.")
    parser.add_argument('--val_file', type=str,
                        help="Path to the validation file.")
    parser.add_argument('--regulon_file', type=str,
                        help="Path to the CollecTF regulon file")
    parser.add_argument('--test_set', type=str,
                        help="Path to the test set file.")
    parser.add_argument('--regulons', type=str,
                        help="Path to the directory containing multiple regulons from CollecTF")
    parser.add_argument('--test_specie', type=str,
                        help="Specie for the test set.")
    args = parser.parse_args()
    return args

def main():
    """ The main function prepares the data and then calls the CNN model.
    """

    motif_dataset, intergenic_folder, train_file, val_file, regulon_file, test_set, regulons, test_specie = vars(parse_arguments()).values()

    file_name = f"Output/CNN_datasets/{test_specie}.pkl"
    try:
        with open(file_name, 'rb') as file:
            loaded_data = pickle.load(file)
        train_data = loaded_data["train_data"][:20000]
        train_labels = loaded_data["train_labels"][:20000]
        validation_data = loaded_data["validation_data"][:4000]
        validation_labels = loaded_data["validation_labels"][:4000]
    except:
        species = []
        sequences = []
        regulon_genes = get_regulon_genes(regulons)

        # Extract the species and sequences from a fasta file
        for record in SeqIO.parse(motif_dataset, "fasta"):
            headerless = record.id.split("|")[0]
            temp_specie = "_".join(headerless.split()[0:2])
            if temp_specie not in test_specie:
                species.append(temp_specie)
                sequences.append(str(record.seq))

        label_dataframe = pandas.DataFrame({'species': species, 'sequence': sequences})
        train_motifs, val_motifs = train_test_split(label_dataframe, test_size=0.2, random_state=42)

        train_motifs = process_motifs(train_motifs, regulon_genes, intergenic_folder)
        val_motifs = process_motifs(val_motifs, regulon_genes, intergenic_folder)

        train_negatives = np.load(train_file, allow_pickle=True)
        train_motifs = train_motifs[:20000]

        val_negatives = np.load(val_file, allow_pickle=True)
        val_motifs = val_motifs[:4000]

        train_data, train_labels = create_labels(train_motifs, train_negatives.tolist())
        validation_data, validation_labels = create_labels(val_motifs, val_negatives.tolist())

        # Create a dictionary to store the variables
        data_to_save = {
            "train_data": train_data,
            "train_labels": train_labels,
            "validation_data": validation_data,
            "validation_labels": validation_labels
        }

        # Open the file in binary write mode and save the data
        with open(file_name, 'wb') as file:
            pickle.dump(data_to_save, file)

    sequences, labels, ids = create_test_set(regulon_file, test_set)

    my_model = "base"
    run_classification(train_data, validation_data, tf.convert_to_tensor(DNA.one_hot_encoder(sequences), dtype=tf.float32),
              train_labels, validation_labels, one_hot_encoder.fit_transform(np.array(labels).reshape(-1, 1)).toarray(), my_model, ids, test_specie)


if __name__ == "__main__":
    sys.exit(main())
