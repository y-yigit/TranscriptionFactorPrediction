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
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

# Enable logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

motif_dataset = "/home/ubuntu/Yaprak/experimental_motifs.fasta"
non_motifs = "/home/ubuntu/Yaprak/non_motifs_filtered.fasta"

train_dataset = "/home/ubuntu/Yaprak/Data/CNN_datasets/train_dataset.csv"
test_dataset = "/home/ubuntu/Yaprak/Data/CNN_datasets/test_dataset.csv"
intergenic_folder = "/home/ubuntu/Yaprak/Data/Intergenic"

max_length = 64
window_size = 50

def extract_non_motifs(fasta_file):
    """ Reads a fasta file and extracts sequences with a length of 64

    The selected sequences are one-hot encoded before they are returned

    :param input_file: A path to a fasta file containing intergenic sequences
    :return: A list of one-hot encoded sequences
    """
    dna_obj = DNA(fasta_file)
    non_motifs = dna_obj.dna_sequences
    negative_labels = []
    for non_motif in non_motifs:
        kmers = []
        for i in range(0, len(non_motif), 3):
            chunk = non_motif[i:i+max_length]
            if len(chunk) == max_length:
                negative_labels.append(chunk)
                # Add the reverse strand as well
                negative_labels.append(chunk[::-1])
        negative_labels += kmers
    random.shuffle(negative_labels)
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
            split_name  = specie.split()[0]
            matches = [os.path.join(intergenic_folder, file) for file in os.listdir(intergenic_folder) if split_name[0] in file]
            if matches == None:
                matches = [os.path.join(intergenic_folder, file) for file in os.listdir(intergenic_folder) if split_name[1] in file]
        matched_sequences = DNA(matches[0])
        nucleotide_percentages = matched_sequences.calculate_nucleotide_percentages()
        sequences = matched_sequences.dna_sequences

        padded_sequences = find_motifs(sequences, motifs, augment)
        encoded_sequences = DNA.one_hot_encoder(padded_sequences)
        positive_labels += encoded_sequences
    return positive_labels


def find_motifs(sequences: list, motifs: list, augment: bool) -> list:
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
    for sequence in sequences:
        for motif in motifs:
            if motif in sequence:
                # Find the index of the motif in the sequence and select upstream and downstream sequences
                motif_index = sequence.find(motif)
                start_point = motif_index - window_size
                end_point = motif_index + window_size
                head = sequence[start_point:motif_index]
                tail = sequence[motif_index:end_point]
                # The data is not augmented if there are less than 50 nucleotides upstream or downstream
                if len(head) < window_size or len(tail) < window_size or augment == False:
                    shortened_head, shortened_tail = (head, tail)
                    halved_window_size = round(window_size/2)
                    if len(head) >= halved_window_size:
                        shortened_head = head[0:halved_window_size]
                    if len(tail) >= halved_window_size:
                        shortened_tail = tail[0:halved_window_size]
                    padded_motif = "0" * (window_size - (len(shortened_head) + len(shortened_tail)))  + shortened_head + motif + shortened_tail
                    padded_motifs.append(padded_motif)
                    padded_motifs.append(padded_motif[::-1])
                # Augment if there are enough surrounding nucleotides
                elif len(head) >= window_size or len(tail) >= window_size and augment == True:
                    for index in range(0, window_size):
                        padded_motif = head[:window_size-index] + motif + tail[0:index]
                        padded_motifs.append(padded_motif)
                        padded_motifs.append(padded_motif[::-1])
    return padded_motifs


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

def create_fake_motifs(fasta_file: str, n_motifs: int) -> list:
    """ Creates sequences that are similar to motif sequences, but cannot be motifs

    The new sequences have 2 to 5 substitutions in the whole motif and additional substitutions
    in all the conserved nucleotides.

    :param fasta_file: A path to a fasta file
    :n_motifs: Number of motifs to mutate

    .. note:: conserved_nucleotides is a list of tuples, each tuple has 2 elements
              The first element is the index of the nucleotide in the motif and the second is the nucleotide

    :return: A list with one-hot encoded sequences that diverged from motifs
    """
    conserved_nucleotides, motifs, dna_set = find_conserved_positions(fasta_file)
    fake_motifs = []
    for i in range(0, n_motifs):
        random_motif = list(random.choice(motifs))
        nucleotide_tuples = random.sample(conserved_nucleotides, 3)
        # Check if all the conserved nucleotides occur in the motif and skip the motif if they do not
        if all(nucleotide_tuples[i][1] != random_motif[nucleotide_tuples[i][0]] for i in range(3)):
            # Increase n_motifs if the motif gets skipped, so that the number of output motifs is equal to n_motifs
            n_motifs += 1
        else:
            # Add random mutations in the whole motif
            random_positions = [random.randint(0, 13) for i in range(0, random.randint(2, 5))]
            for position in random_positions:
                random_motif[position] = random.choice(dna_set)

            # Mutate the conserved motifs
            for nucleotide_tuple in nucleotide_tuples:
                # Create a list of nucleotides without the conserved nucleotide
                possible_mutations =  list(filter(lambda base: base != nucleotide_tuple[1], dna_set))
                random_motif[nucleotide_tuple[0]] = random.choice(possible_mutations)
        # Create random DNA based on the GC content of the organism to surround the fake motif with
        random_intergenic_dna = "".join(random.sample(dna_set, window_size))
        # Insert the fake motif in a random position
        insert_position = random.randint(0, len(random_intergenic_dna))
        final_motif = random_intergenic_dna[:insert_position] + "".join(random_motif) + random_intergenic_dna[insert_position:]
        fake_motifs.append(final_motif)
    return DNA.one_hot_encoder(fake_motifs)

def test_model(train_data, test_data, train_labels, test_labels, model_name):
    """ Trains a CNN model and then tests the model

    The train data and labels are split into train and validation datasets

    :param train_data: A list of numpy arrays containing train data
    :param test_data: A list of numpy arrays containing test data
    :param train_labels: A list of numpy arrays containing train labels
    :param test_labels: A list of numpy arrays containing test labels
    :param model_name: The name of the model defined in the CNN module
    """
    my_model = CNN((max_length, 30,), model_name)
    my_model.set_input(train_data, test_data, train_labels, test_labels)
    my_model.compile_model()
    # The numbers stand for 100 epochs and 32 mini-batches
    my_model.train_model(10, 32)
    my_model.predict_labels(my_model.test_data, my_model.test_labels)

def hypter_tune(train_data, test_data, train_labels, test_labels, model_name):
    tuner = kt.Hyperband(my_model.model_builder,
                         objective='val_accuracy',
                         max_epochs=10,
                         factor=3,
                         directory='my_dir',
                         project_name='intro_to_kt')
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    my_model = CNN((max_length, 30,))
    my_model.set_input(train_data, test_data, train_labels, test_labels)
    tuner.search(my_model.train_data, my_model.train_labels, epochs=100, validation_split=0.2, callbacks=[stop_early])

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Build the model with the optimal hyperparameters and train it on the data for 100 epochs
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(img_train, label_train, epochs=100, validation_split=0.2)

    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

    #Re-instantiate the hypermodel and train it with the optimal number of epochs from above.
    hypermodel = tuner.hypermodel.build(best_hps)
    # Retrain the model
    hypermodel.fit(y_model.train_data, my_model.train_labels, epochs=best_epoch, validation_split=0.2)
    eval_result = hypermodel.evaluate(my_model.test_data, my_model.test_labels)


def main():
    """ The main function prepares the data and then calls the CNN model

    It leaves every organism out of the training and validation dataset once, so that it can be used in the test data.
    The end results such as the F1 and precision score are written to a file.
    """

    species = []
    sequences = []

    fake_motifs = create_fake_motifs(motif_dataset, 5000)

    # parse the fasta file and append headers and sequences to the lists
    for record in SeqIO.parse(motif_dataset, "fasta"):
        species.append(record.id.split("|")[0])
        sequences.append(str(record.seq))

    # create a pandas dataframe with two columns
    label_dataframe = pandas.DataFrame({'species': species, 'sequence': sequences})
    train_sequences = label_dataframe[label_dataframe["species"].str.contains("Streptococcus_pyogenes") == False]
    test_sequences = label_dataframe[label_dataframe["species"].str.contains("Streptococcus_pyogenes")]

    encoded_negatives = extract_non_motifs(non_motifs)
    encoded_positives_train = process_motifs(train_sequences, True)
    encoded_positives_test = process_motifs(test_sequences, False)
    #np.save("train_positives.npy", np.array(encoded_positives_train, dtype=object), allow_pickle=True)
  #  np.load("train_positives.npy", allow_pickle=True)
    #np.save("test_positives.npy", np.array(encoded_positives_test, dtype=object), allow_pickle=True)
   # np.save("train_negatives.npy", np.array(encoded_negatives[0:100000], dtype=object), allow_pickle=True)
    #np.save("test_negatives.npy", np.array(encoded_negatives[100000:200000], dtype=object), allow_pickle=True)
    
    train_data, train_labels = create_labels(encoded_positives_train, encoded_negatives[0:100000])#+ fake_motifs)
    test_data, test_labels = create_labels(encoded_positives_test, encoded_negatives[100000:200000])

    test_model(train_data, test_data, train_labels, test_labels, "model_lstm_bi")
    hyper_tune(train_data, test_data, train_labels, test_labels, "model_lstm_bi")
    # test_model(train_data, test_data, train_labels, test_labels, "model_gaussian")

if __name__ == "__main__":
    sys.exit(main())

