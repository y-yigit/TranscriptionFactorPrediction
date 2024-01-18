#!/bin/bash

#title           :test_new_model.sh
#description     :
#author          :Yaprak Yigit
#date            :23-02-2023
#version         :0.1
#usage           :
#bash_version    :4.2.46(2)-release

#SBATCH --job-name=test_cre_motifs
#SBATCH --output=/home1/p312436/Log/model_out.out
#SBATCH --error=/home1/p312436/Error/model_error.err
#SBATCH --time=23:59:59
#SBATCH --mem=500gb
#SBATCH --nodes=1
#SBATCH --export=NONE
#SBATCH --get-user-env=L
#SBATCH --partition=himem

current_dir=$(pwd)
motif_dataset="${current_dir}/Output/Motifs/all_motifs.fasta"
intergenic_folder="${current_dir}/Data/Intergenic_sequences"
train_file="${current_dir}/Output/CNN_datasets/CcpA_negatives_train30_39.npy"
val_file="${current_dir}/Output/CNN_datasets/CcpA_negatives_val30_39.npy"
regulon_file="${current_dir}/Data/Regulons/s_pneumoniae.tsv"
test_set="${current_dir}/Data/Intergenic_sequences/Streptococcus_pneumoniae_D39_ASM1436v2_genomic.g2d.intergenic.ffn"
regulons="${current_dir}/Data/Regprecise_regulons"
test_specie="Streptococcus_pneumoniae"

# Run the Python script with the variables
python Scripts/Python/test_model.py --motif_dataset "$motif_dataset" --intergenic_folder "$intergenic_folder" \
                      --train_file "$train_file" --val_file "$val_file" --regulon_file "$regulon_file" \
                      --test_set "$test_set" --regulons "$regulons" --test_specie "$test_specie"
