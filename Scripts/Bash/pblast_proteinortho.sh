#!/bin/bash

#title           :pblast_proteinortho.sh
#description     :This script performs a pblast with Proteinortho
#                 Proteinortho is slow and modifies the protein directory, so it is usefull to create a backup directory.
#author          :Yaprak Yigit
#date            :21-02-2023
#version         :0.1
#usage           :bash pblast_proteinortho.sh
#bash_version    :4.2.46(2)-release

current_dir=$(pwd)
protein_dir="${current_dir}/Data/Protein_sequences"
proteinortho_dir="${current_dir}/Output/Proteinortho/"

mkdir -p '$proteinortho_dir'

proteinortho -cpus=12 -p=blastp -clean -project=cre_experimental $protein_dir/*.faa

# Check if the proteinortho files exist
files_to_move=("$current_dir"/cre_experimental)

if [ ${#files_to_move[@]} -gt 0 ]; then
    # Move files to the destination folder
    mv "$current_dir"/cre_experimental* "$proteinortho_dir"
    echo "Files moved successfully."
else
    echo "No matching files found."
fi
