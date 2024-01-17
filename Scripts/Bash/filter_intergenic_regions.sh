#!/bin/bash

#title           :filter_intergenic_regions.sh
#description     :Scan a set of sequences for the motif MX000100
#author          :Yaprak Yigit
#date            :21-02-2023
#version         :0.1
#usage           :bash FIMO.sh -i Data/Intergenic/ -o FIMO_results/
#notes           :MEME version 5.5.0 is required
#bash_version    :4.2.46(2)-release

current_dir=$(pwd)

# Paths related to data or database files
db="${current_dir}/Data/MEME_database/collectf.meme"
gc_report=${current_dir}/Output/Motifs/data.csv
intergenic_dir=${current_dir}/Data/Intergenic_sequences
protein_dir=${current_dir}/Data/Protein_sequences
regulon_dir=${current_dir}/Data/Regulons
proteinortho_file=${current_dir}/Output/Proteinortho/cre_experimental.proteinortho.tsv

# Output paths
cnn_dir=${current_dir}/Output/CNN_datasets
cre_motifs_to_remove=${current_dir}/Output/Non_motifs/cre_motifs_to_remove.fasta
filtered_non_motifs=${current_dir}/Output/Non_motifs/non_motifs_filtered.fasta
meme_dir=${current_dir}/Output/MEME_remove
non_motifs=${current_dir}/Output/Non_motifs/non_cre_motifs.fasta
fimo_dir=${current_dir}/Output/FIMO_remove
proteinortho_dir="${current_dir}/Output/Proteinortho/"

# Other variables
pvalue=1e-4
motif_list="EXPREG_00001520 EXPREG_00001810 EXPREG_000009c0 EXPREG_00000d10 EXPREG_000000e0"
export PATH=$HOME/meme/bin:$HOME/meme/libexec/meme-5.5.2:$PATH

mkdir -p $cnn_dir $fimo_dir ${current_dir}/Output/Non_motifs

# Select the genes found by proteinortho that should not have cre
python3 ${current_dir}/Scripts/Python/extract_blast_genes.py drop $non_motifs $intergenic_dir $proteinortho_file $regulon_dir

########################################################################################################################################################
# Search for cre in case there are still motifs left
for motif_name in $motif_list
do
  fimo -oc $fimo_dir/$motif_name -norc -thresh $pvalue -motif $motif_name $db $non_motifs
done

# Clean the non motifs and generate 
python3 ${current_dir}/Scripts/Python/data_filtering.py $fimo_dir $cre_motifs_to_remove
python3 ${current_dir}/Scripts/Python/clean_non_motifs.py $non_motifs $cre_motifs_to_remove $filtered_non_motifs
python3 ${current_dir}/Scripts/Python/gc_based_sorter.py --intergenic_dir $intergenic_dir --data_dir $cnn_dir --non_motifs $filtered_non_motifs --data_file $gc_report

