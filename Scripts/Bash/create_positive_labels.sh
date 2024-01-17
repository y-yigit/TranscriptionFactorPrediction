#!/bin/bash

#title           :create_positive_labels.sh
#description     :This script created the input dataset of cre motifs for the transcription factor prediction program.
#                 It first combines experimentally found motifs and validates the results through MEME.
#                 Then it will predict new cre motifs with FIMO and MEME.
#                 Lastly, the resulting datasets are combined.
#author          :Yaprak Yigit
#date            :21-02-2023
#version         :0.1
#usage           :bash create_positive_labels.sh
#notes           :MEME version 5.5.0 is required
#bash_version    :4.2.46(2)-release

#SBATCH --job-name=FIMO
#SBATCH --Output=/data/pg-bactprom/Yaprak/Log/FIMO_out.out
#SBATCH --error=/data/pg-bactprom/Yaprak/Error/FIMO_error.err
#SBATCH --time=00:59:59
#SBATCH --mem=8gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --export=NONE
#SBATCH --get-user-env=L
#SBATCH --qos=priority

# Location MEME
export PATH=$HOME/meme/bin:$HOME/meme/libexec/meme-5.5.2:$PATH
current_dir=$(pwd)

# Paths related to data or database files
db="${current_dir}/Data/MEME_database/collectf.meme"
intergenic_dir="${current_dir}/Data/Intergenic_sequences"
protein_dir="${current_dir}/Data/Protein_sequences"
regprecise_dir="${current_dir}/Data/Motifs/Regprecise"
regulon_file="${current_dir}/Data/Regulons/"

# Output paths
raw_motifs="${current_dir}/Output/Motifs/motifs.fasta"
experimental_motifs="${current_dir}/Output/Motifs/experimental_motifs.fasta"
motifs_completed="${current_dir}/Output/Motifs/motifs_completed.fasta"
fimo_motifs="${current_dir}/Output/Motifs/fimo_motifs.fasta"
motifs_w_regprecise="${current_dir}/Output/Motifs/all_motifs.fasta"
predicted_motifs="${current_dir}/Output/Motifs/predicted_motifs.fasta"
fimo_dir="${current_dir}/Output/FIMO"
meme_experimental_dir="${current_dir}/Output/MEME_experimental"
meme_predicted_dir="${current_dir}/Output/MEME_predicted"
proteinortho_dir="${current_dir}/Output/Proteinortho/"

# Regprecise data
files=$(find $regprecise_dir -type f)

# Other variables
pvalue="1e-4"
motif_list="EXPREG_00001520 EXPREG_00001810 EXPREG_000009c0 EXPREG_00000d10 EXPREG_000000e0"
duplicate_species="Bacillus_subtilis Paenibacillus Clostridium_acetobutylicum Clostridium_difficile "\
"Lactobacillus_casei Lactobacillus_sakei Lactobacillus_plantarum Staphylococcus_aureus Lactococcus_lactis "\
"Streptococcus_gordonii Streptococcus_pneumoniae Streptococcus_pyogenes Streptococcus_suis"

# Make output directories
mkdir -p ${current_dir}/Output/Motifs ${current_dir}/Output/FIMO $proteinortho_dir

################################################################################
# Combine and filter data from experimentally predicted cre sites
# The filtering method has been dropped, since it reduces the models F1 score
################################################################################
Rscript ${current_dir}/Scripts/R/combine_data.R

#meme $raw_motifs -oc $meme_experimental_dir -maxw 16 -minw 14
#python3 ${current_dir}/Scripts/Python/meme_reader.py $meme_experimental_dir/meme.txt $experimental_motifs

experimental_motifs=$raw_motifs
################################################################################
# Predict cre sites in ortholog genes of the ccpA regulon and move the proteinortho files
################################################################################

proteinortho -cpus=12 -p=blastp -clean -project=cre_experimental $protein_dir/*.faa

# Check if the proteinortho files exist
files_to_move=("$current_dir"/cre_experimental)

if [ ${#files_to_move[@]} -gt 0 ]; then
    # Move files to the destination folder
    mv "$current_dir"/cre_experimental* "$proteinortho_dir"/
    echo "Files moved successfully."
else
    echo "No matching files found."
fi

python3 ${current_dir}/Scripts/Python/extract_blast_genes.py $fimo_motifs $intergenic_dir $proteinortho_dir/cre_experimental.proteinortho.tsv $regulon_file

for motif_name in $motif_list
do
  fimo -oc $fimo_dir/$motif_name -norc -thresh $pvalue -motif $motif_name $db $fimo_motifs
done

# Filter for the CcpA regulon
python3 ${current_dir}/Scripts/Python/data_filtering.py $fimo_dir $fimo_motifs

meme $fimo_motifs -oc $meme_predicted_dir -maxw 16 -minw 14 -brief 100000#
python3 ${current_dir}/Scripts/Python/meme_reader.py $meme_predicted_dir/meme.txt $predicted_motifs 

################################################################################
# Combine both motif files
################################################################################
python3 ${current_dir}/Scripts/Python/merge_fasta.py $experimental_motifs $motifs_completed --additional_files $predicted_motifs --duplicate_species Bacillus_subtilis

################################################################################
# Add Regprecise data
################################################################################

# Converting newlines to spaces in the 'files' variable
files=${files//$'\n'/ }
python3 ${current_dir}/Scripts/Python/merge_fasta.py $motifs_completed $motifs_w_regprecise --additional_files ${files[@]} --duplicate_species ${duplicate_species[@]}
