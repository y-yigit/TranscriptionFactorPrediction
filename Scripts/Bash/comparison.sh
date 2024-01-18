#!/bin/bash

#title           :comparison.sh
#description     :Compres MAST, MEME, and FIMO
#author          :Yaprak Yigit
#date            :23-05-2023
#version         :0.1
#usage           :bash comparison.sh
#notes           :MEME version 5.5.0 is required
#bash_version    :4.2.46(2)-release

export PATH=$HOME/meme/bin:$HOME/meme/libexec/meme-5.5.2:$PATH
current_dir=$(pwd)

pvalue=1e-4
results=/home/ubuntu/Yaprak/Results/FIMO
motif_file="${current_dir}/Pipeline_output/final_motifs.fasta"
bg_file=/home/ubuntu/Yaprak/converted_motifs.bg
meme_file=/home/ubuntu/Yaprak/output.meme

fasta-get-markov $motif_file > $bg_file
iupac2meme $bg_file > $meme_file

meme $test_file -nmotifs 100 -w 14 -oc $results/MEME -revcomp -pal
mast $meme_file $test_file -oc $results/MAST
fimo -oc $results/FIMO -norc -thresh $pvalue $meme_file $test_file

# List of test files
test_files=(
    /home/ubuntu/Yaprak/Data/Intergenic_all/Lactococcus_lactis_subsp_cremoris_MG1363_ASM942v1_genomic.g2d.intergenic.ffn
    /home/ubuntu/Yaprak/Data/Intergenic_all/Bacillus_subtilis_subsp_subtilis_str_168_ASM904v1_genomic.g2d.intergenic.ffn
    /home/ubuntu/Yaprak/Data/Intergenic_all/Clostridioides_difficile_630_ASM920v1_genomic.g2d.intergenic.ffn
    /home/ubuntu/Yaprak/Data/Intergenic_all/Streptococcus_suis_P1_7_ASM9190v1_genomic.g2d.intergenic.ffn
    /home/ubuntu/Yaprak/Data/Intergenic_all/Streptococcus_pneumoniae_D39_ASM1436v2_genomic.g2d.intergenic.ffn
)

for test_file in "${test_files[@]}"; do
    # Generate a unique identifier for this test file
    test_file_name=$(basename "$test_file" | cut -d. -f1)

    # Create a directory for this test file's results
    test_results_dir="$results/$test_file_name"
    mkdir -p "$test_results_dir"

    # The rest of your commands that process the test file
    fasta-get-markov "$motif_file" > "$test_results_dir/converted_motifs.bg"
    iupac2meme "$test_results_dir/converted_motifs.bg" > "$test_results_dir/output.meme"
    meme "$test_file" -nmotifs 100 -w 14 -oc "$test_results_dir/MEME"
    mast "$test_results_dir/output.meme" "$test_file" -oc "$test_results_dir/MAST"
    fimo -oc "$test_results_dir/FIMO" -norc -thresh "$pvalue" "$test_results_dir/output.meme" "$test_file"
done

