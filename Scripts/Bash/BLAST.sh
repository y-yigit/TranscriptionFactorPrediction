#!/bin/bash

#title           :BLAST.sh
#description     :Performs a bi-directional blastp
#author          :Yaprak Yigit
#date            :23-02-2023
#version         :0.1
#usage           :bash BLAST.sh
#notes           :Requires ProteinOrtho version 6.0.29
#bash_version    :4.2.46(2)-release

#SBATCH --job-name=ProteinOrtho
#SBATCH --output=/data/pg-bactprom/Yaprak/Log/BLAST_out.out
#SBATCH --error=/data/pg-bactprom/Yaprak/Error/BLAST_error.err
#SBATCH --time=23:59:59
#SBATCH --mem=16gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --export=NONE
#SBATCH --get-user-env=L
#SBATCH --qos=priority

# Modules
module load BioPerl/1.7.7-GCCcore-9.3.0
module load BLAST+/2.13.0-gompi-2022a

# Software
software_dir=/data/pg-bactprom/software

# Exports
export PATH=$software_dir/MEME/bin:$software_dir/MEME/libexec/meme-5.5.0:$PATH
export PERL5LIB=$software_dir/molgentools_mirror/lib

# The fasta sequences directory
input_dir=/data/pg-bactprom/Yaprak/Data/Intergenic

# Proteinortho arguments
cpu=$SLURM_NTASKS_PER_NODE
project_name=intergenic

#blastp -query $query -db $genome -evalue 1e-8 -outfmt 6 -out results.txt
$software_dir/proteinortho/proteinortho-master/proteinortho6.pl -cpus="$cpu" -p=blastp -singles -clean -project="$project_name" "$input_dir"/*.ffn
