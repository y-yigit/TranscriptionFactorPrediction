#!/bin/bash

#title           :FIMO.sh
#description     :Scan a set of sequences for the motif MX000100
#author          :Yaprak Yigit
#date            :21-02-2023
#version         :0.1    
#usage           :bash FIMO.sh -i Intergenic/ -o FIMO_results/
#notes           :MEME version 5.5.0 is required
#bash_version    :4.2.46(2)-release

# Modules
export PATH=/data/pg-bactprom/software/MEME/bin:/data/pg-bactprom/software/MEME/libexec/meme-5.5.0:$PATH

# Paths
db=/data/pg-bactprom/software/MEME/motif_databases/PROKARYOTE/prodoric.meme

while getopts i:o: flag
do
    case "${flag}" in
        i) inputdir=${OPTARG};;
        o) outputdir=${OPTARG};;
    esac
done

#output_dir=/data/pg-bactprom/Yaprak/FIMO_results
#datadir=/data/pg-bactprom/Yaprak/Intergenic
pvalue=1e-4

# Make a new output directory if it does not exists
mkdir -p $outputdir

# Execute FIMO on multiple fasta files
for entry in "$inputdir"/*
do
  species=$(echo "$entry" | sed -r "s/.+\/(.+)\..+/\1/")
  fimo -oc $outputdir/${species} -norc -thresh $pvalue -motif MX000100 $db $entry
done
