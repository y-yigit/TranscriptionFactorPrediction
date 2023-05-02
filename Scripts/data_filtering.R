## ----------------------------------------------------------------------------
## Script name: combine_data.R
##
## Purpose: This script filters motifs based on FIMO results
##
## Author: Yaprak Yigit
##
## Date Created: 02-05-2023
##
## Email: y.yigit@st.hanze.nl
## ----------------------------------------------------------------------------
output_file <- "/home/ubuntu/Yaprak/cnn_dataset.csv"

c_difficile <- read.table("/home/ubuntu/Yaprak/FIMO/EXPREG_00000d10/fimo.tsv", sep="\t", header=TRUE)
l_lactis <- read.table("/home/ubuntu/Yaprak/FIMO/EXPREG_00001520/fimo.tsv", sep="\t", header=TRUE)
motif_dataset <- read.table("/home/ubuntu/Yaprak/motif_dataset.csv", sep =",", header=TRUE)

# Find the overlapping motifs and remove them from dataframe l_lactis
overlap <- intersect(l_lactis$sequence_name, c_difficile$sequence_name)
l_lactis <- l_lactis[!l_lactis %in% overlap]

checked_motifs <- rbind(c_difficile, l_lactis)
cnn_dataset <- motif_dataset[motif_dataset$index %in% checked_motifs$sequence_name,]
write.csv(cnn_dataset, output_file)