## ----------------------------------------------------------------------------
## Script name: combine_data.R
##
## Purpose: This script combines motifs found from papers, CollecTF en PRODRIC
## Note: The datasets from papers were already filtered for the sequence 
##       and gene columns.
##
## Author: Yaprak Yigit
##
## Date Created: 11-03-2023
##
## Email: y.yigit@st.hanze.nl
## ----------------------------------------------------------------------------

library("readxl")
library("data.table")
library("plyr")
library("ggseqlogo")
library("tidyr")
library("dplyr")
library("ggplot2")

pubmed_dir <- "/home/ubuntu/Yaprak/Data/Motifs/pubmed_articles/"
prodoric_dir <- "/home/ubuntu/Yaprak/Data/Motifs/Prodoric"
collectf_file <- "/home/ubuntu/Yaprak/Data/Motifs/Collectf/all_ccpa.tsv"
fasta_file <- "/home/ubuntu/Yaprak/motifs.fasta"

streptococcus_pyogenes <- read.table(paste(pubmed_dir, "streptococcus_pyogenes_33325565.csv", sep = ""), sep = "\t", header = TRUE)
streptococcus_pyogenes$species <- "streptococcus_pyogenes"
colnames(streptococcus_pyogenes) <- c("sequence", "gene", "species")
streptococcus_pyogenes_filtered <- streptococcus_pyogenes[streptococcus_pyogenes$sequence != "–", ]
streptococcus_pyogenes <- streptococcus_pyogenes_filtered[, c(2,1,3)]
streptococcus_pyogenes$source <- "PMID:33325565"

bacillus_licheniformis <- read_excel(paste(pubmed_dir, "bacillus_licheniformis_33997685.xlsx", sep = ""))
bacillus_licheniformis$species <- "bacillus_licheniformis"
colnames(bacillus_licheniformis) <- c("gene", "sequence", "species")
bacillus_licheniformis$sequence <- toupper(bacillus_licheniformis$sequence)
bacillus_licheniformis$source <- "PMID:33997685"

clostridium_acetobutylicum <- read.table(paste(pubmed_dir, "clostridium_acetobutylicum_28119470.csv", sep = ""), sep = "\t", header = TRUE)
clostridium_acetobutylicum$species <- "clostridium_acetobutylicum"
clostridium_acetobutylicum <- clostridium_acetobutylicum[,c(2,4,7)]
colnames(clostridium_acetobutylicum) <- c("gene","sequence", "species")
clostridium_acetobutylicum$source <- "PMID: 28119470"

lactococcus_lactis <- read.table(paste(pubmed_dir, "lactococcus_lactis_1702870.csv", sep = ""), sep = "\t", header = TRUE)
lactococcus_lactis$species <- "lactococcus_lactis"
lactococcus_lactis <- gather(lactococcus_lactis, key = "col", value = "sequence", cre.sequence, cre2.sequence)
lactococcus_lactis <- lactococcus_lactis[,c(1,4,2)]
colnames(lactococcus_lactis) <- c("gene", "sequence", "species")
lactococcus_lactis <- lactococcus_lactis[!(lactococcus_lactis$sequence == " " | lactococcus_lactis$sequence == "  "  | lactococcus_lactis$sequence == "<"), ]
lactococcus_lactis$source <- "PMID: 1702870"

# Extra motifs
extra_motifs = read.table(paste(pubmed_dir, "extra.csv", sep = ""), sep = "\t", header = TRUE)
eextra_motifs <- extra_motifs[,c(1,3,2,4)]

# Also present in the CollecTF dataset
#bacillus_subtilis <- read.table(paste(pubmed_dir, "DBRTS/bacillus_subtilis.csv", sep =""), sep = "\t", header = TRUE)
#bacillus_subtilis$species <- "bacillus_subtilis"
#bacillus_subtilis <- bacillus_subtilis[,c(2,3,5)]
#colnames(bacillus_subtilis) <- c("gene", "sequence", "species")
#bacillus_subtilis <- bacillus_subtilis[!(is.na(bacillus_subtilis$sequence) | bacillus_subtilis$sequence=="ND" | bacillus_subtilis$sequence==""), ]
transcription_df <- rbind(streptococcus_pyogenes, clostridium_acetobutylicum, lactococcus_lactis, extra_motifs)#, streptococcus_suis, bacillus_subtilis) 
transcription_df$feature <- "ccpA"

# CollecTF data
# Found articles for streptococcus_pyogenes and lactococcus_lactis
collectf_df <- read.table(collectf_file, sep = "\t", header = TRUE)
collectf_df <- subset(collectf_df, !grepl("Streptococcus pyogenes|Lactococcus lactis", organism))
collectf_df <- collectf_df[,c(8,4,1)]
colnames(collectf_df) <- c("sequence", "species", "feature")
collectf_df$source <- "CollecTF"

# Prodoric data
bacillus_megaterium <- read.table(paste(prodoric_dir, "Bacillus_megaterium.csv", sep = "/"), header = TRUE, sep = ",")
clostridioides_difficile <- read.table(paste(prodoric_dir, "Clostridioides_difficile.csv", sep = "/"), header = TRUE, sep = ",")
prodoric_df <- rbind(bacillus_megaterium, clostridioides_difficile)
prodoric_df <- prodoric_df[,c(6,3,2)]
colnames(prodoric_df) <- c("sequence", "species", "feature")
prodoric_df$source <- "PRODORIC"

complete_dataset <- rbind(transcription_df[,2:5], collectf_df, prodoric_df)
complete_dataset$sequence <- as.character(complete_dataset$sequence)
complete_dataset <- complete_dataset[nchar(complete_dataset$sequence) <= 51, ]
complete_dataset$modified_sequence <- complete_dataset$sequence

# A new column containing indexes
complete_dataset$index <- seq(1, nrow(complete_dataset))

# Create a fasta file containing the motifs
open_file <- file(fasta_file, "w")

# Iterate through the motifs and add each motif to the fasta file with a header
for (i in 1:nrow(complete_dataset)) {
  cat(paste0(">", complete_dataset$index[i], "\n"), file = open_file)
  cat(paste0(complete_dataset$sequence[i], "\n"), file = open_file)
}

# Create a dataset of the motifs and additional information
write.csv(complete_dataset, "cnn_dataset.csv")
