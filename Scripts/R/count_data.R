## ----------------------------------------------------------------------------
## Script name: count_data.R
##
## Purpose: This script counts motif before and after preprocessing
##
## Author: Yaprak Yigit
##
## Date Created: 11-03-2023
##
## Email: y.yigit@st.hanze.nl
## ----------------------------------------------------------------------------

library(seqinr)
library(Biostrings)
library(ggplot2)

filtered_data = "Output/Motifs/experimental_motifs.fasta"
additional_data = "Output/Motifs/predicted_motifs.fasta"
complete_dataset = read.table("Output/Motifs/raw_dataset.csv", sep = ",")[2:7]
names(complete_dataset) <- c("sequence", "species", "source", "feature", "modified_seqeunce", "index")
complete_dataset <- complete_dataset[-1,]
complete_dataset$Species <- sub("Corynebacterium_glutamicumis_", "Corynebacterium_glutamicumis", complete_dataset$species)
raw_counts = data.frame(table(complete_dataset$species))
colnames(raw_counts) <- c("Species", "Amount of motifs found in literature")

count_fasta <- function(fasta_file, df_cols){
  fasta_contents = readDNAStringSet(fasta_file)
  species = names(fasta_contents)
  species <- gsub("\\|.*", "", species)
  sequence = paste(fasta_contents)
  motif_df = data.frame(species, sequence)
  counts_df <- data.frame(table(motif_df$species))
  colnames(counts_df) <- df_cols

  counts_df$Species <- sub("Streptococcus_intermediu", "Streptococcus_intermedius", counts_df$Species)
  counts_df$Species <- sub("Tetragenococcus_halophil", "Tetragenococcus_halophila", counts_df$Species)
  counts_df$Species <- sub("Corynebacterium_glutamic", "Corynebacterium_glutamicum", counts_df$Species)
  counts_df$Species <- sub("Clostridium_acetobutylic", "Clostridium_acetobutylicum", counts_df$Species)
  counts_df$Species <- sub("Streptococcus_oligoferme", "Streptococcus_oligofermentans", counts_df$Species)
  counts_df$Species <- sub("Streptococcus_lutetiensi", "Streptococcus_lutetiensis", counts_df$Species)
  return(counts_df)}

filtered_counts = count_fasta(filtered_data, c("Species", "Amount of motifs after filtering"))
extra_counts = count_fasta(additional_data, c("Species", "Amount of motifs found by orthology"))
merged_df = merge(raw_counts, filtered_counts, by="Species", all=TRUE)
merged_df = merge(merged_df, extra_counts, by="Species", all=TRUE)
merged_df = merged_df[1:26,]
merged_df <- merged_df[merged_df$Species != "species", ]
merged_df <- merged_df[rowSums(is.na(merged_df)) != ncol(merged_df), ]
write.csv(merged_df, "data.csv", row.names = FALSE)

merged_df[is.na(merged_df)] <- 0