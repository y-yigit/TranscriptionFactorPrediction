library(ggplot2)
library(dplyr)
library(tidyr)
library(colorspace)

# Files
c_difficile_matches_fimo <- read.table("/home/ubuntu/Yaprak/Results/Clostridioides_difficile_630_ASM920v1_genomic/FIMO/fimo.tsv", sep="\t", header=TRUE)
b_subtilis_matches_fimo <- read.table("/home/ubuntu/Yaprak/Results/Bacillus_subtilis_subsp_subtilis_str_168_ASM904v1_genomic/FIMO/fimo.tsv", sep="\t", header=TRUE)
l_lactis_matches_fimo <- read.table("/home/ubuntu/Yaprak/Results/Lactococcus_lactis_subsp_cremoris_MG1363_ASM942v1_genomic/FIMO/fimo.tsv", sep="\t", header=TRUE)
s_suis_matches_fimo <- read.table("/home/ubuntu/Yaprak/Results/Streptococcus_suis_P1_7_ASM9190v1_genomic/FIMO/fimo.tsv", sep="\t", header=TRUE)
s_pneumoniae_matches_fimo <- read.table("/home/ubuntu/Yaprak/Results/Streptococcus_pneumoniae_D39_ASM1436v2_genomic/FIMO/fimo.tsv", sep="\t", header=TRUE)

c_difficile_matches_mast <- read.table("/home/ubuntu/Yaprak/Results/Clostridioides_difficile_630_ASM920v1_genomic/MAST/mast_results.csv", sep="\t", header=TRUE)
b_subtilis_matches_mast <- read.table("/home/ubuntu/Yaprak/Results/Bacillus_subtilis_subsp_subtilis_str_168_ASM904v1_genomic/MAST/mast_results.csv", sep="\t", header=TRUE)
l_lactis_matches_mast <- read.table("/home/ubuntu/Yaprak/Results/Lactococcus_lactis_subsp_cremoris_MG1363_ASM942v1_genomic/MAST/mast_results.csv", sep="\t", header=TRUE)
s_suis_matches_mast <- read.table("/home/ubuntu/Yaprak/Results/Streptococcus_suis_P1_7_ASM9190v1_genomic/MAST/mast_results.csv", sep="\t", header=TRUE)
s_pneumoniae_matches_mast <- read.table("/home/ubuntu/Yaprak/Results/Streptococcus_pneumoniae_D39_ASM1436v2_genomic/MAST/mast_results.csv", sep="\t", header=TRUE)


c_difficile_regulon <- read.table("/home/ubuntu/Yaprak/Data/Regulons/collectf_c_difficile.tsv", sep="\t", header=TRUE)
b_subtilis_regulon <- read.table("/home/ubuntu/Yaprak/Data/Regulons/collectf_b_subtilis.tsv", sep="\t", header=TRUE)
l_lactis_regulon <- read.table("/home/ubuntu/Yaprak/Data/Regulons/collectf_l_lactis.tsv", sep="\t", header=TRUE)
s_suis_regulon <- read.table("/home/ubuntu/Yaprak/Data/Regulons/collectf_s_suis.tsv", sep="\t", header=TRUE)
s_pneumoniae_regulon <- read.table("/home/ubuntu/Yaprak/Data/Regulons/s_pneumonia.tsv", sep="\t", header=TRUE)


# For FIMO results
get_metrics <- function(regulon, predictions){
  predictions$sequence_name <- gsub("\\|.*", "", predictions$sequence_name)
  regulon <- regulon %>%
    separate_rows(regulated.genes..locus_tags., sep = ", ")
  # Remove underscores
  predictions$sequence_name <- gsub("_", "", predictions$sequence_name)
  regulon$regulated.genes..locus_tags <- gsub("_", "", regulon$'regulated.genes..locus_tags.')
  
  true_positives <- sum(predictions$sequence_name %in% regulon$'regulated.genes..locus_tags.')
  false_positives <- sum(!predictions$sequence_name %in% regulon$'regulated.genes..locus_tags.')
  precision = true_positives / (true_positives + false_positives)
  false_discovery_rate =  false_positives / (true_positives + false_positives)
  return(list(precision, false_discovery_rate))}

metrics_c_difficile_mast = get_metrics(c_difficile_regulon, c_difficile_matches_mast)
metric_b_subtilis_mast = get_metrics(b_subtilis_regulon, b_subtilis_matches_mast)
metrics_l_lactis_mast = get_metrics(l_lactis_regulon, l_lactis_matches_mast)
metrics_s_suis_mast = get_metrics(s_suis_regulon, s_suis_matches_mast)
metrics_s_pneumoniae_mast = get_metrics(s_pneumoniae_regulon, s_pneumoniae_matches_mast)

metrics_c_difficile_fimo = get_metrics(c_difficile_regulon, c_difficile_matches_fimo)
metric_b_subtilis_fimo = get_metrics(b_subtilis_regulon, b_subtilis_matches_fimo)
metrics_l_lactis_fimo = get_metrics(l_lactis_regulon, l_lactis_matches_fimo)
metrics_s_suis_fimo = get_metrics(s_suis_regulon, s_suis_matches_fimo)
metrics_s_pneumoniae_fimo = get_metrics(s_pneumoniae_regulon, s_pneumoniae_matches_fimo)


precision = unlist(c(0.11, metrics_c_difficile_mast[1], metrics_c_difficile_fimo[1],
              0.18, metric_b_subtilis_mast[1], metric_b_subtilis_fimo[1], 
              0.25 , metrics_l_lactis_mast[1], metrics_l_lactis_fimo[1],
              0.20, metrics_s_pneumoniae_mast[1], metrics_s_pneumoniae_fimo[1],
              0.21, metrics_s_suis_mast[1], metrics_s_suis_fimo[1]))

fdr = unlist(c(0.90, metrics_c_difficile_mast[2], metrics_c_difficile_fimo[2],
              0.82, metric_b_subtilis_mast[2], metric_b_subtilis_fimo[2], 
              0.75 , metrics_l_lactis_mast[2], metrics_l_lactis_fimo[2],
              0.80, metrics_s_pneumoniae_mast[2], metrics_s_pneumoniae_fimo[2],
              0.78, metrics_s_suis_mast[2], metrics_s_suis_fimo[2]))

experiments <- 1:5
# Add pneumonia

# Create a data frame for plot
data <- data.frame(experiments, fdr, precision)

# Create a dataset for precision
specie <- c(rep("Clostridioides difficile" , 3), rep("Bacillus subtilis" , 3), 
            rep("Lactococcus lactis" , 3) , rep("Streptococcus pneumoniae" , 3), 
            rep("Streptococcus suis" , 3))

condition <- rep(c("CNN", "MAST" , "FIMO") , 5)

colourblind_palette <- c("#785EF0", "#DC267F", "#FE6100")
colourblind_palette_muted <-lighten(colourblind_palette, amount = 0.04)

# Create a dataset for FDR
fdr_plot <- ggplot(data, aes(fill=condition, y=fdr, x=specie)) + 
  geom_bar(position="dodge", stat="identity") + 
  ggtitle("Plot of false discovery rate per program") +
  xlab("species") + ylab("False discovery rate")+
  scale_fill_manual(values = colourblind_palette_muted, name = "Program")+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        panel.border = element_blank(), panel.background = element_blank())

precision_plot <- ggplot(data, aes(fill=condition, y=precision, x=specie)) + 
  geom_bar(position="dodge", stat="identity") + 
  ggtitle("Plot of precision per program") +
  xlab("species") + ylab("Precision")+
  scale_fill_manual(values = colourblind_palette, name = "Program")+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        panel.border = element_blank(), panel.background = element_blank())

require(gridExtra)
library(cowplot)
plot_grid(fdr_plot, precision_plot, labels = c("A", "B"))
#grid.arrange(fdr_plot, precision_plot, ncol = 2)
#combined_plot <- grid.arrange(fdr_plot, precision_plot, ncol = 2)

# Save the combined plot to a file
#ggsave("combined_plot.png", combined_plot, width = 10, height = 5)

data <- data.frame(Threshold = c(0.5, 0.6, 0.7, 0.8, 0.9),  
                   B.subtilis = c(0.20, 0.22, 0.23, 0.25, 0.27),  
                   C.difficile = c(0.08, 0.07, 0.06, 0.05, 0.03),   
                   L.Lactis = c(0.25, 0.27, 0.29, 0.30, 0.33),   
                   S.pneumoniae = c(0.25, 0.25, 0.26, 0.27, 0.29),   
                   S.suis = c(0.25, 0.25, 0.25, 0.25, 0.26) )
colourblind_palette <- c("#785EF0", "#DC267F", "#FE6100", "#2FA7E2", "#FFC13D")


ggplot(data, aes(x = Threshold)) +
  geom_line(aes(y = B.subtilis, color = "Bacillus subtilis"), size = 1) +
  geom_line(aes(y = C.difficile, color = "Clostridioides difficile"), size = 1) +
  geom_line(aes(y = L.Lactis, color = "Lactococcus lactis"), size = 1) +
  geom_line(aes(y = S.pneumoniae, color = "Streptococcus pneumoniae"), size = 1) + 
  geom_line(aes(y = S.suis, color = "Streptococcus suis"), size = 1) + 
  labs(title ="F1 score per threshold", x = "Threshold",     y = "F1 Score",     color = "Species"   ) +
  scale_color_manual(values = c("Bacillus subtilis" = "#785EF0", "Clostridioides difficile" = "#DC267F",
                                "Lactococcus lactis" = "#FE6100", "Streptococcus pneumoniae" = "#2FA7E2", "Streptococcus suis" = "#FFC13D")) +
  
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), 
        panel.border = element_blank(), panel.background = element_blank())


