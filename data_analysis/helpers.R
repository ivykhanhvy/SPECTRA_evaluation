library(ggpubr)
library(ggbeeswarm)

library(tidyr)
library(reshape)

#my_3_colors = c("green4", "orange","red")
my_colors = c(random="green4", scaffold="orange",umap="red", spectra="grey50")

datasets = c(
  "bace","bbbp","clintox","delaney","freesolv","lipo","sider","tox21"
)
class_datasets = c("delaney", "freesolv", "lipo")
reg_datasets = c("bace","bbbp","clintox","sider","tox21")

create_performance_v_split_plot <- function(data, model_name, metric_name, 
                        comparisons,
                        my_colors, scales="fixed") {
  p <- ggplot(
    data,
    aes(split_type, metric, color=split_type),
  ) +
    geom_quasirandom() +
    scale_color_manual(
      values=my_colors
    ) +
    stat_compare_means(
      comparisons = comparisons,
      method = "wilcox.test",
      paired = FALSE, 
      label = "p.signif",
      step.increase = 0.1,
      vjust=0.5,
      hide.ns=TRUE
    ) +
    labs(
      x = "Split Type",
      y = metric_name
    ) +
    theme_pubr(base_size = 14) +
    theme(
      legend.position = "none",
      axis.title = element_text(face = "bold")
    ) + facet_wrap(~dataset, scales=scales) +
    ggtitle(model_name)
  
  print(p)
}


create_cso_v_split_plot <- function(data, comparisons,
                                    my_colors, scales="fixed") {
  p <- ggplot(
    data,
    aes(split_type, cso, color=split_type),
  ) +
    geom_quasirandom() +
    scale_color_manual(
      values=my_colors
    ) +
    stat_compare_means(
      comparisons = comparisons,
      method = "wilcox.test",
      paired = FALSE, 
      label = "p.signif",
      step.increase = 0.1,
      vjust=0.5,
      hide.ns=TRUE
    ) +
    labs(
      x = "Split Type",
      y = "Cross-Split Overlap (CSO)"
    ) +
    theme_pubr(base_size = 14) +
    theme(
      legend.position = "none",
      axis.title = element_text(face = "bold")
    ) + facet_wrap(~dataset, scales=scales,nrow=2) +
    ylim(0,0.22) + 
    ggtitle("CSO vs Split Type")
  
  print(p)
}


create_cso_v_sp_plot <- function(data,my_colors) {
  p <- ggplot(cso_vs_sp_df, aes(x = SP, y = spectra_mean)) +
    geom_point(aes(color = "spectra"), size = 2) +
    geom_hline(aes(yintercept = random_mean, color = "random"), 
               linetype = "solid", size = 0.6) +
    geom_hline(aes(yintercept = scaffold_mean, color = "scaffold"), 
               linetype = "dashed", size = 0.6) +
    geom_hline(aes(yintercept = umap_mean, color = "umap"), 
               linetype = "dotdash", size = 0.6) +
    
    facet_wrap(~dataset, nrow = 2) +
    
    scale_x_continuous(limits = c(-0.1, 1.1), breaks = seq(0, 1, 0.5)) +
    scale_y_continuous(limits = c(0, 0.18), breaks = seq(0, 0.18, 0.05)) +
    
    scale_color_manual(
      name = "Split type",
      breaks = c("random", "scaffold", "umap", "spectra"),
      values = my_colors) +
    
    labs(
      x = "Spectral Parameter (SP)",
      y = "Cross-Split Overlap (CSO)") +
    
    theme_pubr(base_size = 14) +
    theme(
      legend.position = "right",
      axis.title = element_text(face = "bold"),
      axis.text.x = element_text(angle = 90, hjust = 1)
    )
  print(p)
}


