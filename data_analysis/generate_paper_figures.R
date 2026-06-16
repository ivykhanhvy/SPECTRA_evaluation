library(xtable)

source("./data_analysis/helpers.R")

perf_dat <- read.csv("./data_analysis/all_split_performance_data.csv")
cso_dat <- read.csv("./data_analysis/all_split_size_cso_data.csv")

comparisons <- list(
  c("random", "scaffold"),
  c("random", "umap"),
  c("scaffold", "umap")
)


#### PERFORMANCE VS SPLIT_TYPE
reg_df = perf_dat[which(perf_dat$dataset %in% reg_datasets &
                            perf_dat$split_type != "spectra_tanimoto"),]
for (model in unique(reg_df$model)) {
  create_performance_v_split_plot(reg_df[which(reg_df$model == model),], 
              model,"AUC", comparisons, my_colors)
}

class_df = perf_dat[which(perf_dat$dataset %in% class_datasets &
                          perf_dat$split_type != "spectra_tanimoto"),]
for (model in unique(class_df$model)) {
  create_performance_v_split_plot(class_df[which(class_df$model == model),], 
                                  model,"RMSE", comparisons, my_colors)
}

# for creating table
df_agg <- aggregate(metric ~ dataset + model + split_type, 
                        FUN=median, data=rbind(reg_df, class_df))
df_agg_wide <- df_agg %>%
  pivot_wider(
    names_from = split_type,
    values_from = metric
  )
df_agg_wide$s_vs_r <- apply(df_agg_wide, 1, function(x) {
  if (x["dataset"] %in% reg_datasets) {
    scaf_dat <- reg_df[which(reg_df$model == x["model"] & 
                               reg_df$split_type == "scaffold" &
                               reg_df$dataset == x["dataset"]),"metric"]
    rand_dat <- reg_df[which(reg_df$model == x["model"] & 
                               reg_df$split_type == "random" &
                               reg_df$dataset == x["dataset"]),"metric"]
    
  } else {
    scaf_dat <- class_df[which(class_df$model == x["model"] & 
                                 class_df$split_type == "scaffold" &
                                 class_df$dataset == x["dataset"]),"metric"]
    rand_dat <- class_df[which(class_df$model == x["model"] & 
                                 class_df$split_type == "random" &
                                 class_df$dataset == x["dataset"]),"metric"]
  }
  wilcox.test(rand_dat, scaf_dat)$p.value
})

df_agg_wide$u_vs_r <- apply(df_agg_wide, 1, function(x) {
  if (x["dataset"] %in% reg_datasets) {
    umap_dat <- reg_df[which(reg_df$model == x["model"] & 
                               reg_df$split_type == "umap" &
                               reg_df$dataset == x["dataset"]),"metric"]
    rand_dat <- reg_df[which(reg_df$model == x["model"] & 
                               reg_df$split_type == "random" &
                               reg_df$dataset == x["dataset"]),"metric"]
    
  } else {
    umap_dat <- class_df[which(class_df$model == x["model"] & 
                                 class_df$split_type == "umap" &
                                 class_df$dataset == x["dataset"]),"metric"]
    rand_dat <- class_df[which(class_df$model == x["model"] & 
                                 class_df$split_type == "random" &
                                 class_df$dataset == x["dataset"]),"metric"]
  }
  wilcox.test(rand_dat, umap_dat)$p.value
})

p_to_stars <- function(x) {
  if (x < 0.001) {
    return("***")
  } else if (x < 0.01) {
    return("**")
  } else if (x < 0.05) {
    return ("*")
  } else {
    return(".")
  }
}

df_agg_wide$s_vs_r <- sapply(
  p.adjust(df_agg_wide$s_vs_r, method="bonferroni"),
  p_to_stars)
df_agg_wide$u_vs_r <- sapply(
  p.adjust(df_agg_wide$u_vs_r, method="bonferroni"),
  p_to_stars)

df_agg_wide <- df_agg_wide[order(df_agg_wide$dataset),]
pt <- xtable(df_agg_wide)
print(pt,
      include.rownames = FALSE,
      digits = 3)


#### CSO VS SPLIT_TYPE, 8-way facet
int_indices = c(
  grep("random", cso_dat$index),
  grep("scaffold", cso_dat$index),
  grep("umap", cso_dat$index))

cso_subset = cso_dat[int_indices,]
cso_subset$dataset <- sapply(strsplit(cso_subset$index, "_"), function(x){x[1]})
cso_subset$split_type <- sapply(strsplit(cso_subset$index, "_"), function(x){x[2]})
create_cso_v_split_plot(cso_subset, comparisons, my_colors)


#### CSO VS SPLIT_TYPE (WITH SPECTRA, not used in paper)
#comparisons_with_spectra = list(
#  c("random", "scaffold"),
#  c("random", "umap"),
#  c("random", "spectra")
#)
#
#cso_subset = cso_dat
#cso_subset$dataset <- sapply(strsplit(cso_subset$index, "_"), function(x){x[1]})
#cso_subset$split_type <- sapply(strsplit(cso_subset$index,"_"), function(x) {x[2]})
#cso_subset$split_type <- factor(cso_subset$split_type, ordered=TRUE,
#                                   levels=c("random","scaffold","umap","spectra"))
#create_cso_v_split_plot(cso_subset, comparisons_with_spectra, my_colors)


#### CSO VS SPECTRAL PARAMETER (with other splits for comparison)
cso_vs_sp_df <- data.frame(
  dataset=character(0),
  SP=numeric(0),
  random_mean=numeric(0),
  random_sd=numeric(0),
  scaffold_mean=numeric(0),
  scaffold_sd=numeric(0),
  umap_mean=numeric(0),
  umap_sd=numeric(0),
  spectra_mean=numeric(0),
  spectra_sd=numeric(0)
)

for (dataset in datasets) {
  random_mean <- mean(cso_dat[grep(paste0(dataset, "_random"), cso_dat$index),"cso"])
  random_sd <- sd(cso_dat[grep(paste0(dataset, "_random"), cso_dat$index),"cso"])
  scaffold_mean <- mean(cso_dat[grep(paste0(dataset, "_scaffold"), cso_dat$index),"cso"])
  scaffold_sd <- sd(cso_dat[grep(paste0(dataset, "_scaffold"), cso_dat$index),"cso"])
  umap_mean <- mean(cso_dat[grep(paste0(dataset, "_umap"), cso_dat$index),"cso"])
  umap_sd <- sd(cso_dat[grep(paste0(dataset, "_umap"), cso_dat$index),"cso"])
  
  SPs <- unique(cso_dat[grep(paste0(dataset, "_spec"), cso_dat$index),"SP"])
  
  tmp_df <- data.frame(cso_dat[grep(paste0(dataset, "_spec"), cso_dat$index),] %>%
    group_by(SP) %>%
    dplyr::summarise(spectra_mean = mean(cso),
              spectra_sd = sd(cso)))
  tmp_df$dataset = dataset
  
  tmp_df$random_mean = random_mean
  tmp_df$random_sd = random_sd
  tmp_df$scaffold_mean = scaffold_mean
  tmp_df$scaffold_sd = scaffold_sd
  tmp_df$umap_mean = umap_mean
  tmp_df$umap_sd = umap_sd
  
  cso_vs_sp_df <- rbind(cso_vs_sp_df, tmp_df)
}

create_cso_v_sp_plot(cso_vs_sp_df, my_colors)




#### PERFORMANCE VS CSO
perf_vs_cso_df <- data.frame(
  dataset=character(0),
  split_type=character(0),
  cso_mean=numeric(0),
  cso_sd=numeric(0),
  metric_mean=numeric(0),
  metric_sd=numeric(0)
)

for (dataset in datasets) {
  ## collate all the info, starting from the CSOs for the spectra splits
  tmp_df1 <- data.frame(cso_dat[grep(paste0(dataset, "_spec"), cso_dat$index),] %>%
                         group_by(SP) %>%
                         dplyr::summarise(cso_mean = mean(cso),
                                   cso_sd = sd(cso)))
  
  tmp_df2 <- data.frame(perf_dat[which(perf_dat$dataset == dataset & 
                                         perf_dat$split_type == "spectra_tanimoto" & 
                                         perf_dat$model == "chemprop"),] %>%
                        group_by(SP) %>%
                          dplyr::summarise(perf_mean = mean(metric), perf_sd = sd(metric)))
  
  tmp_df <- dplyr::full_join(tmp_df1, tmp_df2, by = "SP")
  tmp_df$split_type = "spectra"
  
  for (split_type in c("random","scaffold","umap")) {
    cso_mean <- mean(cso_dat[grep(paste0(dataset, "_", split_type), cso_dat$index),"cso"])
    cso_sd <- sd(cso_dat[grep(paste0(dataset, "_", split_type), cso_dat$index),"cso"])
    
    perf_mean <- mean(perf_dat[which(perf_dat$dataset == dataset & 
                                       perf_dat$split_type == split_type & 
                                       perf_dat$model == "chemprop"), "metric"])
    perf_sd <- sd(perf_dat[which(perf_dat$dataset == dataset & 
                                       perf_dat$split_type == split_type & 
                                       perf_dat$model == "chemprop"), "metric"])
    
    tmp_df <- rbind(tmp_df, data.frame(
      SP=NA,cso_mean=cso_mean, cso_sd=cso_sd,
      perf_mean = perf_mean, perf_sd=perf_sd,
      split_type=split_type
    ))
  }
  
  tmp_df$dataset <- dataset
  
  perf_vs_cso_df <- rbind(perf_vs_cso_df, tmp_df)
}

create_performance_v_cso_plot(perf_vs_cso_df[which(perf_vs_cso_df$dataset %in% reg_datasets),],
                              "AUC",my_colors)
create_performance_v_cso_plot(perf_vs_cso_df[which(perf_vs_cso_df$dataset %in% class_datasets),],
                              "RMSE",my_colors,nrow=1)
