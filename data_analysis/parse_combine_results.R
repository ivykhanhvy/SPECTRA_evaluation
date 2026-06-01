source("./data_analysis/helpers.R")

#### READ IN AND PARSE all performance data from all splits + models

full_df <- data.frame(
  dataset=character(0),
  split_type=character(0),
  split_num=character(0),
  SP=character(0),
  model=character(0),
  metric=numeric(0))

### Read in results from the classical baselines
for (dataset in datasets) {
  classical_baselines = read.csv(paste0("results/classical_results/",
                             "classical_baselines_",dataset,".csv"))
  
  # fill in spectral parameters for spectral splits
  classical_baselines$SP = rep(NA, nrow(classical_baselines))
  classical_baselines[which(classical_baselines$split_type == "spectra_tanimoto"), "SP"] <- 
    sapply(strsplit(
      classical_baselines[which(
        classical_baselines$split_type == "spectra_tanimoto"), 
        "split_num"],"_"), function(x){x[1]})
  
  if (dataset %in% reg_datasets) {
    classical_baselines$metric <- classical_baselines$roc_auc
    classical_baselines$roc_auc <- NULL
  } else {
    classical_baselines$metric <- classical_baselines$root_mean_squared_error
    classical_baselines$root_mean_squared_error <- NULL
  }
  
  # harmonize indices between CSO data and other data; 1-index the splits
  
  classical_baselines[which(classical_baselines$split_type != "spectra_tanimoto"), "split_num"] <- 
    as.character(as.numeric(classical_baselines[which(classical_baselines$split_type != "spectra_tanimoto"), "split_num"]) + 1)
  
  full_df = rbind(full_df, classical_baselines)
}

### Read in results from the chemprop baselines
for (dataset in datasets) {
  for (split_type in c("random", "scaffold", "umap")) {
    chemprop_ex = read.csv(paste0("results/chemprop_results/",split_type,
                               "/",split_type,"_metrics_",dataset,".csv"))
    head(chemprop_ex)
    
    tmp_df <- data.frame(dataset=c(dataset),
                         split_type=c(split_type),
                         split_num=c(paste0(rep(1:5,5), "_ensemble", rep(1:5, each=5))),
                         SP=c(NA),
                         model=c("chemprop"),
                         metric=c(chemprop_ex$Ensemble.1,chemprop_ex$Ensemble.2,
                                  chemprop_ex$Ensemble.3,chemprop_ex$Ensemble.4,
                                  chemprop_ex$Ensemble.5))
    
    full_df = rbind(full_df, tmp_df)
  }
  
  # deal with spectra_tanimoto splits separately
  chemprop_ex = read.csv(paste0("results/chemprop_results/spectra_tanimoto",
                                  "/sheet/spectra_ensemble_",dataset,".csv"))
  
  if (dataset %in% reg_datasets) {
    chemprop_ex$metric <- chemprop_ex$AUC
  } else {
    chemprop_ex$metric <- chemprop_ex$RMSE
  }
  
  tmp_sp <- sapply(strsplit(chemprop_ex$metric,"_"), function(x){x[1]})
  
  tmp_df <- data.frame(dataset=c(dataset),
                       split_type="spectra_tanimoto",
                       split_num=c(paste0(rep(chemprop_ex$metric,5),"_ensemble", rep(1:5, each=nrow(chemprop_ex)))),
                       SP=rep(tmp_sp,5),
                       model=c("chemprop"),
                       metric=c(chemprop_ex$Ensemble.1,chemprop_ex$Ensemble.2,
                                chemprop_ex$Ensemble.3,chemprop_ex$Ensemble.4,
                                chemprop_ex$Ensemble.5))
  
  
  full_df = rbind(full_df, tmp_df)
}

# create indices to match CSO data indices
perf_dat$index <- paste(
  perf_dat$dataset,
  perf_dat$split_type,
  gsub("_ensemble[1-5]","",perf_dat$split_num),
  sep="_"
)

write.csv(full_df, "./data_analysis/all_split_performance_data.csv", 
          quote=FALSE,row.names=FALSE)






#### READ IN AND PARSE all CSO data from all splits
full_cso_df <- data.frame(
  index=character(0),
  train_size=numeric(0),
  val_size=numeric(0),
  test_size=numeric(0),
  SP=character(0),
  cso=numeric(0)
)

for (dataset in datasets) {
  for (split_type in c("random","scaffold","umap","spectra_tanimoto")) {
    cso_df <- read.csv(paste0("./splits_data/cross_split_overlap/",split_type,
                           "/",dataset,"_",split_type,"_cross_split_overlap.csv"))
    
    # harmonize indices between CSO data and other data; 1-index the splits
    tmp_df <- data.frame(
      index=cso_df$index,
      train_size=cso_df$train_size,
      val_size=cso_df$val_size,
      test_size=cso_df$test_size,
      SP=c(NA),
      cso=cso_df$cross_split_overlap
    )
    
    if(split_type == "spectra_tanimoto") {
      tmp_df$SP <- cso_df$SPECTRA_parameter
      # correct index to follow all other indices for other splits
      tmp_df$index <- sapply(strsplit(cso_df$index,"_"), function(x) {
        paste(x[1],x[2],x[3],x[5],x[6], sep="_")})
    } else {
      tmp_df$index <- sapply(strsplit(cso_df$index,"_"), function(x) {
        paste(x[1],x[2],as.numeric(x[3])+1,sep="_")
      })
    }
    
    full_cso_df <- rbind(full_cso_df, tmp_df)
    
  }
}

write.csv(full_cso_df, "./data_analysis/all_split_size_cso_data.csv", 
          quote=FALSE,row.names=FALSE)
