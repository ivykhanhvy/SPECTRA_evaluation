source("./data_analysis/helpers.R")

perf_dat <- read.csv("./data_analysis/all_split_performance_data.csv")
cso_dat <- read.csv("./data_analysis/all_split_size_cso_data.csv")

cso_dat$dataset <- sapply(strsplit(cso_dat$index,"_"), function(x){x[1]})
cso_dat$split_type <- sapply(strsplit(cso_dat$index,"_"), function(x){x[2]})

subset_umap_indices <- sort(paste0(datasets, "_umap_", rep(c(1:5),each=8)))
umap_subset <- cso_dat[which(cso_dat$index %in% subset_umap_indices),]
umap_subset$sil_score <- c(0)

perf_dat_chemprop_umap_only <- perf_dat[which(perf_dat$split_type=="umap" &
                                                perf_dat$model=="chemprop"),]
row.names(perf_dat_chemprop_umap_only) <- paste0(
  perf_dat_chemprop_umap_only$dataset,
  "_",perf_dat_chemprop_umap_only$split_type,
  "_",perf_dat_chemprop_umap_only$split_num)
perf_dat_chemprop_umap_only$sil_score <- c(0)

for (dataset in datasets) {
  sil_dat_tmp <- read.csv(paste0("./splits_data/umap_silhouette_scores/",
                                 dataset,"_silhouette_scores.csv"))
  for (i in 1:5) {
    umap_subset[which(umap_subset$index == paste0(
      dataset,"_umap_",i)),"sil_score"] = sil_dat_tmp[i,"silhouette_score"]
    
    perf_dat_chemprop_umap_only[grep(paste0(dataset,"_umap_",i), rownames(perf_dat_chemprop_umap_only)),
                                "sil_score"] = sil_dat_tmp[i,"silhouette_score"]
  }
}






mean_random_cso <- aggregate(cso ~ dataset, data=cso_dat[which(cso_dat$split_type=="random"),], FUN=median)
mean_umap_cso <- aggregate(cso ~ dataset, data=umap_subset, FUN=median)
mean_sil_score <- aggregate(sil_score ~ dataset, data=umap_subset, FUN=median)


# Okabe-Ito palette
umap_subset$dataset <- sapply(strsplit(umap_subset$index,"_"), function(x){x[1]})
color_map = c(
  "bace"="#E69F00","bbbp"="#56B4E9","clintox"="#009E73",
  "delaney"="#F0E442","freesolv"="#0072B2",
  "lipo"="#D55E00", "sider"="#CC79A7", "tox21"="#000000"
)

plot(cso ~ sil_score, data=umap_subset,
     col=color_map[dataset],pch=19,cex=1.25,
     ylim=c(0,0.17),
     ylab="Cross-Split Overlap",
     xlab="Silhouette Score")
#for(i in 1:8) {
#  abline(h=mean_random_cso[i,"cso"],
#         col=color_map[mean_random_cso[i,"dataset"]],
#         lwd=2)
#}
legend("bottomright", 
       legend = names(color_map), 
       col = color_map, 
       pch = c(19), 
       #bty = "n", 
       pt.cex = 1, 
       cex = 0.8, 
       text.col = "black", 
       horiz = F , 
       inset = c(0.1, 0.1))
cor.test(umap_subset$cso, umap_subset$sil_score, method="spearman")

# plot(metric ~ sil_score, data=perf_dat_chemprop_umap_only[
#   which(perf_dat_chemprop_umap_only$dataset %in% class_datasets),
# ],
#      col=color_map[dataset],pch=19,cex=1.25)
# legend("topright", 
#        legend = names(color_map), 
#        col = color_map, 
#        pch = c(19), 
#        #bty = "n", 
#        pt.cex = 1, 
#        cex = 0.8, 
#        text.col = "black", 
#        horiz = F , 
#        inset = c(0.1, 0.1))
