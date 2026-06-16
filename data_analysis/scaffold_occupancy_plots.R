library(jsonlite)

datasets = c(
  "bace","bbbp","clintox","delaney","freesolv","lipo","sider","tox21"
)

class_datasets = c("delaney", "freesolv", "lipo")

class_aucs = c()
reg_aucs = c()

class_avg_occ = c()
reg_avg_occ = c()
for (dataset in datasets) {
  print(dataset)
  scaffold_occ <- fromJSON(paste0("splits_data/scaffold_info/",dataset,".json"))
  hist(vapply(scaffold_occ, length, 0),breaks=40, 
       main=paste0(dataset, ": Number of molecules per scaffold"),
       xlab="Number of molecules")
  
  tmp <- vapply(scaffold_occ, length, 0)
  tmp_props <- sort(tmp/sum(tmp), decreasing=TRUE)
  tmp_props_cumsum <- cumsum(tmp_props)
  
  avg_occ <- mean(tmp)
  
  plot(tmp_props_cumsum~seq(0,1,length=length(tmp_props_cumsum)),ylim=c(0,1))
  abline(a=0,b=1,col="red")
  # AUC
  auc = sum((tmp_props_cumsum*(1/length(tmp_props_cumsum))))
  print(paste0("AUC: ", auc))
  if (dataset %in% class_datasets) {
    class_aucs = c(class_aucs, auc)
    class_avg_occ = c(class_avg_occ, avg_occ)
  } else {
    reg_aucs = c(reg_aucs, auc)
    reg_avg_occ = c(reg_avg_occ, avg_occ)
  }
  
}

class_changes = c()
reg_changes = c()
for (dataset in datasets) {
  class_ex = read.csv(paste0("results/classical_results/",
                             "classical_baselines_",dataset,".csv"))
  print(dataset)
  if (dataset %in% class_datasets) {
    rand_baseline = class_ex[which(class_ex$split_type=="random" & class_ex$model == "LinReg"),"root_mean_squared_error"]
    scaf_perf = class_ex[which(class_ex$split_type=="scaffold" & class_ex$model == "LinReg"),"root_mean_squared_error"]
    
    change = (scaf_perf - rand_baseline)/rand_baseline
    class_changes = c(class_changes, change)
  } else {
    rand_baseline = class_ex[which(class_ex$split_type=="random" & class_ex$model == "LogReg"),"roc_auc"]
    scaf_perf = class_ex[which(class_ex$split_type=="scaffold" & class_ex$model == "LogReg"),"roc_auc"]
    change = (scaf_perf - rand_baseline)/rand_baseline
    reg_changes = c(reg_changes, change)
  }
}

plot(c(reg_changes, -1*class_changes) ~ c(rep(reg_aucs,each=5), rep(class_aucs,each=5)),
     xlab="Area under the cumulative sum curve", 
     ylab="Model performance (% change in performance)")

plot(c(median(reg_changes[1:5]), median(reg_changes[6:10]), median(reg_changes[10:15]),
       median(reg_changes[15:20]), median(reg_changes[20:25]), median(-1*class_changes[1:5]), 
       median(-1*class_changes[6:10]), median(-1*class_changes[10:15])) ~ c(reg_aucs,class_aucs),
     xlab="Area under the cumulative sum curve", 
     ylab="Model performance (% change in performance)")







reg_csos = c()
class_csos = c()
cso_dat <- read.csv("./data_analysis/all_split_size_cso_data.csv")
for (dataset in datasets) {
  if (dataset %in% class_datasets) {
    class_csos = c(class_csos, mean(cso_dat[grep(paste0(dataset,"_scaffold"), cso_dat$index),"cso"] - 
                                      cso_dat[grep(paste0(dataset,"_random"), cso_dat$index),"cso"]))
  } else {
    reg_csos = c(reg_csos, mean(cso_dat[grep(paste0(dataset,"_scaffold"), cso_dat$index),"cso"] - 
                                  cso_dat[grep(paste0(dataset,"_random"), cso_dat$index),"cso"]))
  }
}




##### PLOTS FOR PAPER 
# Okabe-Ito palette
color_map = c(
  "bace"="#E69F00","bbbp"="#56B4E9","clintox"="#009E73",
  "delaney"="#F0E442","freesolv"="#0072B2",
  "lipo"="#D55E00", "sider"="#CC79A7", "tox21"="#000000"
)
# plot(c(reg_changes, -1*class_changes) ~ c(rep(reg_avg_occ,each=5), rep(class_avg_occ,each=5)),
#      col=rep(color_map[datasets],each=5),
#      pch=19,cex=1.25,
#      xlab="Mean scaffold group occupancy", 
#      ylab="Model performance (% change in performance)")
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


cor.test(c(rep(reg_avg_occ,each=5), rep(class_avg_occ,each=5)), 
         c(reg_changes, -1*class_changes), method="spearman")






plot(c(reg_csos, class_csos) ~ c(reg_avg_occ, class_avg_occ),
     col=color_map[datasets],
     pch=19,cex=1.25,
     xlab="Mean scaffold group occupancy", 
     ylab="Cross-Split Overlap")
legend("topright", 
       legend = names(color_map), 
       col = color_map, 
       pch = c(19), 
       #bty = "n", 
       pt.cex = 1, 
       cex = 0.8, 
       text.col = "black", 
       horiz = F , 
       inset = c(0.1, 0.1))


cor.test(c(reg_csos, class_csos), c(reg_avg_occ, class_avg_occ), method="spearman")



