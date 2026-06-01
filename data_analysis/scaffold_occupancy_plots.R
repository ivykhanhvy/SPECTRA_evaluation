library(jsonlite)

datasets = c(
  "bace","bbbp","clintox","delaney","freesolv","lipo","sider","tox21"
)

class_datasets = c("delaney", "freesolv", "lipo")

class_aucs = c()
reg_aucs = c()
for (dataset in datasets) {
  print(dataset)
  scaffold_occ <- fromJSON(paste0("splits_data/scaffold_info/",dataset,".json"))
  hist(vapply(scaffold_occ, length, 0),breaks=40, 
       main=paste0(dataset, ": Number of molecules per scaffold"),
       xlab="Number of molecules")
  
  tmp <- vapply(scaffold_occ, length, 0)
  tmp_props <- sort(tmp/sum(tmp), decreasing=TRUE)
  tmp_props_cumsum <- cumsum(tmp_props)
  
  plot(tmp_props_cumsum~seq(0,1,length=length(tmp_props_cumsum)),ylim=c(0,1))
  abline(a=0,b=1,col="red")
  # AUC
  auc = sum((tmp_props_cumsum*(1/length(tmp_props_cumsum))))
  print(paste0("AUC: ", auc))
  if (dataset %in% class_datasets) {
    class_aucs = c(class_aucs, auc)
  } else {
    reg_aucs = c(reg_aucs, auc)
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
cor.test(c(rep(reg_aucs,each=5), rep(class_aucs,each=5)), 
         c(reg_changes, -1*class_changes), method="spearman")

plot(c(median(reg_changes[1:5]), median(reg_changes[6:10]), median(reg_changes[10:15]),
       median(reg_changes[15:20]), median(reg_changes[20:25]), median(-1*class_changes[1:5]), 
       median(-1*class_changes[6:10]), median(-1*class_changes[10:15])) ~ c(reg_aucs,class_aucs),
     xlab="Area under the cumulative sum curve", 
     ylab="Model performance (% change in performance)")
