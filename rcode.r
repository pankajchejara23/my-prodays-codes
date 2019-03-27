library(ggplot2)
library(FactoMineR)
library(factoextra)
library(FactoInvestigate)
library(corrplot)
library(RDRToolbox)

data <- read.csv("Data_16_original.csv")

ggplot(data, aes(x=disengaged))+geom_histogram()
ggplot(data, aes(x=looking))+geom_histogram()
ggplot(data, aes(x=talking))+geom_histogram()
ggplot(data, aes(x=intTech))+geom_histogram()
ggplot(data, aes(x=intRes))+geom_histogram()
ggplot(data, aes(x=intExt))+geom_histogram()
ggplot(data, aes(x=Accessed))+geom_histogram()
ggplot(data, aes(x=Create))+geom_histogram()
ggplot(data, aes(x=Open))+geom_histogram()
ggplot(data, aes(x=Update))+geom_histogram()

print(data[0])
pca1 <- PCA(data[,4:13], scale.unit = T, ncp=10, graph=T)  
data$pc1 <- pca1$ind$coord[,1]
data$pc2 <- pca1$ind$coord[,2]
print(pca1$var$cor)
corrplot(cor(data[,4:15]),method="number")

# Isomap code
temp <- matrix(unlist(data[,4:13]),ncol=10,byrow=TRUE)
print(temp)
iso = Isomap(temp,dims=2,k=5)
print(iso$dim2[,0])




if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("RDRToolbox", version = "3.8")


mfa1 <- MFA(data[,4:13], group=c(6,4), type=c("s", "c"), ncp=10, name.group=c("obs","logs"))
print(mfa1$ind)
summary(mfa1)
plot(mfa1)
plotellipses(mfa1)
fviz_screeplot(mfa1) 
data$Engagement <- mfa1$ind$coord[,1]
data$PhysvsTech <- mfa1$ind$coord[,2]

M <- cor(data[,4:17])
corrplot(M, method="number")
# The data is skewed! any transformation to make it a different distribution?
# Leave Accessed out
# Binarize log variables
# ...
print(data[,11:13])
mfa2 <- MFA(data[,c(4:9,11:13)], group=c(6,3), type=c("s", "c"), ncp=9, name.group=c("obs","logs"))
fviz_screeplot(mfa2)
plot(mfa2, axes=c(1,3), choix="var")

names(mfa1)
names(data)
data$time = as.POSIXct(data$timestamp)
ggplot(data[data$group=="1AB",], aes(x=time, y=Engagement, color=group))+geom_line()
ggplot(data[data$group=="1AB",], aes(x=time, y=PhysvsTech, color=group))+geom_line()
