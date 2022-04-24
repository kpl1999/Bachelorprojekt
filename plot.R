setwd("~/Documents/DTU/6. semester/Bachelorprojekt/Code")
library(ggplot2)


data <- read.csv("dataplot.csv", header=TRUE, sep=";")
data_lstm <- read.csv("data_model.csv", header = TRUE, sep=";")
data_ML <- read.csv("data_model_ML.csv", header = TRUE, sep=";")
summary(data_lstm) 

hist(targetData$frustrated)


sd(data_lstm$TEMP)
par(mfrow=c(1,4))
boxplot(data_lstm$HR, col = "coral1", main = "HR")
boxplot(data_lstm$EDA, col = "thistle1", main = "EDA")
boxplot(data_lstm$TEMP, col = "springgreen4", main = "TEMP")
boxplot(data_lstm$BVP, col="khaki2", main = "BVP")

hist(data_lstm$BVP)

(targetData <- read.csv("TargetData.csv", sep=";"))


mean(targetData$frustrated)
sum(targetData$frustrated == 1)
sum(targetData$frustrated == 2)
sum(targetData$frustrated == 7)


hist(targetData$upset, breaks =10)
hist(targetData$hostile, breaks =10)
hist(targetData$alert, breaks =10)
hist(targetData$ashamed, breaks =10)
hist(targetData$inspired, breaks =10)
hist(targetData$nervous, breaks =10)
hist(targetData$determined, breaks =10)
hist(targetData$attentive, breaks =10)
hist(targetData$afraid, breaks =10)
hist(targetData$active, breaks =10)
hist(targetData$frustrated, breaks =10)

ggplot(targetData, aes(x=(frustrated), group = Phase, fill = as.factor(Phase))) + 
  geom_histogram() +
 scale_fill_brewer(name = "Phase", labels = c("1", "2", "3"), palette ="Set1")
ggplot(targetData, aes(x=sqrt(frustrated), group = Phase, fill = as.factor(Phase))) + 
  geom_histogram() +
  scale_fill_brewer(name = "Phase", labels = c("1", "2", "3"), palette ="Set1")
ggplot(targetData, aes(x=log(frustrated), group = Phase, fill = as.factor(Phase))) + 
  geom_histogram() +
  scale_fill_brewer(name = "Phase", labels = c("1", "2", "3"), palette ="Set1")
ggplot(targetData, aes(x=(((frustrated)^0.5)-1)/0.5, group = Phase, fill = as.factor(Phase))) + 
  geom_histogram() +
  scale_fill_brewer(name = "Phase", labels = c("1", "2", "3"), palette ="Set1")


# TEMP - All phases 
targetData$frustrated[targetData$Round==3 & targetData$particpant_ID == 1]

df1 <- data[data$Round == 2 & data$ID == 1, ]
ggplot(df1, aes(X, HR, group = Phase, color= as.factor(Phase))) + 
  geom_line() + scale_colour_brewer(name = "Phase", labels = c("1", "2", "3"), palette ="Set1")

df2 <- data[data$Round == 2 & data$ID == 1, ]
ggplot(df2, aes(X, TEMP, group = Phase, color= as.factor(Phase))) + 
  geom_line() + scale_colour_brewer(name = "Phase", labels = c("1", "2", "3"), palette ="Set1")

df3 <- data[data$Round == 2 & data$ID == 1, ]
ggplot(df3, aes(X, BVP, group = Phase, color= as.factor(Phase))) + 
  geom_line() + scale_colour_brewer(name = "Phase", labels = c("1", "2", "3"), palette ="Set1")

df4 <- data[data$Round == 3 & data$ID == 1, ]
ggplot(df4, aes(X, EDA, group = Phase, color= as.factor(Phase))) + 
  geom_line() + scale_colour_brewer(name = "Phase", labels = c("1", "2", "3"), palette ="Set1")

targetData$frustrated[targetData$Round==3 & targetData$particpant_ID == 1]

 
# BVP - All phases
BVP <- data[data$Measurement == "BVP" & data$Round == 1 & data$Phase == 1 & data$particpant_ID == 1, ]
#BVP <- BVP[50000:51000,]

ggplot(BVP, aes(X, value, group = Phase, color= as.factor(Phase))) + 
  geom_line() + scale_colour_brewer(name = "Phase", labels = c("1", "2", "3"), palette ="Set1")


HR <- data[data$Measurement == "HR" & data$Round == 1 & data$particpant_ID == 1, ]

ggplot(HR, aes(X, value, group = Phase, color= as.factor(Phase))) + 
  geom_line() + scale_colour_brewer(name = "Phase", labels = c("1", "2", "3"), palette ="Set1")


# EDA - Puzzler vs. instructor phase 2
EDA <- data[data$Measurement=="EDA" & data$Round == 4 & data$Phase ==2 & data$team_ID == 4, ]

ggplot(EDA, aes(index, value, color = as.factor(puzzler))) + 
  geom_line() + scale_colour_brewer(name = "Role", labels = c("Instructor", "Puzzler"), palette = "Set1")


# HR - Puzzler vs. instructor phase 2
HR <- data[data$Measurement=="HR" & data$Round == 4 & data$Phase == 2 & data$team_ID == 1, ]

ggplot(HR, aes(index, value, color = as.factor(puzzler))) + 
  geom_line() 


# BVP - Puzzler vs. instructor phase 2
BVP <- data[data$Measurement=="BVP" & data$Round == 3 & data$Phase == 2 & data$team_ID == 3, ]

ggplot(BVP, aes(index, value, color = as.factor(puzzler))) + 
  geom_line() 

# TEMP - Puzzler vs. instructor phase 2
TEMP <- data[data$Measurement=="TEMP" & data$Round == 2 & data$Phase == 2 & data$team_ID == 2, ]

ggplot(TEMP, aes(index, value, color = as.factor(puzzler))) + 
  geom_line() 





df <- data[data$Measurement == "HR" & data$particpant_ID == 2 & data$Phase==2,]
ggplot(df, aes(index, value, colour = Round)) + 
  geom_point() 


unique(data$ID)

summary(data) 
which(is.na(data$EDA))
summary(data[data$Measurement == "HR",]$value)
summary(data[data$Measurement == "EDA",]$value)
summary(data[data$Measurement == "BVP",]$value)
summary(data[data$Measurement == "TEMP",]$value)

sd(data[data$Measurement == "HR",]$value)
sd(data[data$Measurement == "EDA",]$value)
sd(data[data$Measurement == "BVP",]$value)
sd(data[data$Measurement == "TEMP",]$value)

data[data$Measurement == "BVP",]$value

