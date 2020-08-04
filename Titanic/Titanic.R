library(tidyr)
library(dplyr)
library(stringr)
library(neuralnet)
library(pROC) 
library(caret)
library(data.table)
train <- read.csv("/Users/dshenker/Desktop/Kaggle/Titanic/titanic/train.csv", header = T)
train_clean <- na.omit(train)
train_clean <- clean(train_clean)
test <- read.csv("/Users/dshenker/Desktop/Kaggle/Titanic/titanic/test.csv", header = T)
test_clean <- clean(test)


#Train the model on training data
train_clean <- as.data.table(train_clean)
train_clean <- train_clean[, c("PassengerId", "Name", "Ticket", "comma_spot", "dot_spot", "Cabin", "SibSp", "Parch") := NULL]
#maxs <- apply(train_clean[,c("Age", "Fare")], 2, max)
#mins <- apply(train_clean[, c("Age", "Fare")], 2, min)
#train_clean_sub <- as.data.frame(scale(train_clean[, c("Age", "Fare")],center = mins, scale = maxs - mins))
#train_clean <- data.frame(train_clean_sub, train_clean$Pclass, train_clean$SibSp, train_clean$Parch, train_clean$Embarked, 
#                          train_clean$Sex_Num, train_clean$Child, train_clean$Special_Title, train_clean$Survived)
#colnames(train_clean) <- c("Age", "Fare", "Pclass", "SibSp", "Parch", "Embarked", "Sex_Num", "Child", "Special_Title", "Survived")

index <- sample(1:nrow(train_clean),round(0.7*nrow(train_clean))) #get indices for split
train <- train_clean[index,] #get training set
train_part2 <- train_clean[-index,]
train_data <- model.matrix(~ Age + Fare + Pclass + Accompanied + Embarked + Sex_Num + Child + Special_Title + Survived, data = train)
train_data_2 <- model.matrix(~ Age + Fare + Pclass + Accompanied + Embarked + Sex_Num + Child + Special_Title + Survived, data = train_part2)

nn <- neuralnet(Survived ~ Pclass2 + Pclass3 + Age + Accompanied + Fare + Sex_Num1 + Child1 + Special_Title + EmbarkedC + EmbarkedQ + EmbarkedS, 
                data = train_data, 
                hidden = 3, 
                act.fct = "logistic", 
                linear.output = FALSE) #train the neural net

predict_train_2 = neuralnet::compute(nn, train_data_2) #get predictions on training set
prob_train_2 <- predict_train_2$net.result #extract the probabilities
head(prob_train_2)
train_part2$pct_chance <- prob_train_2
model_roc <- roc(Survived ~ pct_chance, data = train_part2)     # apply the ROC function to calculate the ROC information
plot(model_roc) 
as.numeric(model_roc$auc)
coords(model_roc, 'best', 'threshold', transpose = FALSE) #get best threshold -   0.5199618
confusionMatrix_NN <- table(train_part2$Survived, train_part2$pct_chance > 0.5199618)    
confusionMatrix_NN #Print out confusion matrix to see quality of results




#Try out model on the test data
test_clean <- as.data.table(test_clean)
test_clean <- test_clean[, c("Name", "Ticket", "comma_spot", "dot_spot", "Cabin", "SibSp", "Parch") := NULL]
#maxs <- apply(test_clean[,c("Age", "Fare")], 2, max)
#mins <- apply(test_clean[, c("Age", "Fare")], 2, min)
#test_clean_sub <- as.data.frame(scale(test_clean[, c("Age", "Fare")],center = mins, scale = maxs - mins))
#test_clean <- data.frame(test_clean_sub, test_clean$Pclass, test_clean$SibSp, test_clean$Parch, test_clean$Embarked, 
#                          test_clean$Sex_Num, test_clean$Child, test_clean$Special_Title, test_clean$PassengerId)
#colnames(test_clean) <- c("Age", "Fare", "Pclass", "SibSp", "Parch", "Embarked", "Sex_Num", "Child", "Special_Title", "PassengerId")



test_data <- model.matrix(~ Age + Fare + Pclass + Accompanied + Embarked + Sex_Num + Child + Special_Title, data = test_clean)


predict_test = neuralnet::compute(nn, test_data) #get predictions on training set
prob_test <- predict_test$net.result #extract the probabilities
head(prob_test)
test_clean$pct_chance <- prob_test
test_clean$survive_prediction <- ifelse(test_clean$pct_chance > 0.1949438, 1, 0)
submission <- data.frame(test_clean$PassengerId, test_clean$survive_prediction)
colnames(submission) <- c("PassengerId", "Survived")
write.csv(submission, "submission1.csv")


#FUNCTION TO CLEAN DATA
clean <- function(df) {
#df <- train
list_na <- colnames(df)[apply(df, 2, anyNA)]
average_missing <- mean(df[,"Age"], na.rm = TRUE)
average_missing[2] <- mean(df[, "Fare"], na.rm = TRUE)
df_nona <- df %>%
  mutate(Age = ifelse(is.na(Age), average_missing[1], Age),
         Fare = ifelse(is.na(Fare), average_missing[2], Fare))
names(df_nona)
df_nona$Sex <- factor(df_nona$Sex)
df_nona$Pclass <- factor(df_nona$Pclass)
df_nona$Fare <- log(df_nona$Fare + .0001)
#Summary Statistics
nrow(df_nona)
#mean(df_nona$Survived) #40% of passengers survived
summary(df_nona)
sapply(df_nona, class)

#get title 
comma_index <- str_locate(df_nona$Name, ",")
df_nona$comma_spot <- comma_index[,1]

dot_index <- str_locate(df_nona$Name, "\\.")
head(dot_index)
df_nona$dot_spot <- dot_index[,1]
title <- substr(df_nona$Name, df_nona$comma_spot + 2, df_nona$dot_spot - 1)
df_nona$Title <- title

#check if accompanied by someone
accompanied <- ifelse(df_nona$SibSp + df_nona$Parch > 0, 1, 0)
df_nona$Accompanied <- accompanied

#turn sex into numeric
df_nona$Sex_Num <- ifelse(df_nona$Sex == "female", 1, 0)

#make children column
df_nona$Child <- ifelse(df_nona$Age <= 15, 1, 0)

summary(df_nona)
df_nona$Sex_Num <- factor(df_nona$Sex_Num)
df_nona$Child <- factor(df_nona$Child)

#check for a special type of title                       
unique(df_nona$Title)

df_nona$Special_Title = ifelse(df_nona$Title == "Master" || df_nona$Title == "Dr" ||
                                 df_nona$Title == "Major" || df_nona$Title == "Lady" ||
                                 df_nona$Title == "Sir" || df_nona$Title == "Col" ||
                                 df_nona$Title == "Jonkheer", 1, 0)


#df_nona$Survived <- factor(df_nona$Survived)
return(df_nona)
}
