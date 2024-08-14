
###################################
setwd("H:/Lucas/baseline_models")
###################################


#########################
load("income.saved")
#########################
library(fixest)
library(dplyr)

income$year<-income$year+1915

splited<-split(income,income$year)

numberRepetition<-10

mat<-c()
for (j in 1:numberRepetition){
  
  r_squared<-c()
  for (i in 1:11){ # number of years that there are (2011-2021)
    
    # split between training and test:
    training<-sample(splited[[i]]$RINPERSOON,replace=FALSE,size=floor(0.8*length(splited[[i]]$RINPERSOON)))
    test<-splited[[i]]$RINPERSOON[!(splited[[i]]$RINPERSOON%in%training)]
    
    training<-splited[[i]][which(splited[[i]]$RINPERSOON%in%training),]
    test<-splited[[i]][which(splited[[i]]$RINPERSOON%in%test),]
    
    # delete missing values:
    training<-training[complete.cases(training$educSim,training$municipality.x,training$incomeAge30,
                                      training$gender.x,training$month),]
    test<-training[complete.cases(test$educSim,test$municipality.x,test$incomeAge30,
                                  test$gender.x,test$month),]
    
    # in case some municipalities are not in the training set:
    missing_municipalities<-unique(training$municipality.x[training$municipality.x %in%test$municipality.x])
    
    # Only take the non missing municipalities:
    test<-test[which(test$municipality.x%in%missing_municipalities),]
    
    # non missing in training only
    training<-training[complete.cases(training$educSim,training$municipality.x),]
    
    # model with fixed effects:
    m<-feols(data=training,incomeAge30~
               as.factor(month)+as.factor(gender.x)+
               as.factor(municipality.x)+as.factor(educSim))
    
    # predict
    predictions<-predict(m,newdata=test)
    
    # get the r.squared:
    summary(m)$r.squared
    
    r2<-1 - sum((test$incomeAge30-predictions)^2) / sum((test$incomeAge30 - mean(test$incomeAge30))^2)
    
    r_squared<-c(r_squared,r2)
    
  }
  mat<-rbind(mat,r_squared)
  print(j)
}

# mat<-as.data.frame(mat)

mat<-as.matrix(mat)
results<-data.frame(meanR2=colMeans(mat),year=2011:2021)

income<-income%>%
  select(-gender.y,-municipality.y,-daysSinceFirstEvent,-age,-educDet)


