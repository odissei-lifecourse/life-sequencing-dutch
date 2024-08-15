# Baseline models for mariage market

library(dplyr)
library(lubridate)
library(haven)
library(tidyr)

unions<-read.csv("H:/Life_Course_Evaluation/data/raw/unions.csv",sep=";")


birth<-read.csv("H:/data sequences lucas/birth_info.csv",sep=";")

unions<-left_join(unions,birth[,c("RINPERSOON","birthDate","municipality")],
                  by="RINPERSOON")

names(unions)<-c("RINPERSOON","RINPart1","dateBegRelation","typeRel",
                 "genderPart1","genderPart2","birthDatePart1","municipalityPart1")

# I do it again for the second partner (after having changed the RINPERSOON:
unions<-left_join(unions,birth[,c("RINPERSOON","birthDate","municipality")],
                  by="RINPERSOON")

names(unions)<-c("RINPERSOON","RINPart1","dateBegRelation","typeRel",
                 "genderPart1","genderPart2","birthDatePart1","municipalityPart1",
                 "birthDatePart2","municipalityPart2")

rm(birth)

education<-read.csv("H:/data sequences lucas/for Tanzir - safe untouched kopie/education_info_good_format.csv",
                    sep=";")


unions<-left_join(unions,education[,c("RINPERSOON",
                                      "educSim")],by="RINPERSOON")

names(unions)[11]<-"educPart2"

unions<-merge(unions,education[,c("RINPERSOON",
                                  "educSim")],
              by.x="RINPart1" ,
              by.y="RINPERSOON")

names(unions)[12]<-"educPart1"



####################################
setwd("H:/Lucas/baseline_models")
####################################

save(unions,file="unions.saved")

########################
load("unions.saved")
########################

income<-read.table("H:/data sequences lucas/RINPERSOON_and_income_30.txt",header=TRUE)

unions<-left_join(unions,income[,c("RINPERSOON","incomeAge30")],
                  by="RINPERSOON")

names(unions)[13]<-"incomePart2"

unions<-merge(unions,income[,c("RINPERSOON","incomeAge30")],
              by.x="RINPart1",
              by.y="RINPERSOON")


names(unions)[14]<-"incomePart1"

rm(income)

unions$dateRel<-as_date(as.character(unions$dateBegRelation))

unions$yearBirthPart1<-as.numeric(substring(unions$birthDatePart1,1,4))
unions$yearBirthPart2<-as.numeric(substring(unions$birthDatePart2,1,4))
unions$yearUnion<-as.numeric(substring(unions$dateBegRelation,1,4))

unions<-unions%>%
  mutate(agePart1=yearUnion-yearBirthPart1,
         agePart2=yearUnion-yearBirthPart2)

unions<-rename(unions,RINPart2=RINPERSOON)

unions<-distinct(unions)


# equilibrate gender, i.e. make partner1 always the same gender:

s<-unions%>%
  filter(genderPart1==1)
s1<-unions%>%
  filter(genderPart1==2)

names(s1)<-c("RINPart2","RINPart1","dateBegRelation","typeRel","genderPart2","genderPart1",
"birthDatePart2","municipalityPart2", "birthDatePart1","municipalityPart1", "educPart1","educPart2",        
 "incomePart1","incomePart2","dateRel","yearUnion","yearBirthPart2","yearBirthPart1"   ,
 "agePart2","agePart1" )


unions<-bind_rows(s,s1)

rm(s,s1)

save(unions,file="unions.saved")


#######################
load("unions.saved")
#######################

#################################################################################
# addresses for municipality where people live
ad<-read_sav("G:/Bevolking/GBAADRESOBJECTBUS/GBAADRESOBJECT2022BUSV1.sav",
             col_select = c("RINOBJECTNUMMER","RINPERSOON","GBADATUMEINDEADRESHOUDING","SOORTOBJECTNUMMER"))

ad$RINPERSOON<-as.numeric(ad$RINPERSOON)

ad<-rename(ad,year=GBADATUMEINDEADRESHOUDING)
ad$year<-substr(ad$year,1,4)

ids<-data.frame(RINPERSOON=unique(c(unions$RINPart1,unions$RINPart2)))

ids<-left_join(ids,ad,by="RINPERSOON")

rm(ad)
gc()

muni<-read_sav("G:/BouwenWonen/VSLGTAB/VSLG2023TAB03V1.sav",
               col_select = c("RINOBJECTNUMMER","gem2023"))

ids<-left_join(ids,muni,by="RINOBJECTNUMMER")

ids$year<-as.numeric(ids$year)

ids<-ids[which(ids$year>=2010&ids$year<=2021),]

# now I need to make it a balanced panel dataset:

rm(ad,muni)

ids<-ids%>%
  arrange(RINPERSOON,year)%>%
  group_by(RINPERSOON)%>%
  complete(year=full_seq(year,1))%>%
  fill(gem2023, .direction = "down")
  
  
save(ids,file="ids.saved")


####################
load("ids.saved")
####################


# Now I want the municipality where they lived the year before they got married
ids<-ids%>%arrange(RINPERSOON,year)

ids<-ids %>%
  group_by(RINPERSOON) %>%
  mutate(muni_1=lag(gem2023),
         muni_2=lag(muni_1),
         muni_3=lag(muni_2))

unions$RINPERSOON<-unions$RINPart1
unions<-rename(unions,year=yearUnion)

full<-left_join(unions,ids,by=c("RINPERSOON","year"))

# merge to get info on the second partner:
names(full)
unions$RINPERSOON<-unions$RINPart2
full<-rename(full,SOORTOBJECTNUMMER_1=SOORTOBJECTNUMMER,RINOBJECTNUMMER_1=RINOBJECTNUMMER,
             muniLivePart1_1=muni_1,
             muniLivePart1_2=muni_2,
             muniLivePart1_3=muni_3)

full<-left_join(full,ids,by=c("RINPERSOON","year"))
full<-rename(full,SOORTOBJECTNUMMER_2=SOORTOBJECTNUMMER,RINOBJECTNUMMER_2=RINOBJECTNUMMER,
             muniLivePart2_1=muni_1,
             muniLivePart2_2=muni_2,
             muniLivePart2_3=muni_3)

full$muniLivePart1_1<-as.numeric(as.character(full$muniLivePart1_1))
full$muniLivePart1_2<-as.numeric(as.character(full$muniLivePart1_2))
full$muniLivePart1_3<-as.numeric(as.character(full$muniLivePart1_3))
full$muniLivePart2_1<-as.numeric(as.character(full$muniLivePart2_1))
full$muniLivePart2_2<-as.numeric(as.character(full$muniLivePart2_2))
full$muniLivePart2_3<-as.numeric(as.character(full$muniLivePart2_3))

# so we may have an issue here where the people who formed a union where already leaving together

full<-full%>%
  mutate(pb=ifelse(RINOBJECTNUMMER_1==RINOBJECTNUMMER_2,1,0))

mean(full$pb,na.rm=TRUE)
# but actually this is not the right way to check because i did not take RINOBJECT
# in the years before 



# keep only non same sex marriages and one marriage per individual
full<-full%>%
  filter(genderPart1!=genderPart2)%>%
  group_by(RINPart1)%>%
  slice(1)


# leave out missing observations
full<-full[which(complete.cases(full$municipalityPart1)&
                           complete.cases(full$municipalityPart2)&
                           complete.cases(full$educPart1)&
                           complete.cases(full$educPart2)&
                           complete.cases(full$incomePart1)&
                           complete.cases(full$incomePart2)&
                           complete.cases(full$muniLivePart1_3)&
                           complete.cases(full$muniLivePart2_3)),]


save(full,file="full.saved")


#####################
load("full.saved")
#####################


# create a dataset with random 99 other individuals who got married in the same year:
fake_matches<-c()
for (i in sort(unique(full$year))) {
  s<-full[which(full$year==i),c("RINPart1","RINPart2")]
  s$trueCouple<-1
  # Maybe here I should modify slightly to make sure that I don't take the actual exact partner2 of part1
  t<-expand_grid(RINPart1=s$RINPart1,
                 RINPart2=sample(s$RINPart2,99,replace=TRUE))
  t$trueCouple<-0
  fake_matches<-bind_rows(fake_matches,t,s)
  fake_matches$year<-i
  print(i)
}

rm(s,t,i)


full<-full[,c( "RINPart1","RINPart2", "municipalityPart1" ,
                       "municipalityPart2","educPart2","educPart1",          
                       "incomePart2","incomePart1","year",    
                       "agePart1","agePart2",          
                       "muniLivePart1_3","muniLivePart2_3")]

full$trueCouple<-1



# merge fake_matches with relevant info:

fake_matches<-left_join(fake_matches, full[,c( "RINPart1", "municipalityPart1" ,
                                               "educPart1",          
                                               "incomePart1","year",    
                                               "agePart1",          
                                               "muniLivePart1_3")],
                        by=c("RINPart1"))


fake_matches<-left_join(fake_matches, full[,c( "RINPart2", "municipalityPart2" ,
                                               "educPart2",          
                                               "incomePart2","year",    
                                               "agePart2",          
                                               "muniLivePart2_3")],
                        by=c("RINPart2"))

full<-bind_rows(fake_matches,full)

rm(fake_matches)

gc()


full<-full%>%
  mutate(diffEduc=abs(educPart1-educPart2),
         diffIncome=abs(incomePart1-incomePart2),
         diffage=abs(agePart1-agePart2),
         diffMuniBirth=ifelse(municipalityPart1==municipalityPart2,1,0),
         diffMuniLive=ifelse(muniLivePart1_3==muniLivePart2_3,1,0))

training<-full[which(full$year>=2012&full$year<=2016),]
test<-full[which(full$year<2012|full$year>2016),]




# modeling
################################################################################
model<-glm(trueCouple ~ diffEduc+log(diffIncome+1)+log(diffage+1)+diffMuniBirth+diffMuniLive,
           data=training,family="binomial")

# predict on test set:
test$predictions<-predict(model, newdata=test, type="response")

# for some reaon it created 100 fake matches instead of 99 so I have 101 obs per part 1
test<-test%>%
  group_by(year,RINPart1)%>%
  mutate(ranking=102-rank(predictions))

test<-test%>%
  arrange(RINPart1,ranking)

test<-test%>%
  group_by(RINPart1)%>%
  mutate(n=n())


# there is a little thing to be fixed, some persons only have one match (no fake_matches)
# get rid of them for now but need to be corrected in the future

test<-test%>%
  filter(n>=100)

result<-test%>%
  filter(trueCouple==1)%>%
  group_by(year)%>%
  summarize(rank_pred=mean(ranking))

write.csv(result,"results_test_set_union.csv")

training<-training%>%
  select(-year.x,-year.y)

test<-test%>%
  select(-year.x,-year.y)

#############################
write.csv(test,"test.csv")
#############################

####################################
write.csv(training,"training.csv")
####################################

