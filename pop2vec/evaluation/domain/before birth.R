# Income: predict from before birth 
# July 2023

library(dplyr)
library(haven) # package to open dataset from different formats
library(stringr)
library(ggplot2)
library(lubridate) # work with dates
library(tidyr)





setwd("F:/Documents/sequence/data/employment")

# Works only after having run script "income project"
load("edu.saved")

# write the file in stata so I can recode the education into 
# a few categories:
write_dta(edu,"F:/Documents/sequence/data/employment/edu.dta")

edu<-read_dta("edu.dta")

save(edu, file="edu.saved")

#      []

# Get the family chief for which we have education,
# that potentially have children with an income: 
# For this we first match with household file: 

household<-read_sav("G:/Bevolking/GBAHUISHOUDENSBUS/opgesplitste jaarbestanden/GBAHUISHOUDENSBUS2007.sav")

load("edu.saved")

# # Take only the first household per person:
# household<-household%>%
#   group_by(RINPERSOON)%>%
#   slice(1)
# 
# save(household,file="household.saved")
# 
# load("household.saved")

# merge with education
household<-left_join(household, edu[,c("RINPERSOON","isced97field1")],by="RINPERSOON")

rm(edu)


# Merge with the age etc. info:

# demo<-read_sav("G:/Bevolking/VRLGBAPERSOONKTAB/VRLGBAPERSOONKTABV2022121.sav")
# 
# save(demo,file="demo.saved")
load("demo.saved")

household<-left_join(household,demo,by="RINPERSOON")

rm(demo)

# Rename variables:
household<-household%>%
  select(-RINPERSOONS.x,-RINPERSOONS.y)

names(household)<-c( "RINPERSOON","DATUMAANVANGHH", "DATUMEINDEHH", "idHouse", 
                     "typeHouse", "placePers", "refPers", "sizeHouse",
                     "numberChild", "numPersOther","yearBirthYoungestChild","monthBirthYoungestChild",
                     "yearBirthOldestChild","monthBirthOldestChild","houseNotImputed","date_start" ,          
                     "date_end","education","gender","yearBirth"   ,
                     "monthBirth","dayBirth")



household<-household%>%
  arrange(idHouse,desc(refPers),placePers)


save(household,file="household.saved")

load("household.saved")

# To go faster, I will only keep part of the observations 
household$typeHouse<-as.numeric(as.character(household$typeHouse))

# Take only household with children
household<-household[which(household$typeHouse>=4&household$typeHouse<=7),] 

# Take a smaller sample:
household<-household[1:5000000,]


# Find the ref person
household<-household%>%
  group_by(idHouse)%>%
  mutate(refPersId=first(RINPERSOON[refPers=="1"]))


household$placePers<-as.numeric(as.character(household$placePers))

# Find the second parent if any:
household$secondParent<-ifelse(household$refPers=="0"&
                                 (household$placePers>=3&household$placePers<=6),
                               1,0)

household<-household%>%
  group_by(idHouse)%>%
  mutate(secParId=first(RINPERSOON[secondParent==1]))




# Open divorce Files:
years<-list.files("G:/Bevolking/GBASCHEIDINGENMASSATAB/")


divorces<-c()
for (i in years){
  files<-list.files(paste0("G:/Bevolking/GBASCHEIDINGENMASSATAB/",i,"/"))
  # files<-files[-1]
  index_to_remove<-which(files=="geconverteerde data")
  print(index_to_remove)
  if(length(index_to_remove)>0){
    files<-files[-index_to_remove]
  }
  
  print(files)
  
  f<-read_sav(paste0("G:/Bevolking/GBASCHEIDINGENMASSATAB/",i,"/",files))
  
  divorces<-bind_rows(divorces,f)

}

rm(index_to_remove,i,files,years,f,refPersIds)


names(divorces)<-c( "separationYear","SCHEIDINGNUMMER","RINPERSOONSPARTNER1S",
                    "idPart1","RINPERSOONSPARTNER2S","idPart2",               
                    "dateSep","PUBLICATIEJAARSCHEIDING","numDivPart1",        
                    "numDivPart2","AANTALMINDKINDBIJSCHEIDING",
                    "AANTALMINDKINDBIJSCHEIDINGPARTNER1",
                    "AANTALMINDKINDBIJSCHEIDINGPARTNER2")


divorces<-divorces%>%
  select(separationYear,idPart1,idPart2,dateSep,numDivPart1,numDivPart2)


divorces$idPart1<-as.character(divorces$idPart1)
divorces$idPart2<-as.character(divorces$idPart2)
divorces$dateSep<-as.numeric(substr(as.character(divorces$dateSep),1,4))
divorces$numDivPart1<-as.numeric(as.character(divorces$numDivPart1))
divorces$numDivPart2<-as.numeric(as.character(divorces$numDivPart2))



# death
deaths<-read_sav("G:/Bevolking/VRLGBAOVERLIJDENTAB/VRLGBAOVERLIJDENTABV2023061.sav")

deaths<-deaths[,2:3]

deaths$VRLGBADatumOverlijden<-as.numeric(substr(as.character(deaths$VRLGBADatumOverlijden),1,4))

names(deaths)[2]<-"deathYear"



# merge with household data:
div1<-divorces%>%
  select(-idPart2,-numDivPart2)

div2<-divorces%>%
  select(-idPart1,-numDivPart1)

names(div1)[2]<-"RINPERSOON"
names(div2)[2]<-"RINPERSOON"


household<-left_join(household,div1,by="RINPERSOON")

# Here needs to reframe datasets so that there are columns for divorce 1, divorce 2 etc.
# To avoid many to many matching later

household<-left_join(household,div2,by="RINPERSOON")



household<-left_join(household,deaths,by="RINPERSOON")



# Create a subsample of refPersons
refPersIds<-unique(household$refPersId)

refs<-household[which(household$RINPERSOON%in%refPersIds),
                c("RINPERSOON","education","gender","yearBirth","separationYear.x",
                  "numDivPart1","deathYear")]

# We need only one obs per individual to avoid many to many matching
refs<-refs%>%
  group_by(RINPERSOON)%>%
  slice(1)


names(refs)<-c("refPersId","educationRef","genderRef","yearBirthRef","sepYearRef",
               "numDivRef","deathYearRef")



household<-left_join(household,refs,by="refPersId")

rm(refs,refPersIds)




# Same for second parent:
refPersIds<-unique(household$secParId)

refs<-household[which(household$RINPERSOON%in%refPersIds),
                c("RINPERSOON","education","gender","yearBirth","separationYear.x",
                  "numDivPart1","deathYear")]

# We need only one obs per individual to avoid many to many matching
refs<-refs%>%
  group_by(RINPERSOON)%>%
  slice(1)


names(refs)<-c("secParId","educationSec","genderSec","yearBirthSec","sepYearSec",
               "numDivSec","deathYearSec")



household<-left_join(household,refs,by="secParId")

rm(refs,refPersIds)


temporary<-household


save(temporary,file="temporary.saved")

rm(temporary)

# NEXT:
# On peut toujours ajouter les divorce et death de la personne focale

# On peut par ailleurs ajouter les caracteristiques des freres et soeurs





small<-household[which(household$placePers==1),]

small<-small%>%
  group_by(RINPERSOON)%>%
  slice(1)


# Income files:
files<-list.files("G:/InkomenBestedingen/INPATAB/")
files<-files[-1]

for (file_name in files){
  data<-read_sav(paste0("G:/InkomenBestedingen/INPATAB/",file_name),
                 col_select = c("RINPERSOON",
                                "INPBELI"))
  
  # find the two characters after 20 to then name the file
  year<-extract_vec<-str_extract(file_name,"(?<=20)..")
  
  names(data)<-c("RINPERSOON",paste0("inc_20",year))
  
  small<-left_join(small,data,by="RINPERSOON")
  
  print(file_name)
  
  rm(data)
  
  print(Sys.time())
  
}


