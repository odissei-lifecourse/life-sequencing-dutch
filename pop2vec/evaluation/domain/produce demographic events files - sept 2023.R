# Create data for demographic events: divorce, death, birth, union

# Sept 2023

library(dplyr)
library(haven) # package to open dataset from different formats
library(stringr)
library(ggplot2)
library(lubridate) # work with dates
library(tidyr)





setwd("F:/Documents/sequence/data/employment")

# open birthdate info:
load("demo.saved")

# create a single date of birth variable:
demo$birthDate<-ymd(paste(demo$VRLGBAGEBOORTEJAAR,demo$VRLGBAGEBOORTEMAAND,demo$VRLGBAGEBOORTEDAG,sep="-"))


demo<-demo%>%
  select(RINPERSOON,birthDate,VRLGBAGESLACHT)

names(demo)[3]<-"gender"


# open birthplace info:

birthplace<-read_sav("G:/Bevolking/VRLGBAGEBOORTEGEMEENTETAB/VRLGBAGEBOORTEGEMEENTE2022TABV1.sav")


# match the two:

demo<- left_join(demo,birthplace,by="RINPERSOON")

demo<-demo%>%select(-RINPERSOONS)

names(demo)[4]<-"municipality"


write.table(demo,"F:/Documents/sequence/data/birth_info.csv",sep=";",row.names = FALSE)




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
  select(dateSep,idPart1,idPart2,numDivPart1,numDivPart2)

divorces$dateSep<-as_date(divorces$dateSep)


divorces$idPart1<-as.character(divorces$idPart1)
divorces$idPart2<-as.character(divorces$idPart2)
divorces$numDivPart1<-as.numeric(as.character(divorces$numDivPart1))
divorces$numDivPart2<-as.numeric(as.character(divorces$numDivPart2))


divorces2<-divorces



names(divorces)<-c("date","RINPERSOON","idPart2","numberDivorces","numberDivorcesPartner")
names(divorces2)<-c("date","idPart2","RINPERSOON","numberDivorcesPartner","numberDivorces")

divorces<-bind_rows(divorces,divorces2)

divorces<-distinct(divorces)


write.table(divorces,"F:/Documents/sequence/data/divorce_info.csv",sep=";",row.names=FALSE)


# Death

deaths<-read_sav("G:/Bevolking/VRLGBAOVERLIJDENTAB/VRLGBAOVERLIJDENTABV2023061.sav")

deaths<-deaths[,2:3]

deaths$VRLGBADatumOverlijden<-as_date(deaths$VRLGBADatumOverlijden)

names(deaths)[2]<-"date"

write.table(deaths,"F:/Documents/sequence/data/death_info.csv",sep=";",row.names=FALSE)



# Education
# Works only after having run script "income project"
load("edu.saved")

# as we do not have the exat graduation date I put the same month and year for everyone:
# first of july. The year was known from the year of the original file (see income project script)
edu$date<-as_date(paste0(edu$year,"07-01"))

edu<-edu%>%
  select(RINPERSOON,date,oplnr,isced97field1)

names(edu)[3]<-"educationDetailled"
names(edu)[4]<-"educationSimple"


edu$educationDetailled<-as.character(edu$educationDetailled)
edu$educationDetailled<-as.numeric(edu$educationDetailled)

edu$educationSimple<-as.character(edu$educationSimple)
edu$educationSimple<-as.numeric(edu$educationSimple)

edu$educDet<-ifelse(is.na(edu$educationDetailled),-1,as.numeric(factor(edu$educationDetailled)))
edu$educSim<-ifelse(is.na(edu$educationSimple),-1,as.numeric(factor(edu$educationSimple)))


edu<-edu%>%
  select(-educationDetailled,-educationSimple)

edu_bis<-edu

rm(edu)

save(edu_bis,file="edu_bis.saved")


# Reframe everything in the format that Tanzir asked for:

# the oldest event is in the death file (for the moment)

# Load all files 


birth<-read.table("F:/Documents/sequence/data/birth_info.csv",
                sep=";",
                colClasses=c(RINPERSOON = "character"))




# start with birth:
# Slightly different from the other files. 

birth$month<-month(as_date(birth$birthDate))
birth$year<-year(as_date(birth$birthDate))

birth$gender<-ifelse(is.na(birth$gender),-1,as.numeric(factor(birth$gender)))
birth$municipality<-ifelse(is.na(birth$municipality),-1,as.numeric(factor(birth$municipality)))
birth$month<-ifelse(is.na(birth$month),-1,as.numeric(factor(birth$month)))
birth$year<-ifelse(is.na(birth$year),-1,as.numeric(factor(birth$year)))

# For all other files I will need to match it first with birth
# to get everyone birthday and compute their age when the event occured

birth$RINPERSOON<-as.character(birth$RINPERSOON)

oldestEvent<-"1971-12-30"




small_birth<-birth%>%
  select(RINPERSOON,year,month,gender,municipality)

write.table(small_birth,"F:/Documents/sequence/data/birth_info_good_format.csv",sep=";",row.names = FALSE)




# EDUCATION:

load("edu_bis.saved")

edu_bis<-left_join(edu_bis,birth[,c("RINPERSOON","birthDate")])

edu_bis$date<-as_date(edu_bis$date)
edu_bis$date<-as.character(edu_bis$date)

# if we don't know the degree we can't know the date of the graduation. 
edu_bis$date2<-ifelse(edu_bis$educDet<0,NA,edu_bis$date)

edu_bis$age<-as.numeric(interval(edu_bis$birthDate,edu_bis$date2) / years(1))

# there are a few strange (negative values)
# for the time being I recode them as missing
edu_bis$age<-ifelse(edu_bis$age<14,NA,edu_bis$age)

# Now create var for the days since the oldest event in the overall data:
edu_bis$daysSinceFirstEvent<-as.numeric(difftime(edu_bis$date2,oldestEvent,units="days"))

edu_bis<-edu_bis%>%
  select(RINPERSOON,daysSinceFirstEvent,age,educDet,educSim)

edu_bis<-edu_bis[complete.cases(edu_bis[,c("daysSinceFirstEvent","age")]),]

write.table(edu_bis,"F:/Documents/sequence/data/education_info_good_format.csv",sep=";",row.names = FALSE)









# DIVORCE

div<-read.table("F:/Documents/sequence/data/divorce_info.csv",sep=";",
                colClasses=c(RINPERSOON = "character"))


div<-left_join(div,birth[,c("RINPERSOON","birthDate")])


# clean the RINPERSOON

div$age<-as.numeric(interval(div$birthDate,div$date) / years(1))

# if there are negative  values delete them
div$age<-ifelse(div$age<0,NA,div$age)

# Now create var for the days since the oldest event in the overall data:
div$daysSinceFirstEvent<-as.numeric(difftime(div$date,oldestEvent,units="days"))

div<-div%>%
  select(RINPERSOON,daysSinceFirstEvent,age,numberDivorces)

# Delete if we don't know the RINPSERSOON of the first partner:

div$id<-as.numeric(div$RINPERSOON)
div<-div[which(!is.na(div$id)),]
div<-div%>%
  select(-id)

write.table(div,"F:/Documents/sequence/data/divorce_info_good_format.csv",
            sep=";",
            row.names = FALSE)






# DEATH

death$RINPERSOON<-as.character(death$RINPERSOON)
death<-left_join(death,birth[,c("RINPERSOON","birthDate")])

death<-death[complete.cases(death$RINPERSOON),] # did not chage anything

death$age<-as.numeric(interval(death$birthDate,death$date) / years(1))

# if there are negative  values delete them
death$age<-ifelse(death$age<0,NA,death$age)

# Now create var for the days since the oldest event in the overall data:
death$daysSinceFirstEvent<-as.numeric(difftime(death$date,oldestEvent,units="days"))

death<-death%>%
  select(RINPERSOON,daysSinceFirstEvent,age,numberdeathorces)

write.table(death,"F:/Documents/sequence/data/deathorce_info_good_format.csv",sep=";",row.names = FALSE)






