library(dplyr)
library(haven) # package to open dataset from different formats
library(stringr)
library(ggplot2)
library(lubridate) # work with dates



# File that contains the original spolibus data:
path_spolisbus<-"G:/Spolis/SPOLISBUS/"

# The script will produce "intermediary files that will be stored here
# BE CAREFUL: This file should also already contain "birth_info.csv"
path_preprocessed_files<-"F:/Documents/sequence/data/emp/"


# File where the different csv output files will be written
# and can then be used by Tanzir
path_output<-"F:/Documents/sequence/data/"



years<-list.files(path_spolisbus)
for (i in years){
  files<-list.files(paste0(path_spolisbus,i,"/"))
  files<-files[-1]
  print(files)
  data<-read_sav(paste0(path_spolisbus,i,"/",files),
                 col_select = c("RINPERSOON","IKVID","SDATUMAANVANGIKO","SDATUMEINDEIKO",
                                "SAANTVERLU","SLNINGLD","SLNLBPH",
                                "SCDAARD","SCONTRACTSOORT","SSOORTBAAN",
                                "SINDZW","SPOLISDIENSTVERBAND","SSRTIV",
                                "SCAOSECTOR","SSECT"),
                 n_max=50000)
  
  
  # change class problematic variables imported from .sav
  data<-data%>%
    mutate_if(is.factor,as.character)%>%
    mutate_if(function(x) any(class(x)%in%c("haven_labelled")),as.character)
  
  data$beg<-as_date(data$SDATUMAANVANGIKO)
  
  data$end<-as_date(data$SDATUMEINDEIKO)
  
  data<-data%>%
    arrange(RINPERSOON,IKVID,SDATUMAANVANGIKO)
  
  data<-rename(data,
               empRelationship=SCDAARD,
               contractType=SCONTRACTSOORT,
               sicknessInsurance=SINDZW,
               fullTime=SPOLISDIENSTVERBAND,
               sector=SCAOSECTOR,
               industry=SSECT,
               jobType=SSRTIV)
  
  
  d<-data%>%
    group_by(RINPERSOON,IKVID,contractType)%>%
    summarize(begJob=min(beg),
              endJob=max(end),
              inc=sum(SLNINGLD),
              inc2=sum(SLNLBPH),
              nbHours=sum(SAANTVERLU))
  
  d$inc<-findInterval(d$inc,unname(quantile(d$inc,probs=seq(0,1,0.01))))
  d$inc2<-findInterval(d$inc2,unname(quantile(d$inc2,probs=seq(0,1,0.01))))
  d$nbHours<-findInterval(d$nbHours,unname(quantile(d$nbHours,probs=seq(0,1,0.01))))
  
  
  d2<-data%>%
    group_by(RINPERSOON,IKVID,contractType)%>%
    select(contractType,sicknessInsurance,fullTime,sector,industry,jobType,empRelationship)%>%
    slice(1)
  
  d<-left_join(d,d2,by=c("RINPERSOON","IKVID","contractType"))
  rm(d2)
  
  d<-d%>%
    group_by(RINPERSOON)%>%
    mutate(f=1,
           jobNumber=cumsum(f),
           totNumberJobs=n())%>%
    select(-f)
  
  d<-d%>%
    arrange(RINPERSOON,begJob,desc(endJob))
  
  # d$RINPERSOON<-as.character(d$RINPERSOON)
  
  d$year<-i
  
  write.table(d,paste0(path_preprocessed_files,"emp_",as.character(i),".csv"),
              sep=";",
              row.names = FALSE)
  
  # print(d$year[1])
  # print(str(d))
  rm(d)
}





data<-c()
files<-list.files(path_preprocessed_files)

for (i in files){
  d<-read.table(paste0(path_preprocessed_files,i),
                sep=";",
                header = TRUE,
                colClasses=c(RINPERSOON = "character"))
  print(typeof(d$RINPERSOON))
  print(d$year[1])
  head(d)
  data<-bind_rows(data,d)
  rm(d)
}
rm(files,i)



data<-data%>%
  arrange(RINPERSOON,IKVID,contractType,begJob)

d<-data%>%
  group_by(RINPERSOON,IKVID,contractType)%>%
  summarise(begJob=min(begJob),
            endJob=max(endJob))

data<-data%>%
  group_by(RINPERSOON,IKVID,contractType)%>%
  slice(1)%>%
  select(-begJob,-endJob,-inc,-inc2,-nbHours,-jobNumber,-totNumberJobs)

data<-left_join(data,d,by=c("RINPERSOON","IKVID","contractType"))

rm(d)

# Now we want a separate event for begining and end of the job, so basically
# we are just doubling the number of rows and we add a column saying whether
# this is the beginning or the end date.
data2<-data
data$begOrEnd<-1
data2$begOrEnd<-2
data<-bind_rows(data,data2)
rm(data2)

data$date<-ifelse(data$begOrEnd==1,data$begJob,data$endJob)

data<-data%>%
  select(-begJob,-endJob)

# death<-read.csv("F:/Documents/sequence/data/death_info.csv",sep=";")
# oldestEvent<-min(death$date)
oldestEvent<-"1971-12-30"

birth<-read.table(paste0(path_preprocessed_files,"birth_info.csv"),
                  sep=";",
                  colClasses=c(RINPERSOON = "character"))

data<-left_join(data,birth[,c("RINPERSOON","birthDate")])

# Only keep those for whom a match could be find. 
data<-data[which(complete.cases(data$birthDate)),]

data$age<-as.numeric(interval(data$birthDate,data$date) / years(1))

# if there are negative  values delete them
data$age<-ifelse(data$age<0,NA,data$age)

# Now create var for the days since the oldest event in the overall data:
data$daysSinceFirstEvent<-as.numeric(difftime(data$date,oldestEvent,units="days"))

data<-data%>%
  select(RINPERSOON,age,daysSinceFirstEvent,contractType,sicknessInsurance,
         fullTime,sector,industry,jobType,empRelationship,begOrEnd)

# select function kept IKVID even though I did not ask for it so delete it:
data<-data[,-1]

data$empRelationship2 <- ifelse(is.na(data$empRelationship), -1, as.numeric(factor(data$empRelationship)))
data$contractType2 <- ifelse(is.na(data$contractType), -1, as.numeric(factor(data$contractType)))
data$sicknessInsurance2 <- ifelse(is.na(data$sicknessInsurance), -1, as.numeric(factor(data$sicknessInsurance)))
data$fullTime2 <- ifelse(is.na(data$fullTime), -1, as.numeric(factor(data$fullTime)))
data$sector2 <- ifelse(is.na(data$sector), -1, as.numeric(factor(data$sector)))
data$industry2 <- ifelse(is.na(data$industry), -1, as.numeric(factor(data$industry)))
data$jobType2 <- ifelse(is.na(data$jobType), -1, as.numeric(factor(data$jobType)))


selected_vars<-names(data)[4:10]

data<-as.data.frame(data)

for (i in selected_vars){
  j<-paste0(i,2)
  correspondance_table <- data.frame(
    former_categories = levels(factor(data[,i])),
    new_categories = 1:length(levels(factor(data[,j])))
  )
  write.table(correspondance_table,paste0("F:/Documents/sequence/data/corresp_table_",i,".csv"),
              sep=";",
              row.names=FALSE)
}

# Maybe do something with the range
data<-data%>%
  select(RINPERSOON,age,daysSinceFirstEvent,contractType2,sicknessInsurance2,
         fullTime2,sector2,industry2,jobType2,empRelationship2,begOrEnd)

write.table(data,paste0(path_output,"jobs_full_duration.csv"),
            sep=";",
            row.names = FALSE)



# Do the files year by year to get the wage associated with the job on an annual basis


data<-c()
files<-list.files("path_preprocessed_files")

for (i in files){
  data<-read.table(paste0("path_preprocessed_files",i),
                sep=";",
                header = TRUE,
                colClasses=c(RINPERSOON = "character"))
  
  
  data<-data%>%
    arrange(RINPERSOON,IKVID,contractType,begJob)
  
  d<-data%>%
    group_by(RINPERSOON,IKVID,contractType)%>%
    summarise(begJob=min(begJob),
              endJob=max(endJob))
  
  data<-data%>%
    group_by(RINPERSOON,IKVID,contractType)%>%
    slice(1)%>%
    select(-jobNumber,-totNumberJobs,-begJob,-endJob)
  
  data<-left_join(data,d,by=c("RINPERSOON","IKVID","contractType"))
  
  rm(d)
  
  
  oldestEvent<-"1971-12-30"
  
  # birth<-read.table("F:/Documents/sequence/data/birth_info.csv",
  #                   sep=";",
  #                   colClasses=c(RINPERSOON = "character"))
  # 
  data<-left_join(data,birth[,c("RINPERSOON","birthDate")])
  
  # Only keep those for whom a match could be find. 
  data<-data[which(complete.cases(data$birthDate)),]
  
  data$age<-as.numeric(interval(data$birthDate,data$endJob) / years(1))
  
  # if there are negative  values delete them
  data$age<-ifelse(data$age<0,NA,data$age)
  
  # Now create var for the days since the oldest event in the overall data:
  data$daysSinceFirstEvent<-as.numeric(difftime(data$endJob,oldestEvent,units="days"))
  
  data<-data%>%
    select(RINPERSOON,age,daysSinceFirstEvent,contractType,sicknessInsurance,
           fullTime,sector,industry,jobType,empRelationship,inc,inc2,nbHours)
  
  # select function kept IKVID even though I did not ask for it so delete it:
  data<-data[,-1]
  
  data$empRelationship2 <- ifelse(is.na(data$empRelationship), -1, as.numeric(factor(data$empRelationship)))
  data$contractType2 <- ifelse(is.na(data$contractType), -1, as.numeric(factor(data$contractType)))
  data$sicknessInsurance2 <- ifelse(is.na(data$sicknessInsurance), -1, as.numeric(factor(data$sicknessInsurance)))
  data$fullTime2 <- ifelse(is.na(data$fullTime), -1, as.numeric(factor(data$fullTime)))
  data$sector2 <- ifelse(is.na(data$sector), -1, as.numeric(factor(data$sector)))
  data$industry2 <- ifelse(is.na(data$industry), -1, as.numeric(factor(data$industry)))
  data$jobType2 <- ifelse(is.na(data$jobType), -1, as.numeric(factor(data$jobType)))
  
  data<-data%>%
    select(RINPERSOON,age,daysSinceFirstEvent,contractType2,sicknessInsurance2,
           fullTime2,sector2,industry2,jobType2,empRelationship2,inc,inc2,nbHours)
  
  write.table(data,paste0(path_output,"paycheck_",i),
              sep=";",
              row.names=FALSE)
  
}
rm(files,i)



