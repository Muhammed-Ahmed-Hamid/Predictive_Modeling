#Libraries:
library(caret)
library(tidyverse)
library(dplyr)
library(skimr)
library(lubridate)

#Turning off scientific notation:
options(scipen=999)#Turn off scientific notation as global setting


#Loading in data:
Cancelation<-read.csv('Hospitality.csv')

#EDA:
skim(Cancelation) #Missing values in children column.
table(Cancelation$company,Cancelation$is_canceled)
table(Cancelation$agent,Cancelation$is_canceled)
table(Cancelation$country,Cancelation$is_canceled)
table(Cancelation$reservation_status,Cancelation$is_canceled)


#Looking and sorting through NULL and Missing Values:
null_vars<-apply(Cancelation, 2, function(x) any(grepl('NULL', x)))==T

null_vars[null_vars == TRUE]

sort(table(Cancelation$company)) #94% of this variable contains null values. Will remove this predictor variable.

sort(table(Cancelation$agent)) #Listing top agents in terms of frequency. Based on ML results, can change this amount if needed.

sort(table(Cancelation$country)) #Will list the top countries in terms of frequency. Based on ML results, can change this amount if needed.

sort(table(Cancelation$children))#Will impute the missing values with the median value.

sort(table(Cancelation$reservation_status))

sort(table(Cancelation$is_canceled))
#The missing values in children column have been replaced with the median value of 0.
median_value <- median(Cancelation$children, na.rm = TRUE)
Cancelation$children <- replace(Cancelation$children, is.na(Cancelation$children), median_value)

#Listing the top ten countries with the most bookings in order to reduce the abundance of categorical features in country column:
country_freq <- table(Cancelation$country)
top_n <- 20
top_countries <- names(head(sort(country_freq, decreasing = TRUE), top_n))
Cancelation$top_countries <- ifelse(Cancelation$country %in% top_countries, Cancelation$country, "Other")

#Listing the top five most frequent agents:
agent_freq <- table(Cancelation$agent)
top_n <- 5
top_agent <- names(head(sort(agent_freq, decreasing = TRUE), top_n))
top_agent
Cancelation$top_agent <- ifelse(Cancelation$agent %in% top_agent, Cancelation$agent, "Other")
Cancelation <- Cancelation %>%
  mutate(top_agent = ifelse(top_agent == "NULL", "Unknown", top_agent)) #Swapping NULL with unknown for more clarity.

#Removing nonessential columns that have been updated in other columns.
columns_to_remove <- c('country', 'agent' , 'company')
Cancelation <- Cancelation %>% select(-columns_to_remove)


#Converting reservation_status_date from character to date variable:
Cancelation$reservation_status_date <- mdy(Cancelation$reservation_status_date)

#Extracting the year, month, and day values from reservation_status_date to create separate columns:
Cancelation$reservation_status_year <- year(Cancelation$reservation_status_date) #Year of reservation.
Cancelation$reservation_status_day_name <- weekdays(Cancelation$reservation_status_date) #Day name for reservation.
Cancelation$reservation_status_week_of_year <- isoweek(Cancelation$reservation_status_date) #Reservation for week of year.
Cancelation$reservation_status_quarter <- quarter(Cancelation$reservation_status_date) #Reservation status date in quarters.
Cancelation <- Cancelation %>%
  mutate(reservation_status_quarter = case_when(
    reservation_status_quarter == 1 ~ "Q1",
    reservation_status_quarter == 2 ~ "Q2",
    reservation_status_quarter == 3 ~ "Q3",
    reservation_status_quarter == 4 ~ "Q4",
    TRUE ~ NA_character_
  )) #Renaming the quarter values for more clarity.


#Creating a week frequency column to replace week column that has too many values:
week_freq <- table(Cancelation$reservation_status_week_of_year) #Most frequent weeks reserved. Everything else is listed as other category.
top_n <- 5
top_week <- names(head(sort(week_freq, decreasing = TRUE), top_n))
top_week
Cancelation$reservation_status_top_weeks <- ifelse(Cancelation$reservation_status_week_of_year %in% top_week, Cancelation$reservation_status_week_of_year, "Other")

#The reservation itself is more valuable for the given business problem, which is why opted to remove the arrival columns.
#Additionally, after extracting the necessary information from the reservation data variable, we do not need it anymore.
columns_to_remove2 <- c('reservation_status_week_of_year','arrival_date_year','arrival_date_month',
                        'arrival_date_week_number', 'arrival_date_day_of_month',
                        'reservation_status_date')
Cancelation <- Cancelation %>% select(-columns_to_remove2)


#Pre-processing the data/Creating dummy variables:
Cancelation$is_canceled <- ifelse(Cancelation$is_canceled == 1, "canceled", "notcanceled")

#1. change reponse to a factor
Cancelation$is_canceled<-as.factor(Cancelation$is_canceled)

#2. rename resonse 
Cancelation$is_canceled<-fct_recode(Cancelation$is_canceled, canceled = "1",notcanceled = "0")

#3. relevel response
Cancelation$is_canceled<- relevel(Cancelation$is_canceled, ref = "canceled")

#make sure levels are correct
levels(Cancelation$is_canceled)

#Creating dummy variables:
Cancelation_predictors_dummy <- model.matrix(is_canceled~ ., data = Cancelation)#create dummy variables expect for the response.
Cancelation_predictors_dummy<- data.frame(Cancelation_predictors_dummy[,-1]) #get rid of intercept
Cancelation <- cbind(is_canceled=Cancelation$is_canceled, Cancelation_predictors_dummy)

# Get rid of the proxy to the response column which is from the reservation status variable:
Cancelation <- subset(Cancelation, select = -reservation_statusCheck.Out)

#Saving File as a CSV in current working directory.
write.csv(Cancelation, file = "Hospitality_final2.csv", row.names = FALSE)

#General notes: When I get to modeling, I will determine based on the results,
#if I need to re add some columns, or make any other changes to the data set,
#to improve the model's performance.


