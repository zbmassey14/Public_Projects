#Foreign Exchange Projecy


#Clear Space
rm(list=ls())
rm(list = ls(all.names = TRUE))
gc()


--------------------------------------------------------------------------------
  
  #Project Goal:
  
  #Find forex market inefficienties
  #Create triangular arbitrage function to 
  #evaluate these inefficienties
  
  #Create a function that searches for triangular arbitrage oppurtunities
  #in the current foreign exchange market
  
  #Steps:
  #Analyze current forex market
  #Calculate all arbitrage oppurtunities
  #Return/Evaluate the data
  
  
--------------------------------------------------------------------------------
  
  #Load Libraries
library(dplyr)
library(fmpcloudr)
library(TTR)
library(timeSeries)
library(tidyverse)
library(tidyquant)
library(ggplot2)
library(tidyr)
library(quantmod)
library(MASS)
library(quantmod)
library(PerformanceAnalytics)
library(PortfolioAnalytics)
library(ROI.plugin.glpk)
library(ROI.plugin.quadprog)
library(parallel)
library(parallelMap)
library(rvest)
library(BatchGetSymbols)
library(sqldf)
library(ggpubr)
library(data.table)
library(writexl)
fmpc_set_token('Your API Key')
#For faster computation use parallel comp backend - No need for this here.
#parallelStartSocket(cpus = detectCores())
library(gtools)
library(combinat)


--------------------------------------------------------------------------------
    
  
#Model that includes estimated transaction costs
  #Based off IBKR transaction costs
  
forex <- function(investment){
    
    #1. Pulls the forex exchange rates
    forex <- fmpcloudr::fmpc_price_forex()
    
    #2. Get rid of everything EXCEPT currencies
    #The API returned Currency/Curreny but also things like
    #soybean to NZD - We do not need that.
    forex2 <- forex[c(1:27, 33, 35, 40, 42, 50, 54, 55, 56, 59, 60, 62, 66:69,
                      72:76, 81, 83, 87, 89, 91, 95, 98:99, 101:104, 106, 108,
                      113:114, 117, 118, 122:124), c(1:8)]
    
    #3. Create the "base" & new Currencies (Splitting the ticker)
    forex2$base <- substr(x = forex2$ticker, start = 1, stop = 3)
    forex2$new <- substr(x = forex2$ticker, start = 4, stop = 6)
    
    
    #4. Create a list of all unqiue based/new currencies
    curr_list1 <- forex2$base[!duplicated(forex2$base)]
    curr_list2 <- forex2$new[!duplicated(forex2$new)]
    curr <- c(curr_list1, curr_list2)
    currencies <- curr[!duplicated(curr)]
    
    
    #5. 3-way Permutations of currencies (results in all sequences/orders)
    permutations1 <- as.data.frame(permutations(n = 22, r = 3, v = currencies)) 

    
    #6. Make everything readable
    perms <- permutations1
    perms$Curr1 <- perms$V1
    perms$Curr2 <- perms$V2
    perms$Curr3 <- perms$V3
    perms[1:3] <- NULL
    
    
    #7. List each "exchange" that would take place for each Permutation
    #(Exchange = 2 Currency combination that has an exhange rate)
    perms$exchange1 <- paste(perms$Curr1, perms$Curr2, sep = "")
    perms$exchange2 <- paste(perms$Curr2, perms$Curr3, sep = "")
    perms$exchange3 <- paste(perms$Curr3, perms$Curr1, sep = "")
    
    
    #8. Create a new "reference table" with all possible rates/combinations
    forex3 <- forex2[,c(9,10,2, 3)]
    forex3$ticker2 <- paste(forex3$new, forex3$base, sep = "")
    forex3$bid <- as.numeric(forex3$bid)
    forex3$ask <- as.numeric(forex3$ask)
    forex3$inverse <- (1/forex3$bid) #So we can use the inverse of rate provided
    forex3$inverse2 <- (1/forex3$ask)
    forex3$base2 <- substr(x = forex3$ticker2, start = 1, stop = 3)
    forex3$new2 <- substr(x = forex3$ticker2, start = 4, stop = 6)
    
    
    
    #9. Renaming columns for clarity
    forex2$bidRate <- forex2$bid
    forex2$askRate <- forex2$ask
    forex3$bidRate <- forex3$inverse
    forex3$askRate <- forex3$inverse2
    forex3$base <- forex3$base2
    forex3$new <- forex3$new2
    
    
    #10. Make the master reference table with the original
    #exchange rates & their inverse exchange rates
    forex4 <- rbind(forex2[,c(9:12)], forex3[,c(1,2,10,11)])
    #Create the 2-currency "exchange" code
    forex4$exchange <- paste(forex4$base, forex4$new, sep = "")
    
    
    #11. Merge on each exchange
    x <- left_join(x = perms, y = forex4, by = c("exchange1" = "exchange"))
    x <- left_join(x = x, y = forex4, by = c("exchange2" = "exchange"))
    x <- left_join(x = x, y = forex4, by = c("exchange3" = "exchange"))
    
    
    #12. Clear out the non-complete permutations
    #Ones where we do not have data
    x <- na.omit(x)
    
    
    #13. Clean everything up
    x$exhange1base <- x$base.x
    x$exhange1new <- x$new.x
    x$exhange1bid <- x$bidRate.x
    x$exhange1ask <- x$askRate.x
    
    x$exhange2base <- x$base.y
    x$exhange2new <- x$new.y
    x$exhange2bid <- x$bidRate.y
    x$exhange2ask <- x$askRate.y
    
    x$exhange3base <- x$base
    x$exhange3new <- x$new
    x$exhange3bid <- x$bidRate
    x$exhange3ask <- x$askRate
    
    x <- x[,c(1:6, 19:30)]
    
    
    #14. Numeric for the loop below
    x$exhange1bid <- as.numeric(x$exhange1bid)
    x$exhange1ask <- as.numeric(x$exhange1ask)
    x$exhange2bid <- as.numeric(x$exhange2bid)
    x$exhange2ask <- as.numeric(x$exhange2ask)
    x$exhange3bid <- as.numeric(x$exhange3bid)
    x$exhange3ask <- as.numeric(x$exhange3ask)
    
    
    #15. Triangular Arbitrage Loop
    for(i in 1:nrow(x)){
      f <- investment*x$exhange1ask[i] #You buy at the first ask price
      f <- f-(f*0.00002) #Subtract transaction costs 0.20 basis points
      
      a <- f*x$exhange2bid[i]
      a <- a-(a*0.00002)#Subtract transaction costs 0.20 basis points
      
      x$Reutn2Original[i] <- a*x$exhange3bid[i]
      x$Reutn2Original[i] <- x$Reutn2Original[i]-(x$Reutn2Original[i]*0.00002)
      #Subtract transaction costs 0.20 basis points (0.002%)
    }
    
    
    #Reorder to see the most profitable exhange
    x2 <- x %>%
      arrange(desc(Reutn2Original))
    
    #View the results
    print(head(x2))
    
    
    #From here we can select the best arbitrage oppurtunities
    
  }


forex <- commutatio(investment = 10000)



