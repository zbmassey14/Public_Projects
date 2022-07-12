#Portfolio Optimization:

#Clear Space
rm(list=ls())
rm(list = ls(all.names = TRUE))
gc()

#Load Libraries
library(dplyr)
library(finreportr)
library(tseries)
library(quantmod)
library(PerformanceAnalytics)
library(PortfolioAnalytics)
library(ROI.plugin.glpk)
library(ROI.plugin.quadprog)
library(parallel)
library(parallelMap)
library(xts)
library(tidyr)
library(timeSeries)
library(tidyverse)



--------------------------------------------------------------------------------
  
  
  
#Define Portfolio to optimize
tickers <- c("AAPL", "LLY", "MSFT", "AMZN", "TWTR", "UBER", "SPXL",
               "TQQQ", "T", "XOM")

#Pull the past portfolio prices from Yahoo Finance
portolioPrices <- NULL
for(ticker in tickers) {
  portolioPrices <- cbind(portolioPrices,
                          getSymbols.yahoo(ticker, from = '2019-01-03', 
                                           periodicity = 'daily',
                                           auto.assign = F)[,6])
}

#Check for missing data
colSums(is.na(portolioPrices))
portolioPrices <- na.omit(portolioPrices)
colSums(is.na(portolioPrices))



--------------------------------------------------------------------------------
  
  
  
#Create the returns object
stockReturns <- na.omit(Return.calculate(portolioPrices))
stockReturns #Returns for each asset in portfolio
portfolioReturns <- Return.portfolio((stockReturns))
portfolioReturns #Returns for the portfolio as a whole
table.AnnualizedReturns(R = portfolioReturns, Rf = 0.1/250)


#Create a marketbenchmarks
#S&P500
benchmarkPrices <- getSymbols.yahoo('^GSPC', from = '2019-01-03', 
                                    periodicity = 'daily', auto.assign = F)[,6]
colSums(is.na(benchmarkPrices))
benchmarkReturns <- na.omit(Return.calculate(benchmarkPrices))
benchmarkReturns
table.AnnualizedReturns(R = benchmarkReturns, Rf = 0.1/250)
#DJI
benchmarkPrices2 <- getSymbols.yahoo('^DJI', from = '2019-01-03', 
                                     periodicity = 'daily', auto.assign = F)[,6]
colSums(is.na(benchmarkPrices2)) #Check for Nas
benchmarkReturns2 <- na.omit(Return.calculate(benchmarkPrices2))
benchmarkReturns2
table.AnnualizedReturns(R = benchmarkReturns2, Rf = 0.1/250)



--------------------------------------------------------------------------------
  
  
  
#View Financial Ratios for the portfolios returns vs the benchmark
fin_ratios <- function(portfolioRets, benchmarkRets){
  CAPMBETA <- CAPM.beta(portfolioRets, benchmarkRets, .021/252)
  CAPMALPHA <- CAPM.jensenAlpha(portfolioRets, benchmarkRets, .021/252)
  SHAPRE <- SharpeRatio(portfolioRets, 0.021/252)
  INFORATIO <- InformationRatio(portfolioRets, benchmarkRets)
  CALMAR <- CalmarRatio(portfolioRets, scale = 252)
  results = list("*CAPM BETA*", CAPMBETA, 
                 "*CAPM ALPHA*", CAPMALPHA, 
                 "*SHARPE*", SHAPRE, 
                 "*INFO RATIO*", INFORATIO, 
                 "*CALMAR RATIO*", CALMAR)
  return(results)
}

fin_ratios(portfolioRets = portfolioReturns, benchmarkRets = benchmarkReturns)


--------------------------------------------------------------------------------
  
  
#Portfolio optimization
#Set optimization constraints (invesment mins/maxs, costs, etc)
port <- NULL
port <- portfolio.spec(assets = colnames((stockReturns)))
port <- add.constraint(portfolio = port,
                       type = "full_investment")
port <- add.constraint(port, type = "weight-sum", min_sum = 0.99, max_sum = 1.01) 
port <- add.constraint(portfolio = port, type = "transaction_cost", ptc = 0.001)
port <- add.constraint(portfolio = port,
                       type = "long_only")
port <- add.constraint(portfolio = port,
                       type = "box",
                       min = 0.02,
                       max = 0.5)
port


#Create the random portfolio weight instances & the objective
randPort <- port
randport <- add.objective(randPort,
                          type = "return",
                          name = "mean")
randport <- add.objective(randPort, type = "risk",
                          name = "StdDev",
                          target = 0.005)
set.seed(2112)
rp <- random_portfolios(port, 10000, "sample")

#Have to use parallel (it is required for this command)
parallelStartSocket(cpus = detectCores())

#Find the optimal rebalancing of the portfolio
opt_rebal <- optimize.portfolio.rebalancing(stockReturns,
                                            randport,
                                            optimize_method = "random",
                                            rp=rp,
                                            rebalance_on = "months",
                                            training_period = 1,
                                            rolling_window = 10)

opt_rebal


--------------------------------------------------------------------------------
  
  
#View the optimal reweighting strategy
chart.Weights(opt_rebal, main = "Rebalances Weights Over Time",
                cex.legend = 0.45)
#Extract the optimal weights & apply to the portfolio for backtesting
rebal_weights <- extractWeights(opt_rebal)
reabl_returns <- Return.portfolio(stockReturns, weights = rebal_weights)
table.AnnualizedReturns(R = reabl_returns, Rf = 0.1/250)


--------------------------------------------------------------------------------
  
  
#Visualizing the performance
rets.df <- cbind(reabl_returns, benchmarkReturns, benchmarkReturns2)
charts.PerformanceSummary(rets.df, main = "P/L Overtime")


