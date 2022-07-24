#Zak Massey
#Moneyball Project Continuation
#Hierarhical Clustering


#Goal:
#Build off previous supervised learning techniques via unsupervised learning
#with the goal of determining how/who to scout for an MLB roster.
#What players should we pursue?

#Previously we found individual-level features & their relationship with runs per game
#We determined players with certain attributes would provide higher runs per game
#Now we are looking to cluster the players on a more broad level, hoping to segment 
#the players in a way which provides insight regarding what players we should pursue.

--------------------------------------------------------------------------------


#Import Required Starting Packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import stats
import random
import sklearn
import sklearn.cluster
import scipy.cluster.hierarchy as sc
from sklearn.cluster import AgglomerativeClustering

--------------------------------------------------------------------------------

#Import the baseball dataset & clean

#Import Data
baseball = pd.read_csv("your/path/here")
baseball.drop(["Pos Summary", "Name-additional", "Rk", "szn", "Tm", "Lg"], axis=1, inplace=True)

#Clean the data
#Drop values that don't meet critera & duplicates
baseball = baseball.drop_duplicates()
baseball = baseball.dropna()
baseball.drop(["Name"], axis = 1, inplace=True)
baseball.drop(["TB"], axis = 1, inplace=True)
baseball.drop(["PA"], axis = 1, inplace=True)
baseball.drop(["AB"], axis = 1, inplace=True)

#We can do some light feature engineering
baseball["runs_per_game"] = baseball.R/baseball.G

#View the data
baseball


--------------------------------------------------------------------------------

#Further cleaning

#To verify there are outliers according to the zscore
baseball["z"] = np.abs(stats.zscore(baseball.runs_per_game))


#We can see all the values that fall outside the z score range
#This is good because most of them only have 1-3 games played, which is not the best sample
baseball[baseball.z > 3]


#Drop the Runs and Zscore columns
baseball.drop(["z", "R"], axis = 1, inplace=True)

#Drop all obs who played less than 20 games
baseball3 = baseball[baseball["G"]> 20]
baseball3

--------------------------------------------------------------------------------

#Need to standardize the data since we are going to be evaluting distance 
baseball3_mean = baseball3.mean()
baseball3_std = baseball3.std()

baseball4 = (baseball3-baseball3_mean)/baseball3_std
baseball4

#Group the features
baseball_x = baseball4.iloc[:, 0:21].values
baseball_y = baseball4.iloc[:, 21:22].values


--------------------------------------------------------------------------------

#Plot dendrogram
plt.figure(figsize=(30, 10))
plt.title("Dendrograms")

# Create dendrogram
sc.dendrogram(sc.linkage(baseball_x, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Euclidean distance')


--------------------------------------------------------------------------------

#We area wanting to minimize the within cluster varaince
#So how do we determine the optimal number of clusters by looking at the diagram?

#It is somewhat subjective based on what you are trying to seek within the data
#For example, if we were looking to gain insights on credit card defaults, it may be useful to keep the number of clusters small
#Around 2/3/4, because in the back of our head, at the end of the day there are two groups

#For our cases we know from previous projects, that a players runs per game is continious
#And there are a handful of highly impactful features such as OPS, slugging, stolen bases, and age

#For this reason, the number of clusters MAY need to be a bit larger, around 5


#We can look for the largest vertical distance without splitting into a horizontal line
#Here, it looks like the optimal number of clusters would be 3 or 5

#If we were to split the data into two clusters, there would be a lot of overlap betwen groups
#Espcially since we are dealing with mostly continous baseball statistics
#Hence we may not be able to gain much of an insight, because the two groups are less unquie

#Back to the credit card example - in a dataset like that, much of the data might be factors
#Like single/married, employed/unemployed, male/female etc
#Which have defined separation in the data

#But in our case, we have ONLY continous features such as # of RBIS, H, SBs, etc etc
#So there is no definate separation, each "threshold" smoothly transistions into the next,
#And each threshold is subjective based on the player & their competition
#The only separation is subjective, which could be defined by clusters - but there WILL be much more overlap

#For that reason we are going to have to increase the number of clusters beyond what would be considered "optimal"
#According to quantitiacive metrics like the silhouette score

#In my opinion you should not sacrifice interpretability/gained insights for quantitativley optimal settings


--------------------------------------------------------------------------------

#Define how many clusters we should use based on the graph above
#Running each cluster through the silhouette scores
cluster_3 = AgglomerativeClustering(
    n_clusters=3, affinity='euclidean', linkage='ward')

cluster_4 = AgglomerativeClustering(
    n_clusters=4, affinity='euclidean', linkage='ward')

cluster_5 = AgglomerativeClustering(
    n_clusters=5, affinity='euclidean', linkage='ward')

cluster_6 = AgglomerativeClustering(
    n_clusters=6, affinity='euclidean', linkage='ward')

--------------------------------------------------------------------------------

#Need to find the appropriate number of clusters to use via silhouette score
from sklearn.metrics import silhouette_score
k = [3,4,5,6]
silhouette_scores = []

#View the silhouette scores for each cluster
silhouette_scores.append(
        silhouette_score(baseball_x, cluster_3.fit_predict(baseball_x)))
silhouette_scores.append(
        silhouette_score(baseball_x, cluster_4.fit_predict(baseball_x)))
silhouette_scores.append(
        silhouette_score(baseball_x, cluster_5.fit_predict(baseball_x)))
silhouette_scores.append(
        silhouette_score(baseball_x, cluster_6.fit_predict(baseball_x)))

# Plotting a bar graph to compare the results
plt.bar(k, silhouette_scores)
plt.xlabel('Number of clusters', fontsize = 20)
plt.ylabel('S(i)', fontsize = 20)
plt.show()


--------------------------------------------------------------------------------

#Define how many clusters we should use based on the graph above
cluster_5.fit(baseball_x)
labels_5 = cluster_5.labels_
labels_5


#Need to inverse transform the dataset back for interpretability
baseball5 = (baseball4*baseball3_std)+baseball3_mean
baseball6 = baseball5
labels_5_2 = pd.DataFrame(labels_5)
baseball6.reset_index(inplace = True)
baseball6["labels"] = labels_5_2[0]+1


--------------------------------------------------------------------------------

#Creating visualization from the clustering
#Many more graphs were made, but it was just the same code with different axis. 

#2D scatter
plt.figure(figsize=(50, 30))
fig, ax = plt.subplots()
groups = baseball6.groupby('labels')
for name, group in groups:
    ax.plot(group.runs_per_game, group.OPS, marker='o', linestyle='', ms=5, label=name)
    ax.legend()
    ax.set_xlabel("Runs Per Game")
    ax.set_ylabel("OPS")


#3d scatter
ax.clear()
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(projection='3d')
groups = baseball6.groupby('labels')
for name, group in groups:
    ax.scatter(group.runs_per_game, group.OPS, group.HR)
    #ax.plot(group.runs_per_game, group.HR, marker='o', linestyle='', ms=5, label=name)
    ax.legend()
    ax.set_xlabel("Runs Per Game")
    ax.set_ylabel("OPS")
    ax.set_zlabel("HR")


ax.clear()
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(projection='3d')
groups = baseball6.groupby('labels')
for name, group in groups:
    ax.scatter(group.runs_per_game, group.OPS, group.SB)
    #ax.plot(group.runs_per_game, group.HR, marker='o', linestyle='', ms=5, label=name)
    ax.legend()
    ax.set_xlabel("Runs Per Game")
    ax.set_ylabel("OPS")
    ax.set_zlabel("Stolen Bases")


--------------------------------------------------------------------------------

#Group the observations by their assocaited cluster
#View each clusters summary statistics for more understanding
groups = baseball6.groupby('labels')
summary = groups.describe()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
summary

