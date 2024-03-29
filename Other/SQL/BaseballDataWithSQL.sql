--Data Science Technical Test Section 1
--Zak Massey
--8/12/2022


-- Data preprocessing & cleaning:

--Part 1:
-- Starting off I saved the data as a csv/.txt file so I could import each set as a flat file
-- Used import/export wizard to import the two datasets to SSMS
-- Selected the appropriate data formats (varchar, char, int, num, etc) and imported

--Part 2:
-- How you clean the data should depend on what you plan on doing with the data
-- In this case, there was no statistical modeling required
	-- So there was no reason to address outliers/anomalies, do normality testing, or data transformation. 
	-- If we needed to model, these steps might be required

-- But there were still items that could be addressed
-- Removing duplicates, standardized formatting, addressing NULLs
-- With this data, all of the above could be executed



-- Checking for duplicates:

-- Method 1: Check for same number of rows using select & select distinct
select count(*) from Batting1 -- 110495
select distinct count(*) from Batting1 -- 110495

--Method 2: Create a cte table, segment data by inherently distinct features, count the rows
with cte2 as (
    select playerID, yearID, teamID, stint,
        row_number() over (
            partition by playerID, yearID, teamID, stint
            order by yearID, playerID) row_num
     from Batting1)
select count(*) from cte2
WHERE cte2.row_num < 2; -- 110495

select count(*) from Batting1 -- 110495


--If we were to exclude stint from the query above, we can see that different values returned
-- This means that there were 72 instances where a player played on the same team during a different stint in the same year
with cte2 as (
    select playerID, yearID, teamID,
        row_number() over (
            partition by playerID, yearID, teamID
            order by yearID, playerID) row_num
     from Batting1)
select count(*) from cte2
WHERE cte2.row_num < 2; -- 110423

select count(*) from Batting1 -- 110495

-- We can conclude there are no duplicated values in the Batting Data


-- Standardized Formatting
-- Getting the Dates in a consistent format:
-- Dates in the 1800s were listed as yyyy-dd-mm and dates after 1900 were shown as d/mm/yyyy
-- So they needed to be consistent

--Debut
select t1.playerID, case when newdebut is null then debut else newdebut end as New_debut
	into #temp_table_debut
	from (select * from People1) as t1
	left join (select playerID, debut as 'debut2', concat(substring(t1.month2, patindex('%[^0]%',t1.month2), 10), '/', substring(t1.day2, patindex('%[^0]%',t1.day2), 10), '/', t1.year2) as 'newdebut'
	from (select playerID, debut, substring(debut, 0, charindex('-', debut))year2,substring(debut, 6, 2)month2,substring(debut, 9, charindex('-', debut, 1))day2 from People1) as t1
	where t1.year2 < 1900 and t1.year2 > 0) as t2
	on t1.debut = t2.debut2 and t1.playerID = t2.playerID

--Final Game
select t1.playerID, case when newfinalGame is null then finalGame else newfinalGame end as New_FinalGame
	into #temp_table_finalgame
	from (select * from People1) as t1
	left join (select playerID, finalGame as 'finalGame2', concat(substring(t1.month2, patindex('%[^0]%',t1.month2), 10), '/', substring(t1.day2, patindex('%[^0]%',t1.day2), 10), '/', t1.year2) as 'newfinalGame'
	from (select playerID, finalGame, substring(finalGame, 0, charindex('-', finalGame))year2,substring(finalGame, 6, 2)month2,substring(finalGame, 9, charindex('-', finalGame, 1))day2 from People1) as t1
	where t1.year2 < 1900 and t1.year2 > 0) as t2
	on t1.finalGame = t2.finalGame2 and t1.playerID = t2.playerID


--Add the new columns to the original People1 dataset
alter table dbo.People1 
add NewDebut varchar(50) NULL, NewFinalGame varchar(50) NULL ;

update People1
set People1.NewDebut = #temp_table_debut.New_debut, People1.NewFinalGame = #temp_table_finalgame.New_FinalGame
from People1
inner join #temp_table_debut on People1.playerID = #temp_table_debut.playerID
inner join #temp_table_finalgame on People1.playerID = #temp_table_finalgame.playerID;
drop table #temp_table_debut, #temp_table_finalgame
select * from People1

--Besides this, that was really all the data cleaning that was necessary to answer the questions we were interested in
-- From this point we could turn the dates formatted above into actual date datatypes, but this preprocessing section is getting a bit long already
--Concerning null values in the data - there are many things that could've been done but...
--There was no real use for columns such as intentional bases on balls or times hit by pitches so if they were null it would not matter
-- For this reason I left them as-is



-- Question 1:
--For each year, give the number of MLB players who had a batting average over .300. Only 
--consider players with more than 50 at bats

--Query
select yearID as 'Year', count(*) as 'Players with BA above 0.300'
from (select sum(AB) as sumAB, sum(H) as sumH, yearID, playerID from Batting1 where AB > 0 group by playerID, yearID) as b1
where b1.sumAB > 50 and b1.sumH/b1.sumAB > .300
group by yearID
order by yearID;

--Explanation:
-- Selected the 'year' and defined a 'count' column which would count the returned values given some condition
-- Had to use a subquery to aggregate  At-Bats and Hits at a more granular  level.
-- Summed At-Bats and Hits under the condition that the player had At-Bats recorded and grouped by the players ID and season so we would sum each players At-Bats and Hits per season
-- This step was necessary to account for any players who might�ve had two or more records for a given season and batted above a 0.300 with 50 or more at bats for a given stint.
-- For instance, Douglas Allison (allisdo01) played for two different teams in 1872 where he batted over a 0.300 with over 50 At-Bats with both teams.
-- If this step was left out, he would've been counted twice
-- The following where clause defined the query conditions to be met, having over 50 At-Bats along with a 0.300 Batting Average (Hits/At Bats)
-- Then grouped by the year to keep the seasons independent
-- And ordered by the year for visibility 






-- Question 2:
--For each year since 1930, give the MLB player with the most home runs from each league. 
--Please include their league, team, and batting average. 
	--� If the player played on more than 1 team in a season, give the team information for the 
	--last team he played for
		--o For example, if a player gets traded from the National League to the American 
		--league, his home run total would be considered in the American League.
	--� If there is a tie, include the player with the higher number of RBI�s
	--� If there is still a tie, pick the player with the lower playerID

--Query
with cte as (
select playerID, yearID, lgID, teamID, nameFirst, nameLast, HRs, BattingAverage, RBIS, row_number() over (
            partition by yearID, lgID, HRs
            order by 
				RBIs desc, playerID -- Tie breaker for higher RBI and if tie then use playerID
        ) row_num
			from(select t1.yearID, t1.lgID, t2.playerID, t2.sumHR2 as 'HRs', t3.nameFirst, t3.nameLast, t4.batavg as 'BattingAverage', t4.rbi2 as 'RBIs', t5.teamID
				from(select yearID, lgID, max(sumHR) as 'Most HRs'
					from (select sum(HR) as sumHR, yearID, lgID, playerID from Batting1 where yearID > 1930 group by yearID, lgID, playerID)
							as t0 where yearID > 1930 group by yearID, lgID)
							as t1
						inner join (select sum(HR) as 'sumHR2', yearID, lgID, playerID
									from Batting1
									where AB > 0 and yearID > 1930
									group by yearID, lgID, playerID)
							as t2 
								on t1.yearID=t2.yearID and t1.lgID=t2.lgID and t1.[Most HRs] = t2.sumHR2
						inner join (select nameFirst, nameLast, playerID
									from People1)
							as t3
								on t2.playerID = t3.playerID
						inner join (select yearID, playerID, rbi2, h2/ab2 as 'batavg'
									from(select sum(RBI) as 'rbi2', SUM(AB) as 'ab2', SUM(H) as 'h2', yearID, playerID from Batting1 group by yearID, playerID) as tt where tt.ab2> 0)
							as t4
								on t2.playerID = t4.playerID and t2.yearID = t4.yearID
						inner join (select stint1.playerID, stint1.yearID, final_stint, base1.teamID
									from(select max(stint) as 'final_stint', yearID, playerID from Batting1 group by yearID, playerID) as stint1
									inner join (select playerID, teamID, stint, yearID from Batting1) as base1
									on stint1.final_stint = base1.stint and stint1.yearID = base1.yearID and stint1.playerID = base1.playerID) as t5
							on t5.playerID = t4.playerID and t5.yearID = t4.yearID
								)as temp)
					select yearID, lgID, teamID, nameFirst, nameLast, HRs, BattingAverage from cte
					where cte.row_num < 2;

--Explaination:
-- This one got a little more complex, had to use a common table expression (cte) or a �temp table� to make the joins & subqueries easier & more efficient. 
-- It is a lot cleaner than making many different tables and then pulling from them at the end. It allows you not to clutter up the number of tables in your folder.
-- There were multiple things we had to calculate and include while organizing the output in proper groups. 
-- But this is essentially what was going on:


-- First we had to define the cte and determine what we wanted in the end. All of which would be coming from different tables inside our cte.
-- The �over� and �partition by� clauses in conjunction with the �row_number� command basically just determines how you want to segment the data 
	-- and then provides an index number for each observation you have segmented. This was all done to implement the �tie breaker�. 
-- In our case, we wanted home runs by year and league, so that is how we segmented the data. The next �order by� clause ordered each of those segments based on RBIs and then player ID.
-- Each segment consisted of a year, a league, and a number of home runs and was indexed. 
-- In the instance that there was more than one observation in a given segment, the segment was ordered by RBIs then player ID and the indexes inside that segment were given accordingly. 

-- For instance, in 1931 Lou Gehrig and Babe Ruth both played in the same league (and same team) and both hit 46 home runs.
-- So they were in the same segment, both given an index of 1. But since each segment was ordered by RBIs then player ID, Lou Gehrig was indexed as a 1 and Babe Ruth was indexed as a 2. 
-- This is how the �tie breaker� was implemented in the query.
-- Now on to the actual query.

-- First defined a table which recorded the most home runs hit for each league by year.
-- This was done by summing the home runs and grouping by the year league and player, then returning the maximum for each year and league.
-- Once we had that, we created another table which contained each players total home runs for a year and their league
-- These two tables were joined so we were left with a table that had the most home runs for a player by year in each league
-- Then utilized the People data provided so the players names could be used in the results - joined on players ID and pulled the first and last name
-- Since the batting average was requested to be included & the RBI total was needed for the segmenting/index the next table focused on getting those values and joining
-- They were summed by player ID and year in-case the player moved around the league and Batting average calculation was made when selecting the columns
-- This was then joined with the previous table, so we now included the players total RBIs for the season along with their Batting Average
-- The final table to be joined addressed the stints issue, we want the most recent information for a player, so if they got traded, their new information should be used.
-- To solve this problem one last table was created which found a players "max stint" which was just the largest number representing the final chronological location.
-- There had to be a sub query inside of the sub query on this one which joined the maximum stint and total stints while only keeping the maximum's information for each player
-- Had to test this logic would work & used the query below to be sure - No duplicates with respect to player & year, and the final stint/team is provided

select stint1.playerID, stint1.yearID, final_stint, base1.teamID
from(select max(stint) as 'final_stint', yearID, playerID from Batting1 group by yearID, playerID) as stint1
inner join (select playerID, teamID, stint, yearID from Batting1) as base1
on stint1.final_stint = base1.stint and stint1.yearID = base1.yearID and stint1.playerID = base1.playerID
order by yearID

-- After that everything was sorted out & we had all that was needed
-- Selected the columns we wanted and referenced the cte table we made
-- Then dropped all cte segmented row indexes that were above 1, which applied the tie breaker and we were left with our output







-- Question 3:
--What teams have had the highest slugging percentage? Your output should give the top 10 
--historical team slugging percentages, the year it occurred, and the team name.

select top(10) ((sum(H)-sum(Batting1.[2B])-sum(Batting1.[3B])-sum(Batting1.HR))+2*sum(Batting1.[2B])+3*sum(Batting1.[3B])+4*sum(Batting1.HR))/sum(AB) as 'Slugging Pct',
				 yearID as 'Year',
				 teamID as 'Team' from Batting1
group by teamID, yearID
order by 'Slugging Pct' desc;

--Explaination:
-- Used the "top" function to only return the "top x" values of the related query
-- Then had to define Slugging Percentage since it was not included in the given data. 
-- SLG is a measurement of batting productivity that gives more weight/value to hits which resulted in extra bases
-- It can be calculated using the following formula ((1*Singles + 2*Doubles + 3*Triples + 4*Home Runs)/At-Bats)
-- Two things needed to be done before we could calculate the SLG percentage for each player. 
-- First Needed �singles� since it was not provided in the data and also needed to sum the values on a yearly/team basis

--Calculated singles by taking the total hits and subtracting all extra bases hits.
-- I was interested in whether sacrifice hits (bunts and sac flies) were added to the hits total, if they were � they needed to be deducted as well. 
-- After referencing Wikipedia, MLB.com and the Baseball Reference website, I learned that sacrifice hits are not used in the batting average calculation of H/AB.
-- The rationale being that a player should not be penalized for a successful action. 
-- Since they are not used in the Batting average calculation, I did not use them in the SLG calculation either. (To be sure I checked my top 10 were matching up with the published top SLG by team)

-- Then summed up each variable used in the SLG calculation on a yearly & per team basis
--Did this by including a Group By clause that defined the categories (by team and year)
-- Then ordered by the calculated SLG in descending order to list the top 10 highest percentages

