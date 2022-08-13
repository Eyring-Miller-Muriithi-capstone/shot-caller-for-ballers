## Data Science Capstone--Take The Shot or Not.
<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

### Project Summary 

In this Capstone project, our team has gathered data from the NBA api with a goal to predict whether or not a player should take a shot. The goal is to create a machine learning model that will allow players the opportunity to take a shot based on previous activities leading to the chance. 

You can view our team's presentation deck here: <a href="https://www.canva.com/design/DAFJC32y92U/b6rqmt93S_OJEJWzffA3Wg/edit?utm_content=DAFJC32y92U&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton">Linux: </a>
<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

### Initial Questions

> - <br> Can a model predict when a player should choose to make a 3 point field goal or not? <br>
> - <br> Is home court advantage really an advantage?
> - <br> Do players have a favorite shooting position 
> - <br> Do players need rest in order for them to perform better 


#### Project Objectives
> - avors: Visualizing players 3 point shooting locations <br>
> - avors: Creating tiers for the league's 3 point shooters <br>
> - avors: Predicting whether a player should take the shot or not <br>

#### Data Dictionary
>
>
player	player_id	team	team_id	game_id	home	period	abs_time	play_time	since_rest	loc_x	loc_y	zone	shot_type	score_margin	points	fg_pct	shot_result	games_played	game_3pa	game_3pm	game_3miss	cum_3pa	cum_3pm	cum_3miss	cum_3pct	tm_v1	tm_v2	tm_v3	distance	game_event_id
>
|Feature|Datatype|Definition|
|:-------     |:--------        |:---------|:
|player       |:  84228 non-null|: object  |: player's official name 
|player_id    |:  84228 non-null|: int64   |: player's unique id 
|team         |:  84228 non-null|: object  |: team name
|team_id      |:  84228 non-null|: int64   |: team's unique id 
|game_id      |:  84228 non-null|: int64   |: game's unique id 
|home         |:  84228 non-null|: bool    |: home games identified by 1
|period       |:  84228 non-null|: int64   |: 1st 2nd 3rd and 4th periods, data on overtime also included
|abs_time     |:  84228 non-null|: int64   |: engineered: how much a player is on the court based on their rotational data 
|play_time    |:  84228 non-null|: float64 |: engineered: how much a player is on the court based on rotational data 
|since_rest   |:  84228 non-null|: float64 |: player's time on the court since the last time they rested 
|loc_x        |:  84228 non-null|: int64   |: location of three point shot on the court
|loc_y        |:  84228 non-null|: int64   |: location of three point shot on the court
|zone         |:  84228 non-null|: object  |: engineered: shooting clusters engineered with KMeans 
|shot_type    |:  84228 non-null|: object  |: the type of shot based on the player's shooting mechanics 
|score_margin |:  84228 non-null|: int64   |: the difference in scores of the winning vs losing team 
|points       |:  84228 non-null|: int64   |: points per player per game 
|shot_result  |:  84228 non-null|: object  |: boolean shot missed/ made 
|games_played |:  84228 non-null|: int64   |: number of games played in the season 
|game_3pa     |:  84228 non-null|: int64   |: 3point shots attempted per game 
|game_3pm     |:  84228 non-null|: int64   |: 3point shots made per game 
|game_3miss   |:  84228 non-null|: int64   |: 3point shots missed in a game
|cum_3pa      |:  84228 non-null|: int64   |: cumulative 3 point shots attempts in a season
|cum_3pm      |:  84228 non-null|: int64   |: cumulative 3 point shots made in a season
|cum_3miss    |:  84228 non-null|: int64   |: cumulative 3 point shots missed in a season
|cum_3pct     |:  84228 non-null|: float64 |: cumulative 3 point shot percent for the season 
|tm_v1        |:  84228 non-null|: float64 |: engineered tiers for shooters
|tm_v2        |:  84228 non-null|: float64 |: engineered tiers for shooters, most accurate
|tm_v3        |:  84228 non-null|: float64 |: engineered tiers for shooters
|distance     |:  84228 non-null|: float64 |: distance of shots from the rim 
|game_event_id|:  84228 non-null|: int64   |: chronological game events 

##### Plan|:|: 
> <br>We plan to use the NBA api to acquire individual players game statistics for the model.<br>
> <br>Clean the data<br>
> <br>Visualize the data through the EDA process <br>
> <br>Conduct a number of hypothesis tests <br>
> <br>Conduct an ensemble of classification models  <br>
> <br>Draw conclusions <br>




#### Initial Hypotheses
> - **Hypothesis 1 -**
- H_0: Is there a statistical relationship on made shots between home and away games 
> - **Hypothesis 2 -** 
- H_0: Does the period/quarter have a statistical relationship with made shots 
> - **Hypothesis 3 -**
- H_0: Is there a relationship between the zones and how accurate a player is 
> - **Hypothesis 4 -**
- H_0: Is there a relationship between distance the shot is taken and the shot accuracy
> - **Hypothesis 5 -**
- H_0: Does rest affect a players shooting ability
> - **Hypothesis 6 -**
- H_0: Is the engineered feature tm_v2 accurately showing which players are better at 3 point shooting 
> - **Hypothesis 7 -**
- H_0: Is there a relationship between score margin and whether or not a shot is made

### Executive Summary - Conclusions & Next Steps



<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>


### Reproduce Our Project

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

In order to reproduce this project, you will need all the necessary files listed below to run the final project notebook. 
- [x] Read this README.md
- [ ] use the acquire.py to run the tome_prep function 
- [ ] use the wrangle.py file to run the wrangle_prep function to get the cleaned, split, encoded and scaled data
- [ ] use the explore.py to do the univariate and bivariate visualization
