## Data Science Capstone--Take The Shot or Not.
<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

### Project Summary 

In this Capstone project, our team has gathered data from the NBA api with a goal to predict whether or not a player should take a shot. The goal is to create a machine learning model that will allow players the opportunity to take a shot based on previous activities leading to the chance. 

You can view our team's presentation deck here: <a href="https://www.canva.com/design/DAFJC32y92U/b6rqmt93S_OJEJWzffA3Wg/edit?utm_content=DAFJC32y92U&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton">Take the shot or not presentation</a>
<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

### Initial Questions

> - Can a model predict when a player should choose to make a 3 point field goal or not? <br>
> - Is home court advantage really an advantage?
> - Do players have a favorite shooting position 
> - Do players need rest in order for them to perform better 


#### Project Objectives
> - avors: Visualizing players 3 point shooting locations <br>
> - avors: Creating tiers for the league's 3 point shooters <br>
> - avors: Predicting whether a player should take the shot or not <br>

#### Data Dictionary
|Feature|Datatype|Definition|
|:-------|:--------|:----------|
|player       | 84228 non-null: object  | player's official name |
|player_id    | 84228 non-null: int64   | player's unique id | 
|team         | 84228 non-null: object  | team name |
|team_id      | 84228 non-null: int64   | team's unique id |
|game_id      | 84228 non-null: int64   | game's unique id |
|home         | 84228 non-null: bool    | home games identified by 1 |
|period       | 84228 non-null: int64   | 1st 2nd 3rd and 4th periods, data on overtime also included |
|abs_time     | 84228 non-null: int64   | engineered: how much a player is on the court based on their rotational data | 
|play_time    | 84228 non-null: float64 | engineered: how much a player is on the court based on rotational data |
|since_rest   | 84228 non-null: float64 | player's time on the court since the last time they rested |
|loc_x        | 84228 non-null: int64   | location of three point shot on the court |
|loc_y        | 84228 non-null: int64   | location of three point shot on the court |
|zone         | 84228 non-null: object  | engineered: shooting clusters engineered with KMeans | 
|shot_type    | 84228 non-null: object  | the type of shot based on the player's shooting mechanics | 
|score_margin | 84228 non-null: int64   | the difference in scores of the winning vs losing team |
|points       | 84228 non-null: int64   | points per player per game | 
|shot_result  | 84228 non-null: object  | boolean shot missed/ made |
|games_played | 84228 non-null: int64   | number of games played in the season |
|game_3pa     | 84228 non-null: int64   | 3point shots attempted per game |
|game_3pm     | 84228 non-null: int64   | 3point shots made per game |
|game_3miss   | 84228 non-null: int64   | 3point shots missed in a game |
|cum_3pa      | 84228 non-null: int64   | cumulative 3 point shots attempts in a season |
|cum_3pm      | 84228 non-null: int64   | cumulative 3 point shots made in a season |
|cum_3miss    | 84228 non-null: int64   | cumulative 3 point shots missed in a season |
|cum_3pct     | 84228 non-null: float64 | cumulative 3 point shot percent for the season |
|tm_v1        | 84228 non-null: float64 | engineered tiers for shooters |
|tm_v2        | 84228 non-null: float64 | engineered tiers for shooters, most accurate |
|tm_v3        | 84228 non-null: float64 | engineered tiers for shooters |
|distance     | 84228 non-null: float64 | distance of shots from the rim |
|game_event_id| 84228 non-null: int64   | chronological game events |

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

> - Actions:  Using official stats from the [NBA.com/stats](https://www.nba.com/stats/) database, with endpoints accessed through a third party [API client](https://github.com/swar/nba_api), we collected data on shots taken in the 2021-2022 season and filtered for three point shots.  This data includes a number of in-game and player specific features, including both raw data and engineered features.  After an in-depth analysis, we split the dataset into training, validation and unseen test sets and applied a number of classification models varying hyperparameters.  We also created a informational dashboard for players and coaches to use.

> - Conclusions:  We discovered that ‘total game playing time for a player, score margin, points, games played, distance of shot, shooting zone, if it’s the fourth period, and our unique player metric are all significant factors of making a three point shot (as measured at the time of shot).  Unfortunately, even with extensive analysis, for a general league-wide model we were only able to beat baseline by 0.9% (0.6% absolute improvement), and thus we feel it is inadequate to recommend at this time.  However, when modeling by individual player, we typically get better results.

> -  Recommendations:  There is a lot of additional data found in the stats that we feel have a strong impact on three point success, such as distance from defender, how many dribbles before the shot, and time remaining on the shot clock.  However they are filters, not returned data, so a more rigorous (and computationally intensive) acquisition can be performed to get these features.  Also, to improve player-specific models, we can use multiple years of data, including playoffs and even other leagues contained in the database (G-League, WNBA).

## Project Steps:
<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

### Acquire

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>
> - Our analysis and model are focused on three pointers taken in the most recent (2021-2022) regular season.  Out goal was to acquire a dataframe that had every three point shot taken by eveyr player who took one, along with the result (Made or Missed) and a number of features calculated for the player and game right up to the moment the shot was taken.  The [Acquire Workbook](*/acquire_workbook.ipynb) associated with this report contains the step by step process of building our initial, raw dataset.  In brief:
    1. Get team-player data from 'teamplayerdashboard' endpoint and use it to acquire all shots taken in the '21-'22 Regular Season using shot information from the 'shotchartdetail' endpoint.
    2. Isolate 3-pt shots from this data and remove outliers. We used shot distance, acquired from 'shotchartdetail', as the metric to determine outliers.  Outliers were shots taken from a distance 1.5 times the IQR of all three point shots.  This eliminated all shots taken from 30 feet and beyond (2188 shots, or 2.5% of all three pointers).
    3. Apply a KMeans clustering algorithm on all remaining three point attempts to identify the natural 'zones' from which players are shooting.  Using a k-elbow test we determined seven (7) clusteres were optimal, improving upon the NBA standard five (5) shot zones.
    
 
### Prepare
 
<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

> - For additional pre-processing/cleaning, we perform the following actions detailed in the [Wrangle Workbook](*/wrangle_workbook.ipynb):
    1. Create cumulative three point stats for all players across all games.
    2. Add location back in using Pythagorean Theorem on loc_x and loc_y (was lost in the acqusition process; this is also an improvement as it is a float and does not bin by integer foot).
    3. Add back in 'game_event_id', as it is needed for the Tableau Dashboards.
    4. Encode categoricals, scale numericals, split into train/validate/test, and finally seperate the X_ (features) from the y_ (target).
    5. Return the following dataframes:
    - df: Complete initial dataframe with pre-processing perfomed, used for Univariate Distribution Analysis
    - df_outlier: A dataframe containing the outlier three point shots, in case needed in future
    - X_train_exp: A dataframe for exploring the train data split (uneconded and unscaled)
    - X_train, X_validate, X_test: Scaled and encoded features for train/validate/test, used for bi/multivariate EDA and modeling
    - y_train, y_validate, y_test: Target (actual outcomes), used for bi/multivariate EDA and modeling
    
    
### EDA

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>
> - From the visualization exploration combined with statistical testing these are the features that were significant for modeling
     - Location on court (zone and distance)
     - Playtime (how long a player has been on court)
     - Score margin
     - Number of games played
     - Calculated player three point shooting ability (Jem-metric)

### Modeling 

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

After isolating the scaled and encoded features into our modeling feature (X) dataframes, we built a function to train a total of 125 model-hyperparameter combinations, including ensemble models, on our data.  The function applies the model to the validate dataset to check for overfitting of the training model. The function and associated modeling details are found in the [Modeling Workbook](*/modeling_workbook.ipynb).  

Base models include:
- K-Nearest Neighbors (k = 1 to 25)
- Decision Tree (max_depth = 1 to 25)
- Logistical Regression
- Random Forest (leaf = 1 to 3; depth = 2 to 5)
- Extra Trees (leaf = 1 to 3; depth = 2 to 5; n_estimators = 100, 150, 200, 250)
- Ensemble models of the above.

#### Model results
Taking our key features, we used a Decision Tree Classification Model that returned a 0.7% increase (overall 1.1% over baseline) in predicting whether a shot would be made or missed from each zone on the court. 
Individual player models seem more promising, but need more data to run modeling on.

#### Individual player 3pt assessment 

We created a new metric, Jem-metric of the Top 3pt shooters scale could give the NBA a better estimation of which players have the best 3pt shot efficiency from game-to-game.

### Takeaways
<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

    - For a general, league-wide three point shooting model, using the drivers above as key features, we exceeded the predictive power of a baseline model by 0.6 percentage points (0.9% improvement over base).
    - Applying modeling to individual players, the best models vary as well as the results, although almost all beat baseline on validation data (including one by 8.8%).  Unfornately, running the model on the test data of a single individual example, we actually performed more poorly than the league-wide.  However this can be due to a signifigant smaller data size whcih can be improved upon by brining in additional years of data on the player.



### Reproduce Our Project

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

In order to reproduce this project, you will need all the necessary files listed below to run the final project notebook. 
- [x] Read this README.md
- [ ] Use the acquire.py to run the tome_prep function 
- [ ] Use the wrangle.py file to run the wrangle_prep function to get the cleaned, split, encoded and scaled data
- [ ] Use the explore.py to do the univariate, bivariate analysis  
- [ ] Continue with your analysis and draw new conclusions 