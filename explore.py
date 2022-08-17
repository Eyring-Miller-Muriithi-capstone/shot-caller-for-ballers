'''
Univariate Distributions, Bivariate Analysis and Hypothesis Testing, Multivariate Analysis.
Also, any other code that needs to be run in the final notebook.
'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
import scipy.stats as stats
from nba_api.stats.endpoints import leaguegamefinder


# ------------------------------------------------------------------------------------------------
# Univariate Analysis Functions
# ------------------------------------------------------------------------------------------------


def univariate(df):
    '''
    This function creates univariate histograms of all the NBA players variables.
    Call in by importing this explore.py file, then type: explore.univariate(df)
    '''
    df.hist(bins = 30, figsize = (20, 20), color= 'blue')


# ------------------------------------------------------------------------------------------------
# Bivariate/Multivariate Graphing and Hypothesis Testing
# ------------------------------------------------------------------------------------------------


def heatmap(X_train_exp):
    '''
    This function returns a heatmap of continuous numerical features.
    '''
    # Identify the continuous features of interest 
    cont_cols = ['since_rest','score_margin','games_played','cum_3pct', 'tm_v2', 'distance','abs_time',
                 'play_time', 'points']

    # Create a correlation matrix from the continuous numerical columns
    df_cont_cols = X_train_exp[cont_cols]
    corr = df_cont_cols.corr()

    # Pass correlation matrix on to sns heatmap
    plt.figure(figsize=(12,12))
    sns.heatmap(corr, annot=True, cmap="flare", mask=np.triu(corr))
    plt.show()


# Categorical--------------------------------------------------------------------------------------

def plot_categorical(X_train_exp):
    '''
    This finction creates a bin for time since rest and plots the significant categorical features.
    '''
    # Setting for plots
    sns.set(rc={'figure.figsize':(17.7,8.27)})

    # Create plots
    cat_cols = X_train_exp[['team','zone']]
    for i, predictor in enumerate(cat_cols):
        plt.figure(i)
        plt.xlabel(i)
        plot= sns.countplot(data=X_train_exp, x=predictor, hue='shot_made_flag')
        plt.setp(plot.get_xticklabels(), rotation=90) 
    plt.show()

    return plot


def home_vs_target(X_train_exp):
    '''
    Creates combo histogram and barplot, along with chi-squared hypothesis test
    '''
    # Set alpha:
    a = .05

    # 'Encoded' target
    target_encoded = 'shot_made_flag'

    # Modify home to no be boolean for plotting
    exp = X_train_exp.copy()
    exp['home'] = np.where(exp.home == True, "Home","Away")

    # Plot
    plt.figure(figsize = (20,10))
    plt.subplot(121)
    plt.title('Home vs Away Shots')
    plt.ylabel('Shots')
    plt.xlabel('Home/Away')
    sns.histplot(data = exp, x = exp['home'], hue = 'shot_result', multiple = 'stack')   
    plt.subplot(122)
    sns.barplot(x = 'home', y = target_encoded, data=exp, alpha=.8, color='lightblue')
    plt.axhline(exp[target_encoded].mean(), ls='--', color='gray')
    plt.title('Home/Away vs 3PT%')
    plt.ylabel('3PT%')
    plt.xlabel('Home/Away')
    plt.show()

    # Hypothesis Test
    observed = pd.crosstab(X_train_exp.home, X_train_exp.shot_result)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print(f'Chi-square = {chi2}')
    if p < a:
        print(f'p = {p}, reject the null hypothesis.')
    else:
        print(f'p = {p}, fails to reject the null hypothesis.') 


def period_vs_target(exp):
    '''
    Creates combo histogram and barplot, along with chi-squared hypothesis test
    '''
    # Set alpha:
    a = .05

    # 'Encoded' target
    target_encoded = 'shot_made_flag'

    # Plot
    plt.figure(figsize = (20,10))
    plt.subplot(121)
    plt.title('Shots by Period')
    plt.ylabel('Shots')
    plt.xlabel('Period')
    sns.histplot(data = exp, x = exp['period'], hue = 'shot_result', multiple = 'stack')   
    plt.subplot(122)
    sns.barplot(x = 'period', y = target_encoded, data=exp, alpha=.8, color='lightblue')
    plt.axhline(exp[target_encoded].mean(), ls='--', color='gray')
    plt.title('Period vs 3PT%')
    plt.ylabel('3PT%')
    plt.xlabel('Period')
    plt.show()

    # Hypothesis Test
    observed = pd.crosstab(exp.home, exp.shot_result)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print(f'Chi-square = {chi2}')
    if p < a:
        print(f'p = {p}, reject the null hypothesis.')
    else:
        print(f'p = {p}, fails to reject the null hypothesis.') 


def zone_vs_target(exp):
    '''
    Creates combo histogram and barplot, along with chi-squared hypothesis test
    '''
    # Set alpha:
    a = .05

    # 'Encoded' target
    target_encoded = 'shot_made_flag'

    # Plot
    plt.figure(figsize = (30,15))
    plt.subplot(121)
    plt.title('Shots by 3PT Zone')
    plt.ylabel('Shots')
    plt.xlabel('3PT Zone')
    sns.histplot(data = exp, x = exp['zone'], hue = 'shot_result', multiple = 'stack')   
    plt.subplot(122)
    sns.barplot(x = 'zone', y = target_encoded, data=exp, alpha=.8, color='lightblue')
    plt.axhline(exp[target_encoded].mean(), ls='--', color='gray')
    plt.title('3PT Zone vs 3PT%')
    plt.ylabel('3PT%')
    plt.xlabel('3PT Zone')
    plt.show()

    # Hypothesis Test
    observed = pd.crosstab(exp.home, exp.shot_result)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print(f'Chi-square = {chi2}')
    if p < a:
        print(f'p = {p}, reject the null hypothesis.')
    else:
        print(f'p = {p}, fails to reject the null hypothesis.') 


# Continuous--------------------------------------------------------------------------------------


def distance_vs_target(exp):
    '''
    Creates combo histogram and boxenplot, along with t-test
    '''
    # Set alpha
    a = .05

    # Create charts - hist of numerical then boxenplot vs target
    plt.figure(figsize=(20,10))   
    plt.subplot(121)
    plt.title('Shots by Distance')
    plt.ylabel('Shots')
    plt.xlabel('Distance (feet)')
    sns.histplot(exp.distance)
    plt.subplot(122)
    sns.boxenplot(y = exp.distance, x = 'shot_result', data=exp)
    plt.axhline(exp.distance.mean(), ls='--', color='gray')
    plt.title('Distance vs Shot Result')
    plt.ylabel('Distance (feet)')
    plt.xlabel('Shot Result')
    plt.show()

    # Hypothesis Test
    distance_made = exp[exp.shot_result == 'Made Shot'].distance
    distance_missed = exp[exp.shot_result == 'Missed Shot'].distance
    t, p = stats.ttest_ind(distance_made, distance_missed, equal_var=True)
    print(f't = {t}')
    if p/2 < a:
        print(f'p = {p}, reject the null hypothesis.')
    else:
        print(f'p = {p}, fails to reject the null hypothesis.')


def tmv2_vs_target(exp):
    '''
    Creates combo histogram and boxenplot, along with t-test
    '''    
    # Set alpha
    a = .05

    # Create charts - hist of numerical then boxenplot vs target
    plt.figure(figsize=(20,10)) 
        
    plt.subplot(121)
    plt.title('Jem-Metric Shots')   
    plt.ylabel('Shots')
    plt.xlabel('Jem-Metric')
    sns.histplot(exp.tm_v2)
    plt.subplot(122)
    sns.boxenplot(y = exp.tm_v2, x = 'shot_result', data=exp)
    plt.axhline(exp.tm_v2.mean(), ls='--', color='gray')
    plt.title('Jem-Metric vs Shot Result')
    plt.ylabel('Jem-Metric')
    plt.xlabel('Shot Result')
    plt.show()

    # Hypothesis Test
    tmv2made = exp[exp.shot_result == 'Made Shot'].tm_v2
    tmv2missed = exp[exp.shot_result == 'Missed Shot'].tm_v2
    t, p = stats.ttest_ind(tmv2made, tmv2missed, equal_var=True)
    print(f't = {t}')
    if p/2 < a:
        print(f'p = {p}, reject the null hypothesis.')
    else:
        print(f'p = {p}, fail to reject the null hypothesis.')


def play_time_vs_target(exp):
    '''
    Creates combo histogram and boxenplot, along with t-test
    '''    
    # Set alpha
    a = .05

    # Create charts - hist of numerical then boxenplot vs target
    plt.figure(figsize=(20,10))     
    plt.subplot(121)
    plt.title('Shot by Play Time')
    plt.ylabel('Shots')
    plt.xlabel('Play Time (seconds)')
    sns.histplot(exp.play_time)
    plt.subplot(122)
    sns.boxenplot(y = exp.play_time, x = 'shot_result', data=exp)
    plt.axhline(exp.play_time.mean(), ls='--', color='gray')
    plt.title('Play Time vs Shot Result')   
    plt.ylabel('Play Time (seconds)')
    plt.xlabel('Shot Result')
    plt.show()

    # Hypothesis Test
    ptmade = exp[exp.shot_result == 'Made Shot'].tm_v2
    ptmissed = exp[exp.shot_result == 'Missed Shot'].tm_v2
    t, p = stats.ttest_ind(ptmade, ptmissed, equal_var=True)
    print(f't = {t}')
    if p/2 < a:
        print(f'p = {p}, reject the null hypothesis.')
    else:
        print(f'p = {p}, fail to reject the null hypothesis.')


def rest_vs_target(exp):
    '''
    Creates combo histogram and boxenplot, along with t-test
    '''    
    # Set alpha
    a = .05

    # Create charts - hist of numerical then boxenplot vs target
    plt.figure(figsize=(20,10))    
    plt.subplot(121)
    plt.title('Shot by Time Since Rest')
    plt.ylabel('Shots')
    plt.xlabel('Time Since Rest (seconds)')
    sns.histplot(exp.since_rest)
    plt.subplot(122)
    sns.boxenplot(y = exp.since_rest, x = 'shot_result', data=exp)
    plt.axhline(exp.since_rest.mean(), ls='--', color='gray')
    plt.title('Time Since Rest vs Shot Result')  
    plt.ylabel('Time Since Rest (seconds)')
    plt.xlabel('Shot Result')
    plt.show()

    # Hypothesis Test
    restmade = exp[exp.shot_result == 'Made Shot'].since_rest
    restmissed = exp[exp.shot_result == 'Missed Shot'].since_rest
    t, p = stats.ttest_ind(restmade, restmissed, equal_var=True)
    print(f't = {t}')
    if p/2 < a:
        print(f'p = {p}, reject the null hypothesis.')
    else:
        print(f'p = {p}, fail to reject the null hypothesis.')


def score_margin_vs_target(exp):
    '''
    Creates combo histogram and boxenplot, along with t-test
    '''    
    # Set alpha
    a = .05

    # Create charts - hist of numerical then boxenplot vs target
    plt.figure(figsize=(24,12))
    plt.subplot(121)
    plt.title('Shots by Score Margin')
    plt.ylabel('Shots')
    plt.xlabel('Score Margin')
    sns.histplot(exp.score_margin)
    plt.subplot(122)
    sns.boxenplot(y = exp.score_margin, x = 'shot_result', data=exp)
    plt.axhline(exp.score_margin.mean(), ls='--', color='gray')
    plt.title('Score Margin vs Shot Result')   
    plt.ylabel('Score Margin')
    plt.xlabel('Shot Result')
    plt.show()

    # Hypothesis Test
    scoremarginmade = exp[exp.shot_result == 'Made Shot'].score_margin
    scoremarginmissed = exp[exp.shot_result == 'Missed Shot'].score_margin
    t, p = stats.ttest_ind(scoremarginmade, scoremarginmissed, equal_var=True)
    print(f't = {t}')
    if p/2 < a:
        print(f'p = {p}, reject the null hypothesis.')
    else:
        print(f'p = {p}, fail to reject the null hypothesis.')


def games_played_vs_target(exp):
    '''
    Creates combo histogram and boxenplot, along with t-test
    '''    
    # Set alpha
    a = .05

    # Create charts - hist of numerical then boxenplot vs target
    plt.figure(figsize=(24,12))       
    plt.subplot(121)
    plt.title('Shots by Games Played')
    plt.ylabel('Shots')
    plt.xlabel('Games Played')
    sns.histplot(exp.games_played)
    plt.subplot(122)
    sns.boxenplot(y = exp.games_played, x = 'shot_result', data=exp)
    plt.axhline(exp.games_played.mean(), ls='--', color='gray')
    plt.title('Games Played vs Shot Result') 
    plt.ylabel('Games Played')
    plt.xlabel('Shot Result')
    plt.show()

    # Hypothesis Test
    gpmade = exp[exp.shot_result == 'Made Shot'].games_played
    gpmissed = exp[exp.shot_result == 'Missed Shot'].games_played
    t, p = stats.ttest_ind(gpmade, gpmissed, equal_var=True)
    print(f't = {t}')
    if p/2 < a:
        print(f'p = {p}, reject the null hypothesis.')
    else:
        print(f'p = {p}, fail to reject the null hypothesis.')
  

# ------------------------------------------------------------------------------------------------
# Additional Helper Functions
# ------------------------------------------------------------------------------------------------


def find_elites(df):
    '''
    Using our Jemetric, aka tm_v2, we determine the elite players as those whose Jemetric is 2 stadard deviations
    above the league average.
    '''
    # Create a Series of v2 scores, binned by player
    tm_v2_scores = df.groupby('player').tm_v2.mean()
    # Calculate the std and mean
    stddev = tm_v2_scores.std()
    meanscore = tm_v2_scores.mean()
    # Create an elite cutoff score at two standard deviations above the mean
    elites = meanscore + 2 * stddev
    # Print the list of 'elite' players
    elites_list = tm_v2_scores[tm_v2_scores > elites].index

    return elites_list


def jem_graph(df, player_list):
    '''
    Takes a list of players and returns their season Jemetric
    '''
    # Creates graphs for inputted players
    plt.figure(figsize = (15,5))
    for player in player_list:
        df_p = df[df['player'] == player]
        sns.lineplot(data = df_p, x = 'games_played',y = 'tm_v2')
    plt.title('Jem-Metric Score over Season')
    plt.xlabel('Games Played')
    plt.ylabel('Jem-Metric Score')
    plt.legend(player_list)
    plt.show()

    return 


# ------------------------------------------------------------------------------------------------
# Functions showing the importance of 3pt shots
# ------------------------------------------------------------------------------------------------


def winner_3pct():
    '''
    Emits a dataframe comparing winning teams average 3-point percentage with the losing teams 3-point percentage,
    for each of the past ten years, as well as cumulatively.
    '''
    # Select years to use
    year_list=['2010-11','2011-12','2012-13','2013-14','2014-15','2015-16','2016-17','2017-18','2018-19','2020-21']
    
    # Initialize some container objects
    past_ten_cum = pd.DataFrame()
    past_ten = []
    
    # Pull all games from the years selected above, year by year
    for year in year_list:
        year_holder= {}
        df_teams = leaguegamefinder.LeagueGameFinder(league_id_nullable = '00',
                                            season_nullable = year,
                                            season_type_nullable = 'Regular Season').get_data_frames()
        df_season = df_teams[0]
        df_season['WL'] = np.where(df_season.WL == 'W', 'Win', 'Loss')
        past_ten_cum = pd.concat([past_ten_cum,df_season])

        

        # Seperate between winners and losers
        season_winners = df_season[df_season.WL == 'Win']
        season_losers = df_season[df_season.WL == 'Loss']

        # Create df
        year_holder['season'] = year
        year_holder['winner_3pct'] = season_winners.FG3_PCT.mean()
        year_holder['loser_3pct'] = season_losers.FG3_PCT.mean()
        year_holder['difference'] = year_holder['winner_3pct'] - year_holder['loser_3pct']

        past_ten.append(year_holder)

        if year == '2020-21':
            winner_pct_h_test(df_season)
            continue
    
    # Create aggregate column of last ten years
    season_cum_winners = past_ten_cum[past_ten_cum.WL == 'Win']
    season_cum_losers = past_ten_cum[past_ten_cum.WL == 'Loss']
    
    # Add cumulative to df
    past_ten.append({'season':'cumulative',
                    'winner_3pct':season_cum_winners.FG3_PCT.mean(),
                    'loser_3pct':season_cum_losers.FG3_PCT.mean(),
                    'difference':(season_cum_winners.FG3_PCT.mean()-season_cum_losers.FG3_PCT.mean())})
    
    past_ten = pd.DataFrame(past_ten).set_index('season')

    past_ten = round(past_ten.T,3)

    return past_ten


def winner_pct_h_test(df_season):
    '''
    This function performs a hypothesis test on the winning team and losing team series from a given year,
    in this case 2020-21.  It also shows a fun chart!
    '''
    a = .05

    # Seperate winning and losing teams
    season_winners = df_season[df_season.WL == 'Win']
    season_losers = df_season[df_season.WL == 'Loss']

    # Get three percentage for each type
    winners = season_winners.FG3_PCT
    losers = season_losers.FG3_PCT

    # Plot barplot with mean
    plt.figure(figsize = (12,6))
    plt.title('3-point Percentage, Winning vs. Losing Teams, 2020-21 Regular Season')
    plt.axhline(df_season['FG3_PCT'].mean(), ls='--', color='gray')
    sns.barplot(data = df_season, x = 'WL', y = 'FG3_PCT')
    plt.xlabel('Game Result')
    plt.ylabel('3-point Field Goal Percentage')
    plt.show()

    # Hypothesis test
    t, p = stats.ttest_ind(winners, losers, equal_var=True)
    print(f'p/2 = {p/2} (t = {t})')
    if p/2 > a:
        print('Null hypothesis confirmed - the mean 3-point percentage is the same or lower for winning teams.')
    else:
        print('Null hypothesis rejected - the mean 3-point percentage is the higher for winning teams.')
    t, p / 2

    return


# ------------------------------------------------------------------------------------------------
# Shot-Court visualization functions
# ------------------------------------------------------------------------------------------------


def draw_court(ax=None, color='black', lw=2, outer_lines=False):
    """This function comes from Savvas Tjortjoglou of how to create
     an NBA sized court to plot our findings/data points on."""
    # If an axes object isn't provided to plot onto, just get current one
    if ax is None:
        ax = plt.gca()

    ####Creating the various parts of an NBA basketball court:

    # 1) Create the basketball hoop:
    # Diameter of a hoop is 18" so it has a radius of 9", which is a value
    # 7.5 in our coordinate system
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)

    # 2) Create the backboard (rectangle parameters)
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)

    # 3) The paint (on the court edges):
    # Create the outer box 0f the paint, width=16ft, height=19ft
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
                          fill=False)
    # Create the inner box of the paint, widt=12ft, height=19ft
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
                          fill=False)

    # 4) Create free throw top arc:
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
                         linewidth=lw, color=color, fill=False)
    # 5) Create free throw bottom arc:
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle='dashed')
    # 6) Restricted Zone, it is an arc with 4ft radius from center of the hoop
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,
                     color=color)

    # 7) Three point line:
    # Create the side 3pt lines, they are 14ft long before they begin to arc
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,
                               color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop
    # I just played around with the theta values until they lined up with the 
    # threes
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,
                    color=color)

    # 8) Create Center Court:
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
                           linewidth=lw, color=color)

    # Here is a List of the court elements to be plotted onto the axes:
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc,
                      center_inner_arc]

    if outer_lines:
        # Draw the half court line, baseline and side out bound lines
        outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,
                                color=color, fill=False)
        court_elements.append(outer_lines)

    # Add the court elements onto the axes
    for element in court_elements:
        ax.add_patch(element)

    return ax


def scatter_plot_player_shots(df_player):
    '''
    For scatter plotting a players shots. The player needs to be held in their
    own df for this, or all shots will be charted. 
    Example: (for Steph Curry) df_player = df[df.player == 'Stephen Curry'] 
    Then the player's df can be inputted to the plot.
    '''
    # Create basic scatterplot
    g=sns.relplot(data=df_player, kind = 'scatter',
    x = df_player.loc_x, y= df_player.loc_y, hue= df_player.shot_result)

    # Place scatterplot on court model
    for i, ax in enumerate(g.axes.flat):
        ax = draw_court(ax, outer_lines=True)
        ax.set_xlim(-300, 300)
        ax.set_ylim(-100, 500)


# ------------------------------------------------------------------------------------------------
# Fun Analysis
# ------------------------------------------------------------------------------------------------


def plot_curry_bros(X_train_exp):
    '''
    This function creates dataframes for two tier one players and charts comparisons
    '''
    # player dfs
    steph_curry = X_train_exp[X_train_exp['player']=="Stephen Curry"]
    seth_curry = X_train_exp[X_train_exp['player']== "Seth Curry"]

    # steph_curry shots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True)
    fig.suptitle('Steph Shots')
    chart_steph = sns.countplot(ax = axes[0], data = steph_curry, x=steph_curry.zone, hue =steph_curry.shot_made_flag )
    axes[0].set_title('Steph')
    chart_steph.set_xticklabels(chart_steph.get_xticklabels(), rotation=90)
    # chart_steph = sns.countplot(ax=axes[1], x=miami.zone)

    # seth curry shots
    fig.suptitle("Curry Brothers Shots")
    seth_curry = sns.countplot(data = seth_curry, x=seth_curry.zone, hue =seth_curry.shot_made_flag)
    axes[1].set_title('Seth')
    seth_curry.set_xticklabels(seth_curry.get_xticklabels(), rotation=90)