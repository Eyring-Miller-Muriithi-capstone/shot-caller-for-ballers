########################
#These functions are for the exploration of Take the Shot or Not capstone project
###########################

#imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
import wrangle
# def univariate(df):
#     """This function creates univariate histograms of all the NBA players variables.
#     Call in by importing this explore.py file, then type: explore.univariate(df)"""
#     df.hist(bins = 30, figsize = (20, 20), color= 'blue')
#     return plot





########## Court Drawing and player Scatter plots ################

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

def scatter_plot_player_shots(df):
    """For scatter plotting a players shots. The player needs to be held in their
    own df for this, or all shots will be charted. 
    Example: (for Steph Curry) df_curry = df.player == 'Stephen Curry' 
    Then the player's df can be inputted to the plot."""
    g=sns.relplot(data=df, kind = 'scatter',
    x = df.loc_x, y= df.loc_y, hue= df.shot_result)

    for i, ax in enumerate(g.axes.flat):
        ax = draw_court(ax, outer_lines=True)
        ax.set_xlim(-300, 300)
        ax.set_ylim(-100, 500)

################# Hypothesis Testing ##########################

def chi_square_test(col1, col2):
    """This function runs a chi-square test on two variables to find any 
    statistical relationship of dependancy. 
    To call this function, input your df.(column_1) and df.(column_2)"""

    alpha = 0.05
    null_hypothesis = "{col1} and {col2} are independent"
    alternative_hypothesis = "there is a relationship between {col1} and {col2}"

# Setup a crosstab of observed survival to pclass
    observed = pd.crosstab(col1, col2)

    chi2, p, degf, expected = stats.chi2_contingency(observed)

    if p < alpha:
        print("Reject the null hypothesis that", null_hypothesis)
        print("Sufficient evidence to move forward understanding that", alternative_hypothesis)
    else:
        print("Fail to reject the null")
        print("Insufficient evidence to reject the null")
    p
    
#############################  EDA   ####################################
import itertools
df, df_outlier_3pt, X_train_exp, X_train, y_train, X_validate, y_validate, X_test, y_test = wrangle.wrangle_prep()
sns.set(rc={'figure.figsize':(17.7,8.27)})

def plot_categorical():
    """this finction creates a bin for time since rest and plots the significant categorical features"""
    X_train_exp['rest_bin'] = pd.cut(X_train_exp.since_rest,[0,250,500,750,1000,2000,3000])
    cat_cols = X_train_exp[['team','shot_type','zone','rest_bin']]
    for i, predictor in enumerate(cat_cols):
        plt.figure(i)
        plot= sns.countplot(data=X_train_exp, x=predictor, hue='shot_made_flag')
        plt.setp(plot.get_xticklabels(), rotation=90) 

    return plot

def team_df():
    """this function creates dataframes for selected teams for eda purposes"""
    dallas = X_train_exp[X_train_exp['team']=="Dallas Mavericks"]
    miami =  X_train_exp[X_train_exp['team']=="Miami Heat"]
    boston = X_train_exp[X_train_exp['team']=="Boston Celtics"]
    golden_state = X_train_exp[X_train_exp['team'] =='Golden State Warriors']
    detroit = X_train_exp[X_train_exp['team'] =='Detroit Pistons']
    orlando = X_train_exp[X_train_exp['team'] =='Orlando Magic']
    oklahoma = X_train_exp[X_train_exp['team'] =='Oklahoma City Thunder']
    san_antonio = X_train_exp[X_train_exp['team'] =='San Antonio Spurs']
    return dallas, miami,boston, golden_state,detroit,orlando,oklahoma, san_antonio

def plot_curry_bros():
    """this function creates dataframes for two tier one players and charts comparisons"""
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

def plot_numerical_made():
    """this function plots numerical features with relationship to the players time since rest"""
    numerics = X_train_exp[['score_margin','cum_3pa','points','tm_v1', 'tm_v2', 'tm_v3']]
    sns.set(rc={'figure.figsize':(17.7,8.27)})

    for i, predictor in enumerate(numerics):
        plt.figure(i)
        plot= sns.lmplot(data=X_train_exp.sample(900), y=predictor, x='since_rest')#, hue = 'shot_made_flag')
        sns.set(rc={'figure.figsize':(17.7,8.27)})
    return plot

def plot_numerical_distance():
    """this function plots numerical features with relationship to the players time since rest"""
    numerics = X_train_exp[['since_rest','score_margin','cum_3pa','points','tm_v1', 'tm_v2', 'tm_v3']]
    sns.set(rc={'figure.figsize':(17.7,8.27)})

    for i, predictor in enumerate(numerics):
        plt.figure(i)
        plot= sns.lmplot(data=X_train_exp.sample(900), y=predictor, x='distance')#, hue = 'shot_made_flag')
        sns.set(rc={'figure.figsize':(17.7,8.27)})
    return plot

def shot_zone_comparison():
    """this function charts and compares zon shots among four of the best teams and 4 regular season teams """
    dallas, miami,boston, golden_state,detroit,orlando,oklahoma, san_antonio = team_df()
        # Shot Comparison Best 4 Teams
    # dallas
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharex=True)
    palette = itertools.cycle(sns.color_palette())
    fig.suptitle('Shot Comparison Best 3 Teams')
    chart1 = sns.countplot(ax=axes[0], x=dallas.zone)
    axes[0].set_title('Dallas')

    chart1.set_xticklabels(chart1.get_xticklabels(), rotation=90)
    c= next(palette)

    chart3 = sns.countplot(ax=axes[1], x=golden_state.zone)
    axes[1].set_title('Golden State')
    chart3.set_xticklabels(chart3.get_xticklabels(), rotation=90)

    #boston
    chart4 = sns.countplot(ax=axes[2], x=boston.zone)
    axes[2].set_title('Boston')
    chart4.set_xticklabels(chart4.get_xticklabels(), rotation=90)

            # Shot Comparison Worst 4 Teams
    # detroit
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharex=True)
    fig.suptitle('Shot Comparison of 3 Regular Teams')
    
    chart5 = sns.countplot(ax=axes[0], x=detroit.zone)
    axes[0].set_title('Detroit')
    chart5.set_xticklabels(chart5.get_xticklabels(), rotation=90)

    #orlando
    chart6 = sns.countplot(ax=axes[1], x=orlando.zone)
    axes[1].set_title('Orlando')
    chart6.set_xticklabels(chart6.get_xticklabels(), rotation=90)


    #san antonio
    chart7 = sns.countplot(ax=axes[2], x=san_antonio.zone)
    axes[2].set_title('San Antonio')
    chart7.set_xticklabels(chart7.get_xticklabels(), rotation=90)
    return plt

############# From Ryan####################

'''
Univariate Distributions, Bivariate Analysis and Hypothesis Testing, Multivariate Analysis.
Alos, any other code that needs to be run in the final notebook.
'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
import scipy.stats as stats
from nba_api.stats.endpoints import leaguegamefinder

def winner_3pct():
    '''
    Emits a dataframe comparing winning teams average 3-point percentage with the losing teams 3-point percentage,
    for each of the past ten years, as well as cumulatively.
    '''
    year_list=['2010-11','2011-12','2012-13','2013-14','2014-15','2015-16','2016-17','2017-18','2018-19','2020-21']
    past_ten_cum = pd.DataFrame()
    past_ten = []
    
    for year in year_list:
        year_holder= {}
        df_teams = leaguegamefinder.LeagueGameFinder(league_id_nullable = '00',
                                            season_nullable = year,
                                            season_type_nullable = 'Regular Season').get_data_frames()
        df_season = df_teams[0]
        past_ten_cum = pd.concat([past_ten_cum,df_season])

        season_winners = df_season[df_season.WL == 'W']
        season_losers = df_season[df_season.WL == 'L']

        year_holder['season'] = year
        year_holder['winner_3pct'] = season_winners.FG3_PCT.mean()
        year_holder['loser_3pct'] = season_losers.FG3_PCT.mean()
        year_holder['difference'] = year_holder['winner_3pct'] - year_holder['loser_3pct']

        past_ten.append(year_holder)

        if year == '2020-21':
            winner_pct_h_test(df_season)
            continue
    
    season_cum_winners = past_ten_cum[past_ten_cum.WL == 'W']
    season_cum_losers = past_ten_cum[past_ten_cum.WL == 'L']
    
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
    # Can use the following to test for equal variances (done, all are equal)
    # var1 = winner.var()  
    # var2 = winner.var() 
    # print(f'Variances Assumption Check: {var1} , {var2}')
    season_winners = df_season[df_season.WL == 'W']
    season_losers = df_season[df_season.WL == 'L']

    winners = season_winners.FG3_PCT
    losers = season_losers.FG3_PCT
    plt.figure(figsize = (12,6))
    plt.title('3-point Percentage, Winning vs. Losing Teams, 2020-21 Regular Season')
    plt.axhline(df_season['FG3_PCT'].mean(), ls='--', color='gray')
    sns.barplot(data = df_season, x = 'WL', y = 'FG3_PCT')
    plt.xlabel('(W)inner or (L)oser')
    plt.ylabel('3-point Field Goal Percentage')
    plt.show()
    t, p = stats.ttest_ind(winners, losers, equal_var=True)
    print(f'p/2 = {p/2} (t = {t})')
    if p/2 > a:
        print('Null hypothesis confirmed - the mean 3-point percentage is the same or lower for winning teams.')
    else:
        print('Null hypothesis rejected - the mean 3-point percentage is the higher for winning teams.')
    t, p / 2

    return



def univariate(df):
    """This function creates univariate histograms of all the NBA players variables.
    Call in by importing this explore.py file, then type: explore.univariate(df)"""
    df.hist(bins = 30, figsize = (20, 20), color= 'blue')






########## Court Drawing and player Scatter plots ################

# def draw_court(ax=None, color='black', lw=2, outer_lines=False):
#     """This function comes from Savvas Tjortjoglou of how to create
#      an NBA sized court to plot our findings/data points on."""
#     # If an axes object isn't provided to plot onto, just get current one
#     if ax is None:
#         ax = plt.gca()

#     ####Creating the various parts of an NBA basketball court:

#     # 1) Create the basketball hoop:
#     # Diameter of a hoop is 18" so it has a radius of 9", which is a value
#     # 7.5 in our coordinate system
#     hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)

#     # 2) Create the backboard (rectangle parameters)
#     backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)

#     # 3) The paint (on the court edges):
#     # Create the outer box 0f the paint, width=16ft, height=19ft
#     outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
#                           fill=False)
#     # Create the inner box of the paint, widt=12ft, height=19ft
#     inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
#                           fill=False)

#     # 4) Create free throw top arc:
#     top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
#                          linewidth=lw, color=color, fill=False)
#     # 5) Create free throw bottom arc:
#     bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,
#                             linewidth=lw, color=color, linestyle='dashed')
#     # 6) Restricted Zone, it is an arc with 4ft radius from center of the hoop
#     restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,
#                      color=color)

#     # 7) Three point line:
#     # Create the side 3pt lines, they are 14ft long before they begin to arc
#     corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,
#                                color=color)
#     corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
#     # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop
#     # I just played around with the theta values until they lined up with the 
#     # threes
#     three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,
#                     color=color)

#     # 8) Create Center Court:
#     center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
#                            linewidth=lw, color=color)
#     center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
#                            linewidth=lw, color=color)

#     # Here is a List of the court elements to be plotted onto the axes:
#     court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
#                       bottom_free_throw, restricted, corner_three_a,
#                       corner_three_b, three_arc, center_outer_arc,
#                       center_inner_arc]

#     if outer_lines:
#         # Draw the half court line, baseline and side out bound lines
#         outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,
#                                 color=color, fill=False)
#         court_elements.append(outer_lines)

#     # Add the court elements onto the axes
#     for element in court_elements:
#         ax.add_patch(element)

#     return ax

# def scatter_plot_player_shots(df):
#     """For scatter plotting a players shots. The player needs to be held in their
#     own df for this, or all shots will be charted. 
#     Example: (for Steph Curry) df_curry = df.player == 'Stephen Curry' 
#     Then the player's df can be inputted to the plot."""
#     g=sns.relplot(data=df, kind = 'scatter',
#     x = df.loc_x, y= df.loc_y, hue= df.shot_result)

#     for i, ax in enumerate(g.axes.flat):
#         ax = draw_court(ax, outer_lines=True)
#         ax.set_xlim(-300, 300)
#         ax.set_ylim(-100, 500)

################# Hypothesis Testing ##########################

def chi_square_test(col1, col2):
    """This function runs a chi-square test on two variables to find any 
    statistical relationship of dependancy. 
    To call this function, input your df.(column_1) and df.(column_2)"""

    alpha = 0.05
    null_hypothesis = "{col1} and {col2} are independent"
    alternative_hypothesis = "there is a relationship between {col1} and {col2}"

# Setup a crosstab of observed survival to pclass
    observed = pd.crosstab(col1, col2)

    chi2, p, degf, expected = stats.chi2_contingency(observed)

    if p < alpha:
        print("Reject the null hypothesis that", null_hypothesis)
        print("Sufficient evidence to move forward understanding that", alternative_hypothesis)
    else:
        print("Fail to reject the null")
        print("Insufficient evidence to reject the null")
    p