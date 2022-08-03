import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns

from nba_api.stats.static import teams
from nba_api.stats.endpoints import gamerotation
from nba_api.stats.endpoints import shotchartdetail
from nba_api.stats.endpoints import teamplayerdashboard
from nba_api.stats.endpoints import winprobabilitypbp

from sklearn.cluster import KMeans

# ------------------------------------------------------------------------------------------------
# CREATING THE DATAFRAME WITH ALL SHOTS FOR THE 2021-2022 SEASON 
# ------------------------------------------------------------------------------------------------

def get_team_player_ids():
    '''
    1
    Acquires a dataframe from cached .csv, or NBA API calls (and caches it if it doesn't exist),
    then returns the data as as dataframe with the columns 'team_id' and 'player_id'. 
    '''
    filename = 'team_player_ids.csv'
    if os.path.isfile(filename):
        team_player_ids =  pd.read_csv(filename, index_col=0)
        players_list = team_player_ids.values.tolist()
    else:
        df_teams = pd.DataFrame(teams.get_teams())
        team_id_list = list(df_teams.id)
        players_list = []
        for team in team_id_list:
            df_tpd = teamplayerdashboard.TeamPlayerDashboard(team,
                                                            season = '2021-22').get_data_frames()
            player_list = list(df_tpd[1].PLAYER_ID)
            for player in player_list:
                row = [team,player]
                players_list.append(row)
        team_player_ids = pd.DataFrame(players_list, columns = ['team_id','player_id'])
        team_player_ids.to_csv('team_player_ids.csv')
        players_list = team_player_ids.values.tolist()
    
    return players_list

def all_21_22_shots():
    '''
    2
    Acquires a dataframe of all shot attempts made in the 2021-2022 season by all players.
    '''
    players_list = get_team_player_ids()
    filename = 'all_last_season_shots.csv'
    if os.path.isfile(filename):
        df_shots =  pd.read_csv(filename, index_col=0)
    else:
        df_shots = pd.DataFrame()
        index = 0
        for player in players_list:
            print(f'\rFetching index {index} of 714', end='')
            df_pl = shotchartdetail.ShotChartDetail(team_id = player[0],
                                                            player_id = player[1],
                                                            season_type_all_star='Regular Season',
                                                            season_nullable='2021-22',
                                                            context_measure_simple = 'FG3A').get_data_frames()
            time.sleep(.5)
            index += 1
            df_shots = pd.concat([df_shots, df_pl[0]])
        df_shots.to_csv(filename)
    # Add abs time
    df_shots = get_absolute_time_shots(df_shots)
    # Reset Index
    df_shots = df_shots.reset_index(drop = True)
    
    return df_shots

def get_absolute_time_shots(df):
    '''
    3
    Takes in a dataframe based on 'ShotChartDetail' NBA-API endpoint and adds a column with the absolute game time, in seconds.
    '''
    # Covers overtime edge cases by having seperate function if game is in overtime (periods > 4)
    df['abs_time'] = np.where(df.PERIOD < 5,
                                (df.PERIOD - 1) * 720 + (720 - (60 * df.MINUTES_REMAINING) - (df.SECONDS_REMAINING)),
                                2880 + ((df.PERIOD - 5) * 300) + (300 - (60 * df.MINUTES_REMAINING) - (df.SECONDS_REMAINING)))
    return df

def create_3pt_shot_zones(df_shots):
    '''
    Removes 3pt outliers and creates a 3pt shot location feature through a KMeans Clustering algorithm
    '''
    # Create a dataframe that is 3 pointers only
    df_all_3pt = df_shots[df_shots.SHOT_TYPE == '3PT Field Goal']

    # Remove 3pt outliers using 1.5 x IQR or all 3pts attempted
    # Since minimum distance is bounded by the three point line, only using a high bound
    low = df_all_3pt.SHOT_DISTANCE.quantile(.25)
    high = df_all_3pt.SHOT_DISTANCE.quantile(.75)
    add = (high-low) * 1.5
    bound = high + add
    # Create outlier dataframe for possible future analysis
    df_outlier_3pt = df_all_3pt[df_all_3pt.SHOT_DISTANCE > bound]
    # df with outliers removed
    df_shots = df_shots[df_shots.SHOT_DISTANCE <= bound]

    # Create the clusters - ***Create sub functions for all these components
    df_3pt = df_all_3pt[df_all_3pt.SHOT_DISTANCE <= bound]
    X = df_3pt[['LOC_X','LOC_Y']]
    kmeans = KMeans(n_clusters=7)
    kmeans.fit(X)
    clusters = kmeans.predict(X)
    df_3pt['location'] = clusters
    plt.figure(figsize = (12,12))
    sns.scatterplot(data =df_3pt, x='LOC_X', y = 'LOC_Y', hue = 'location')
    plt.show()

    df_3pt['zone'] = df_3pt['location'].map({0: 'R Above Break', 1: 'L Above Break',2:'L Below Break/Corner',3:'R Center',4:'R Below Break/Corner',5:'Center',6:'L Center'})
    zone_column = df_3pt[['zone']]

    df_shots = df_shots.merge(zone_column, how = 'left', left_index = True, right_index = True)

    return df_shots, df_outlier_3pt

def acquire_shots():
    df_shots = all_21_22_shots()
    df_shots, df_outlier_3pt = create_3pt_shot_zones(df_shots)

    return df_shots, df_outlier_3pt


# ------------------------------------------------------------------------------------------------
# CREATING A DETAILED PLAYER-GAME DATAFRAME
# ------------------------------------------------------------------------------------------------

def create_base_df(game_id):
    '''
    Acquires a second-by-second record of the given game creating a foundation for our dataset.
    '''
    # Use the NBA API endpoint for Win Probability
    df = winprobabilitypbp.WinProbabilityPBP(game_id).get_data_frames()[0]
    # Use function below to get the absolute time
    df_base = get_absolute_time_winprob(df)

    return df_base

def get_absolute_time_winprob(df):
    '''
    Takes in a dataframe based on 'WinProbabilityPBP' NBA-API endpoint and adds a column with the absolute game time, in seconds.
    '''
    # Covers overtime edge cases by having seperate function if game is in overtime (periods > 4)
    df['abs_time'] = np.where(df.PERIOD <5,
                     ((df.PERIOD - 1) * 720 + (720 - df.SECONDS_REMAINING)),
                     (2880 + (df.PERIOD - 5) * 300 + (300 - df.SECONDS_REMAINING)))
    df = df.astype({'abs_time':'int'})
    
    return df

def get_player_rotation_data(game_id, player_id):
    '''
    *SHould I indicate step somewhere in here - maybe after return 
    '''
    # Use NBA API endpoint for Game Rotation
    df_rotation = gamerotation.GameRotation(game_id).get_data_frames()
    # Searches through the rotation datasets for the player of interest
    for i in range(2):
        for player in df_rotation[i].PERSON_ID:
            if player == player_id:
                df_player_roto = df_rotation[i][df_rotation[i].PERSON_ID == player]
    # Create abs_time indexed columns out of the two columns pulled from the rotation dataset, then return
    df_player_roto['abs_in_time'] = df_player_roto.IN_TIME_REAL/10
    df_player_roto['abs_out_time'] = df_player_roto.OUT_TIME_REAL/10
    df_player_roto_times = df_player_roto[['abs_in_time','abs_out_time']].reset_index(drop = 'True')

    return df_player_roto_times

#
#
# ADD FUNCTION WHICH CREATES ACTIVE TIME AND TIME SINCE REST!
#
#

def build_player_gameline(df_player_roto_times, df_base):
    '''
    *Remember, this is an intermediate step that is solely designed to create the features 
    '''
    zipped = list(zip(df_player_roto_times.abs_in_time, df_player_roto_times.abs_out_time))
    # Let me create a holder dataframe as I pull slices off from the base
    df_player_game = pd.DataFrame()
    for times in zipped:
        df_slice = df_base[(df_base.abs_time >= times[0]) & (df_base.abs_time <= times[1])]
        df_player_game = pd.concat([df_player_game, df_slice])
    
    return df_player_game

def acquire_player_game(game_id, player_id):
    df_base = create_base_df(game_id)
    df_player_roto_times = get_player_rotation_data(game_id, player_id)
    df_player_game = build_player_gameline(df_player_roto_times, df_base)

    return df_player_game

# ------------------------------------------------------------------------------------------------
# PLAYER-GAME SHOTS
# ------------------------------------------------------------------------------------------------_

def get_player_game_shots(game_id, player_id, df_shots):
    df_game_shots = df_shots[df_shots.GAME_ID == int(game_id)]
    df_game_shots = df_game_shots[df_game_shots.PLAYER_ID == player_id]

    df_game_shots = id_home_team(df_game_shots)

    return df_game_shots

def id_home_team(df_game_shots):
    teams.find_teams_by_full_name(df_game_shots.TEAM_NAME.max())[0]['abbreviation']
    df_game_shots['home'] = np.where(teams.find_teams_by_full_name(df_game_shots.TEAM_NAME.max())[0]['abbreviation'] == df_game_shots.HTM, True, False)

    return df_game_shots

def player_game(df_game_shots, df_player_game):
    df_player_game = df_player_game.merge(df_game_shots, how = 'inner', on = 'abs_time')
    df_player_game['score_margin'] = np.where(df_player_game.home == True, df_player_game.HOME_SCORE_MARGIN, df_player_game.HOME_SCORE_MARGIN * -1)
    if df_player_game.loc[0,'home'] == True:
        df_player_game = df_player_game.drop(columns = ['VISITOR_PCT'])
        df_player_game = df_player_game.rename(columns = {'HOME_PCT':"WIN_PCT"})
    else:
        df_player_game = df_player_game.drop(columns = ['HOME_PCT'])
        df_player_game = df_player_game.rename(columns = {'VISITOR_PCT':"WIN_PCT"})
        
    df_player_game['play_points'] = np.where(df_player_game.SHOT_TYPE == '2PT Field Goal',
                                    np.where(df_player_game.SHOT_MADE_FLAG == 1, 2,0),
                                    np.where(df_player_game.SHOT_MADE_FLAG == 1, 3,0))

    df_player_game['points'] = df_player_game['play_points'].cumsum()
    df_player_game['shots_taken'] = df_player_game['SHOT_ATTEMPTED_FLAG'].cumsum()
    df_player_game['shots_hit'] = df_player_game['SHOT_MADE_FLAG'].cumsum()
    # Need to count the previous row's shooting percentage to not include current shot.
    df_player_game['fg_pct'] = np.where(df_player_game['SHOT_MADE_FLAG'] == 1, 
                                        round((df_player_game['shots_hit']-1)/(df_player_game['shots_taken']-1),2),
                                        round((df_player_game['shots_hit'])/(df_player_game['shots_taken']-1),2))
    df_player_game.fg_pct = df_player_game.fg_pct.fillna(1)
    return df_player_game

def clean_game(df_player_game):
    #coumns to rename
    columns_to_rename = {'WIN_PCT':'win_prob',
                     'GAME_ID_y':'game_id',
                     'PLAYER_ID':'player_id',
                     'PLAYER_NAME':'player',
                     'TEAM_ID':'team_id',
                     'TEAM_NAME':'team',
                     'PERIOD_y':'period',
                     'ACTION_TYPE':'shot_type',
                     'SHOT_TYPE':'fg_type',
                     'LOC_X':'loc_x',
                     'LOC_Y':'loc_y',
                     'EVENT_TYPE':'shot_result'}
    #columns to remove
    columns_to_drop = [
        'GAME_ID_x',
        'EVENT_NUM',
        'HOME_PTS',
        'VISITOR_PTS',
        'HOME_SCORE_MARGIN',
        'PERIOD_x',
        'DESCRIPTION',
        'SHOT_DISTANCE',
        'SECONDS_REMAINING_x',
        'HOME_POSS_IND',
        'HOME_G',
        'LOCATION',
        'PCTIMESTRING',
        'ISVISIBLE',
        'GRID_TYPE',
        'GAME_EVENT_ID',
        'MINUTES_REMAINING',
        'SECONDS_REMAINING_y',
        'play_points',
        'shots_taken',
        'shots_hit',
        'SHOT_ZONE_BASIC',
        'SHOT_ZONE_AREA',
        'SHOT_ZONE_RANGE',
        'SHOT_ATTEMPTED_FLAG',
        'SHOT_MADE_FLAG',
        'GAME_DATE',
        'HTM',
        'VTM']
    #columns to type change - None
    df_player_game = df_player_game.drop(columns = columns_to_drop)
    df_player_game = df_player_game.rename(columns = columns_to_rename)
    # filter out the 3pointers
    df_player_game_target = df_player_game[df_player_game['fg_type'] == '3PT Field Goal']

    return df_player_game_target

# ------------------------------------------------------------------------------------------------
# Putting it together - Acquire 1 player, 1 game
# ------------------------------------------------------------------------------------------------

def acquire_player_game_target(game_id, player_id):
    df_shots, df_outlier_3pt = acquire_shots()
    df_player_game = acquire_player_game(game_id, player_id)

    df_game_shots = get_player_game_shots(game_id, player_id, df_shots)

    df_player_game = player_game(df_game_shots, df_player_game)

    df_player_game_target = clean_game(df_player_game)

    return df_player_game_target

# ------------------------------------------------------------------------------------------------
# Need function to take in a player name and return a seasons worth of clean games
# ------------------------------------------------------------------------------------------------





