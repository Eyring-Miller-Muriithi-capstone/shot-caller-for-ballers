'''
These functions drive the complicated process of calling multiple NBA statistical endpoints
using an open source program called NBA-API.
It is encapsulated in a single function, tome_prep(), which emits a dataframe ready for further pre-processing.
'''
# DS Libraries
import pandas as pd
import numpy as np
import os
import time

# Plotting Functions
import matplotlib.pyplot as plt
import seaborn as sns

# NBA-API Endpoint Analyzers
from nba_api.stats.static import teams
from nba_api.stats.static import players
from nba_api.stats.endpoints import gamerotation
from nba_api.stats.endpoints import shotchartdetail
from nba_api.stats.endpoints import teamplayerdashboard
from nba_api.stats.endpoints import winprobabilitypbp

# For clustering
from sklearn.cluster import KMeans

# ------------------------------------------------------------------------------------------------
# CREATING THE DATAFRAME WITH ALL SHOTS FOR THE 2021-2022 SEASON 
# ------------------------------------------------------------------------------------------------

def get_team_player_ids():
    '''
    Acquires a dataframe from cached .csv, or NBA API calls (and caches it if it doesn't exist),
    then returns the data as as dataframe with the columns 'team_id' and 'player_id'. 
    Needed to account for players who played for multiple teams in the season.
    '''
    filename = 'team_player_ids.csv'
    # Checks if .csv already exists
    if os.path.isfile(filename):
        team_player_ids =  pd.read_csv(filename, index_col=0)
        players_list = team_player_ids.values.tolist()
    else:
        df_teams = pd.DataFrame(teams.get_teams())
        # Create a list of team ids
        team_id_list = list(df_teams.id)
        players_list = []
        # Attaches team_id to player_id
        for team in team_id_list:
            df_tpd = teamplayerdashboard.TeamPlayerDashboard(team,
                                                            season = '2021-22').get_data_frames()
            player_list = list(df_tpd[1].PLAYER_ID)
            for player in player_list:
                row = [team,player]
                players_list.append(row)
        # Isolates the relevant columns (team and player id) and saves to .csv and returns a list
        team_player_ids = pd.DataFrame(players_list, columns = ['team_id','player_id'])
        team_player_ids.to_csv('team_player_ids.csv')
        players_list = team_player_ids.values.tolist()
    
    return players_list

def all_21_22_shots():
    '''
    Acquires a dataframe of all shot attempts made in the 2021-2022 season by all players.
    '''
    # Use above function to get list of players + teams ids.  Each is a player's entire shot record for the season.
    players_list = get_team_player_ids()
    filename = 'all_last_season_shots.csv'
    # Check if file exists
    if os.path.isfile(filename):
        df_shots =  pd.read_csv(filename, index_col=0)
    else:
        df_shots = pd.DataFrame()
        index = 0
        for player in players_list:
            print(f'\rFetching index {index} of 714', end='')
            # Pulls each player's season shot record, one by one
            df_pl = shotchartdetail.ShotChartDetail(team_id = player[0],
                                                            player_id = player[1],
                                                            season_type_all_star='Regular Season',
                                                            season_nullable='2021-22',
                                                            context_measure_simple = 'FG3A').get_data_frames()
            time.sleep(.5)
            index += 1
            # Combines player records with previous player records pulled
            df_shots = pd.concat([df_shots, df_pl[0]])
        df_shots.to_csv(filename)
    # Add in absolute time
    df_shots = get_absolute_time_shots(df_shots)
    # Reset Index
    df_shots = df_shots.reset_index(drop = True)
    
    return df_shots

def get_absolute_time_shots(df):
    '''
    Takes in a dataframe based on 'ShotChartDetail' NBA-API endpoint and adds a column with the absolute game time, in seconds.
    '''
    # Covers overtime edge cases by having seperate calculation if game is in overtime (periods > 4)
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
    # Create outlier dataframe for future analysis
    df_outlier_3pt = df_all_3pt[df_all_3pt.SHOT_DISTANCE > bound]
    # df with outliers removed
    df_shots = df_shots[df_shots.SHOT_DISTANCE <= bound]

    # Create the clusters
    df_3pt = df_all_3pt[df_all_3pt.SHOT_DISTANCE <= bound]
    X = df_3pt[['LOC_X','LOC_Y']]
    kmeans = KMeans(n_clusters=7)
    kmeans.fit(X)
    clusters = kmeans.predict(X)
    df_3pt['location'] = clusters
    # Emit a graph for funsies
    plt.figure(figsize = (12,12))
    sns.scatterplot(data =df_3pt, x='LOC_X', y = 'LOC_Y', hue = 'location')
    plt.show()

    # Naming the zones and creating the column to merge back to df_shots
    df_3pt['zone'] = df_3pt['location'].map({0: 'R Above Break', 1: 'L Above Break',2:'L Below Break/Corner',3:'R Center',4:'R Below Break/Corner',5:'Center',6:'L Center'})
    zone_column = df_3pt[['zone']]

    # Merge
    df_shots = df_shots.merge(zone_column, how = 'left', left_index = True, right_index = True)

    # Returns outliers as well
    return df_shots, df_outlier_3pt

def acquire_shots():
    '''
    Performs all steps above to acquire shots with clusters and caches to .csv
    '''
    filename1 = 'all_last_season_shots_3pt_clusters.csv'
    filename2 = 'last_season_outlier_3PA.csv'
    # Check to see if .csv is in directory - asssumes if one is, both are
    if os.path.isfile(filename1):
        df_shots =  pd.read_csv(filename1, index_col=0)
        df_outlier_3pt = pd.read_csv(filename2, index_col = 0)
    # If no .csv, calls function to create them
    else:
        df_shots = all_21_22_shots()
        df_shots, df_outlier_3pt = create_3pt_shot_zones(df_shots)
        df_shots.to_csv(filename1)
        df_outlier_3pt.to_csv(filename2)

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
    # Covers overtime edge cases by having seperate calculations if game is in overtime (periods > 4)
    df['abs_time'] = np.where(df.PERIOD <5,
                     ((df.PERIOD - 1) * 720 + (720 - df.SECONDS_REMAINING)),
                     (2880 + (df.PERIOD - 5) * 300 + (300 - df.SECONDS_REMAINING)))
    df = df.astype({'abs_time':'int'})
    
    return df

def get_player_rotation_data(game_id, player_id):
    '''
    Identifies the time periods, in absolute time, that the player was in a given game.  Returns a df with this data.
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

def build_player_gameline(df_player_roto_times, df_base):
    '''
    This builds the player gameline to include play time
    '''
    # Creates a list of sublists, each sublist being the time bounds of their in-game appearances
    zipped = list(zip(df_player_roto_times.abs_in_time, df_player_roto_times.abs_out_time))

    # Initialize a list starting with zero for rest times
    rest_time = [0]
    for i in range(len(zipped)-1):
        rest_time.append(zipped[i+1][0] - zipped[i][1])
    # Create a cumulative rest
    rest_time = list(pd.Series(rest_time).cumsum())
    # Let me create a holder dataframe as I pull slices off from the base
    df_player_game = pd.DataFrame()
    counter = 0
    # Create columns based on playing time and rest
    for times in zipped:
        in_time = times[0]
        df_slice = df_base[(df_base.abs_time >= times[0]) & (df_base.abs_time <= times[1])]
        df_slice['slice'] = counter
        df_slice['rest'] = rest_time[counter]
        df_slice['play_time'] = df_slice['abs_time'] - df_slice['rest']
        df_slice['since_rest'] = df_slice['abs_time'] - in_time
        df_player_game = pd.concat([df_player_game, df_slice])
        counter += 1
    
    return df_player_game

def acquire_player_game(game_id, player_id):
    '''
    This gives a second by second record of the game for the times in the game the player appears
    '''
    df_base = create_base_df(game_id)
    df_player_roto_times = get_player_rotation_data(game_id, player_id)
    df_player_game = build_player_gameline(df_player_roto_times, df_base)

    return df_player_game

# ------------------------------------------------------------------------------------------------
# PLAYER-GAME SHOTS
# ------------------------------------------------------------------------------------------------_

def get_player_game_shots(game_id, player_id, df_shots):
    '''
    Creates a df of all a players shots in a game
    '''
    df_game_shots = df_shots[df_shots.GAME_ID == int(game_id)]
    df_game_shots = df_game_shots[df_game_shots.PLAYER_ID == player_id]

    # Add in home category (True or False)
    df_game_shots = id_home_team(df_game_shots)

    return df_game_shots

def id_home_team(df_game_shots):
    '''
    Identifies home team and creates a boolean column
    '''
    # Discovered Clippers are mis-entered in the API and need to be changed to work
    if df_game_shots.TEAM_NAME.max() == 'LA Clippers':
        df_game_shots.TEAM_NAME = 'Los Angeles Clippers'
    df_game_shots['home'] = np.where(teams.find_teams_by_full_name(df_game_shots.TEAM_NAME.max())[0]['abbreviation'] == df_game_shots.HTM, True, False)
    
    return df_game_shots

def player_game(df_game_shots, df_player_game):
    '''
    Merges a player's shots with their second-by-second stats
    '''
    # Merge on absolute time, remove all home vs visitor columns and make them team focused
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

    df_player_game['points'] = df_player_game['play_points'].cumsum() - df_player_game['play_points']
    df_player_game['shots_taken'] = df_player_game['SHOT_ATTEMPTED_FLAG'].cumsum()
    df_player_game['shots_hit'] = df_player_game['SHOT_MADE_FLAG'].cumsum()

    # Need to count the previous row's shooting percentage to not include current shot.
    df_player_game['fg_pct'] = np.where(df_player_game['SHOT_MADE_FLAG'] == 1, 
                                        round((df_player_game['shots_hit']-1)/(df_player_game['shots_taken']-1),2),
                                        round((df_player_game['shots_hit'])/(df_player_game['shots_taken']-1),2))
    df_player_game.fg_pct = df_player_game.fg_pct.fillna(1)
    return df_player_game

def clean_game(df_player_game):
    '''
    Rename and drop columns, while also removing all non-3pt shots.
    '''
    # Columns to rename
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
    # Columns to remove
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
        'slice',
        'rest',
        'SHOT_ZONE_BASIC',
        'SHOT_ZONE_AREA',
        'SHOT_ZONE_RANGE',
        'SHOT_ATTEMPTED_FLAG',
        'SHOT_MADE_FLAG',
        'GAME_DATE',
        'HTM',
        'VTM']
    # Columns to type change - None

    df_player_game = df_player_game.drop(columns = columns_to_drop)
    df_player_game = df_player_game.rename(columns = columns_to_rename)
    # Filter out the 3pointers
    df_player_game_target = df_player_game[df_player_game['fg_type'] == '3PT Field Goal']

    return df_player_game_target

# ------------------------------------------------------------------------------------------------
# Putting it together - Acquire 1 player, 1 game
# ------------------------------------------------------------------------------------------------

def acquire_player_game_target(game_id, player_id, df_shots): #moved df_shots to here to not have to call it every time
    '''
    Acquires a single player's 3 point shot and featurs around that shot for a single game
    '''
    df_player_game = acquire_player_game(game_id, player_id)

    df_game_shots = get_player_game_shots(game_id, player_id, df_shots)

    df_player_game = player_game(df_game_shots, df_player_game)

    df_player_game_target = clean_game(df_player_game)

    return df_player_game_target

# ------------------------------------------------------------------------------------------------
# Need function to take in a player name and return a seasons worth of clean games
# ------------------------------------------------------------------------------------------------

def player_season_3pa(player_full_name):
    '''
    Creates an entire season of 3-point shots for the player using the NBA API
    '''

    # We used player_full_name as the input parameter (for Tableau purposes) so need to get back player_id
    player_id = players.find_players_by_full_name(player_full_name)[0]['id']

    # Will use df_shots only
    df_shots, df_outlier_3pt = acquire_shots()

    # Unique game_ids for the player (games they shot a three in this season)
    game_id_list = df_shots[df_shots.PLAYER_ID == player_id].GAME_ID.unique()
    
    # Add in the leading zeros, as this format is required
    game_id_list = [f'00{n}' for n in game_id_list]

    # Initialize and empty dataframe
    df_player_season = pd.DataFrame()

    count = 1

    # Loop for actual scraping
    for game in game_id_list:
        print(f'\rLoading game {count} of {len(game_id_list)} for {player_full_name} in 2021-2022 Regular Season', end='')
        try:
            df_game = acquire_player_game_target(game, player_id, df_shots)
            df_player_season = pd.concat([df_player_season,df_game])  
            count += 1
            time.sleep(.5)
        except:
            # When it fails, capture the spot for restarting
            print(f'\nLoad of game {count} for {player_full_name} failed.\n')
            continue
    
    # Dataframe for a player-season
    df_player_season.reset_index(drop = True, inplace = True)

    return df_player_season

def the_tome():
    '''
    The tome is the name for the full document with all player's three point shots (outliers removed)
    for the 2021-2022 regular season.
    Caches to an in_process_tome so as to not lose data on restarts.
    '''
    # Acquire all 3pt shots for the season
    df_shots, df_outlier_3pt = acquire_shots()

    # Need player names and ids zipped into a list of tuples
    player_id_list = df_shots.PLAYER_ID.unique()
    player_name_list = df_shots.PLAYER_NAME.unique()
    player_tuple = list(zip(player_id_list, player_name_list))

    # Check for cached file, if none initialize
    if os.path.isfile('in_progress_tome.csv'):
        df_league_3pa =  pd.read_csv('in_progress_tome.csv', index_col=0)
    else:
        df_league_3pa = pd.DataFrame()

    player_count = 1

    # Loop through each player using NBA API and all their games and collect data
    for player in player_tuple:
        game_id_list = df_shots[df_shots.PLAYER_ID == player[0]].GAME_ID.unique()
    
        game_id_list = [f'00{n}' for n in game_id_list]
         
        df_player_season = pd.DataFrame()

        # Game count, mostly for tracking when scrape fails
        game_count = 1

        # Error counter - rather than stop on every error, we have a counter to give the scrape a chance to restart on its own
        error_count = 0

        # 596 player records
        for game in game_id_list:
            print(f'\rLoading game {game_count} of {len(game_id_list)} for {player[1]} (player {player_count} of 596) in 2021-2022 Regular Season.                    ', end='')
            try:
                df_game = acquire_player_game_target(game, player[0], df_shots)
                df_player_season = pd.concat([df_player_season,df_game])  
                game_count += 1
                time.sleep(.5)
            # Need this for some errors or it will not stop
            except KeyboardInterrupt:
                print(f'\nLoad of game {game_count} for {player[1]} failed.\n')
                print('Keyboard Interrupt')
                return df_league_3pa
            # Try again!
            except:
                print(f'\nLoad of game {game_count} for {player[1]} failed.\n')
                time.sleep(5)
                game_count += 1
                error_count += 1
                if error_count == 5:
                    print("Too many failures.  Stopping download.")
                    return df_league_3pa
                continue

        df_player_season.reset_index(drop = True, inplace = True)

        df_league_3pa = pd.concat([df_league_3pa, df_player_season]) 

        df_league_3pa.to_csv('in_progress_tome.csv')

        player_count += 1
    
    df_league_3pa.reset_index(drop = True, inplace = True)

    return df_league_3pa

def get_tome():
    '''
    Once tome is finished, this is the easier way to get it's cached data in league_3pa.csv
    '''
    filename = 'league_3pa.csv'
    if os.path.isfile(filename):
        df_league_3pa =  pd.read_csv(filename, index_col=0)
    else:
        df_league_3pa = the_tome()
        df_league_3pa.to_csv(filename)
    
    return df_league_3pa

def tome_prep():
    '''
    Calls get_tome() above and then removes some columns, reordering and adds in their game running point total
    '''
    df = get_tome()
    # A number of duplicates were accidentally created in league_3pa.csv, this removes them
    df = df.drop_duplicates(subset = ['game_id','player_id','period','shot_result','loc_x','loc_y'])
    # Win prob is dropped because it doesn't cover the last 500 seconds of a game.  fg_type and fg_pct not effective
    df = df.drop(columns = ['win_prob','fg_type','fg_pct'])
    df = df[['player',
            'player_id',
            'team',
            'team_id',
            'game_id',
            'home',
            'period',
            'abs_time',
            'play_time',
            'since_rest',
            'loc_x',
            'loc_y',
            'zone',
            'shot_type',
            'score_margin',
            'points',    
            'shot_result']]
    # Correcting for mistake in earlier scraping code to remove the shot value itself from the player's points
    df['points'] = np.where(df.shot_result == 'Made Shot', df.points - 3, df.points)
    
    return df