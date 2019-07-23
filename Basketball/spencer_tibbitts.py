import numpy as np
import pandas as pd
import csv
from progressbar import ProgressBar
pbar = ProgressBar()

def add_points(points,players):
	players.Points_For = players.Points_For + (players.where((players.Person_id.isin(on_court)) & (players.Team_id==team_off))['Points_For'].apply(lambda x: x*0) + points).fillna(0)
	players.Points_Allowed = players.Points_Allowed + (players.where((players.Person_id.isin(on_court)) & (players.Team_id!=team_off))['Points_Allowed'].apply(lambda x: x*0) + points).fillna(0)

def add_poss(players):
	players.Offensive_Possessions = players.Offensive_Possessions + (players.where((players.Person_id.isin(on_court)) & (players.Team_id==team_off))['Offensive_Possessions'].apply(lambda x: x*0) + 1).fillna(0)
	players.Defensive_Possessions = players.Defensive_Possessions + (players.where((players.Person_id.isin(on_court)) & (players.Team_id!=team_off))['Defensive_Possessions'].apply(lambda x: x*0) + 1).fillna(0)

#### Clean the Object data.
game_lineup = pd.read_csv('Game_Lineup.txt',sep='\s+')
play_by_play = pd.read_csv('Play_by_Play.txt',sep='\s+')
event_codes = pd.read_csv('Event_Codes.txt',sep='\s+',)

for col in game_lineup.columns.values:
	if game_lineup[col].dtype=='object':
		game_lineup[col] = game_lineup[col].str.strip()

for col in play_by_play.columns.values:
	if play_by_play[col].dtype=='object':
		play_by_play[col] = play_by_play[col].str.strip()

for col in event_codes.columns.values:
	if event_codes[col].dtype=='object':
		event_codes[col] = event_codes[col].str.strip()

#### Order the games chronologically.
play_by_play = play_by_play.sort_values(["Game_id","Period","PC_Time","Event_Num"],ascending=[True,True,False,True])

#### Map each player to the correct team using the Game_Lineup.csv file as suggested.
team_mapper = {}
for player in play_by_play['Person1'].unique():
	if len(game_lineup[game_lineup['Person_id']==player]["Team_id"].unique()) > 0:
		team_mapper[player] = game_lineup[game_lineup['Person_id']==player]["Team_id"].unique()[0]

team_id2 = play_by_play.Person1.apply(lambda x: team_mapper[x] if x in team_mapper else np.NaN) 
play_by_play.Team_id = team_id2.fillna(play_by_play.Team_id)  # Use the mapped team identifier where possible, if not possible go with what the data gave us.

	# A possession is ended by 
	# *  made field goal attempts
	# *  made final free throw attempt
	# *  missed final free throw attempt that results in a defensive rebound,
	# *  missed field goal attempt that results in a defensive rebound
	# *  turnover
	# *  end of time period.

#### Calculate off/def efficiency for each player, in each game, and write to a .csv file.
lines = []
for game_id in pbar(play_by_play.Game_id.unique()):

	game = play_by_play[play_by_play.Game_id==game_id].reset_index()
	cols = game.columns.tolist()
	players = pd.DataFrame(index=game_lineup[(game_lineup.Game_id==game_id) & (game_lineup.Period==0)]['Person_id'].tolist(),columns=['Offensive_Possessions','Points_For','Defensive_Possessions','Points_Allowed'])
	players['Person_id'] = players.index
	players = players.fillna(0)
	players['Team_id'] = game_lineup[(game_lineup.Game_id==game_id) & (game_lineup.Period==0)]['Team_id'].tolist()

	incoming_sub = False
	entering = []
	leaving = []

	for row in game.iterrows():

		instance = row[1].tolist() # Get all of the attributes for this event.
		event_msg_type = instance[cols.index("Event_Msg_Type")] # Number associated with the type of event.
		
		# Check if any new players are on the court.
		if incoming_sub and instance[cols.index("PC_Time")] != substitution_time:
			[on_court.append(player) for player in entering] # Adds players coming into game to the 'on_court' flagger.
			[on_court.remove(player) for player in leaving] # Removes players exiting the game from the 'on_court' flagger.
		# Reset the substituter for the next group of substitutions.
			incoming_sub = False 
			entering = []
			leaving = []

		if event_msg_type==1: # made shot
			points = instance[cols.index("Option1")] # points associated with the shot
			team_off = instance[cols.index("Team_id")] # team that made the shot
			add_points(points,players)
			add_poss(players)

		elif event_msg_type==2 and game.iloc[(row[0]+1)]["Event_Msg_Type"]==4: # missed shot followed by a rebound
			team_off = instance[cols.index("Team_id")] # team that missed the shot
			team_reb = game.iloc[(row[0]+1)]["Team_id"] # team that got the rebound
			if team_off != team_reb: # defensive rebound, so a change of possession occured
				add_poss(players)

		elif event_msg_type==3: # free throws

			team_off = instance[cols.index("Team_id")] # team shooting at the line
			points = instance[cols.index("Option1")] # 1 if made, 2 if missed

			if points == 1:
				add_points(points,players)

			if instance[cols.index("Action_Type")] in [10,12,15,19,20,22,26,29]:  # all of these action types signify the last free throw
				if game.iloc[(row[0]+1)]["Event_Msg_Type"]!=4 or game.iloc[(row[0]+1)]["Team_id"]!=team_off:  # anything but an offensive rebound occured
					add_poss(players)

		elif event_msg_type==5: # the team associated with the event number is the team who LOST possesion
			team_off = instance[cols.index("Team_id")]
			add_poss(players)

		elif event_msg_type==8: # substitution! Person1 LEAVES the court, Person2 ENTERS
					# new players are not technically on the court until the game clock starts moving
			incoming_sub = True
			leave = instance[cols.index("Person1")]
			enter = instance[cols.index("Person2")]
			substitution_time = instance[cols.index("PC_Time")]
			entering.append(enter)
			leaving.append(leave)

		elif event_msg_type==12: # start of period
			period = instance[cols.index("Period")]
			on_court = game_lineup[(game_lineup.Game_id==game_id) & (game_lineup.Period==period)]['Person_id'].tolist()

		elif event_msg_type==13: # end of period
			add_poss(players)
	# Calculate each player's offensive/defensive rating.  Players who did not play receive a rating of NaN.
	players['Offensive_Rating'] = (100* players.Points_For/players.Offensive_Possessions)
	players['Defensive_Rating'] = (100* players.Points_Allowed/players.Defensive_Possessions)
	for i in range(len(players)):
		lines.append([game_id]+ [players.iloc[i]["Person_id"]] + [players.iloc[i]["Offensive_Rating"]] + [players.iloc[i]["Defensive_Rating"]])

with open('spencer_tibbitts.csv', 'w') as writeFile:
	writer = csv.writer(writeFile)
	writer.writerow(["Game_ID","Player_ID","OffRtg","DefRtg"])
	writer.writerows(lines)	
