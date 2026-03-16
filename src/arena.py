import csv
import os

# Import the auctioneer, to run your auctions
from auctioneer import Auctioneer

# Import some bots to play with
# We have given you some basic bots to start with in the bots folder
# You can also add your own bots to test against
from bots import flat_bot_10
from bots import random_bot
from bots import u1234321
from bots import claude

def run_basic_auction():
	"""
	An example function that runs a basic auction with 3 bots
	"""
	# Setup a room of bots to play against each other, imported above from the bots folder
	room = [u1234321, flat_bot_10, random_bot, claude]
	#room = [u1234321, random_bot, claude]
	# Setup the auction
	my_auction = Auctioneer(room=room)
	# Play the auction
	my_auction.run_auction()


def run_lots_of_auctions():
	"""
	An example if you want to run alot of auctions at once
	"""
	# A large room with a few bots of the same type
	room = [random_bot, random_bot, claude, u1234321]
	#room = [u1234321, random_bot, claude]
	# Set the game type between "value" or "collection"
	
	win_count = 0
	run_count = 50
	for i in range(run_count):
		# Setup the auction
		# slowdown = 0 makes it fast
		my_auction = Auctioneer(room=room, slowdown=1)
		# run_auction() returns a list of winners, sometimes there are more than one winner if there is a tie
		winners = my_auction.run_auction()
		# Check if the bot's name, "my_bot", was a winner 
		if "1234321" in winners:
			win_count +=1
	print("My bot won {} of {} games".format(win_count, run_count))	

# modified by claude opus 4.6
def modified_run_lots_of_auctions():
	room = [u1234321, random_bot, claude, random_bot]
	
	# Persistent Q-table across all games
	qler = u1234321.ql.QLearner()
	
	win_count = 0
	run_count = 2000
	
	for i in range(run_count):
			my_auction = Auctioneer(room=room, slowdown=0, verbose=False)
			
			# Inject persistent Q-table into the new bot instance
			my_auction.bots[0]["bot_instance"].KB.qler = qler
			
			winners = my_auction.run_auction()
			
			# Reset Q-learner state for next episode, keep table
			qler.state = [0, 0, 0, 0, "high"]
			qler.lastState = [0, 0, 0, 0, "high"]
			qler.lastBid = 0
			
			if "5714960" in winners:
					win_count += 1
			
			if i % 5 == 0:
					print(f"Episode {i}, wins: {win_count}, Q-table size: {len(qler.Qtable)}")
	
	# Save after training
	path = os.path.join(
					os.path.dirname(__file__), "data/qtable.json"
			)
	qler.saveQtable(path)
	print(f"Training complete. Won {win_count}/{run_count}")


if __name__=="__main__":
	modified_run_lots_of_auctions()
	#run_basic_auction()