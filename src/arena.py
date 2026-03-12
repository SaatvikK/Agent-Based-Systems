
import csv

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
	room = [u1234321, flat_bot_10, random_bot]
	room = [u1234321, random_bot, claude]
	# Setup the auction
	my_auction = Auctioneer(room=room)
	# Play the auction
	my_auction.run_auction()


def run_lots_of_auctions():
	"""
	An example if you want to run alot of auctions at once
	"""
	# A large room with a few bots of the same type
	room = [random_bot, random_bot, random_bot, u1234321]
	
	# Set the game type between "value" or "collection"
	
	win_count = 0
	run_count = 50
	for i in range(run_count):
		# Setup the auction
		# slowdown = 0 makes it fast
		my_auction = Auctioneer(room=room, slowdown=0)
		# run_auction() returns a list of winners, sometimes there are more than one winner if there is a tie
		winners = my_auction.run_auction()
		# Check if the bot's name, "my_bot", was a winner 
		if "1234321" in winners:
			win_count +=1
	print("My bot won {} of {} games".format(win_count, run_count))	


if __name__=="__main__":
	run_basic_auction()