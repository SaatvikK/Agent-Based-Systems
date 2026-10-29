import csv
import os
import random 
import json 

# Import the auctioneer, to run your auctions
from auctioneer import Auctioneer

# Import some bots to play with
# We have given you some basic bots to start with in the bots folder
# You can also add your own bots to test against
from bots import flat_bot_10
from bots import random_bot
from bots import u5714960
from bots import claude
from bots import claude2
from bots import rohan
from bots import gemini
from bots import selfbot
from bots import poisonbot

def run_basic_auction(room = [u5714960, rohan, claude2, gemini], verbose = True):
	"""
	An example function that runs a basic auction with 3 bots
	"""

	# Setup the auction
	my_auction = Auctioneer(room=room, slowdown = 0, verbose = verbose)
	# Play the auction
	winners = my_auction.run_auction()
	return my_auction.run_auction()


def run_lots_of_auctions():
	"""
	An example if you want to run alot of auctions at once
	"""
	# A large room with a few bots of the same type
	room = [random_bot, random_bot, claude, u5714960]
	#room = [u1234321, random_bot, claude]
	# Set the game type between "value" or "collection"
	
	win_count = 0
	run_count = 50
	for i in range(run_count):
		# Setup the auction
		# slowdown = 0 makes it fast
		my_auction = Auctioneer(room=room, slowdown=1, verbose = False)
		# run_auction() returns a list of winners, sometimes there are more than one winner if there is a tie
		winners = my_auction.run_auction()
		# Check if the bot's name, "my_bot", was a winner 
		if "1234321" in winners:
			win_count +=1
	print("My bot won {} of {} games".format(win_count, run_count))	

# modified run lots of auctions for training by claude opus 4.6
import csv
import os

def train(epochs=1000):
    opponents = [rohan, claude, claude2, flat_bot_10, random_bot]
    room = [u5714960] + random.sample(opponents, 3)

    qler = u5714960.ql.QLearner()
    win_count = 0
    run_count = epochs

    # Define paths
    base_path = os.path.dirname(__file__)
    qtable_path = os.path.join(base_path, "data/qtable.json")
    csv_path = os.path.join(base_path, "data/training_log.csv")

    # Ensure the data directory exists
    os.makedirs(os.path.join(base_path, "data"), exist_ok=True)

    # Initialize CSV file with headers
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'win_count'])

    for i in range(run_count):
        my_auction = Auctioneer(room=room, slowdown=0, verbose=False)
        my_auction.bots[0]["bot_instance"].KB.qler = qler
        
        winners = my_auction.run_auction()
        
        # Reset Q-learner state
        qler.state = [0, 0, 0, 0, "high"]
        qler.lastState = [0, 0, 0, 0, "high"]
        qler.lastBid = 0
        
        # Check for winner (Using the ID you provided in your logic)
        if "5714960" in winners:
            win_count += 1
        
        # --- NEW: Log to CSV ---
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([i, win_count])
        
        if i % 5 == 0:
            print(f"Episode {i}, wins: {win_count}, Q-table size: {len(qler.Qtable)}")
            qler.saveQtable(qtable_path)

    # Final Save
    qler.saveQtable(qtable_path)
    print(f"Training complete. Won {win_count}/{run_count}")


def test(room, epochs=500, verbose = True):
	wins = 0
	winslist = []
	path = 'data/testresults/3.csv'
	
	# Run the auction loop
	for i in range(epochs):
		winners = run_basic_auction(room, verbose = verbose)
		if "5714960" in winners:
			wins += 1   

		# Ensure the directory exists
		os.makedirs(os.path.dirname(path), exist_ok=True)

		# Write to CSV
		file_exists = os.path.isfile(path)
		with open(path, mode='a', newline='') as file:
			writer = csv.writer(file)
			# Write header only if the file is new
			if not file_exists:
				writer.writerow(['episode', 'win_count'])
			
			writer.writerow([i, wins])
	
	win_rate = wins / epochs

	#res = { "opponents": [ bot.name for bot in opponents ], "winrate": win_rate, "cumwins": winslist, "patiences": patiences }

	return win_rate

def overalltest(numtests = 6, epochs = 500):
	path = "data/testresults"

	for testidx in range(numtests):
		print(f"TEST {testidx}")
		opponents = [rohan, claude, claude2, flat_bot_10, random_bot, gemini]
		chosenOps = random.sample(opponents, random.randint(1, len(opponents)))
		room = [u5714960] + random.sample(opponents, random.randint(1, len(opponents)))

		res = test(room, chosenOps, epochs, verbose = False)

		filepath = os.path.join(path, f"{testidx}.json")
        
		with open(filepath, "w") as json_file:
			json.dump(res, json_file, indent=4)
            
		print(f"Test {testidx} complete: Saved to {filepath}")


if __name__== "__main__":
	#train(epochs = 5000)
	#run_basic_auction()
	testepochs = 10
	room = [claude, selfbot, poisonbot, gemini]

	print(f"Bot win rate after {testepochs} auctions = {test([u5714960] + room, epochs = testepochs, verbose = True)}")

	#overalltest(epochs = 50)