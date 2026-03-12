
import signal

import csv


def export_to_csv(output_csv_file, data_list):

	with open(output_csv_file, 'a', newline='') as fp:
		a = csv.writer(fp, delimiter=';')
		data = [data_list]
		a.writerows(data)

# Timeout class used to stop bots if they take too long to bid
class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)