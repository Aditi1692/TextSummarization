import os
import json

INPUT_FILE_PATH = "data.json"
OUTPUT_FILE_PATH = "extracted_opinions.txt"

if __name__ == "__main__":
	# Get the file from the parent directory
	with open(os.path.join(os.pardir, INPUT_FILE_PATH)) as data_file:
		data = json.load(data_file)
		
	write_output = open(OUTPUT_FILE_PATH, 'w')

	for line in data:
		normal_words = line["content"].encode('ascii', 'ignore')
		write_output.write("%s\n" % normal_words)
		
	write_output.close()
