import sys
from parser import Parser
from model import Model

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print("Too few arguments.")
	elif sys.argv[1] == "-g":
		parser = Parser("filters/tags.txt", "filters/stopwords.txt")
		parser.parse_directory(sys.argv[2])
	elif sys.argv[1] == "train":
		model = Model()
		model.train()