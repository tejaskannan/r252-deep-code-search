import sys
from parser import Parser

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print("Too few arguments.")
	elif sys.argv[1] == "-g":
		parser = Parser("filters/tags.txt", "filters/stopwords.txt")
		parser.parse_directory(sys.argv[2])