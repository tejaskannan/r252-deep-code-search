import sys
from parser import Parser

if __name__ == '__main__':
	parser = Parser("filters/tags.txt", "filters/stopwords.txt")
	parser.parse_directory(sys.argv[1])