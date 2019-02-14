import sys
from parser import Parser
from model import Model

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print("Too few arguments.")
	elif sys.argv[1] == "-g":
		parser = Parser("filters/tags.txt", "filters/stopwords.txt")

		out_folder = "data"
		if len(sys.argv) > 3:
			out_folder = sys.argv[3]

		if sys.argv[2][-1] == "/":
			written = parser.parse_directory(sys.argv[2], out_folder)
		else:
			written = parser.parse_file(sys.argv[2], out_folder)
		print("Dataset size: {0}".format(written))
	elif sys.argv[1] == "train":
		model = Model()
		model.train()
	elif sys.argv[1] == "test":
		file_name = sys.argv[2]
		model = Model()
		model.restore(file_name)