import re

from dpu_utils import codeutils
from html.parser import HTMLParser

class TextFilter:

    def __init__(self, tags_file, stopwords_file):
        self.java_keywords = set(codeutils.get_language_keywords("java"))
        self.javadoc_tags = self._load_from_file(tags_file)
        self.stopwords = self._load_from_file(stopwords_file)
        self.token_regex = re.compile(r"[\{\}\[\]\(\)\*\n\t,><\\]") 

    def apply_to_token_lst(self, tokens, use_keywords=True, use_stopwords=True, use_tags=False):
        tokens = [self.token_regex.sub("", t.strip()) for t in tokens]
        all_tokens = []
        for token in tokens:
            all_tokens += token.split("_")
        return list(filter(lambda t: self.filter_single_token(t, use_keywords, use_stopwords),
                           all_tokens))

    def apply_to_javadoc(self, text, use_stopwords=True, use_tags=True):
        lines = text.split("\n")
        cleaned_lines = []

        html_parser = JavadocHTMLParser()
        for line in lines:
            tokens = line.strip().split()
            if len(tokens) == 0:
                continue
            if use_tags and tokens[0] in self.javadoc_tags:
                continue
            stripped_tokens = self.apply_to_token_lst(tokens,
                                                      use_keywords=False,
                                                      use_stopwords=use_stopwords,
                                                      use_tags=True)

            cleaned_tokens = []
            for token in stripped_tokens:
                html_parser.feed(token)
                cleaned_tokens.append(html_parser.get_data())
                html_parser.reset()

            cleaned_lines += cleaned_tokens
        return cleaned_lines

    def filter_single_token(self, token, use_keywords=True, use_stopwords=True, use_tags=True):
        if len(token.strip()) == 0:
            return False
        if use_keywords and token in self.java_keywords:
            return False
        if use_stopwords and token in self.stopwords:
            return False
        if use_tags and token in self.javadoc_tags:
            return False
        return True


    def _load_from_file(self, file_name):
        tokens = set()
        with open(file_name, "r") as token_file:
            for token in token_file:
                tokens.add(token.strip())
        return tokens


class JavadocHTMLParser(HTMLParser):

    def __init__(self):
        super().__init__()
        self.reset()
        self.contents = []

    def handle_data(self, data):
        self.contents.append(data)

    def get_data(self):
        return "".join(self.contents)

    def reset(self):
        super().reset()
        self.contents = []
