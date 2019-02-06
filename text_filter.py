from dpu_utils import codeutils

class TextFilter:

    def __init__(self, tags_file, stopwords_file):
        self.java_keywords = set(codeutils.get_language_keywords("java"))
        self.javadoc_tags = self._load_from_file(tags_file)
        self.stopwords = self._load_from_file(stopwords_file)

    def apply_to_token_lst(self, tokens, use_keywords=True, use_stopwords=True, use_tags=False):
        return list(filter(lambda t: self.filter_single_token(t, use_keywords, use_stopwords),
                           tokens))

    def apply_to_javadoc(self, text, use_stopwords=True, use_tags=True):
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            tokens = line.strip().split()
            if len(tokens) == 0:
                continue
            if use_tags and tokens[0] in self.javadoc_tags:
                continue
            cleaned_tokens = self.apply_to_token_lst(tokens,
                                                     use_keywords=False,
                                                     use_stopwords=use_stopwords,
                                                     use_tags=True)
            cleaned_lines += cleaned_tokens
        return cleaned_lines

    def filter_single_token(self, token, use_keywords=True, use_stopwords=True, use_tags=True):
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