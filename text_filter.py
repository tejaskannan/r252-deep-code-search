import re
from dpu_utils import codeutils
from html.parser import HTMLParser


class TextFilter:

    def __init__(self, tags_file, stopwords_file):
        self.java_keywords = set(codeutils.get_language_keywords('java'))
        self.javadoc_tags = self._load_from_file(tags_file)
        self.stopwords = self._load_from_file(stopwords_file)
        self.token_regex = re.compile(r'[\{\}\[\]\(\)\*\n\t,><\\]')
        self.alpha_numeric = re.compile(r'[^a-zA-Z0-9]')
        self.camel_case = re.compile(r'([a-z])([A-Z])')

    # Accepts a single method name (string) and returns a list of cleaned tokens
    def apply_to_method_name(self, method_name):
        return [t.lower() for t in self.split_camel_case(method_name.strip())]

    def apply_to_api_calls(self, api_calls, lowercase_api):
        if lowercase_api:
            return [call.strip().lower() for call in api_calls if len(call.strip()) > 0]
        return [call.strip() for call in api_calls if len(call.strip()) > 0]

    # Accepts a list of method tokens and returns a cleaned list
    def apply_to_token_lst(self, tokens, use_keywords=True, use_stopwords=True, use_tags=False):
        split_on_camel_case = []
        for token in tokens:
            t = token.strip()
            if len(t) > 0:
                split_on_camel_case += self.split_camel_case(t)

        split_on_underscore = []
        for token in split_on_camel_case:
            if len(token) > 0:
                split_on_underscore += token.split('_')

        cleaned_tokens = [self.alpha_numeric.sub('', t).lower() for t in split_on_underscore]
        return set(filter(lambda t: self.filter_single_token(t, use_keywords, use_stopwords, use_tags),
                           cleaned_tokens))

    def apply_to_javadoc(self, javadoc_text, use_stopwords=True, use_tags=True):
        lines = javadoc_text.split('\n')
        cleaned_lines = []

        html_parser = JavadocHTMLParser()
        for line in lines:
            tokens = line.strip().split()
            if len(tokens) == 0:
                continue

            # We omit lines which start with javadoc tags.
            if use_tags and tokens[0] in self.javadoc_tags:
                continue

            tokens = [t.lower() for t in tokens]
            filtered_tokens = list(filter(lambda t: self.filter_single_token(t, False, use_stopwords, use_tags),
                                          tokens))

            stripped_tokens = []
            for token in filtered_tokens:
                html_parser.feed(token)
                stripped_tokens.append(html_parser.get_data())
                html_parser.reset()

            cleaned_tokens = []
            for token in stripped_tokens:
                token = token.strip()
                if len(token) == 0:
                    continue

                for tag in self.javadoc_tags:
                    if tag in token:
                        token = token.replace(tag, '')
                        break
                token = self.alpha_numeric.sub('', token)
                if len(token) > 0:
                    cleaned_tokens.append(token)

            cleaned_lines += cleaned_tokens
        return cleaned_lines

    def filter_single_token(self, token, use_keywords=True, use_stopwords=True, use_tags=True):
        token = token.strip()
        if len(token) == 0:
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
        with open(file_name, 'r') as token_file:
            for token in token_file:
                tokens.add(token.strip())
        return tokens

    def split_camel_case(self, text):
        return self.camel_case.sub(r'\1 \2', text).split()


class JavadocHTMLParser(HTMLParser):

    def __init__(self):
        super().__init__()
        self.reset()
        self.contents = []

    def handle_data(self, data):
        self.contents.append(data)

    def get_data(self):
        return ''.join(self.contents)

    def reset(self):
        super().reset()
        self.contents = []
