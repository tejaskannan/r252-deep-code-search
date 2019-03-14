import re
from dpu_utils import codeutils
from html.parser import HTMLParser


class TextFilter:
    """Class which filters token sequences."""

    def __init__(self, tags_file, stopwords_file):

        # Fetch list of Java keyworkds, Javadoc tags, and English Stopwords
        self.java_keywords = set(codeutils.get_language_keywords('java'))
        self.javadoc_tags = self._load_from_file(tags_file)
        self.stopwords = self._load_from_file(stopwords_file)

        self.alpha_numeric = re.compile(r'[^a-zA-Z0-9]')
        self.camel_case = re.compile(r'([a-z])([A-Z])')

    def apply_to_method_name(self, method_name):
        """
        Returns lowercase method name tokens which are formed from splitting the given name
        based on camel case.
        """
        return [t.lower() for t in self.split_camel_case(method_name.strip())]

    def apply_to_api_calls(self, api_calls, should_lowercase, should_subtokenize=False):
        """
        Returns a flattened list of cleaned API call tokens given a list of API calls.
        All tokens are lowercased and all whitespace is removed. The subtokenize flag denotes
        whether or not API calls should be split based on camel casing.
        """
        if should_subtokenize:
            api_calls = self._split_token_lst(api_calls, split_chars='_.', use_camel_case=True)
        if should_lowercase:
            return [call.strip().lower() for call in api_calls if len(call.strip()) > 0]
        return [call.strip() for call in api_calls if len(call.strip()) > 0]

    def apply_to_token_lst(self, tokens, use_keywords=True, use_stopwords=True, use_tags=False):
        """
        Returns a cleaned set of method tokens. All Java keywords and English stopwords are removed,
        and all tokens are lowercased.
        """
        split_lst = self._split_token_lst(tokens, split_chars='_', use_camel_case=True)
        cleaned_tokens = [self.alpha_numeric.sub('', t).lower() for t in split_lst]
        return set(filter(lambda t: self.filter_single_token(t, use_keywords, use_stopwords, use_tags),
                          cleaned_tokens))

    def apply_to_javadoc(self, javadoc_text, use_stopwords=True, use_tags=True):
        """
        Returns a list of cleaned lines for the given Javadoc text. All javadoc and HTML tags are removed,
        all tokens are lowercased, all English stopwords are omitted and all non-alphanumeric
        characters are removed. 
        """
        lines = javadoc_text.split('\n')
        cleaned_lines = []

        html_parser = JavadocHTMLParser()
        for line in lines:
            tokens = line.strip().split()
            if len(tokens) == 0:
                continue

            # Omit lines which start with javadoc tags.
            if use_tags and tokens[0] in self.javadoc_tags:
                continue

            # Filter out tokens which are English topwords or Javadoc tags
            tokens = [t.lower() for t in tokens]
            filtered_tokens = list(filter(lambda t: self.filter_single_token(t, False, use_stopwords, use_tags),
                                          tokens))

            # Remove all HTML tags
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

                # Remove all Javadoc tags within a given line. This scenario happens
                # with the @code tag.
                for tag in self.javadoc_tags:
                    if tag in token:
                        token = token.replace(tag, '')
                        break

                # Only keep alphanumeric characters.
                token = self.alpha_numeric.sub('', token)
                if len(token) > 0:
                    cleaned_tokens.append(token)

            cleaned_lines += cleaned_tokens
        return cleaned_lines

    def filter_single_token(self, token, use_keywords=True, use_stopwords=True, use_tags=True):
        """Returns True if the given token passes the filter"""
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
        """Returns a set of tokens from the file where each token is on its own line."""
        tokens = set()
        with open(file_name, 'r') as token_file:
            for token in token_file:
                tokens.add(token.strip())
        return tokens

    def _split_token_lst(self, token_lst, split_chars, use_camel_case):
        """
        Returns a flattened list of tokens from the token_lst which are
        split based on the characters in split_chars as well as on camel case (if true)
        """
        if use_camel_case:
            split_lst = []
            for token in token_lst:
                split_lst += self.split_camel_case(token)
            token_lst = split_lst

        split_lst = []
        for char in split_chars:
            for token in token_lst:
                split_lst += token.split(char)
            token_lst = split_lst
            split_lst = []
        return token_lst

    def split_camel_case(self, text):
        """Returns a list of tokens after splitting the text based on camel case."""
        return self.camel_case.sub(r'\1 \2', text).split()


class JavadocHTMLParser(HTMLParser):
    """HTML Parser subclass used to remove HTML tags from Javadoc comments."""

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
