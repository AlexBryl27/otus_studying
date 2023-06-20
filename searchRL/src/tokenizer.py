import re
from urllib.parse import unquote
from nltk.stem import SnowballStemmer


class Tokenizer:

    def __init__(
        self,
        path_to_stopwords=None,
        filter_stopwords=False,
        stem_tokens=True,
    ):

        self.stopwords = []
        if path_to_stopwords:
            with open(path_to_stopwords, 'r') as f:
                self.stopwords = f.read().split('\n')

        self.filter_stopwords = filter_stopwords
        self.stem_tokens = stem_tokens
        self.stemmer = SnowballStemmer(language="russian")

        self.SUB_PATTERN = re.compile("[^а-яёa-z0-9]")

    def tokenize(self, s):

        s = unquote(s)
        tokens = self._splitting(s)
        if self.filter_stopwords:
            tokens = self._filtering(tokens)
        if self.stem_tokens:
            tokens = self._stemming(tokens)

        return tokens

    def _splitting(self, s):
        s = self.SUB_PATTERN.sub(" ", s.lower())
        s = re.sub('\\s{2,}', ' ', s).strip()
        return s.split()

    def _stemming(self, tokens):

        return [self.stemmer.stem(token) for token in tokens]

    def _filtering(self, tokens):

        return [token for token in tokens if token not in self.stopwords]
