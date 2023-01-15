__author__ = ["Byungseon Choi"]

import itertools
import typing as t

from PyKomoran import Komoran
from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import sent_tokenize


class WordTokenizer(object):

    def __init__(self, stopwords: t.Optional[t.Sequence[str]] = None):
        self.stopwords = list()
        if stopwords is not None:
            self.stopwords.extend(list(stopwords))

        self._passtags = ("NNG", "NNP", "NNB", "SL")
        self._en_stopwords = list(set(nltk_stopwords.words("english")))  # noqa
        self._ko_tag = Komoran(model_path="EXP")

    def tokenize(self, document: str, flatten: bool = True) -> t.List[t.List[t.Union[t.Tuple[str, str], str]]]:
        tokens = list()
        for sentence in sent_tokenize(document):
            tokens.append([list()])
            for token_tag in [token.split("/") for token in self._ko_tag.get_plain_text(sentence).split(" ")]:
                tag = token_tag[-1]
                token = "/".join(token_tag[:-1])

                flag = True
                if tag == "SL" and token.lower() in self._en_stopwords:
                    flag = False
                if token in self.stopwords:
                    flag = False
                if tag not in self._passtags:
                    flag = False

                if flag:
                    tokens[-1][-1].append(token)
                else:
                    if len(tokens[-1][-1]) > 0:
                        tokens[-1].append(list())

            if len(tokens[-1][-1]) == 0:
                tokens[-1].pop()

            for i in range(0, len(tokens[-1])):
                if len(tokens[-1][i]) > 1:
                    tokens[-1][i].append("-".join(tokens[-1][i]))

            tokens[-1] = list(itertools.chain.from_iterable(tokens[-1]))

        if flatten:
            tokens = list(itertools.chain.from_iterable(tokens))

        return tokens
