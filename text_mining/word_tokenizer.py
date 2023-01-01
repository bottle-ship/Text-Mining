__author__ = ["Byungseon Choi"]

import itertools
import typing as t

from PyKomoran import Komoran
from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import sent_tokenize


class WordTokenizer(object):

    def __init__(
            self,
            stopwords: t.Optional[t.Sequence[str]] = None,
            passtags: t.Optional[t.Sequence[str]] = ("NNG", "NNP", "SL")
    ):
        self.stopwords = list()
        if stopwords is not None:
            self.stopwords.extend(list(stopwords))
        self.passtags = passtags

        self._en_stopwords = list(set(nltk_stopwords.words("english")))  # noqa
        self._ko_tag = Komoran(model_path="EXP")
        # self._ko_tag.set_user_dic()

    def tokenize(self, document: str, flatten: bool = True) -> t.List[t.List[t.Union[t.Tuple[str, str], str]]]:
        tokens = list()
        for sentence in sent_tokenize(document):
            tokens.append(list())
            for token_tag in [token.split("/") for token in self._ko_tag.get_plain_text(sentence).split(" ")]:
                tag = token_tag[-1]
                token = "/".join(token_tag[:-1])

                flag = True
                if tag == "SL" and token.lower() in self._en_stopwords:
                    flag = False
                if token in self.stopwords:
                    flag = False
                if tag not in self.passtags:
                    flag = False

                if flag:
                    if self.passtags is None:
                        tokens[-1].append((token, tag))
                    else:
                        tokens[-1].append(token)

        if flatten:
            tokens = list(itertools.chain.from_iterable(tokens))

        return tokens
