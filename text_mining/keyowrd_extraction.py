__author__ = ["Byungseon Choi"]

import typing as t

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from soykeyword.proportion import MatrixbasedKeywordExtractor
from soykeyword.proportion._proportion import KeywordScore  # noqa


class ProportionBasedKeywordExtractor(object):

    def __init__(
            self,
            corpus: t.List[str],
            tokenizer: t.Callable,
            stop_words: t.Optional[t.List[str]] = None,
            ngram_range: t.Tuple[int, int] = (1, 1),
            min_tf: int = 20,
            min_df: int = 2
    ):
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.stop_words = stop_words
        self.ngram_range = ngram_range
        self.min_tf = min_tf
        self.min_df = min_df

        self._keyword_extractor = self._set_keyword_extractor()

    def extract_keyword_from_doc_index(
            self,
            index: int,
            min_score: float = 0.8,
            min_frequency: int = 20
    ) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
        doc_keywords = self._keyword_extractor.extract_from_docs(
            [index],
            min_score=min_score,
            min_frequency=min_frequency
        )
        frequency = dict()
        score = dict()
        for keyword in doc_keywords:
            word = keyword.word
            frequency[word] = [keyword.frequency]
            score[word] = [keyword.score]
        frequency = pd.DataFrame.from_dict(frequency)
        score = pd.DataFrame.from_dict(score)

        return frequency, score

    def extract_keyword_from_corpus(
            self,
            min_score: float = 0.8,
            min_frequency: int = 20
    ) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
        doc_index = list()
        frequency = list()
        score = list()
        for i in range(0, len(self.corpus)):
            doc_frequency, doc_score = self.extract_keyword_from_doc_index(
                index=i,
                min_score=min_score,
                min_frequency=min_frequency
            )
            if len(doc_frequency) > 0:
                doc_index.append(i)
                frequency.append(doc_frequency)
                score.append(doc_score)
        frequency = pd.concat(frequency, ignore_index=True)
        frequency.index = doc_index
        score = pd.concat(score, ignore_index=True)
        score.index = doc_index

        return frequency, score

    def extract_keyword_from_word(
            self,
            word: str,
            min_score: float = 0.8,
            min_frequency: int = 20
    ) -> pd.DataFrame:
        keywords = self._keyword_extractor.extract_from_word(
            word=word,
            min_score=min_score,
            min_frequency=min_frequency
        )

        keyword_score = dict()
        for keyword in keywords:
            keyword_score[keyword.word] = [keyword.frequency, keyword.score]
        keyword_score = pd.DataFrame.from_dict(keyword_score)
        keyword_score.index = ["frequency", "score"]

        return keyword_score

    def _set_keyword_extractor(self):
        count_vectorizer = CountVectorizer(
            tokenizer=self.tokenizer,
            token_pattern=None,
            ngram_range=self.ngram_range,
            stop_words=self.stop_words
        )
        x = count_vectorizer.fit_transform(self.corpus)

        index2word = [feature_name.replace(" ", "-") for feature_name in count_vectorizer.get_feature_names_out()]

        keyword_extractor = MatrixbasedKeywordExtractor(
            min_tf=self.min_tf,
            min_df=self.min_df,
            verbose=False
        )
        keyword_extractor.train(x, index2word=index2word)

        return keyword_extractor
