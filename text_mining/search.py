__author__ = ["Byungseon Choi"]

import pandas as pd
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity


def search_similar_documents(
        doc_index: int,
        score_matrix: pd.DataFrame,
        n_components: int = 20
) -> pd.DataFrame:
    nmf = NMF(n_components=n_components)
    features = nmf.fit_transform(score_matrix)

    normalizer = Normalizer()
    norm_features = normalizer.fit_transform(features)

    df_features = pd.DataFrame(norm_features, index=score_matrix.index.tolist())

    score = cosine_similarity(df_features.loc[doc_index, :].values.reshape(1, -1), df_features.values)
    similarity = pd.DataFrame(score.T, columns=["score"], index=score_matrix.index.tolist())
    similarity = similarity.sort_values(by=["score"], ascending=False)

    return similarity
