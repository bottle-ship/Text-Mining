__author__ = ["Byungseon Choi"]

import os
import typing as t
from pathlib import Path

import matplotlib.pyplot as plt
from wordcloud import WordCloud


def draw_word_cloud(frequencies: t.Dict[str, float]):
    wordcloud = WordCloud(
        font_path=os.path.join(Path(os.path.abspath(__file__)).parents[0], "한글틀고딕.ttf"),
        width=800,
        height=800,
        background_color="white",
    )
    wc = wordcloud.generate_from_frequencies(frequencies=frequencies)

    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.imshow(wc.to_array(), interpolation="bilinear")
    plt.tight_layout()
    plt.show()
