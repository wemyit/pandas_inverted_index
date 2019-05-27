"""
Build simple inverted index using pandas
"""

import pandas as pd

# Tweets dataset from here https://github.com/hafidhfikri/Practice-Twitter-Sentiment-Analysis/blob/master/train_E6oV3lV.csv
df = pd.read_csv('text.zip')

WORD_INDEX = (
    df[['id', 'tweet']]
    .set_index('id')
    .tweet.str.replace(r'\W+', ' ')
    .str.lower()
    .str.split().apply(pd.Series)
    .unstack().dropna()
)

WORD_INDEX.name = 'word'


def occurrence(x):
    return set(x.values)


occurrence.__name__ = 'occurrence'

WORD_INDEX = WORD_INDEX.reset_index('id').set_index('word')

WORD_INDEX = (
    WORD_INDEX
    .groupby('word')
    .agg({'id': ['count', occurrence]})
    .id.reset_index()
)


def matching_tweet_ids(words, percent=75):
    """
    Finds ids of the words in index with specific probability
    """

    found = WORD_INDEX[
        WORD_INDEX.word.isin(words)
    ].set_index('word')

    if found.empty:
        return pd.Series()

    frequency = found[['count']].T

    total = frequency.values.sum()

    frequency = total - frequency

    # only one result
    frequency[frequency == 0] = total

    total = frequency.values.sum()

    probability = (frequency / total) * 100

    ids = pd.Series(
        found.occurrence
        .apply(lambda x: pd.Series(list(x)))
        .unstack().dropna().unique()
    )

    found_words = found[['occurrence']].T

    occurrences = pd.DataFrame({
        column: (
            ids.apply(
                lambda x: x in found_words[column].occurrence
            )
            .astype(int)
        )
        for column in found_words.columns
    })

    occurrences['total'] = (occurrences * probability.values).sum(axis=1)

    occurrences['id'] = ids

    return occurrences[occurrences.total >= percent].id


matching_tweets = df[
    df.id.isin(
        matching_tweet_ids('phone sit an'.split())
    )
].tweet
