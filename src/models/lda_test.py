# %%
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.preprocess.text import TextDataProcessor, Corpus
from src.models.lsa import LSAModel
import pandas as pd
from gensim import corpora

# %%
df_text = pd.read_csv(
    "/Users/Mikhail_Bulgakov/GitRepo/topic-modeling/data/clean_title_preprocessed_None_russian.csv"
)
# %%
data_words = df_text["clean_title_array_norm"].str.split(" ").to_list()
# %%
# Build the bigram and trigram models
bigram = Phrases(
    data_words, min_count=5, threshold=100
)  # higher threshold fewer phrases.
# trigram = Phrases(bigram[data_words], threshold=100)
# %%
bigram

# %%
# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = Phraser(bigram)
# trigram_mod = Phraser(trigram)

# %%
print(bigram_mod[data_words[0]])
# %%
def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]


# %%

# Form Bigrams
data_words_bigrams = make_bigrams(data_words)
# %%
id2word = corpora.Dictionary(data_words_bigrams)
# %%
texts = data_words_bigrams
# %%
corpus = [id2word.doc2bow(text) for text in texts][:2000]
# %%
print(corpus[:1])
# %%
lda_model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    num_topics=20,
    random_state=100,
    update_every=1,
    chunksize=100,
    passes=10,
    alpha="auto",
    per_word_topics=True,
)

# %%
lda_model.print_topics()
# %%
