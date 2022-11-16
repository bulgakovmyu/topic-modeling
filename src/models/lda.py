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


class LDAModel(object):
    def __init__(self, framework, bigrams=True):
        self.framework = framework
        self.model = None
        self.bigrams = bigrams
        self.bigram_words = None
        self.corpus = None
        self.id2word = None

    @staticmethod
    def create_bigrams(data_words, min_count=10, threshold=100):
        bigram = Phrases(data_words, min_count=min_count, threshold=threshold)
        bigram_mod = Phraser(bigram)
        return [bigram_mod[doc] for doc in data_words]

    def make_corpus(self, data_words, min_count=10, threshold=100):
        self.bigram_words = self.create_bigrams(
            data_words, min_count=min_count, threshold=threshold
        )
        self.id2word = corpora.Dictionary(self.bigram_words)
        self.corpus = [self.id2word.doc2bow(text) for text in self.bigram_words]

    def init_model(self, num_topics):
        self.model = LdaModel(
            corpus=self.corpus,
            id2word=self.id2word,
            num_topics=num_topics,
            random_state=100,
            update_every=1,
            chunksize=100,
            passes=10,
            alpha="auto",
            per_word_topics=True,
        )

    def return_model_params(self, num_of_topics, num_of_words):
        topics = self.model.print_topics(
            num_topics=num_of_topics, num_words=num_of_words
        )
        return [
            {
                "weights": [float(x.split("*")[0]) for x in topics[i][1].split(" + ")],
                "words": [x.split("*")[1][1:-1] for x in topics[i][1].split(" + ")],
            }
            for i in range(num_of_topics)
        ]

    def compute_coherence_values(self, stop, start=2, step=3, coherence="c_v"):
        coherence_values = []
        model_list = []
        for num_of_topics in tqdm(range(start, stop, step)):
            self.init_model(num_of_topics)
            model_list.append(self.model)
            coherencemodel = CoherenceModel(
                model=self.model,
                texts=self.bigram_words,
                dictionary=self.id2word,
                coherence=coherence,
            )
            coherence_values.append(coherencemodel.get_coherence())
        return model_list, coherence_values

    def plot_coherence_graph(self, start, stop, step):
        _, coherence_values = self.compute_coherence_values(stop, start, step)
        x = range(start, stop, step)
        plt.plot(x, coherence_values)
        plt.xlabel("Number of Topics")
        plt.ylabel("Coherence score")
        plt.legend(("coherence_values"), loc="best")
        plt.show()
