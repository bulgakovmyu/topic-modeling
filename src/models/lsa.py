from gensim.models import LsiModel
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
from tqdm import tqdm


class LSAModel(object):
    def __init__(self, framework):
        self.framework = framework
        self.model = None

    def init_model(self, dictionary, doc_term_matrix, num_of_topics):

        if self.framework == "gensim":
            self.model = LsiModel(
                doc_term_matrix, num_topics=num_of_topics, id2word=dictionary
            )

    def print_model(self, num_of_topics, num_of_words):
        return self.model.print_topics(num_topics=num_of_topics, num_words=num_of_words)

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

    def compute_coherence_values(
        self,
        dictionary,
        doc_term_matrix,
        doc_clean,
        stop,
        start=2,
        step=3,
        coherence="c_v",
    ):
        coherence_values = []
        model_list = []
        for num_of_topics in tqdm(range(start, stop, step)):
            self.init_model(dictionary, doc_term_matrix, num_of_topics)
            model_list.append(self.model)
            coherencemodel = CoherenceModel(
                model=self.model,
                texts=doc_clean,
                dictionary=dictionary,
                coherence=coherence,
            )
            coherence_values.append(coherencemodel.get_coherence())
        return model_list, coherence_values

    def plot_coherence_graph(
        self, doc_clean, dictionary, doc_term_matrix, start, stop, step, coherence="c_v"
    ):
        _, coherence_values = self.compute_coherence_values(
            dictionary, doc_term_matrix, doc_clean, stop, start, step, coherence
        )
        x = range(start, stop, step)
        plt.plot(x, coherence_values)
        plt.xlabel("Number of Topics")
        plt.ylabel("Coherence score")
        plt.legend(("coherence_values"), loc="best")
        plt.show()
