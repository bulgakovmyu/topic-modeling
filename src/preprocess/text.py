import re
import itertools
from turtle import st
import contractions
from gensim.parsing.preprocessing import remove_stopwords
import nltk
from nltk.stem import (
    PorterStemmer,
    LancasterStemmer,
    WordNetLemmatizer,
    SnowballStemmer,
)
from nltk.corpus import words
from nltk.corpus import stopwords
import pandas as pd
from pandarallel import pandarallel
from time import time
from gensim import corpora
from sklearn.model_selection import train_test_split

from pymystem3 import Mystem
from nltk.stem import WordNetLemmatizer

lemm = WordNetLemmatizer()
mystem = Mystem()

words_list = words.words()
RANDOM_SEED = 100

stop_words = stopwords.words("russian")


class Corpus(object):
    def prepare(self, series: pd.Series):
        dic = corpora.Dictionary(series.to_list())
        doc_term_matrix = [dic.doc2bow(doc) for doc in series.to_list()]
        return series.to_list(), dic, doc_term_matrix


class TextDataProcessor(object):
    def __init__(self, filepath, language, sample_n):
        self.filepath = filepath
        self.language = language
        self.sample_n = sample_n
        pandarallel.initialize()

    def read_textdata(self):
        with open(self.filepath, "r") as f:
            return pd.read_csv(f).sample(n=self.sample_n, random_state=RANDOM_SEED)

    def run(
        self,
        text_field,
        columns=None,
        norm_type="stemming",
        subtype="porter",
        simple_preprocess=True,
        for_bert=False,
        load_ready=True,
        save=False,
    ):
        if norm_type == "lemma":
            subtype = ""

        if load_ready:
            type_of_preprocess = "simple_" if simple_preprocess else ""
            bert_prefix = "for_bert_" if for_bert else ""
            return pd.read_csv(
                f"../data/{text_field}_{type_of_preprocess}preprocessed_{bert_prefix}{norm_type}_{subtype}.csv"
            )
        if self.language == "en":
            nltk.download("wordnet")
            nltk.download("omw-1.4")
            nltk.download("words")
        dataframe = self.drop_empty_arrays_data(
            self.make_normalization(
                self.clean_text_as_list(
                    self.clean_text_as_string(
                        self.read_textdata()[columns]
                        if columns
                        else self.read_textdata().dropna(subset=[text_field]),
                        column_to_clean=text_field,
                    ),
                    column_to_clean=text_field,
                    language=self.language,
                    simple=simple_preprocess,
                    for_bert=for_bert,
                ),
                column_to_clean=text_field,
                norm_type=norm_type,
                subtype=subtype,
            ),
            column_to_clean=text_field,
        )
        if save:
            type_of_preprocess = "simple_" if simple_preprocess else ""
            bert_prefix = "for_bert_" if for_bert else ""
            name = f"../data/{text_field}_{type_of_preprocess}preprocessed_{bert_prefix}{norm_type}_{subtype}.csv"
            dataframe[[f"{text_field}_array_norm"]].to_csv(name)
        return dataframe[[f"{text_field}_array_norm"]]

    @staticmethod
    def clean_text_as_string(dataframe, column_to_clean):
        start = time()
        regex_pat = re.compile(r"[^\w\s\']", flags=re.IGNORECASE)
        dataframe[f"{column_to_clean}_cleaned"] = (
            (
                dataframe[column_to_clean]
                .str.replace(
                    r'https?://[^\s<>"]+|www\.[^\s<>"]+', "", regex=True
                )  # remove web addresses
                .str.replace(r"(\s[\'\"])|([\'\"]\s)", " ", regex=True)  # remove quotes
                .str.replace(
                    regex_pat, "", regex=True
                )  # leave only letters, spaces and apostrophes
                .str.replace(r"\d", "", regex=True)
                .str.replace("\n", " ", regex=False)  # remove next line characters
                .str.replace(r"\s+", " ", regex=True)  # remove multiple spaces
                .str.replace("_", " ", regex=False)
            )
            .str.lower()  # lower case
            .str.strip()  # remove spaces from borders
        )

        print("String cleaning techniques done :::", time() - start)
        return dataframe.dropna(subset=[f"{column_to_clean}_cleaned"])

    @staticmethod
    def clean_text_as_list(
        dataframe, column_to_clean, language, simple=True, for_bert=False
    ):
        start = time()
        if language == "en":
            if simple:
                dataframe[f"{column_to_clean}_array"] = (
                    dataframe[f"{column_to_clean}_cleaned"]
                    .str.split(" ")
                    .parallel_apply(drop_unknown_and_stop_word)
                )
            else:

                if for_bert:
                    dataframe[f"{column_to_clean}_array"] = (
                        dataframe[f"{column_to_clean}_cleaned"]
                        .str.split(" ")
                        .parallel_apply(transform_text_list_for_bert)
                    )
                else:
                    dataframe[f"{column_to_clean}_array"] = (
                        dataframe[f"{column_to_clean}_cleaned"]
                        .str.split(" ")
                        .parallel_apply(transform_text_list)
                    )
        elif language == "rus":
            dataframe[f"{column_to_clean}_array"] = (
                dataframe[f"{column_to_clean}_cleaned"]
                .str.split(" ")
                .parallel_apply(transform_text_list_rus)
            )
        else:
            raise ValueError("Unknown language to process: %s" % language)

        print("List cleaning techniques done :::", time() - start)
        return dataframe

    @staticmethod
    def make_normalization(dataframe, column_to_clean, norm_type, subtype):
        start = time()
        if not norm_type:
            dataframe[f"{column_to_clean}_array_norm"] = dataframe[
                f"{column_to_clean}_array"
            ].apply(lambda lst: " ".join(lst))
            return dataframe

        if norm_type == "stemming":
            if subtype == "porter":
                stemmer = PorterStemmer()
            elif subtype == "lancaster":
                stemmer = LancasterStemmer()
            elif subtype == "russian":
                stemmer = SnowballStemmer("russian")
            else:
                raise ValueError("Unknown stemmer type: %s" % subtype)
            dataframe["norm_type"] = norm_type + "_" + subtype
            dataframe[f"{column_to_clean}_array_norm"] = dataframe[
                f"{column_to_clean}_array"
            ].parallel_apply(lambda lst: " ".join([stemmer.stem(x) for x in lst]))

        elif norm_type == "lemma":
            dataframe["norm_type"] = norm_type
            dataframe[f"{column_to_clean}_array_norm"] = dataframe[
                f"{column_to_clean}_array"
            ].parallel_apply(lemmatize_all)
        else:
            raise ValueError("Unknown normalizing type: %s" % norm_type)
        print("Normalisation done :::", time() - start)
        return dataframe

    @staticmethod
    def drop_empty_arrays_data(dataframe, column_to_clean):
        return dataframe[
            dataframe[f"{column_to_clean}_array_norm"].apply(lambda lst: len(lst) > 0)
        ]


def transform_text_list_rus(text_list):
    text_list = remove_repeating_patterns(text_list)
    text_list = remove_stopwords_from_text_list_en_rus(text_list)
    text_list = remove_too_short_words(text_list)
    text_list = remove_too_long_words(text_list)

    return text_list


def transform_text_list(text_list):
    text_list = remove_non_latin_characters(text_list)
    text_list = fix_contractions(text_list)
    text_list = remove_mult_equal_letters(text_list)
    text_list = remove_repeating_patterns(text_list)
    text_list = remove_stopwords_from_text_list(text_list)
    text_list = remove_too_short_words(text_list)
    text_list = remove_too_long_words(text_list)

    return text_list


def transform_text_list_for_bert(text_list):
    text_list = remove_non_latin_characters(text_list)
    text_list = fix_contractions(text_list)
    text_list = remove_mult_equal_letters(text_list)
    text_list = remove_repeating_patterns(text_list)

    return text_list


def fix_contractions(text_list):
    #  fix contractions
    return [
        contractions.fix(word)
        for word in text_list
        if not "'" in contractions.fix(word)
    ]


def remove_mult_equal_letters(text_list):
    # remove multiple equal letters
    return [
        word if word in words_list else "".join([c[0] for c in itertools.groupby(word)])
        for word in text_list
    ]


def remove_repeating_patterns(text_list):
    # remove reapeating patterns
    return [re.sub(r"\b(.+?)\1+|(.+?)\1+", r"\1", word) for word in text_list]


def remove_non_latin_characters(text_list):
    # remove non-latin characters
    return [re.sub(r"[^\x00-\x7f]", "", word) for word in text_list]


def remove_stopwords_from_text_list(text_list):
    return remove_stopwords(" ".join(text_list)).split(" ")


def remove_stopwords_from_text_list_rus(text_list):
    return [i for i in text_list if i not in stop_words]


def remove_stopwords_from_text_list_en_rus(text_list):
    return [
        i
        for i in remove_stopwords(" ".join(text_list)).split(" ")
        if i not in stop_words
    ]


def remove_too_short_words(text_list, short_thld=1):
    return [word for word in text_list if len(word) > short_thld]


def remove_too_long_words(text_list, long_thld=20):
    return [word for word in text_list if len(word) < long_thld]


def drop_unknown_and_stop_word(lst, voc=words_list):
    return [
        remove_stopwords(word)
        for word in lst
        if (word in voc and len(remove_stopwords(word)) > 2)
    ]


def lemmatize_all(string_list):
    return " ".join([lemm.lemmatize(mystem.lemmatize(x)[0]) for x in string_list])
