# %%
import pandas as pd

# %%
df_text = pd.read_csv(
    "/Users/Mikhail_Bulgakov/GitRepo/topic-modeling/data/clean_text_preprocessed_None_russian.csv"
)
# %%
df_text
# %%
from pymystem3 import Mystem
from nltk.stem import WordNetLemmatizer

# %%
lemm = WordNetLemmatizer()
m = Mystem()
# %%
def lemmatize_rus(string):
    return " ".join([x for x in m.lemmatize(string) if x not in [" ", "\n"]])


# %%
def lemmatize_en(string):
    return " ".join([lemm.lemmatize(x) for x in string.split(" ")])


# %%
def lemmatize_all(string):
    return " ".join([lemm.lemmatize(m.lemmatize(x)[0]) for x in string.split(" ")])


# %%
text = "следующем году лет момента создания worlds rolling well"
print(lemmatize_rus(text))
print(lemmatize_en(text))
print(lemmatize_all(text))
# %%
from tqdm.notebook import tqdm

tqdm.pandas()
# %%
test = df_text.iloc[:100]

# %%
test["clean_text_array_norm"].progress_apply(lemmatize_all)
# %%
