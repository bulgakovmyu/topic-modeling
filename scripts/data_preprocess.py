# %%
from src.preprocess.text import TextDataProcessor

# %%
# dp = TextDataProcessor(
#     "/Users/Mikhail_Bulgakov/GitRepo/topic-modeling/data/recipes.csv", language="rus"
# )
# # %%
# dp.run(
#     text_field="ingredients",
#     columns=["ingredients", "category"],
#     norm_type=None,
#     subtype="russian",
#     simple_preprocess=False,
#     load_ready=False,
#     save=True,
# )
# # %%
# dp.run(
#     text_field="ingredients",
#     columns=["ingredients", "category"],
#     norm_type="stemming",
#     subtype="russian",
#     simple_preprocess=False,
#     load_ready=False,
#     save=True,
# )
# # %%
# dp.run(
#     text_field="steps",
#     columns=["steps", "category"],
#     norm_type=None,
#     subtype="russian",
#     simple_preprocess=False,
#     load_ready=False,
#     save=True,
# )
# # %%
# dp.run(
#     text_field="steps",
#     columns=["steps", "category"],
#     norm_type="stemming",
#     subtype="russian",
#     simple_preprocess=False,
#     load_ready=False,
#     save=True,
# )
# %%
dp = TextDataProcessor(
    "/Users/Mikhail_Bulgakov/GitRepo/topic-modeling/data/posts_1.csv",
    language="rus",
    sample_n=43866,
)
# # %%
# dp.run(
#     text_field="clean_title",
#     columns=["clean_title"],
#     norm_type="lemma",
#     subtype="russian",
#     simple_preprocess=False,
#     load_ready=False,
#     save=True,
# )
# %%
dp.run(
    text_field="clean_text_title",
    columns=["clean_text_title"],
    norm_type="lemma",
    subtype="russian",
    simple_preprocess=False,
    load_ready=False,
    save=True,
)
# %%
