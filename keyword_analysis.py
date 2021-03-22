# %% [markdown]
# # Keyword Analysis of CHI Best Papers 2016-2020

# In this analysis, we will collect and chart the author keywords used for the Best Paper winners at the Conference on Human Factors in Computing Systems (CHI) from 2016 to 2020.
# The aim of this research is to identify commonalities between those top 1% of articles honoured by the Best Paper Committee and to explore signficant trends among those.

# %% [markdown]
# ## Reading in the data

# First, we import the necessary packages and read in the keyword data from the respective file.

# Excluded (missing author keywords):

# DOI | Title | Author(s)
# -|-|-|-
# https://doi.org/10.1145/2858036.2858075 | Object-Oriented Drawing | Xia et al. 2016
# https://doi.org/10.1145/2858036.2858063 | The Effect of Visual Appearance on the Performance of Continuous Sliders and Visual Analogue Scales | Matejka et al. 2016
# https://doi.org/10.1145/3313831.3376510 | Integrating the Digital and the Traditional to Deliver Therapy for Depression: Lessons from a Pragmatic Study | Stawarz et al. 2020
# https://doi.org/10.1145/3313831.3376244 | AutoGain: Gain Function Adaptation with Submovement Efficiency Optimization | Lee et al. 2020
# https://doi.org/10.1145/3313831.3376363 | Community Collectives: Low-tech Social Support for Digitally-Engaged Entrepreneurship | Hui et al. 2020
# https://doi.org/10.1145/3290605.3300771 | Saliency Deficit and Motion Outlier Detection in Animated Scatterplots | Veras and Collins 2019
# https://doi.org/10.1145/3290605.3300748 | Put Your Warning Where Your Link Is: Improving and Evaluating Email Phishing Warnings | Petelka et al. 2019
# https://doi.org/10.1145/3290605.3300256 | Impact of Contextual Factors on Snapchat Public Sharing | Habib et al. 2019
# https://doi.org/10.1145/3173574.3174226 | Cognitive Load Estimation in the Wild | Fridman et al. 2019
# https://doi.org/10.1145/3173574.3173797 | DataInk: Direct and Creative Data-Oriented Drawing | Xia et al. 2019
# https://doi.org/10.1145/3025453.3025912 | Same Stats, Different Graphs: Generating Datasets with Varied Appearance and Identical Statistics through Simulated Annealing | Matejka and Fitzmaurice 2017
# https://doi.org/10.1145/3025453.3025554 | Collection Objects: Enabling Fluid Formation and Manipulation of Aggregate Selections | Xia et al. 2017

# %%
# Imports
import pandas as pd

# Utility Functions
def string_to_list(list_as_string: str, delimiter=", ") -> list[str]:
    """A simple function that splits a list input as a string by the specified delimiter and returns the list."""
    return list_as_string.split(delimiter)

def print_dataframe_summary(dataframe: pd.DataFrame, include_description=False, head_nrows=10) -> None:
    """Uses pandas' info, head, and---depending on the description flag---describe methods to output information about a DataFrame."""
    print("Information about dataframe:")
    print(dataframe.info())
    #print()
    #print("Top 10 rows of dataframe:")
    #print(dataframe.head(n=head_nrows))
    if include_description:
        print()
        print("Description of dataframe:")
        print(dataframe.describe())

# %%
# Best Papers
best_papers = pd.read_csv("best_papers.csv", delimiter=";", converters={"keywords": string_to_list})
best_papers["is_best_paper"] = True

print_dataframe_summary(best_papers)
best_papers

# %%
# Honorable mentions
honorable_mentions = pd.read_csv("honorable_mentions.csv", delimiter=";", converters={"keywords": string_to_list})
honorable_mentions["is_best_paper"] = False

print_dataframe_summary(honorable_mentions)
honorable_mentions

# %%
papers = pd.concat([best_papers, honorable_mentions])
papers.rename(columns={"keywords": "original_keywords"}, inplace=True)

print_dataframe_summary(papers)
papers

# %% [markdown]
# ## Normalisation

# Next, since we are interested in individual keywords and their co-occurences, we normalise the DataFrame to have one keyword per row.

# %%
# Explode (have one row for every keyword)
keywords = papers.explode("original_keywords").rename(columns={"original_keywords": "original_keyword"})

print_dataframe_summary(keywords)
keywords

# %% [markdown]
# ## Pre-processing

# There are several possible pitfalls with the raw author keywords that have to be addressed.
# Therefore, we will pre-process the keywords using the following transformations:

# - Removing any possible leading and trailing whitespace
# - Transforming all keywords to their lowercase form
# - Replacing abbreviations which might be used in either short- or long-form (e.g. AI) with their long-form (e.g. artificial intelligence)
# - Replacing all dialect spellings (e.g. British English) with their American English forms (CHI allows any consistent dialect but American English is most common)

# %%
# Remove possible leading and trailing whitespace
keywords["original_keyword"].replace(to_replace="(^\s* | \s*$)", inplace=True)
keywords = keywords.reset_index(drop=True)

# Copy original keyword into keyword column
keywords["keyword"] = keywords["original_keyword"]

# Transform to lowercase
keywords["lowercase_keyword"] = keywords["keyword"].str.lower()

# Replace abbreviations
abbreviations = {
    "adi": "around-device interaction",
    "ccbt": "computerised cognitive behavioural therapy",
    "bci": "brain-computer interface",
    "rpg": "role-playing game",
    "fps": "first-person shooter",
    "adhd": "attention deficit hyperactivity disorder",
    "ssd": "sensory substitution devices",
    "mooc": "massive open online course",
    "lgbtq": "lesbian, gay, bisexual, transgender, and queer",
    "lgbtq+": "lesbian, gay, bisexual, transgender, queer, and other queer-identifying community",
    "lgbt": "lesbian, gay, bisexual, and transgender",
    "p2p": "peer-to-peer",
    "cam": "computer-aided manufacturing",
    "cad": "computer-aided design",
    "gui": "graphical user interface",
    "asha": "accredited social health activist",
    "chw": "community health workers",
    "cmc": "computer-mediated communication",
    "als": "amyotrophic lateral sclerosis",
    "wifi": "wi-fi",
    "tui": "tangible user interface",
    "cscw": "computer supported cooperative work",
    "scg": "seismocardiography",
    "ppg": "photoplethysmogram",
    "ptt": "pulse transit time",
    "ux": "user experience",
    "pnprpg": "pen and paper role playing game",
    "iot": "internet of things",
    "uav": "unmanned aerial vehicle",
    "hmd": "head-mounted display",
    "hcid": "human-computer interaction for developing contexts",
    "ores": "objective revision evaluation system",
    "anti-elab": "anti-extradition law amendment bill",
    "vsd": "value sensitive design",
    "sns": "social network services",
    "tv": "television",
    "ivr": "interactive voice response",
    "ecg": "electrocardiogram",
    "diy": "do-it-yourself",
    "ui": "user interface",
    "ipv": "intimate partner violence",
    "catme": "comprehensive assessment for team-member effectiveness",
    "stem": "science, technology, engineering, and mathematics",
    "rfid": "radio-frequency identification",
    "mhealth": "mobile health",
    "sel": "social-emotional learning",
    "nhst": "null hypothesis significance testing",
    "em": "electromagnetic",
    "ictd": "information communication technologies for development",
    "hci4d": "human-computer interaction for development",
    "cci": "child-computer interaction",
    "gis": "geographic information systems",
    "esm": "experience sampling method",
    "ema": "ecological momentary assessment",
    "vru": "vulnerable road user",
    "ehmi": "external human-machine-interface",
    "ai": "artificial intelligence",
    "ml": "machine learning",
    "aac": "alternative and augmentative communication",
    "nicu": "neonatal intensive care unit",
    "hci": "human-computer interaction",
    "ar": "augmented reality",
    "vr": "virtual reality"
}
regex_leading = r"(^|(?<=\/)|(?<=\s))"
regex_trailing = r"($|(?=\/)|(?=,)|(?=\s))"
regex_abbreviations = {f"{regex_leading}{key}{regex_trailing}": value.lower() for key, value in abbreviations.items()}
keywords["keyword"] = keywords["lowercase_keyword"].replace(to_replace=regex_abbreviations, regex=True)

# Replace dialect spellings
dialect_spellings = {
    "customisation": "customization",
}
keywords["keyword"].replace(to_replace=dialect_spellings, inplace=True)

keywords = keywords.drop("lowercase_keyword", axis=1)

print_dataframe_summary(keywords)
keywords

# %% [markdown]
# ## Merging back into original DataFrame

# After the pre-processing was successfully completed, we can reintroduce the keywords to the original papers DataFrame.

# %%
# Returning list of keywords back together into papers DataFrame
papers = papers.set_index("doi").join(keywords[["doi", "keyword"]].groupby("doi").agg({"keyword": lambda keyword: keyword.tolist()})).rename(columns={"keyword": "keywords"})
papers = papers.reset_index()

print_dataframe_summary(papers)
papers

# %% [markdown]
# ## Analysis

# First, we will look at some simple descriptive measures that characterise the dataset.
# Afterwards, we will start analyzing the keywords to explore relationships between papers.

# %% [markdown]
# ### Descriptive analysis

# Let us look at some descriptive measure first.

# %%
# Number of keywords per paper
papers["number_of_keywords"] = papers["keywords"].apply(lambda keywords_list: len(keywords_list))

print("Number of author keywords per paper:")
papers["number_of_keywords"].describe()

# %%
# Number of keywords per Best Paper / Honorable Mention
number_of_keywords_best_papers = papers[["is_best_paper", "number_of_keywords"]].groupby("is_best_paper").describe()
number_of_keywords_best_papers

# %%
# Number of keywords per paper per year
number_of_keywords_per_year = papers[["year", "number_of_keywords"]].groupby("year").describe()
number_of_keywords_per_year

# %%
# Number of keywords per paper per year
number_of_keywords_per_year_and_type = papers[["year", "is_best_paper", "number_of_keywords"]].groupby(["year", "is_best_paper"]).describe()
number_of_keywords_per_year_and_type

# %% [markdown]
# ### Further analysis

# Next, we will look at pairwise cosine similarities to identify similar Best Papers based on their author keywords.

# %%
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

papers["joined_keywords"] = papers["keywords"].apply(lambda keywords: "; ".join(keywords))

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(papers["joined_keywords"])
#print(vectorizer.get_feature_names())
#print(X.toarray())

cosine_similarities = pd.DataFrame(cosine_similarity(X.toarray()), index=list(papers["doi"]), columns=list(papers["doi"]))
cosine_similarities

#papers = pd.concat([papers, cosine_similarities], axis=1)

#print_dataframe_summary(papers)

# %%
# Melting the upper triangular matrix
cosine_similarities_unpivoted = cosine_similarities.where(np.triu(np.ones(cosine_similarities.shape)).astype(bool))
cosine_similarities_unpivoted = cosine_similarities_unpivoted.stack().reset_index()
cosine_similarities_unpivoted.columns = ["doi_1", "doi_2", "cosine_similarity"]
cosine_similarities_unpivoted

# %%
# Filtering only high similarities (but removing same pairs with similarity of 1)
SIMILARITY_THRESHOLD = 0.33
high_similarity = (cosine_similarities_unpivoted["cosine_similarity"] > SIMILARITY_THRESHOLD) & (cosine_similarities_unpivoted["doi_1"] != cosine_similarities_unpivoted["doi_2"])
highest_cosine_similarities = cosine_similarities_unpivoted[high_similarity].sort_values(by="cosine_similarity", ascending=False).head(10)
highest_cosine_similarities = highest_cosine_similarities.join(papers[["doi", "joined_keywords"]].set_index("doi"), on="doi_1").rename(columns={"joined_keywords": "joined_keywords_1"})
highest_cosine_similarities = highest_cosine_similarities.join(papers[["doi", "joined_keywords"]].set_index("doi"), on="doi_2").rename(columns={"joined_keywords": "joined_keywords_2"})
highest_cosine_similarities

# %% [markdown]
# ### Visualisation

# %%
# Heatmap of paper keywords pairwise cosine similarities
import seaborn as sns

heatmap = sns.heatmap(cosine_similarities)

# %%
# Filter out highest cosine similarities from full similarity matrix
cosine_similarities_high = cosine_similarities.filter(items=highest_cosine_similarities["doi_1"].unique()).loc[highest_cosine_similarities["doi_2"].unique()]
cosine_similarities_high

# %%
# Create similarity matrix with keywords instead of DOIs
cosine_similarities_high_keywords = cosine_similarities_high
cosine_similarities_high_keywords.index.name = "doi"
cosine_similarities_high_keywords = cosine_similarities_high_keywords.join(papers[["doi", "joined_keywords"]].set_index("doi"))
cosine_similarities_high_keywords = cosine_similarities_high_keywords.reset_index()
dois_to_lookup = cosine_similarities_high_keywords.drop(["doi", "joined_keywords"], axis=1).columns
dois_to_lookup_with_keywords = papers[["doi", "joined_keywords"]].set_index("doi").loc[dois_to_lookup]
dois_to_keywords_mapping = {f"{doi}": values["joined_keywords"] for doi, values in dois_to_lookup_with_keywords.to_dict('index').items()}
cosine_similarities_high_keywords = cosine_similarities_high_keywords.rename(columns=dois_to_keywords_mapping)
cosine_similarities_high_keywords = cosine_similarities_high_keywords.drop(["doi"], axis=1)
cosine_similarities_high_keywords = cosine_similarities_high_keywords.set_index("joined_keywords")
cosine_similarities_high_keywords

# %%
# Create similarity matrix with keywords and DOIs
cosine_similarities_high_keywords_and_doi = cosine_similarities_high

index_mapper = {}
for i in range(len(cosine_similarities_high.index)):
    doi = cosine_similarities_high.index[i]
    keywords = cosine_similarities_high_keywords.index[i]
    index_mapper[doi] = f"DOI: {doi}\nKeywords: {keywords}"

columns_mapper = {}
for c in range(len(cosine_similarities_high.columns)):
    doi = cosine_similarities_high.columns[c]
    keywords = cosine_similarities_high_keywords.columns[c]
    columns_mapper[doi] = f"DOI: {doi}\nKeywords: {keywords}"

cosine_similarities_high_keywords_and_doi = cosine_similarities_high_keywords_and_doi.rename(columns=columns_mapper, index=index_mapper)

cosine_similarities_high_keywords_and_doi

# %%
# Display heatmap (DOIs)
heatmap_highest_dois = sns.heatmap(cosine_similarities_high, linewidths=.5)

# %%
# Display heatmap (keywords)
heatmap_highest_keywords = sns.heatmap(cosine_similarities_high_keywords, linewidths=.5)

# %%
# Display heatmap (DOI and keywords)
heatmap_highest_keywords_and_doi = sns.heatmap(cosine_similarities_high_keywords_and_doi, linewidths=.5)

# %% [markdown]
# ## Topic Modeling

# %% [markdown]
# ### LDA topic modeling using gensim

# First, let us make a small test how LDA fairs on our dataset.

# %%
# Logging
import logging

# for gensim to output some progress information while it's training
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# %%
# Utility functions
def build_corpus(keywords_list:[[str]], separator=" ") -> [[str]]:
    """Splits multi-word keyboards by separator."""
    corpus = []
    for keywords in keywords_list:
        updated_keywords = []
        for keyword in keywords:
            split_keyword = keyword.split(separator)
            for word in split_keyword:
                updated_keywords.append(word)
        corpus.append(updated_keywords)
    return corpus

# %%
corpus = list(papers["keywords"])
#corpus = build_corpus(list(papers["keywords"]))
for index, keywords in enumerate(corpus[:10]):
    joined_keywords = ', '.join(keywords)
    print(f'{index}: {joined_keywords}')


# %%
# Gensim
import gensim

num_topics = 10

common_dictionary = gensim.corpora.dictionary.Dictionary(corpus)
common_corpus = [common_dictionary.doc2bow(keywords) for keywords in corpus]

lda = gensim.models.LdaMulticore(corpus=common_corpus, id2word=common_dictionary, num_topics=num_topics, passes=10, random_state=100, workers=4)

# %%
# Showing top topics
topics_list = []
for index, (topic_keywords, coherence_score) in enumerate(lda.top_topics(common_corpus)):
    keywords = [ keyword for _score, keyword in topic_keywords ]
    joined_keywords = ', '.join(keywords)
    topic = {"coherence_score": coherence_score, "keywords": joined_keywords}
    #print(f'{index}: {coherence_score.round(3)}\t|\t{joined_keywords}')
    topics_list.append(topic)
topics = pd.DataFrame(topics_list)
topics

# %%
# PyLDAvis
import pyLDAvis.gensim_models
import pyLDAvis
pyLDAvis.enable_notebook()
# Visualize the topics
pyLDAvis.gensim_models.prepare(topic_model=lda, corpus=common_corpus, dictionary=common_dictionary)

# %%
# Fit alternative HDP model (infers number of topics)
# The optional parameter T here indicates that HDP should find no more than 50 topics
# if there exists any.
hdp = gensim.models.hdpmodel.HdpModel(corpus=common_corpus, id2word=common_dictionary, T=20)

# %%
# PyLDAvis
# Visualize the topics
pyLDAvis.gensim_models.prepare(topic_model=hdp, corpus=common_corpus, dictionary=common_dictionary)

# %% [markdown]
# # Hyperparameter Tuning

# %%
from gensim.models import CoherenceModel
# Compute Coherence Score
COHERENCE_MEASURE = "c_v"
if COHERENCE_MEASURE == "u_mass":
    coherence_model_lda = CoherenceModel(model=lda, corpus=common_corpus, coherence='u_mass')
else: # c_v
    coherence_model_lda = CoherenceModel(model=lda, texts=corpus, dictionary=common_dictionary, coherence='c_v')

coherence_lda = coherence_model_lda.get_coherence()
print(f'Coherence Score: {coherence_lda}')

# %%
# supporting function
def compute_coherence_values(common_corpus, common_dictionary, k, a, b, coherence_measure="c_v"):
    lda_model = gensim.models.LdaMulticore(corpus=common_corpus,
                                           id2word=common_dictionary,
                                           num_topics=k, 
                                           random_state=100,
                                           passes=10,
                                           workers=2,
                                           alpha=a,
                                           eta=b)

    if COHERENCE_MEASURE == "u_mass":
        coherence_model_lda = CoherenceModel(model=lda_model, corpus=common_corpus, coherence="u_mass")
    else: # c_v
        coherence_model_lda = CoherenceModel(model=lda_model, texts=corpus, dictionary=common_dictionary, coherence="c_v")
    
    return coherence_model_lda.get_coherence()

# %%
# Grid search
import tqdm
grid = {}
grid['Validation_Set'] = {}
# Topics range
min_topics = 5
max_topics = 25
step_size = 1
topics_range = range(min_topics, max_topics, step_size)
# Alpha parameter
alpha = list(np.arange(0.01, 1, 0.3))
alpha.append('symmetric')
alpha.append('asymmetric')
# Beta parameter
beta = list(np.arange(0.01, 1, 0.3))
beta.append('symmetric')
# Validation sets
num_of_docs = len(common_corpus)
corpus_sets = [# gensim.utils.ClippedCorpus(common_corpus, int(num_of_docs*0.25)), 
               # gensim.utils.ClippedCorpus(common_corpus, int(num_of_docs*0.5)), 
               gensim.utils.ClippedCorpus(common_corpus, int(num_of_docs*0.75)), 
               common_corpus]
corpus_title = [# '25% Corpus',
                # '50% Corpus',
                '75% Corpus',
                '100% Corpus']
model_results = {'Validation_Set': [],
                 'Topics': [],
                 'Alpha': [],
                 'Beta': [],
                 'Coherence_(c_v)': [],
                 'Coherence_(u_mass)': []
                }
# Can take a long time to run
if 1 == 1:
    pbar = tqdm.tqdm(total=len(corpus_sets)*len(topics_range)*len(alpha)*len(beta))
    
    # iterate through validation corpuses
    for i in range(len(corpus_sets)):
        # iterate through number of topics
        for k in topics_range:
            # iterate through alpha values
            for a in alpha:
                # iterare through beta values
                for b in beta:
                    # get the coherence score for the given parameters
                    cv = compute_coherence_values(common_corpus=corpus_sets[i], common_dictionary=common_dictionary, 
                                                  k=k, a=a, b=b)
                    u_mass = compute_coherence_values(common_corpus=corpus_sets[i], common_dictionary=common_dictionary, 
                                                  k=k, a=a, b=b, coherence_measure="u_mass")
                    # Save the model results
                    model_results['Validation_Set'].append(corpus_title[i])
                    model_results['Topics'].append(k)
                    model_results['Alpha'].append(a)
                    model_results['Beta'].append(b)
                    model_results['Coherence_(c_v)'].append(cv)
                    model_results['Coherence_(u_mass)'].append(u_mass)
                    
                    pbar.update(1)
    pd.DataFrame(model_results).to_csv('lda_tuning_results.csv', index=False)
    pbar.close()

# %%
results = pd.DataFrame(model_results)
results

# %% [markdown]
# ## Word2Vec

# %%
SIZE = 5
WINDOW = 3
WORKERS = 4

model = gensim.models.word2vec.Word2Vec(corpus, size=SIZE, window=WINDOW, min_count=1, workers=WORKERS)
#print(model.wv["privacy"])

# %%
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def tsne_plot(model, perplexity=20):
    """Creates a TSNE model and plots it."""
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=perplexity, n_components=2, init='pca', n_iter=1000, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

tsne_plot(model)
#tsne_plot(model, perplexity=10)
#tsne_plot(model, perplexity=20)

# %%
# A more selective model
selective_model = word2vec.Word2Vec(corpus, size=SIZE, window=WINDOW, min_count=2, workers=WORKERS)
tsne_plot(selective_model)
#tsne_plot(selective_model, perplexity=10)
#tsne_plot(selective_model, perplexity=20)

# %%
# An even more selective model
more_selective_model = word2vec.Word2Vec(corpus, size=SIZE, window=WINDOW, min_count=4, workers=WORKERS)
tsne_plot(more_selective_model)

# %% [markdown]
# ## To Do

# - [ ] topic modeling using LSA vs LDA
# - [ ] training/test split (e.g. 90/10)
# - [ ] repeatedly evaluate cosine similarity on test set between count vectors (co-occurence of words) of
#     - papers within topics (should be high; similar keywords within topics)
#     - topics (should be low; different keywords between topics)
# - [ ] heatmap of similar individual keywords
# - [x] heatmap of similar papers (keyword lists) => table of, for example, top 10 similar keyword lists
# - [x] include more best papers
# - [ ] try lemmatization
# - [ ] try stemming?
# - [ ] include honorable mentions?
# - [ ] maybe retrieve some info via CrossRef API?
# - [ ] splitting combined expressions (e.g. AR/VR)?
# - [ ] check correlation with CCS concepts?

# %% [markdown]
# ## Limitations

# - Counter very prone to over-evaluate repeated words in keywords (e.g. design research, design fiction)
# - Paper might not include fitting and expected keywords (e.g. "These are not my hands!”: Effect of Gender on the Perception of Avatar Hands in Virtual Reality" does not include "gender") and get misclassified as a result
# - Different spellings using dashes (e.g. research through design vs research-through-design) are interpreted as independent
