from sklearn.metrics.pairwise import cosine_similarity
import matplotlib

matplotlib.use('Agg')
import string
import re
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import defaultdict
from flask import Flask, request, render_template, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd
import os
import json
import matplotlib
from sklearn.decomposition import TruncatedSVD

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gensim.models import Word2Vec

app = Flask(__name__)


# Preprocessing functions

def lemmatize_text(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w, pos="v") for w in tokens]


def stem_text(tokens):
    stemmer = SnowballStemmer("english")
    return [stemmer.stem(w) for w in tokens]


def data_preprocessing(df):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Handle missing values
    df['text'] = df['text'].fillna("")

    # Lowercase
    df['text'] = df['text'].str.lower()

    # Remove stop words
    df['text'] = df['text'].apply(lambda x: " ".join([w for w in x.split() if w not in stop_words]))

    # Remove punctuation
    df['text'] = df['text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

    # Remove numbers, links, and non-Latin characters using regular expressions
    df['text'] = df['text'].apply(lambda x: re.sub(r'\b\d+\b', '', x))  # Remove numbers
    df['text'] = df['text'].apply(lambda x: re.sub(r'http\S+', '', x))  # Remove links
    df['text'] = df['text'].apply(lambda x: re.sub(r'www\S+', '', x))  # Remove websites starting with www
    df['text'] = df['text'].apply(lambda x: re.sub(r'[^\x00-\x7F]+', ' ', x))  # Remove non-Latin characters
    df['text'] = df['text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))  # Remove symbols

    # Remove short words with only one or two letters and very long words
    df['text'] = df['text'].apply(lambda x: re.sub(r'\b\w{1,2}\b', '', x))  # Remove words with only one or two letters
    df['text'] = df['text'].apply(lambda x: re.sub(r'\b\w{15,}\b', '', x))  # Remove very long words

    # Remove numbers with specific suffixes and mathematical expressions or numbers with characters
    df['text'] = df['text'].apply(
        lambda x: re.sub(r'\b\d+(st|nd|rd|th|s|l|k|bi)\b', '', x))  # Remove numbers with suffixes
    df['text'] = df['text'].apply(lambda x: re.sub(r'\b\d+[a-zA-Z]+\d*\b', '', x))  # Remove mathematical expressions
    df['text'] = df['text'].apply(
        lambda x: re.sub(r'\b\d+[a-zA-Z]+\d+[a-zA-Z]*\b', '', x))  # Remove complex expressions

    # Tokenize, lemmatize, and stem text
    df['text'] = df['text'].apply(word_tokenize)
    df['text'] = df['text'].apply(lemmatize_text)
    # df['text'] = df['text'].apply(correct_sentence_spelling)
    df['text'] = df['text'].apply(stem_text)

    return df


def dummy(tokens):
    return tokens


def data_representation(df, vectorizer=None):
    # if vectorizer is None:
    #     vectorizer = TfidfVectorizer(tokenizer=dummy, preprocessor=dummy, token_pattern=None)
    # X = vectorizer.fit_transform(df["text"])
    # df_tfidf = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out(), index=df.index)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)
    # return df_tfidf , vectorizer
    if vectorizer is None:
        vectorizer = TfidfVectorizer(tokenizer=dummy, preprocessor=dummy, token_pattern=None)
    X = vectorizer.fit_transform(df["text"])
    return X, vectorizer


def data_representation1(df, vectorizer=None):
    if vectorizer is None:
        vectorizer = TfidfVectorizer(tokenizer=dummy, preprocessor=dummy, token_pattern=None)
    X = vectorizer.fit_transform(df["text"])
    df_tfidf = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out(), index=df.index)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    return df_tfidf, vectorizer


def load_dataset_from_tsv(file_path, num_docs):
    df = pd.read_csv(file_path, sep='\t', nrows=num_docs)
    df.columns = ['doc_id', 'text']  # Ensure columns are named correctly
    return df


def query_process(query):
    data = [{'doc_id': 1, 'text': query}]
    q_df = pd.DataFrame(data, columns=['doc_id', 'text'])
    q_df = data_preprocessing(q_df)
    return q_df


def cosine_sim(i_df, qi_df):
    cosine_sim = cosine_similarity(i_df, qi_df)
    return cosine_sim


def inverted_index(df):
    df['text'] = df['text'].apply(lambda x: " ".join(x))
    vectorizer = CountVectorizer()
    tf = vectorizer.fit_transform(df['text'])
    terms = vectorizer.get_feature_names_out()
    inv_index = defaultdict(list)

    for i, term in enumerate(terms):
        doc_ids = list(tf[:, i].nonzero()[0])
        inv_index[term] = doc_ids

    return pd.DataFrame(inv_index.items(), columns=['Term', 'Documents'])


def results(i_df, q_df, df, vectorizer):
    query_vector = vectorizer.transform(q_df["text"])
    cosine_similarities = cosine_similarity(i_df, query_vector)
    matching_qids = cosine_similarities[:, 0].argsort()[::-1]  # Get indices of sorted similarities in descending order
    res = [{'doc_id': df.loc[idx, 'doc_id'], 'doc': df.loc[idx, 'text'], 'similarity': cosine_similarities[idx, 0]} for
           idx in matching_qids[:10]]  # Return top results
    res_df = pd.DataFrame(res, columns=['doc_id', 'doc', 'similarity'])
    return res_df

def load_ground_truth(ground_truth_path):
    ground_truth = defaultdict(set)
    with open(ground_truth_path, 'r') as f:
        for line in f:
            query_id, _, doc_id, relevance = line.strip().split()
            if int(relevance) > 0:
                ground_truth[int(query_id)].add(int(doc_id))
    return ground_truth


def load_queries(queries_path):
    queries = {}
    with open(queries_path, 'r') as f:
        for line in f:
            query_id, text = line.strip().split('\t')
            queries[int(query_id)] = text
    return queries


def precision_at_k(retrieved_docs, relevant_docs, k):
    retrieved_doc_ids = {int(re.sub(r'_(?=\d)', '', doc)) for doc in retrieved_docs[:k]}
    print("retrieved_docs", retrieved_doc_ids)
    relevant_retrieved = retrieved_doc_ids.intersection(relevant_docs)
    print(len(relevant_retrieved))
    precision = len(relevant_retrieved) / k if k != 0 else 0
    return precision
    # """Calculate precision at k"""
    # assert k >= 1
    # r = np.asarray(r)[:k]  # take the first k elements
    # return np.mean(r)


def recall_at_k(retrieved_docs, relevant_docs):
    retrieved_doc_ids = {int(re.sub(r'_(?=\d)', '', doc)) for doc in retrieved_docs}
    relevant_retrieved = set(retrieved_doc_ids).intersection(relevant_docs)
    recall = len(relevant_retrieved) / len(relevant_docs) if len(relevant_docs) != 0 else 0
    return recall
    # """Calculate recall at k"""
    # assert k >= 1
    # r = np.asarray(r)[:k]
    # return np.sum(r) / all_rel


def calculate_map(precision_list):
    num_queries = len(precision_list)
    return sum(precision_list) / num_queries


def calculate_mrr(reciprocal_ranks):
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def reciprocal_rank(retrieved_docs, relevant_docs):
    retrieved_doc_ids = {int(re.sub(r'_(?=\d)', '', doc)) for doc in retrieved_docs}
    for i, doc_id in enumerate(retrieved_docs):
        if int(re.sub(r'_(?=\d)', '', doc_id)) in relevant_docs:
            return 1 / (i + 1)
    return 0  # If no relevant document found


def evaluate_model(ground_truth, queries, df_docs_preprocessed, df_tfidf, vectorizer):
    precision_list = []
    recall_list = []
    reciprocal_ranks = []
    results_data = []

    for query_id, query_text in queries.items():
        query_preprocessed = query_process(query_text)
        results_df = results(df_tfidf, query_preprocessed, df_docs_preprocessed, vectorizer)

        relevant_docs = ground_truth[query_id]
        retrieved_docs = results_df['doc_id'].tolist()

        precision_10 = precision_at_k(retrieved_docs, relevant_docs, 10)
        recall = recall_at_k(retrieved_docs, relevant_docs)
        reciprocal_rank_value = reciprocal_rank(retrieved_docs, relevant_docs)
        results_data.append({
            'Query ID': query_id,
            'Query Text': query_text,
            'Precision@10': precision_10,
            'Recall': recall,
            'Reciprocal Rank': reciprocal_rank_value
        })
        precision_list.append(precision_10)
        recall_list.append(recall)
        reciprocal_ranks.append(reciprocal_rank_value)

        print("Query ID:", query_id)
        print("Precision@10:", precision_10)
        print("Recall:", recall)
        print("Reciprocal Rank:", reciprocal_rank_value)

    # Calculate MAP and MRR
    map_value = calculate_map(precision_list)
    mrr_value = calculate_mrr(reciprocal_ranks)

    print("\nMean Average Precision (MAP):", map_value)
    print("Mean Reciprocal Rank (MRR):", mrr_value)
    results_data.append({
        'Query ID': 'Overall',
        'Query Text': 'Overall',
        'Precision@10': '',
        'Recall': '',
        'Reciprocal Rank': '',
        'MAP': map_value,
        'MRR': mrr_value
    })
    # Save results to CSV
    results_df = pd.DataFrame(results_data)
    results_df.to_csv('output_file.csv', index=False, encoding='utf-8')


def cluster_documents(df, num_clusters):
    df_tfidf, vectorizer = data_representation(df)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(df_tfidf)
    return df, df_tfidf, kmeans


def load_ground_truth2(ground_truth_path):
    ground_truth = defaultdict(set)
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                query_id = data['qid']
                answer_pids = data['answer_pids']
                for doc_id in answer_pids:
                    ground_truth[query_id].add(doc_id)
            except json.JSONDecodeError as e:
                print("Skipping line with invalid JSON: {line.strip()}")
                print(line.strip())
    return ground_truth


def load_queries2(queries_path):
    queries = {}
    with open(queries_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                query_id, text = line.strip().split('\t')
                queries[int(query_id)] = text
            except ValueError:
                print("Skipping line with invalid format: {line.strip()}")
                print(line.strip())
    return queries


def precision_at_k2(retrieved_docs, relevant_docs, k):
    retrieved_doc_ids = set(retrieved_docs[:k])  # Convert to set and take the first k elements
    relevant_retrieved = retrieved_doc_ids.intersection(relevant_docs)
    precision = len(relevant_retrieved) / k if k != 0 else 0
    return precision


def recall_at_k2(retrieved_docs, relevant_docs):
    # retrieved_doc_ids = {int(re.sub(r'_(?=\d)', '', doc)) for doc in retrieved_docs}
    relevant_retrieved = set(retrieved_docs).intersection(relevant_docs)
    recall = len(relevant_retrieved) / len(relevant_docs) if len(relevant_docs) != 0 else 0
    return recall
    # """Calculate recall at k"""
    # assert k >= 1
    # r = np.asarray(r)[:k]
    # return np.sum(r) / all_rel


def calculate_map2(precision_list):
    num_queries = len(precision_list)
    return sum(precision_list) / num_queries


def calculate_mrr2(reciprocal_ranks):
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def reciprocal_rank2(retrieved_docs, relevant_docs):
    # retrieved_doc_ids = {int(re.sub(r'_(?=\d)', '', doc)) for doc in retrieved_docs}
    for i, doc_id in enumerate(retrieved_docs):
        if doc_id in relevant_docs:
            return 1 / (i + 1)
    return 0  # If no relevant document found


def evaluate_model2(ground_truth, queries, df_docs_preprocessed, df_tfidf, vectorizer):
    precision_list = []
    recall_list = []
    reciprocal_ranks = []
    results_data = []

    for query_id, query_text in queries.items():
        query_preprocessed = query_process(query_text)
        results_df = results(df_tfidf, query_preprocessed, df_docs_preprocessed, vectorizer)

        relevant_docs = ground_truth[query_id]
        retrieved_docs = results_df['doc_id'].tolist()

        precision_10 = precision_at_k2(retrieved_docs, relevant_docs, 10)
        recall = recall_at_k2(retrieved_docs, relevant_docs)
        reciprocal_rank_value = reciprocal_rank2(retrieved_docs, relevant_docs)
        results_data.append({
            'Query ID': query_id,
            'Query Text': query_text,
            'Precision@10': precision_10,
            'Recall': recall,
            'Reciprocal Rank': reciprocal_rank_value
        })
        precision_list.append(precision_10)
        recall_list.append(recall)
        reciprocal_ranks.append(reciprocal_rank_value)

        print("Query ID:", query_id)
        print("Precision@10:", precision_10)
        print("Recall:", recall)
        print("Reciprocal Rank:", reciprocal_rank_value)

    # Calculate MAP and MRR
    map_value = calculate_map2(precision_list)
    mrr_value = calculate_mrr2(reciprocal_ranks)

    print("\nMean Average Precision (MAP):", map_value)
    print("Mean Reciprocal Rank (MRR):", mrr_value)
    results_data.append({
        'Query ID': 'Overall',
        'Query Text': 'Overall',
        'Precision@10': '',
        'Recall': '',
        'Reciprocal Rank': '',
        'MAP': map_value,
        'MRR': mrr_value
    })
    # Save results to CSV
    results_df = pd.DataFrame(results_data)
    results_df.to_csv('output_file1.csv', index=False, encoding='utf-8')


def plot_clusters(df, df_tfidf, kmeans, filename='static/images/cluster_plot.png'):
    plt.figure(figsize=(10, 7))

    # Use TruncatedSVD to reduce the dimensionality of the sparse matrix
    svd = TruncatedSVD(n_components=2, random_state=42)
    reduced_tfidf = svd.fit_transform(df_tfidf)

    plt.scatter(reduced_tfidf[:, 0], reduced_tfidf[:, 1], c=df['cluster'], cmap='viridis', marker='o')
    plt.title('Document Clusters')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    if not os.path.exists('static/images'):
        os.makedirs('static/images')
    plt.savefig(filename)
    plt.close()


# def train_word2vec(df):
#     documents = df['text'].tolist()
#     model = Word2Vec(sentences=documents, vector_size=100, window=5, min_count=1, workers=4)
#     model.save("word2vec.model")
#     return model

def get_doc_vectors(df, model):
    vectors = []
    for tokens in df['text']:
        vector = sum([model.wv[word] for word in tokens if word in model.wv]) / len(tokens)
        vectors.append(vector)
    return pd.DataFrame(vectors)


# -------------------------------------------------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/preprocess', methods=['POST'])
def preprocess():
    dataset_name = request.form['dataset']
    if dataset_name == "Antique":
        file_path = 'C:/Users/ASUS/.ir_datasets/antique/collection.tsv'

    elif dataset_name == "Lifestyle":
        file_path = 'C:/Users/ASUS/Desktop/lotte/lifestyle/dev/collection.tsv'
    num_docs = 100
    # Load dataset
    df_docs = load_dataset_from_tsv(file_path, num_docs)
    # Preprocess dataset
    df_docs_preprocessed = data_preprocessing(df_docs)

    # Convert preprocessed text to display
    preprocessed_text = df_docs_preprocessed.head(10)

    return render_template('index.html', preprocessed_text=preprocessed_text, show_preprocessed=True)


@app.route('/representation', methods=['POST'])
def representation():
    dataset_name = request.form['dataset']
    if dataset_name == "Antique":
        file_path = 'C:/Users/ASUS/.ir_datasets/antique/collection.tsv'

    elif dataset_name == "Lifestyle":
        file_path = 'C:/Users/ASUS/Desktop/lotte/lifestyle/dev/collection.tsv'
    num_docs = 100
    # Load dataset
    df_docs = load_dataset_from_tsv(file_path, num_docs)

    # Preprocess dataset
    df_docs_preprocessed = data_preprocessing(df_docs)
    df_docs_representation = data_representation1(df_docs_preprocessed.head())

    return render_template('index.html', df_docs_representation=df_docs_representation, show_representation=True)


@app.route('/invert', methods=['POST'])
def invert():
    dataset_name = request.form['dataset']
    if dataset_name == "Antique":
        file_path = 'C:/Users/ASUS/.ir_datasets/antique/collection.tsv'

    elif dataset_name == "Lifestyle":
        file_path = 'C:/Users/ASUS/Desktop/lotte/lifestyle/dev/collection.tsv'
    num_docs = 100
    # Load dataset
    df_docs = load_dataset_from_tsv(file_path, num_docs)

    # Preprocess dataset
    df_docs_preprocessed = data_preprocessing(df_docs)
    df_inverted_index = inverted_index(df_docs_preprocessed)

    return render_template('index.html', invert_text=df_inverted_index, show_invert=True)


@app.route('/query', methods=['POST'])
def queryProsses():
    dataset_name = request.form['dataset']
    if dataset_name == "Antique":
        file_path = 'C:/Users/ASUS/.ir_datasets/antique/collection.tsv'

    elif dataset_name == "Lifestyle":
        file_path = 'C:/Users/ASUS/Desktop/lotte/lifestyle/dev/collection.tsv'
    num_docs = 100
    # Load dataset
    df_docs = load_dataset_from_tsv(file_path, num_docs)

    query_text = request.form['query']  # Retrieve the query from the form

    query_preprocessed = query_process(query_text)
    query_inverted_index = inverted_index(query_preprocessed)

    return render_template('index.html', query_results=query_inverted_index, query_text=query_text, show_query=True)


@app.route('/result', methods=['POST'])
def result():
    dataset_name = request.form['dataset']
    if dataset_name == "Antique":
        file_path = 'C:/Users/ASUS/.ir_datasets/antique/collection.tsv'

    elif dataset_name == "Lifestyle":
        file_path = 'C:/Users/ASUS/Desktop/lotte/lifestyle/dev/collection.tsv'
    num_docs = 100
    # Load dataset
    df_docs = load_dataset_from_tsv(file_path, num_docs)
    df_docs_preprocessed = data_preprocessing(df_docs)
    vectorizer1,df_tfidf1 = data_representation1(df_docs_preprocessed.head())
    query_text = request.form['query']  # Retrieve the query from the form
    query_preprocessed = query_process(query_text)
    #     print(query_preprocessed)
    query_inverted_index = inverted_index(query_preprocessed)
    results_df1 = results(df_tfidf1, query_preprocessed, df_docs_preprocessed, vectorizer1)

    return render_template('index.html', results=results_df1, show_results=True)


@app.route('/evaluation', methods=['POST'])
def evaluation():
    dataset_name = request.form['dataset']
    if dataset_name == "Antique":
        file_path = 'C:/Users/ASUS/Desktop/antique/collection.tsv'
        queries_path = 'C:/Users/ASUS/Desktop/antique/train/queries.txt'
        ground_truth_path = 'C:/Users/ASUS/Desktop/antique/train/qrels'
        num_docs = None
        # Load dataset
        df_docs = load_dataset_from_tsv(file_path, num_docs)
        ground_truth = load_ground_truth(ground_truth_path)
        queries = load_queries(queries_path)
        # Preprocess datasets
        df_docs1_preprocessed = data_preprocessing(df_docs)
        df_docs1_preprocessed.to_csv('df_docs1_preprocessed.csv', index=False, encoding='utf-8')
        df_tfidf1, vectorizer1 = data_representation(df_docs1_preprocessed)
        print(df_tfidf1)
        evaluate_model(ground_truth, queries, df_docs1_preprocessed, df_tfidf1, vectorizer1)

    elif dataset_name == "Lifestyle":
        file_path = 'C:/Users/ASUS/Desktop/lotte/lifestyle/dev/collection.tsv'
        queries_path = 'C:/Users/ASUS/Desktop/lotte/lifestyle/dev/questions.forum.tsv'
        ground_truth_path = 'C:/Users/ASUS/Desktop/lotte/lifestyle/dev/qas.forum.jsonl'
        num_docs = None
        # Load dataset
        df_docs = load_dataset_from_tsv(file_path, num_docs)
        ground_truth = load_ground_truth2(ground_truth_path)
        queries = load_queries2(queries_path)
        # Preprocess datasets
        df_docs2_preprocessed = data_preprocessing(df_docs)
        df_docs2_preprocessed.to_csv('df_docs2_preprocessed.csv', index=False, encoding='utf-8')
        df_tfidf2, vectorizer2 = data_representation(df_docs2_preprocessed)
        evaluate_model2(ground_truth, queries, df_docs2_preprocessed, df_tfidf2, vectorizer2)
    return render_template('index.html', query_results=query_inverted_index, query_text=query_text, show_query=True)


@app.route('/cluster', methods=['POST'])
def cluster():
    dataset_name = request.form['dataset']
    if dataset_name == "Antique":
        file_path = 'C:/Users/ASUS/.ir_datasets/antique/collection.tsv'

    elif dataset_name == "Lifestyle":
        file_path = 'C:/Users/ASUS/Desktop/lotte/lifestyle/dev/collection.tsv'
    num_docs = 100
    # Load dataset
    df_docs = load_dataset_from_tsv(file_path, num_docs)

    # Preprocess dataset
    df_docs_preprocessed = data_preprocessing(df_docs)

    # Cluster documents
    df_clustered, df_tfidf, kmeans = cluster_documents(df_docs_preprocessed, num_clusters=5)

    # Plot clusters
    image_path = 'static/images/cluster_plot.png'
    plot_clusters(df_clustered, df_tfidf, kmeans, filename=image_path)

    return render_template('index.html', cluster_results=df_clustered, show_clusters=True, image_path=image_path)


@app.route('/word2vec', methods=['POST'])
def word2vec_train():
    dataset_name = request.form['dataset']
    if dataset_name == "Antique":
        file_path = 'C:/Users/ASUS/.ir_datasets/antique/collection.tsv'
    elif dataset_name == "Lifestyle":
        file_path = 'C:/Users/ASUS/Desktop/lotte/lifestyle/dev/collection.tsv'
    num_docs = 100
    word = request.form.get('word')

    # Load dataset
    df_docs = load_dataset_from_tsv(file_path, num_docs)
    df_docs_preprocessed = data_preprocessing(df_docs)

    # Extract texts for Word2Vec training
    documents = df_docs_preprocessed['text'].tolist()

    # Train Word2Vec model
    model = Word2Vec(sentences=documents, vector_size=100, window=5, min_count=1, workers=4)
    model.save("word2vec.model")

    # Get most common words
    most_common_words = model.wv.index_to_key[:10]

    # Calculate similarity if word is provided
    similarity_scores = None
    if word:
        similarity = model.wv.most_similar(word) if word in model.wv else None
        if similarity:
            similarity_scores = [score for _, score in similarity]

    return render_template('index.html', most_common_words=most_common_words, similarity_scores=similarity_scores,
                           show_word2vec=True)


# ------------------------------------------------------------------------------


if __name__ == '__main__':
    app.run(debug=True)
