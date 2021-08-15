import os
import re
import jieba
import numpy as np
from doc import doc
from tqdm import tqdm
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

es = Elasticsearch(['https://sunho:Dunkel6eit!!@i-o-optimized-deployment-84c1c6.es.us-east-1.aws.found.io:9243'], timeout=30)

stopwords = [line.strip() for line in open('../stoplist.txt')]

def generate_documents(es, index_name):
    """ Create training data/labels given the Elasticsearch client. """
    doc_ids, training_data, training_labels = [], [], []
    res = es.search(index=index_name, size=20000)
    hits = res['hits']['hits']
    for hit in hits:
        doc_id = hit['_id']
        text = hit['_source']['text']
        category = hit['_source']['category']
        training_data.append(text)
    return np.array(training_data)

def preprocessing(documents):
    word2vec = {}
    vec2word = {}
    docs = []
    currentDocument = []
    currentWordId = 0
    
    for document in documents:
        segList = jieba.cut(document)
        for word in segList: 
            word = word.lower().strip()
            if len(word) > 1 and word.isalpha() and word not in stopwords:
                if word in word2vec:
                    currentDocument.append(word2vec[word])
                else:
                    currentDocument.append(currentWordId)
                    word2vec[word] = currentWordId
                    vec2word[currentWordId] = word
                    currentWordId += 1
        docs.append(currentDocument);
        currentDocument = []
    return docs, word2vec, vec2word

def gibbs_sampling(docs, Z, ndz, nzw, nz):
	for d, doc in enumerate(docs):
		for index, w in enumerate(doc):
			z = Z[d][index]
			ndz[d, z] -= 1
			nzw[z, w] -= 1
			nz[z] -= 1
			pz = np.divide(np.multiply(ndz[d, :], nzw[:, w]), nz)
			z = np.random.multinomial(1, pz / pz.sum()).argmax()
			Z[d][index] = z 
			ndz[d, z] += 1
			nzw[z, w] += 1
			nz[z] += 1

def LDA(num_topics, docs, word2vec, vec2word, iteration):
    alpha, beta = 5, 0.1
    N, M , K = len(docs), len(word2vec), num_topics
    M = len(word2vec)
    Z = []
    ndz = np.zeros([N, K]) + alpha
    nzw = np.zeros([K, M]) + beta
    nz = np.zeros([K]) + M * beta

    for d, doc in enumerate(docs):
        cur_z = []
        for w in doc:
            pz = np.divide(np.multiply(ndz[d, :], nzw[:, w]), nz)
            z = np.random.multinomial(1, pz / pz.sum()).argmax()
            cur_z.append(z)
            ndz[d, z] += 1
            nzw[z, w] += 1
            nz[z] += 1
        Z.append(cur_z)

    for i in tqdm(range(iteration)):
        gibbs_sampling(docs, Z, ndz, nzw, nz)

    for i in range(num_topics):
        vectors = nzw[i, :].argsort()
        topic_words = [vec2word[vector] for vector in vectors[:10]]
        print('topic', i + 1, ' '.join(topic_words))

if __name__ == "__main__":
    documents = generate_documents(es, '20-ng')
    docs, word2vec, vec2word = preprocessing(documents)
    LDA(num_topics=20, docs=docs, word2vec=word2vec, vec2word=vec2word, iteration=30)