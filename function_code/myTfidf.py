import numpy as np
import scipy.sparse as sp
from  sklearn.preprocessing import Normalizer

from itertools import chain
from collections import Counter

class myTfidfVectorizer:
    def __init__(self,max_df=0.9,min_df=0.1):
        self.max_df=max_df
        self.min_df=min_df
        self.word_dict={}

    def _get_vocabulary(self, corpus):
        words=[]
        temp=self.get_df(corpus)
        for k ,v in temp.items():
            words.append(k)
        self.word_dict={value:i for i,value in enumerate(words)}

        return self.word_dict, len(words)

    def count_word_appearance(self, corpus):
        """
        返回每个词的文档频率
        """
        return dict(Counter(chain.from_iterable(set(doc.split(" ")) for doc in corpus)))
    def get_df(self,corpus):
        df=self.count_word_appearance(corpus)
        nf=len(corpus)
        choose_df={}#过滤
        for k ,v in df.items():
            if   self.min_df<(v/nf)<self.max_df:
                choose_df[k]=v
        return choose_df


    def _tf(self, corpus):
        """
        求tf
        """
        bag_vector = np.zeros(self.n_vocab, dtype=np.float64)

        doc = corpus.split(" ")
        for word in doc:
            if word in self.word_dict.keys():
                 bag_vector[self.word_dict[word]] += 1

       # bag_vector /= len(doc)
        return bag_vector

    def _idf_smooth(self, corpus):
        """
        求IDF
        """
        bag_vector = np.zeros(self.n_vocab, dtype=np.float64)

        for word, count in self.get_df(corpus).items():
            bag_vector[self.word_dict[word]] += count

        bag_vector = np.log((np.shape(corpus)[0] +1)/ (bag_vector + 1))+1
        return bag_vector

    def fit_transform(self, corpus):
        """
        返回tfidf矩阵
        """
        self.word_dict, self.n_vocab = self._get_vocabulary(corpus)

        c_matrix = sp.csr_matrix(([], ([], [])), shape=(0, self.n_vocab))

        idf_vec = self._idf_smooth(corpus)
        for doc in corpus:
            tf_vec = self._tf(doc)
            tfidf_vec = tf_vec * idf_vec
            c_matrix = sp.vstack((c_matrix, sp.csr_matrix(tfidf_vec)))
        co= c_matrix.todense()
        coN=Normalizer().fit_transform(X=co)
        return coN



if __name__ == '__main__':
    tfidf=myTfidfVectorizer()
    cor=['中国 中国 加油 加油 加油','武汉 加油','中国 武汉','美国']
    import numpy
    x=tfidf.fit_transform(cor)
    print(tfidf.word_dict)
    print(x)
    print(tfidf.count_word_appearance(cor))
    print(tfidf.get_df(cor))



    print('-'*50)
    from sklearn.feature_extraction.text import TfidfVectorizer
    tf=TfidfVectorizer(stop_words=None)
    tfx=tf.fit_transform(cor)
    print(tfx.toarray())
    print(tf.vocabulary_)