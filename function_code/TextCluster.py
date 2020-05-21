from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import  TfidfVectorizer
from function_code.text_process import  *
import random
from sklearn.cluster import KMeans,AgglomerativeClustering
import numpy as np
import math


class textClusterEnsemble:
    def __init__(self,file_path,label_path):
        self.file_path=file_path
        self.text_handle=text(self.file_path)
        self.corpus=self.text_handle.get_corpus()
        self.file_num = len(self.corpus)
        trueLabelDic = get_true_label(label_path)
        self.true_lable=[trueLabelDic[name] for name in self.text_handle.text_name]
        '''
        vec = TfidfVectorizer(max_df=0.8, min_df=0.1)  # ,max_features=3，会优先选择高频词 max_df=0.8,min_df=0.2
        tfidf = vec.fit_transform(self.corpus)  #
        tfidf_matrix = tfidf.toarray()
        '''
        #my tfidf
        from  function_code import myTfidf
        vec=myTfidf.myTfidfVectorizer(max_df=0.8, min_df=0.1)
        tfidf_matrix=vec.fit_transform(self.corpus)

        print("words len:",len(vec.word_dict))
        '''
        import pandas as pd
        df = pd.DataFrame(tfidf_matrix,index=self.text_handle.text_name,columns=vec.word_dict.keys())
        df.to_excel('tfidf_matrix.xlsx')
        '''
        #降维

        self.pca = PCA(n_components=0.95)
        if np.shape(tfidf_matrix)[1]>100:
            self.red_tfidf = self.pca.fit_transform(tfidf_matrix)
            print('降维：',self.pca.n_components_)
        else:
            self.red_tfidf = tfidf_matrix

        '''
        voc = vec.get_feature_names()
        
        print("words_len:", len(voc))
        print('filenum:',self.file_num)
        print(voc)
        print('true lebel:',self.true_lable)
        '''



    def simpleKmeans(self,aimK=4):

        km_pred = KMeans(n_clusters=aimK, init='random', n_init=1,
                               algorithm='auto').fit_predict(self.red_tfidf)  # 估计器

        return km_pred


    def averEnsemble(self,aimK=4,iter=50):
        co_mat=np.zeros((self.file_num,self.file_num))
        for i in range(iter):
            k= random.randint(2, int(math.sqrt(self.file_num)))
            km_pred = KMeans(n_clusters=k, init='random', n_init=1,
                           algorithm='auto').fit_predict(self.red_tfidf)  # 估计器

            for j in range(len(km_pred)):
                for k in range(j, len(km_pred)):
                    if km_pred[j] == km_pred[k]:
                        co_mat[j][k] += 1.0 / iter
                        co_mat[k][j] = co_mat[j][k]
        if self.file_num>=100:
            red_co_mat = self.pca.fit_transform(co_mat)
        else :red_co_mat=co_mat
        '''
        averEnsemble_pred = AgglomerativeClustering(linkage='average',
                                          n_clusters=aimK,
                                          # affinity='cosine'
                                          ).fit_predict(red_co_mat)
       # enscore_in.append(self.ARI_evaluation(self.true_lable,averEnsemble_pred))


        '''
        from function_code.myAGNES import AGNES
        myagens=AGNES(k=aimK)
        myagens.fit(red_co_mat)
        averEnsemble_pred=myagens.labels

        return averEnsemble_pred



    def voting(self,aimK=4,iter=50):
        from function_code.relable import best_map
        cluster_components = np.zeros((iter, self.file_num)).astype(int)
        vote_label=[None]*self.file_num
        for i in range(iter):
            single_pred=self.simpleKmeans(aimK=aimK)
            if i==0:    base=single_pred
            relabel_pred=best_map(base,single_pred)
            cluster_components[i,:]=relabel_pred
        for i in range(len(vote_label)):
            vote_label[i]=np.argmax(np.bincount(cluster_components[:,i]))

        return vote_label


    def ARI_evaluation(self,pred):
        return metrics.adjusted_rand_score(self.true_lable,pred)

    def save_clustered_file(self,save_lable_path,label):
        import shutil
        import os.path
        if not os.path.exists(save_lable_path):
            os.mkdir(save_lable_path)
        elif os.listdir(save_lable_path):
            save_lable_path=os.path.join(save_lable_path,os.path.basename(save_lable_path))
            os.mkdir(save_lable_path)

        for i in range(len(label)):
            dest=os.path.join(save_lable_path,str(label[i]))
            if not os.path.exists(dest):
                os.mkdir(dest)
            else :
                shutil.copy(self.text_handle.text_path[i],dest)






if __name__ == '__main__':
    tce=textClusterEnsemble(r'E:\毕设\text_cluster\experiment_coupus\cor_200',
                            r'E:\毕设\text_cluster\experiment_coupus\cor_200_tureLabel.txt')
    av_score=[]
    km_score=[]
    vt_score=[]
    weight_score=[]
    runs=20
    k=4
    for x in range(runs):
        av_label=tce.averEnsemble(aimK=k,iter=50)
        kmlable=tce.simpleKmeans(k)
        av=tce.ARI_evaluation(av_label)

        av_score.append(av)
        km_score.append(tce.ARI_evaluation( kmlable))

    import matplotlib.pyplot as plt

    e = [i for i in range(runs)]
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(e,km_score,marker='.')
    plt.plot(e,[max(km_score)]*runs)
    plt.plot(e,[np.mean(km_score)]*runs)
    plt.legend(['km4_score', 'maxKm4',
                'average4'
                ], loc='best')
    plt.ylabel('ARI')
    plt.xlabel('RUN')
    plt.title("single kmeans")
    print('min KM:%s,max KM:%s,avg Km:%s'%(min(km_score),max(km_score),np.mean(km_score)))

    plt.subplot(1,3,2)
    plt.plot(e, av_score,marker='.')
    plt.plot(e, [max(av_score)] * runs)
    plt.plot(e, [np.mean(av_score)] * runs)
    plt.legend(['AL_ensemble', 'max_aver',
                'average'
                ], loc='best')
    plt.ylabel('ARI')
    plt.xlabel('RUN')
    plt.title('AL_ensemble')
    print('min EN:%s,max EN:%s,avg EN:%s' % (min(av_score),max(av_score), np.mean(av_score)))
   # plt.show()

    plt.subplot(1, 3, 3)
    plt.plot(e, km_score,marker='.')
    plt.plot(e, av_score,marker='.')
    plt.legend(['km_score', 'av_score',
                ], loc='best')
    plt.ylabel('ARI')
    plt.xlabel('RUN')
    plt.title("compare")
    plt.show()
    print(km_score,av_score,vt_score,sep='\n')
