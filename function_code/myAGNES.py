import numpy as np
import matplotlib.pyplot as plt

class AGNES:
    def __init__(self,k=3,dist_type='AVG'):
        self.k=k
        self.labels=None
        self.C={}
        self.dist_func=None
        if dist_type=='MIN':
            self.dist_func=self.mindist
        elif dist_type=='MAX':
            self.dist_func=self.maxdist
        else:
            self.dist_func=self.avgdist


    def fit(self,X):
        for j in range(X.shape[0]):
            self.C[j]=set()
            self.C[j].add(j)
        M=1e10*np.ones((X.shape[0],X.shape[0]),dtype=np.float32)
        for i in range(X.shape[0]):
            for j in range(i+1,X.shape[0]):
                M[i,j]=self.dist_func(X,self.C[i],self.C[j])
                M[j,i]=M[i,j]
        q=X.shape[0]
        while q>self.k:
            index=np.argmin(M)
            i_=index//M.shape[1]#最小距离对应的行
            j_=index%M.shape[1]#对应的列
            self.C[i_]=set(self.C[i_].union(self.C[j_]))#合并
            #print(self.C[i_])
            for j in range(j_+1,q):
                self.C[j-1]=set(self.C[j])
            del self.C[q-1]#将簇向前移动
            M=np.delete(M,[j_],axis=0)#删除距离矩阵中j_簇
            M=np.delete(M,[j_],axis=1)
            for j in range(q-1):
                if i_!=j:
                    M[i_,j]=self.dist_func(X,self.C[i_],self.C[j])
                    M[j,i_]=M[i_,j]
            q-=1
        self.labels=np.zeros((X.shape[0],),dtype=np.int32)
        for i in range(self.k):
            self.labels[list(self.C[i])] = i    #分配簇标签

    @classmethod
    def mindist(cls,X,Ci,Cj):
        Xi=X[list(Ci)]
        Xj=X[list(Cj)]
        min=1e10
        for i in range(len(Xi)):
            d=np.sqrt(np.sum((Xi[i]-Xj)**2,axis=1))
            dmin=np.min(d)
            if dmin<min:
                min=dmin
        return min

    @classmethod
    def maxdist(cls,X,Ci,Cj):
        Xi=X[list(Ci)]
        Xj=X[list(Cj)]
        max=0
        for i in range(len(Xi)):
            d=np.sqrt(np.sum((Xi[i]-Xj)**2,axis=1))
            dmax=np.max(d)
            if dmax>max:
                max=dmax
        return max

    @classmethod
    def avgdist(cls,X,Ci,Cj):
        Xi=X[list(Ci)]
        Xj=X[list(Cj)]
        sum=0.
        for i in range(len(Xi)):
            d=np.sqrt(np.sum((Xi[i]-Xj)**2,axis=1))
            sum+=np.sum(d)
        dist=sum/(len(Ci)*len(Cj))
        return dist




if __name__=='__main__':
    from sklearn import datasets
    import matplotlib.pyplot as plt

    n = 200
    x, y = datasets.make_moons(n_samples=n, noise=0.05, random_state=None)
    X=x
    X_test=x
    agnes=AGNES()
    agnes.fit(X)
    print('C:', agnes.C)
    print(agnes.labels)
    plt.figure(12)
    plt.subplot(121)
    plt.scatter(X[:, 0], X[:, 1], c=agnes.labels)
    plt.title('tinyml')

    from sklearn.cluster.hierarchical import AgglomerativeClustering
    sklearn_agnes=AgglomerativeClustering(n_clusters=7,affinity='l2',linkage='average')
    sklearn_agnes.fit(X)
    print(sklearn_agnes.labels)
    plt.subplot(122)
    plt.scatter(X[:,0],X[:,1],c=sklearn_agnes.labels)
    plt.title('sklearn')
    plt.show()
