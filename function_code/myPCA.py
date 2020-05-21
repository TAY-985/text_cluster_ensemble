from numpy import *
import numpy

class my_PCA:
    def __init__(self,CRate):
        self.CRate=CRate
        self.n_component=None

    def fit_transform(self,X):
        meanValue = mean(X, axis=0)
        X -= meanValue
        C = cov(X, rowvar=0)
        print(C.shape)
        eigvalue, eigvector = linalg.eigh(mat(C))
        sumEigValue = sum(eigvalue)
        sortedeigvalue = eigvalue[::-1]
        for i in range(sortedeigvalue.size):
            j = i + 1
            rate = sum(sortedeigvalue[0:j]) / sumEigValue
            if rate > self.CRate:
                break
        self.n_component=j
        indexVec = numpy.argsort(-eigvalue)
        nLargestIndex = indexVec[:j]
        T = eigvector[:,nLargestIndex]
        newX = numpy.dot(X, T)
        return newX

if __name__ == '__main__':
    X=[[-1,-1,0,2,1],[2,0,0,-1,-1],[2,0,1,1,0]]
    myPCA=my_PCA(0.95)
    newX =myPCA.fit_transform(X)
    print(newX)
    print(myPCA.n_component)
    from sklearn.decomposition import PCA

    pca = PCA(n_components=0.95)
    skX=pca.fit_transform(X)
    print(skX)
    print('降维：', pca.n_components_)
    print(X-mean(X,axis=0))
