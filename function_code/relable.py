from munkres import Munkres
import numpy as np
def best_map(L1,L2):
	#L1 should be the labels and L2 should be the clustering number we got
	L1=np.array(L1)
	L2=np.array(L2)
	Label1 = np.unique(L1)       # 去除重复的元素，由小大大排列
	nClass1 = len(Label1)        # 标签的大小
	Label2 = np.unique(L2)
	nClass2 = len(Label2)
	nClass = np.maximum(nClass1,nClass2)
	G = np.zeros((nClass,nClass))
	for i in range(nClass1):
		ind_cla1 = L1 == Label1[i]
		ind_cla1 = ind_cla1.astype(int)
		for j in range(nClass2):
			ind_cla2 = L2 == Label2[j]
			ind_cla2 = ind_cla2.astype(int)
			G[i,j] = np.sum(ind_cla2 * ind_cla1)
	m = Munkres()
	index = m.compute(-G.T)
	index = np.array(index)
	c = index[:,1]
	newL2 = np.zeros(L2.shape)
	for i in range(nClass2):
		newL2[L2 == Label2[i]] = Label1[c[i]]
	return list(newL2.astype(int))

if __name__ == '__main__':
	a=([1,1,2,2,3,3])
	b=([3,3,1,1,2,2])
	c=best_map(a,b)
	print(b,c)
	print((a)==(c))