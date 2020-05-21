import os
import re
import jieba


def get_true_label(label_path=None):
    true_label=None
    if os.path.exists(label_path):
        true_label = {}
        with open(label_path, 'r')as id:
            for line in id:
                temp = line.strip().split('\t')
                true_label[temp[0]] = int(temp[1])
    return true_label


def stop_words(st_word_path=r'stop_word.txt'):

    with open(st_word_path, 'r', encoding='utf-8') as st:
        stw = st.read()
        stwlist = stw.splitlines()  # 按照行划分成list

    return stwlist

class text:
    def __init__(self,file_path=None):
        self.file_path=file_path#文件夹路径
        self.text_path=[]#每个文本文件路径
        self.text_name=os.listdir(file_path)#文本名字，方便对照label
        self.ori_text_content=[]#暂时未用到
        self.corpus=[]#语料库
       # self.true_label=None#字典类型
        if file_path:
            self.get_text_path()

    def get_text_path(self):
        for f in self.text_name:
            self.text_path.append(os.path.join(self.file_path,f))
        return self.text_path

    def get_CNtext(self,text_path):
        f = open(text_path, 'r', encoding='ansi')
        ori_text = f.read().replace('\n', '')
       # print(text_path)
        data = ''.join(re.findall(u'[\u4e00-\u9fff]+', ori_text))  # 必须Unicode,取出所有中文字符
        f.close()
        return data

    def get_corpus(self):
        #b = (get_true_label(r'E:\毕设\text_cluster\experiment_coupus\cor_200_tureLabel.txt'))

        for f in self.text_path:#text_path 文件i 就是corpus 语料i

           # print(f.split('\\')[-1],b[f.split('\\')[-1]])
            text=self.get_CNtext(f)
            seg_words=jieba.lcut(text)
            stw=stop_words()
            text_del_stw = [wd for wd in seg_words if wd not in stw]
            self.corpus.append(' '.join(text_del_stw))
        return self.corpus




if __name__ == '__main__':


    print(stop_words())




