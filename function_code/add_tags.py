import os


def add_trueLabel(dir_path,label_path):
    if os.path.exists(label_path):
        os.remove(label_path)

    dir_list = os.listdir(dir_path)
    i = -1
    id = open(label_path, 'w', encoding='utf-8')
    for dir in dir_list:
        i = i + 1
        file_list = os.listdir(dir_path + '\\' + dir)
        for file in file_list:
            id.write(file + '\t' + str(i) + '\n')
    id.close()

if __name__ == '__main__':
    dir_path1 = r'E:\毕设\text_cluster\cor_500'
    true_lable1=r'E:\毕设\text_cluster\cor_500\cor_500_tureLabel.txt'

    add_trueLabel(dir_path1,true_lable1)


