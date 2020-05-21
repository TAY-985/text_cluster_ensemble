#界面、业务逻辑分离
import sys
from PyQt5.QtWidgets import   QApplication ,QMainWindow,QFileDialog
from PyQt5.QtWidgets import QMessageBox,QTreeWidget,QTreeWidgetItem
from PyQt5.QtGui import  QIcon,QPixmap
import os.path

from function_code.TextCluster import textClusterEnsemble
from text_gui.try_txt_gui import Ui_MainWindow
import numpy as np


class mywin(QMainWindow,Ui_MainWindow):
    def __init__(self,parent=None):
        super(mywin,self).__init__(parent)

        self.setupUi(self)
        self.setWindowTitle("文本聚类")

        self.algorithm =None
        self.aimK=None
        self.file_path=None
        self.label_path=None
        self.tce=None
        self.pred_label=[]

        self.chose_file_btn.clicked.connect(self.choose_file)
        self.label_btn.clicked.connect(self.true_label)
        self.file_path_txt.textChanged.connect(self.set_file_path)
        self.label_path_txt.textChanged.connect(self.set_label_path)
        self.km_radio.toggled.connect(self.choose_algorithm)
        self.av_ensembel_radio.toggled.connect(self.choose_algorithm)
        self.aimK_txt.textChanged.connect(self.input_aimK)
        self.iters_txt.textChanged.connect(self.inpu_iter)
        self.run_btn.clicked.connect(self.run_cluster)
        self.save_btn.clicked.connect(self.save_result)
        self.tree.doubleClicked.connect(self.show_text_content)


    def choose_file(self):
        file_path=QFileDialog.getExistingDirectory(self,"选择文件",'/')
        self.file_path_txt.setText(file_path)
        self.file_path=file_path
    def set_file_path(self):
        self.file_path=self.file_path_txt.text().strip()
    def set_label_path(self):
        self.label_path=self.label_path_txt.text().strip()

    def true_label(self):
        label_path,_ = QFileDialog.getOpenFileName(self, "选择标签", '/','Text File (*.txt)')
        self.label_path_txt.setText(label_path)
        self.label_path=label_path
    def choose_algorithm(self):
        sender=self.sender()
        if sender.text()=='simpleKmeans':
            self.algorithm='simpleKmeans'
            self.label_iter.setVisible(False)
            self.iters_txt.setVisible(False)
        elif sender.text()=='Average-Ensemble':
            self.label_iter.setVisible(True)
            self.iters_txt.setVisible(True)
            self.algorithm='Average-Ensemble'
    def input_aimK(self):
        if self.aimK_txt.text():
            self.aimK=int(self.aimK_txt.text())
    def inpu_iter(self):
        self.iter=None
        if self.iters_txt.text():
            self.iter=int(self.iters_txt.text())




    def run_cluster(self):
        import os.path as op
        if self.file_path and self.label_path \
                and op.isdir(self.file_path) and op.isfile(self.label_path):
            if self.algorithm and self.aimK :
                if not self.tce:
                    self.tce = textClusterEnsemble(file_path=self.file_path, label_path=self.label_path)
                if self.algorithm=='simpleKmeans':
                    self.ARI_show.clear()
                    self.text_content.clear()
                    label=self.tce.simpleKmeans(self.aimK)
                    self.pred_label=label
                    self.list_tree(label)
                    score = self.tce.ARI_evaluation(label)
                    #print(score)
                    self.ARI_show.setText(str(score))
                    self.ARI_show.adjustSize()

                elif self.algorithm=='Average-Ensemble' and self.iter:
                    self.ARI_show.clear()
                    self.text_content.clear()
                    label=avlable=self.tce.averEnsemble(aimK=self.aimK, iter=self.iter)
                    self.pred_label=label
                    self.list_tree(label)
                    score=self.tce.ARI_evaluation(label)

                    self.ARI_show.setText(str(score))
                    self.ARI_show.adjustSize()

                else:QMessageBox.critical(self, "错误", 'error')
            else :QMessageBox.critical(self, "错误", 'error')
        else :
            QMessageBox.critical(self, "错误", 'error')

    def clear_tree(self):
        self.tree.clear()

    def list_tree(self,pred_label):
        self.clear_tree()
        num_class=list(np.unique(pred_label))
        root=[]
        for i in num_class:
            rt=QTreeWidgetItem(self.tree)
            rt.setText(0,str(i))
           # rt.setIcon(0,QIcon('./image/folder.png'))
            root.append(rt)
        #root=[QTreeWidgetItem(self.tree).setText(0,str(i)) for i in num_class]
        for i in range(len(pred_label)):
            child=QTreeWidgetItem(root[pred_label[i]])
            child.setText(0,str(self.tce.text_handle.text_name[i]))
            child.setText(1,str(pred_label[i]))
           # print(child.text(0))

    def show_text_content(self):
        item=self.tree.currentItem()
        name=item.text(0)
        path=os.path.join(self.file_path,name)
        with open(path, 'r', encoding='ansi') as f:
            content=f.read()
        self.text_content.setText(content)


    def save_result(self):
        save_path=QFileDialog.getExistingDirectory(self,'文件保存','/')
        self.save_path_txt.setText(save_path)
        if self.save_path_txt.text():
            if list(self.pred_label):
                self.tce.save_clustered_file(save_path,self.pred_label)
            else:QMessageBox.critical(self, "错误", 'error')
        else:
            QMessageBox.critical(self, "错误", 'error')



if __name__ == '__main__':

    app=QApplication(sys.argv)
    wy=mywin()
    wy.show()
    sys.exit(app.exec_())