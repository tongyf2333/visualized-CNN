import sys
import os
from train import CNN
from PyQt6.QtWidgets import (QWidget,QPushButton,QApplication,QLineEdit,QLabel)
from PyQt6.QtGui import (QImage,QPainter)
from PyQt6.QtCore import (QTimer,QRect)

class mywidget(QWidget):
    
    def __init__(self):
        super().__init__()
        self.timer=QTimer(self)
        self.timer.start(1000)
        self.neural=CNN()
        self.initUI()
        if os.path.isfile('images/image.png'):
            os.remove('images/image.png')
        
    def initUI(self):
        #开始训练按钮
        btn=QPushButton('开始训练',self)
        btn.setGeometry(625,475,150,100)
        btn.clicked.connect(self.train)
        
        #超参数输入框
        self.input_batch=QLineEdit(self)
        self.input_batch.setGeometry(600,50,150,50)
        self.input_batch.textChanged.connect(self.change_batch)
        
        self.input_lr=QLineEdit(self)
        self.input_lr.setGeometry(600,150,150,50)
        self.input_lr.textChanged.connect(self.change_lr)
        
        self.input_epoch=QLineEdit(self)
        self.input_epoch.setGeometry(600,250,150,50)
        self.input_epoch.textChanged.connect(self.change_epoch)
        
        #超参数标签
        label_batch=QLabel('batch size(at least 16):',self)
        label_batch.setGeometry(450,50,150,50)
        
        label_lr=QLabel('learning rate:',self)
        label_lr.setGeometry(450,150,150,50)
        
        label_epoch=QLabel('number of epoch:',self)
        label_epoch.setGeometry(450,250,150,50)
        
        #设置窗口参数
        self.setGeometry(300,300,800,600)
        self.setWindowTitle('可视化衣物识别工具')
        self.show()
    
    def change_batch(self):
        self.neural.batch_size=int(self.input_batch.text())
        
    def change_lr(self):
        self.neural.learning_rate=float(self.input_lr.text())
        
    def change_epoch(self):
        self.neural.epochs=int(self.input_epoch.text())
        
    def train(self):
        self.neural.train()
        
        print(self.neural.pred)
        
        label_pred0=QLabel("0:"+self.neural.pred[0],self)
        label_pred0.setGeometry(0,400,120,20)
        label_pred0.show()
        
        label_pred1=QLabel("1:"+self.neural.pred[1],self)
        label_pred1.setGeometry(0,420,120,20)
        label_pred1.show()
        
        label_pred2=QLabel("2:"+self.neural.pred[2],self)
        label_pred2.setGeometry(0,440,120,20)
        label_pred2.show()
        
        label_pred3=QLabel("3:"+self.neural.pred[3],self)
        label_pred3.setGeometry(0,460,120,20)
        label_pred3.show()
        
        label_pred4=QLabel("4:"+self.neural.pred[4],self)
        label_pred4.setGeometry(0,480,120,20)
        label_pred4.show()
        
        label_pred5=QLabel("5:"+self.neural.pred[5],self)
        label_pred5.setGeometry(0,500,120,20)
        label_pred5.show()
        
        label_pred6=QLabel("6:"+self.neural.pred[6],self)
        label_pred6.setGeometry(0,520,120,20)
        label_pred6.show()
        
        label_pred7=QLabel("7:"+self.neural.pred[7],self)
        label_pred7.setGeometry(0,540,120,20)
        label_pred7.show()
        
        label_pred8=QLabel("8:"+self.neural.pred[8],self)
        label_pred8.setGeometry(120,400,120,20)
        label_pred8.show()
        
        label_pred9=QLabel("9:"+self.neural.pred[9],self)
        label_pred9.setGeometry(120,420,120,20)
        label_pred9.show()
        
        label_pred10=QLabel("10:"+self.neural.pred[10],self)
        label_pred10.setGeometry(120,440,120,20)
        label_pred10.show()
        
        label_pred11=QLabel("11:"+self.neural.pred[11],self)
        label_pred11.setGeometry(120,460,120,20)
        label_pred11.show()
        
        label_pred12=QLabel("12:"+self.neural.pred[12],self)
        label_pred12.setGeometry(120,480,120,20)
        label_pred12.show()
        
        label_pred13=QLabel("13:"+self.neural.pred[13],self)
        label_pred13.setGeometry(120,500,120,20)
        label_pred13.show()
        
        label_pred14=QLabel("14:"+self.neural.pred[14],self)
        label_pred14.setGeometry(120,520,120,20)
        label_pred14.show()
        
        label_pred15=QLabel("15:"+self.neural.pred[15],self)
        label_pred15.setGeometry(120,540,120,20)
        label_pred15.show()
        
    def paintEvent(self,e):
        qp=QPainter()
        qp.begin(self)
        self.drawPic(qp)
        qp.end()
        
    def drawPic(self,qp):
        img=QImage('images/image.png')
        rect=QRect(0,0,img.width(),img.height())
        qp.drawImage(rect,img)
        
def main():
    app=QApplication(sys.argv)
    window=mywidget()
    sys.exit(app.exec())
    
if __name__=='__main__':
    main()