import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

class my_cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs=nn.Sequential(
            nn.Conv2d(1,16,3,stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(16,32,4,stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2,2),
            
            nn.Flatten(),
            nn.Linear(800,10),
            nn.Softmax(dim=-1)
        )
        
    def forward(self,x):
        x=self.convs(x)
        return x

class CNN:
    def __init__(self,learing_rate=1e-3,batch_size=32,epochs=10):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self.training_data = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor()
        )

        self.test_data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor()
        )
        
        self.model = my_cnn().to(self.device)

        self.learning_rate=learing_rate
        self.batch_size=batch_size
        self.epochs=epochs

        self.loss_fn=nn.CrossEntropyLoss()
        
    def train_loop(self,dataloader,model,loss_fn,optimizer):
        model.train()
        acc,total=0,0
        for batch,(X,y) in enumerate(dataloader):
            if torch.cuda.is_available():
                X=X.to("cuda")
                y=y.to("cuda")
            pred=model(X)
            loss=loss_fn(pred,y)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total+=self.batch_size
            acc+=(pred.argmax(1) == y).type(torch.float).sum().item()
            
            if batch%100==0 :
                print(f"loss:{loss:>7f}")
                print(f"accuracy:{acc/total:>7f}")
                acc,total=0,0
        
    def train(self):
        self.optimizer=torch.optim.SGD(self.model.parameters(),lr=self.learning_rate,momentum=0.9)
        self.train_dataloader=DataLoader(self.training_data,batch_size=self.batch_size,shuffle=True)
        self.test_dataloader=DataLoader(self.test_data,batch_size=self.batch_size,shuffle=True)
        for t in range(self.epochs):
            self.train_loop(self.train_dataloader,self.model,self.loss_fn,self.optimizer)
        print("finished training")
        self.predict(self.test_dataloader,self.model)
            
    def predict(self,dataloader,model):
        model.eval()
        with torch.no_grad():
            for batch,(X,y) in enumerate(dataloader):
                if torch.cuda.is_available():
                    X=X.to("cuda")
                    y=y.to("cuda")
                pred=model(X)
                fig=plt.figure(figsize=(4,4))
                self.pred=[]
                for i in range(16):
                    if i>=16 :
                        break
                    plt.subplot(4,4,i+1)
                    plt.imshow(X[i,0].cpu().numpy()*127.5+127.5,cmap='gray')
                    plt.axis('off')
                    if pred[i].argmax()==0 :
                        self.pred.append("T-Shirt")
                    if pred[i].argmax()==1 :
                        self.pred.append("Trouser")
                    if pred[i].argmax()==2 :
                        self.pred.append("Pullover")
                    if pred[i].argmax()==3 :
                        self.pred.append("Dress")
                    if pred[i].argmax()==4 :
                        self.pred.append("Coat")
                    if pred[i].argmax()==5 :
                        self.pred.append("Sandal")
                    if pred[i].argmax()==6 :
                        self.pred.append("Shirt")
                    if pred[i].argmax()==7 :
                        self.pred.append("Sneaker")
                    if pred[i].argmax()==8 :
                        self.pred.append("Bag")
                    if pred[i].argmax()==9 :
                        self.pred.append("Ankle Boot")
                    
                plt.savefig('images/image.png')
                
                if batch==0 :
                    break
            
