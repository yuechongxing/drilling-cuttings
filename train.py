import time
from torchstat import stat

import numpy as np
import torch    # torch核心以依赖
import torch.backends.cudnn as cudnn

from torch.functional import Tensor

from torch import Tensor #修改

import torch.nn.functional as F
import torch.optim as optim    # pytorch优化工具箱
from torch import nn   # pytorch网络核心依赖
from torch.utils.data import DataLoader
from tqdm import tqdm

from nets.mobilenet import mobilenet_v2
from nets.resnet50 import resnet50
# from nets.vgg16 import vgg16
# from nets.resnet18 import resnet18
# from nets.mobilenet_psa import mobilenet_psa
# from nets.pro_mobilenet import pro_mobilenet_v2
from nets.mobilenet_v3 import mobilenet_v3
# from nets.mobilenet_pspsa import mobilenet_psspa
from nets.mobilenet_bile import mobilenet_bile
from nets.convext import convnext_tiny
# from nets.shufflenetv2_origin import shufflenet_v2_x1_0
from nets.shufflecent_modify import shufflenet_v2_x1_0
from nets.efficientNet_model import efficientnetv2_s

from nets.mobilelenetv2_modify import modelilev2_modify
# from nets.mobile_bile_v2 import mobilenet_bile_v2
# from nets.mobile_bile_v3 import mobilenet_bile_v3

# from nets.mobile_bile_v21 import mobilenet_bile_v21



from utils.utils import weights_init
from utils.dataloader import DataGenerator, detection_collate

import matplotlib.pyplot as plt



#以字典形式存放模型的名字
get_model_from_name = {
    "mobilenet" : mobilenet_v2,
    "resnet50"  : resnet50,
    # "vgg16"     : vgg16,
    # "resnet18"  : resnet18,
    # "pro_mobilenet" : pro_mobilenet_v2 ,
    "mobilenet_v3"  : mobilenet_v3,
    # "mobilenet_psa" : mobilenet_psa,
    # "mobilenet_psspa":mobilenet_psspa,
    "mobilenet_bile":mobilenet_bile,
    # "mobilenet_bile_v2" : mobilenet_bile_v2,
    # "mobilenet_bile_v3" : mobilenet_bile_v3,
    # "mobilenet_bile_v21" : mobilenet_bile_v21,
    "convext":convnext_tiny,
    "shufflenet_v2_x1_0":shufflenet_v2_x1_0,
    "efficientnetv2_s":efficientnetv2_s,
    "modelilev2_modify":modelilev2_modify
}

#这是迁移学习用的
freeze_layers = {   #以字典的形式存放模型层数
    "mobilenet" :81,
    "resnet50"  :173,
    "vgg16"     :19,
}

#动态修改学习率------------->得到学习率   optimizer.param_groups[0]["lr"] 
def get_lr(optimizer):     
    for param_group in optimizer.param_groups:  #optimizer.param_groups这是一个长度为2的list 
        return param_group['lr']

def fit_one_epoch(net,epoch,epoch_size,epoch_size_val,gen,genval,Epoch,cuda):
    total_loss = 0
    total_accuracy = 0
    val_toal_loss = 0
    
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:  #tqdm作用是扩展进度条   desc用来说明进度条显示什么
        for iteration, batch in enumerate(gen):   #enumerate  这个是为了遍历列表gen里边的索引和值
            if iteration >= epoch_size:    #iteration迭代的意思
                break
            images, targets = batch   #batch[0], batch[1]
            
            with torch.no_grad():    #torch.no_grad 在该模块下，所有计算得出的tensor的requires_grad都自动设置为False
                images      = torch.from_numpy(images).type(torch.FloatTensor)
                targets     = torch.from_numpy(targets).type(torch.FloatTensor).long()
                if cuda:
                    images  = images.cuda()
                    targets = targets.cuda()

            optimizer.zero_grad()  #梯度初始化为零，把loss关于weight的导数变成0
            outputs = net(images)   #forward：将数据传入模型，前向传播求出预测的值
            loss    = nn.CrossEntropyLoss()(outputs, targets)    #求loss
            loss.backward()    #backward：反向传播求梯度
            optimizer.step()    #梯度值计算好后，调用该函数更新  ptimizer：更新所有参数
            total_loss += loss.item()  # loss的累加  ，减少内存的消耗，因为loss是一个variable
            
            with torch.no_grad():
                accuracy = torch.mean((torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
                total_accuracy += accuracy.item()  #准确率的累加
    
            ACCU = total_accuracy / (iteration + 1)  #一个迭代周期的准确率
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'accuracy'  : total_accuracy / (iteration + 1), 
                                'lr'        : get_lr(optimizer)})     #输入一个字典，显示实验指标
            pbar.update(1)

    print('Start Validation')   #开始验证
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:   #扩展进度条
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images, targets = batch
            with torch.no_grad():
                images = torch.from_numpy(images).type(torch.FloatTensor)
                targets = torch.from_numpy(targets).type(torch.FloatTensor).long()
                if cuda:
                    images = images.cuda()
                    targets = targets.cuda()

                optimizer.zero_grad()

                outputs = net(images)
                val_loss = nn.CrossEntropyLoss()(outputs, targets)
                
                val_toal_loss += val_loss.item()
                
            pbar.set_postfix(**{'total_loss': val_toal_loss / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
    print('Finish Validation')   #完成验证
    
    Total_loss=total_loss/(epoch_size+1) 
    Val_loss=val_toal_loss/(epoch_size_val+1)
    
  
    
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (Total_loss,Val_loss))
    print('Saving state, iter:', str(epoch+1))  
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f-Acc%.4f.pth'%((epoch+1),
                                                                                           total_loss/(epoch_size+1),
                                                                                           val_toal_loss/(epoch_size_val+1),
                                                                                         ACCU))
 
    # torch.save(model.state_dict(), 'E:/vgg16log/Epoch%d-Total_Loss%.4f-Val_Loss%.4f-Acc%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_toal_loss/(epoch_size_val+1),ACCU))
    
    



    
    

#---------------------------------------------------#
#   获得我们所要区分的类别
#---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes 加载我们所要区分的类'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]   #默认删除字符串头和尾的空格和换行符   strip()函数用于删除指定字符串头尾信息（默认为空格）
    return class_names

#----------------------------------------#
#   主函数
#----------------------------------------#
if __name__ == "__main__":
    log_dir = "./logs/"     #权值文件存储路径
    #---------------------#
    #   所用模型种类
    #---------------------#
    backbone = "shufflenet_v2_x1_0" 
    # backbone = "mobilenet"
    #---------------------#
    #   输入的图片大小
    #---------------------#
    input_shape = [224,224,3]
    
    
    #-------------------------------#
    #   Cuda的使用
    #-------------------------------#
    Cuda = False

    #-------------------------------#
    #   是否使用网络的imagenet
    #   预训练权重
    #-------------------------------#
    pretrained = True    

    classes_path = './model_data/new_classes.txt' #需要区分类别的路径
    class_names = get_classes(classes_path)   #['jz', 'kz', 'pz', 'tz']
    num_classes = len(class_names)   # 4


    # backbone 选择模型
    assert backbone in ["mobilenet_bile_v21","mobilenet",
                        "resnet50", "vgg16","resnet18",
                        "pro_mobilenet","mobilenet_v3",
                        "mobilenet_psa","mobilenet_psspa",
                        "mobilenet_bile","mobilenet_bile_v2",
                        "mobilenet_bile_v3","convext","shufflenet_v2_x1_0",
                        "efficientnetv2_s","model_v2","modelilev2_modify"]

    model = get_model_from_name[backbone](num_classes=num_classes)
    
    if not pretrained:
        weights_init(model) #模型权重初始化

    #------------------------------------------#
    #   注释部分可用于断点训练
    #   将训练好的模型重新载入
    #------------------------------------------#
    # # 加快模型训练的效率
    # model_path = "logs\Epoch45-Total_Loss0.3655-Val_Loss0.5183-Acc0.8752.pth"
    # print('Loading weights into state dict...')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model_dict = model.state_dict()
    # pretrained_dict = torch.load(model_path, map_location=device)
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # model_dict.update(pretrained_dict)
    
    # model.load_state_dict(model_dict)

    with open(r"./cls_train.txt","r") as f:
        lines = f.readlines()

    np.random.seed(10101)  #随机数种子，使后边生成的随机数一直相同
    np.random.shuffle(lines)  #shuffle() 方法将序列的所有元素随机排序。
    np.random.seed(None)   #不再使用随机数种子需要将种子置为空
    num_val = int(len(lines)*0.1)     #将数据量缩小0.1倍
    num_train = len(lines) - num_val
 
    net = model.train()    #使用BatchNormalizetion()和Dropout()
    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()
        
    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if False:
        #--------------------------------------------#
        #   BATCH_SIZE不要太小，不然训练效果很差
        #--------------------------------------------#
        lr              = 1e-3
        Batch_size      = 16
        Init_Epoch      = 0  #网络最开始训练的次数
        Freeze_Epoch    = 50 #网络冻结起来所需要训练的次数
        
        optimizer       = optim.Adam(net.parameters(),lr,weight_decay=5e-4)
        lr_scheduler    = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        train_dataset   = DataGenerator(input_shape,lines[:num_train])
        val_dataset     = DataGenerator(input_shape,lines[num_train:], False)
        gen             = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                drop_last=True, collate_fn=detection_collate)
        gen_val         = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                drop_last=True, collate_fn=detection_collate)

        epoch_size      = train_dataset.get_len()//Batch_size
        epoch_size_val  = val_dataset.get_len()//Batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
        #------------------------------------#
        #   冻结一定部分训练
        #------------------------------------#
        model.freeze_backbone()

        for epoch in range(Init_Epoch,Freeze_Epoch):
            fit_one_epoch(model,epoch,epoch_size,epoch_size_val,gen,gen_val,Freeze_Epoch,Cuda)
            lr_scheduler.step()

    if True:
        #--------------------------------------------#
        #   BATCH_SIZE不要太小，不然训练效果很差
        #--------------------------------------------#
        lr              = 1e-3
        Batch_size      = 32      #每一次训练喂入的数据
        Freeze_Epoch    = 0
        Epoch           = 100    #解冻之后训练的轮数

        optimizer       = optim.Adam(net.parameters(),lr,weight_decay=5e-4)
        lr_scheduler    = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        train_dataset   = DataGenerator(input_shape,lines[:num_train])
        val_dataset     = DataGenerator(input_shape,lines[num_train:], False)
        gen             = DataLoader(train_dataset, batch_size=Batch_size, num_workers=2, pin_memory=True,
                                drop_last=True, collate_fn=detection_collate)
        gen_val         = DataLoader(val_dataset, batch_size=Batch_size, num_workers=2, pin_memory=True,
                                drop_last=True, collate_fn=detection_collate)

        
        epoch_size      = train_dataset.get_len()//Batch_size
        epoch_size_val  = val_dataset.get_len()//Batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
        #------------------------------------#
        #   解冻后训练
        #------------------------------------#
        # model.Unfreeze_backbone()

        for epoch in range(Freeze_Epoch,Epoch):  
            
          
            print("第",epoch+1,"轮")
       
            fit_one_epoch(model,epoch,epoch_size,epoch_size_val,gen,gen_val,Epoch,Cuda)
            lr_scheduler.step()
            
            