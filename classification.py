import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from nets.mobilenet import mobilenet_v2
# from nets.resnet18 import resnet18
from nets.resnet50 import resnet50
# from nets.mobilenet_psa import mobilenet_psa
# from nets.pro_mobilenet import pro_mobilenet_v2
from nets.mobilenet_v3 import mobilenet_v3
# from nets.vgg16 import vgg16
# from nets.mobilenet_pspsa import mobilenet_psspa
# from nets.mobilenet_bile import mobilenet_bile
# from nets.mobile_bile_v2 import mobilenet_bile_v2
# from nets.mobile_bile_v3 import mobilenet_bile_v3


# from nets.mobile_bile_v21 import mobilenet_bile_v21
from nets.shufflecent_modify  import shufflenet_v2_x1_0

from utils.utils import letterbox_image

get_model_from_name = {
    "mobilenet":mobilenet_v2,
    "resnet50":resnet50,
    # "vgg16":vgg16,
    # "resnet18":resnet18,
    # "pro_mobilenet":pro_mobilenet_v2,
    "mobilenet_v3":mobilenet_v3,
    # "mobilenet_psa":mobilenet_psa,
    # "mobilenet_psspa":mobilenet_psspa,
    # "mobilenet_bile":mobilenet_bile,
    # "mobilenet_bile_v2" : mobilenet_bile_v2,
    # "mobilenet_bile_v3" : mobilenet_bile_v3,
    # "mobilenet_bile_v21" : mobilenet_bile_v21,
    "shufflenet_v2_x1_0":shufflenet_v2_x1_0,
}

#----------------------------------------#
#   预处理训练图片
#----------------------------------------#
def _preprocess_input(x,):
    x /= 127.5
    x -= 1.
    return x

#--------------------------------------------#
#   使用自己训练好的模型预测需要修改3个参数
#   model_path和classes_path和backbone都需要修改！
#--------------------------------------------#
class Classification(object):
    _defaults = {
        "model_path"    : 'logs\Epoch148-Total_Loss0.2828-Val_Loss0.4502-Acc0.9032.pth',
        "classes_path"  : 'model_data\\new_classes.txt',
        "input_shape"   : [224,224,3],
        "backbone"      : "shufflenet_v2_x1_0",
        "cuda"          : False
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化classification
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.generate()

    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)

        # 计算总的种类
        self.num_classes = len(self.class_names)

        assert self.backbone in ["mobilenet_bile_v21",
                                 "mobilenet", "resnet50",
                                 "vgg16","resnet18","pro_mobilenet",
                                 "mobilenet_v3","mobilenet_psa","mobilenet_psspa",
                                 "mobilenet_bile","mobilenet_bile_v2",
                                 "mobilenet_bile_v3","shufflenet_v2_x1_0"]

        self.model = get_model_from_name[self.backbone](num_classes=self.num_classes, pretrained=False)

        self.model = self.model.eval()     #禁止使用BatchNormalizetion()和Dropout()
        #state_dict = torch.load(self.model_path)

        state_dict = torch.load(self.model_path, map_location=lambda storage, loc: storage)  #gpu模型转为cpu模型运行

        self.model.load_state_dict(state_dict,False)
        if self.cuda:
            self.model = nn.DataParallel(self.model)
            self.model = self.model.cuda()
        print('{} model, and classes loaded.'.format(model_path))

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        old_image = copy.deepcopy(image)
        
        crop_img = letterbox_image(image, [self.input_shape[0],self.input_shape[1]])
        photo = np.array(crop_img,dtype = np.float32)

        # 图片预处理，归一化
        photo = np.reshape(_preprocess_input(photo),[1,self.input_shape[0],self.input_shape[1],self.input_shape[2]])
        photo = np.transpose(photo,(0,3,1,2))

        with torch.no_grad():
            photo = Variable(torch.from_numpy(photo).type(torch.FloatTensor))
            if self.cuda:
                photo = photo.cuda()
            preds = torch.softmax(self.model(photo)[0], dim=-1).cpu().numpy()

        class_name = self.class_names[np.argmax(preds)]
        probability = np.max(preds)
        
        
        # plt.rcParams两行是用于解决标签不能显示汉字的问题
        # plt.rcParams['font.sans-serif']=['SimHei']
        # plt.rcParams["font.family"] = "sans-serif"
        # plt.rcParams['axes.unicode_minus'] = False

        plt.subplot(1, 1, 1)
        plt.imshow(np.array(old_image))
        plt.title('%s %.1f %%' %(class_name, probability*100),x=0.175,y=0.92,bbox=dict(facecolor='black', edgecolor='black', alpha=0.65 ),color='red')
        plt.show()
        
        return class_name

    def close_session(self):
        self.sess.close()
