from PIL import Image

from classification import Classification

import torch,time
import torch
import torchvision.transforms as transforms


classfication = Classification() 

while True:
    img = input('Input image filename:')
    
    
    start = time.time()
    
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        class_name = classfication.detect_image(image)
        print(class_name)
       
        end = time.time()
        total_time = end - start
        print('total_time:{:.2f}'.format(total_time))
        
        







