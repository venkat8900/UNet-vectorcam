import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sklearn
from detecto import core, utils
from torchvision import transforms
import matplotlib.pyplot as plt
from detecto.visualize import show_labeled_image
from detecto.utils import read_image


#provide labels path and images path 
labels_path =r"D:\JHU\CBID Lab\VectorCam\UNet-Scratch\data\test_labels"
images_path = r"D:\JHU\CBID Lab\VectorCam\UNet-Scratch\data\test_images"

#converting all the xml files to single csv file
labels_df = utils.xml_to_csv(labels_path)

full_df=labels_df[labels_df['class']=='f']

# Images
images_path=r'D:\JHU\CBID Lab\VectorCam\UNet-Scratch\data\test_images'
full_path=r'D:\JHU\CBID Lab\VectorCam\UNet-Scratch\data\test_full_image'

for index,rows in full_df.iterrows():
    try:
        name=rows["filename"]
        ImagePath=os.path.join(images_path,name)
        #reading the image
        image=read_image(ImagePath)
        #selecting the cropped part
        image=image[int(rows['ymin']):int(rows['ymax']),int(rows['xmin']):int(rows['xmax'])]  
        name=name[:-4]
        #saving the images in the output folder
        cv2.imwrite(os.path.join(full_path,f'{name}.png'),cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(e)  


# Body - Images
images_path=r'D:\JHU\CBID Lab\VectorCam\UNet-Scratch\data\test_images'
body_path=r'D:\JHU\CBID Lab\VectorCam\UNet-Scratch\data\test_body_image'

for index,rows in body_df.iterrows():
    try:
        name=rows["filename"]
        ImagePath=os.path.join(images_path,name)
        #reading the image
        image=read_image(ImagePath)
        #selecting the cropped part
        image=image[int(rows['ymin']):int(rows['ymax']),int(rows['xmin']):int(rows['xmax'])]  
        #saving the images in the output folder
        name=name[:-4]
        cv2.imwrite(os.path.join(body_path,f'{name}.jpg'),cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(e)  

# masks
masks_path=r'D:\JHU\CBID Lab\VectorCam\UNet-Scratch\data\test_masks'
full_path=r'D:\JHU\CBID Lab\VectorCam\UNet-Scratch\data\test_full_mask'

for index,rows in full_df.iterrows():
    try:
        name=rows["filename"]
        ImagePath=os.path.join(masks_path,name)
        #reading the image
        image=read_image(ImagePath)
        #selecting the cropped part
        image=image[int(rows['ymin']):int(rows['ymax']),int(rows['xmin']):int(rows['xmax'])]  
        #saving the images in the output folder
        name=name[:-4]
        cv2.imwrite(os.path.join(full_path,f'{name}.png'),image)
    except Exception as e:
        print(e)  

# Body - masks
masks_path=r'D:\JHU\CBID Lab\VectorCam\UNet-Scratch\data\test_masks'
body_path=r'D:\JHU\CBID Lab\VectorCam\UNet-Scratch\data\test_body_mask'

for index,rows in body_df.iterrows():
    try:
        name=rows["filename"]
        ImagePath=os.path.join(masks_path,name)
        #reading the image
        image=read_image(ImagePath)
        #selecting the cropped part
        image=image[int(rows['ymin']):int(rows['ymax']),int(rows['xmin']):int(rows['xmax'])]  
        #saving the images in the output folder
        name=name[:-4]
        cv2.imwrite(os.path.join(body_path,f'{name}.jpg'),cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(e)  
