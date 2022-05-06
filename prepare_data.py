
from pathlib import Path

from tqdm import tqdm
import cv2
import numpy as np



data_path = Path('D:/JHU/CBID Lab/VectorCam/Data')

train_path = Path('D:/JHU/CBID Lab/VectorCam/Data')

cropped_train_images = Path('D:/JHU/CBID Lab/VectorCam/UNet-Scratch/data/train_images')

cropped_train_masks = Path('D:/JHU/CBID Lab/VectorCam/UNet-Scratch/data/train_masks')

cropped_test_images = Path('D:/JHU/CBID Lab/VectorCam/UNet-Scratch/data/test_images')

cropped_test_masks = Path('D:/JHU/CBID Lab/VectorCam/UNet-Scratch/data/test_masks')


binary_factor = 255

if __name__ == '__main__':

    counter = 0

    for file_name in tqdm(list((train_path / 'ANNOTATED').glob('*'))):

        if counter%5 != 0:

            img = cv2.imread(str(file_name))
            cv2.imwrite(str(cropped_train_images / (str(counter) + '.jpg')), img) #,[cv2.IMWRITE_JPEG_QUALITY, 100])

            counter += 1

        else:

            img = cv2.imread(str(file_name))
    
            cv2.imwrite(str(cropped_test_images / (str(counter) + '.jpg')), img) # ,[cv2.IMWRITE_JPEG_QUALITY, 100])
            counter += 1

    counter = 0
    for file_name in tqdm(list((train_path / 'ground truth').glob('*'))):
        if counter%5 != 0:

            img = cv2.imread(str(file_name))

            cv2.imwrite(str(cropped_train_masks / (str(counter) + '.jpg')), img)

            counter += 1

        else:

            img = cv2.imread(str(file_name))
    
            cv2.imwrite(str(cropped_test_masks / (str(counter) + '.jpg')), img)

            counter += 1

