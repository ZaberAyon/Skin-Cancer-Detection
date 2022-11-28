import cv2
import csv
import os
import math
import shutil
import pandas as pd
from tqdm import tqdm

basePath = "D:\Projects\Skin_cancer\PH2Dataset\PH2Dataset"
dest_dir = "D:\Projects\Skin_cancer\Skin-Cancer-Detection\skinLesion"
def dataset_info_from_csv(filepath):
    diagnoses = []
    dataFrame = pd.read_csv(filepath)
    print(dataFrame.columns)
    filenames = dataFrame['Image Name'].to_list()
    common_nevus = dataFrame['Common Nevus'].to_list()
    atypical_nevus = dataFrame['Atypical Nevus'].to_list()
    melanoma = dataFrame['Melanoma'].to_list()
    asymmetry = dataFrame['Asymmetry\n(0/1/2)'].to_list()


    for i in range(len(filenames)):
        if(common_nevus[i]=='X'):
            diagnoses.append('benign')
        if(atypical_nevus[i]=='X'):
            diagnoses.append('benign')
        if(melanoma[i]=='X'):
            diagnoses.append("melanoma")

    return filenames, diagnoses


filenames, diagnoses = dataset_info_from_csv("D:\Projects\Skin_cancer\PH2Dataset\PH2Dataset\PH2_dataset.csv")
resultDf = pd.DataFrame(list(zip(filenames,diagnoses)),columns=['filename','diagnoses'])
resultDf.to_csv("minimized.csv")

for i in filenames:
    image_path = os.path.join(basePath, "PH2 Dataset images", i, f"{i}_Dermoscopic_Image", f"{i}.bmp")
    dest = os.path.join(dest_dir, f"{i}.jpg")
    image = cv2.imread(image_path)
    cv2.imwrite(dest, image)


print(filenames)
print(diagnoses)