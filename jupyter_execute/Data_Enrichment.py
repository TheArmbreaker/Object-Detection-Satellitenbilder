#!/usr/bin/env python
# coding: utf-8

# # Data Enrichment - Tiling and Rotating 

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import os
import re
import numpy as np
import subprocess
import cv2
import math
import shutil


# In[ ]:


myPathLabel = '/content/drive/MyDrive/dataset/labels/validation'
myPathImages = '/content/drive/MyDrive/dataset/images/validation'


# ## Tiling

# Wie im Abschnitt Model Evaluation erläutert, ist eine Möglichkeit zum Umgang mit großen Bilder das "Kacheln" (englisch tiling). Die besondere Herausforderung ist hierbei, dass sowohl die Bild-, wie auch die Labeldatein zerlegt werden müssen. Hierbei wird auf ein bereits vorhandenes Github-Repository von {cite:t}`YOLO_Dataset_Tiling` zurückgegriffen. 

# In[ ]:


#! git clone https://github.com/slanj/yolo-tiling.git


# In[ ]:


label_list = os.listdir(myPathLabel)
#image_list = os.listdir(myPathImages)


# In[ ]:


image_list = list()
for x in label_list:
  png = str.split(x, ".")[0] + ".png"
  image_list.append(png)


# In[ ]:


for file1 in image_list:
    src_path = '/content/drive/MyDrive/dataset/images/validation/' + file1
    dst_path = '/content/drive/MyDrive/dataset_tile/org/validation/' + file1
    shutil.copy(src_path, dst_path)
    #print('Copied')
for file1 in label_list:
    src_path = '/content/drive/MyDrive/dataset/labels/validation/' + file1
    dst_path = '/content/drive/MyDrive/dataset_tile/org/validation/' + file1
    shutil.copy(src_path, dst_path)
    #print('Copied')


# Mit diesem Repository können nur Bilder mit Label (keine Backgroundbilder), deswegen werden diese herausgefiltert. 

# In[ ]:


get_ipython().run_line_magic('cd', 'yolo-tiling/')


# Mit Aufruf der `tile_yolo.py` werden die Bilder und Labels im entsprechenden Ordner in Kacheln der Größe 640x640 Pixel zerlegt und in einem entsprechenden Zielordner ablegt. Dies wird für Trainings- und Validierungsdaten durchgeführt. 

# In[ ]:


get_ipython().system(' python3 tile_yolo.py -source /content/drive/MyDrive/dataset_tile/org/validation -target /content/drive/MyDrive/dataset_tile/tiled/validation -ext .png -size 640')


# In[ ]:


tiled_list = os.listdir('/content/drive/MyDrive/dataset_tile/tiled/validation')


# Die enstandene Ordnerstruktur (validation -> images) entspricht nicht der Zielstruktur (images -> validation). Aus diesem Grund werden die Dateien erneut verschoben. 

# In[ ]:


tiled_img = list()
tiled_txt = list()
for file in tiled_list: 
  if str.split(file, ".")[1] == "png":
    tiled_img.append(file)
  elif str.split(file, ".")[1] == "txt":
    tiled_txt.append(file)


# In[ ]:


for file1 in tiled_img:
    src_path = '/content/drive/MyDrive/dataset_tile/tiled/validation/' + file1
    dst_path = '/content/drive/MyDrive/dataset_tile/tiled/validation/images/' + file1
    shutil.move(src_path, dst_path)
    #print('Copied')
for file1 in tiled_txt:
    src_path = '/content/drive/MyDrive/dataset_tile/tiled/validation/' + file1
    dst_path = '/content/drive/MyDrive/dataset_tile/tiled/validation/labels/' + file1
    shutil.move(src_path, dst_path)
    #print('Copied')


# ## Tiling Background

# Wie oben erwähnt, funktioniert das erwähnte Repo nur für Bilder mit Labels. Da die Backgroundbilder keine Labels haben, müssen diese seperat "getilt" werden. Dafür werden die Bibliotheken `cv2` und `math` genutzt.  

# In[ ]:


def get_tiles(src_path,dst_path,file):
  img = cv2.cvtColor(cv2.imread(f"{src_path}{file}"),cv2.COLOR_BGR2RGB)
  img_shape = img.shape
  tile_size = (640, 640)
  offset = (640, 640)
  
  for i in list(range(int(math.ceil(img_shape[0]/(offset[1] * 1.0))))):
      for j in list(range(int(math.ceil(img_shape[1]/(offset[0] * 1.0))))):
          cropped_img = img[offset[1]*i:min(offset[1]*i+tile_size[1], img_shape[0]), offset[0]*j:min(offset[0]*j+tile_size[0], img_shape[1])]
          # Debugging the tiles
          cv2.imwrite(dst_path + "debug_" + str(i) + "_" + str(j) + file, cv2.cvtColor(cropped_img,cv2.COLOR_BGR2RGB))
  return (i+1)*(j+1)


# Mit der oben aufgeführten Funktion werden die Bilder zerlegt. Da sich bei Bildern implizit um mehrdimensionale Listen handelt, werden diese entsprechend der gewünschten Teilung durch itteriert. 
# Wichtig hier ist es, die Farbkanäle mit cv2.COLOR_BGR2RGB anzupassen. Da diese  durch die `cv2`-Bibliothek vertauscht werden. 

# In[ ]:


dir_path = "/content/drive/MyDrive/dataset/labels/validation" #GoogleDrive Data Folders
img_path = "/content/drive/MyDrive/dataset/images/removed_background_validation/"
img_list_all = os.listdir(img_path)
labels_list_all  = os.listdir(dir_path)


# In[ ]:


image_all = list()
for img in img_list_all:
  image_all.append(str.split(img, ".")[0])


# In[ ]:


label_all = list()
for label in labels_list_all:
  label_all.append(str.split(label, ".")[0])


# In[ ]:


background_img = list()
no_background_img = list()
for filename in image_all:
  pic_name = filename+".png" 
  if filename in label_all:
    no_background_img.append(pic_name)
  else: 
    background_img.append(pic_name)


# Um die Backgroundbilder zu ermitteln, werden alle Bilder die keine Labeldatei haben in eine Liste gespeichert. 

# In[ ]:


count_tiles = 0
source_path = img_path
dest_path = '/content/drive/MyDrive/dataset_tile/tiled/images/validation/'
for i in background_img:
  count_tiles += get_tiles(source_path,dest_path,i)
  if count_tiles >= 207:
    break


# Anschließend werden die Backgroundbilder zerlegt bis 207 Kacheln entstanden sind. Die 207 entspricht der 10%-Anreicherung der Daten mit Backgroundbildern. 

# ## Rotate

# Um die Anzahl an Instanzen der Klasse Helikopter zu erhöhen, werden die vorhandenen Bilder und Label zusätzlich zu den YOLOv5 interen Funktionen um einen beliebigen Winkel gedreht. 

# In[ ]:


img_list = list()
text_list = list()
label_list = os.listdir(myPath)
for filename in label_list:
  path =  myPath + "/" + filename
  with open(path) as f:
    for lines in f:
      if (re.match("^1",lines)):
        png = str.split(filename, ".")[0] + ".png"
        txt = str.split(filename, ".")[0] + ".txt"
        text_list.append(txt)
        img_list.append(png)
        continue
img_list = np.unique(img_list)
label_list = np.unique(text_list)
# return img_list, text_list


# Dafür werden im ersten Schritt die Datein mit Helikoptern über die Einträge in der Labeldatei ermittelt. 

# Ähnlich wie bei dem Tiling ist auch beim Rotieren die Herausforderung das Label und Bilder rotiert werden müssen. Aus diesem Grund wird ebenfalls auf ein vorhandenes Gihtub-Repository zurückgegriffen. {cite:p}`Bbox_rotate`

# In[ ]:


get_ipython().system(' git clone https://github.com/usmanr149/Yolo_bbox_manipulation.git')


# Vorrausetzung für die Ausführung des Codes ist, dass sich die Bilder und Label im gleichen Ordner befinden. Aus diesem Grund müssen diese erneut verschoben werden. 

# In[ ]:


for file1 in img_list:
    src_path = '/content/drive/MyDrive/dataset/images/train/' + file1
    dst_path = '/content/Yolo_bbox_manipulation/' + file1
    shutil.copy(src_path, dst_path)
    print('Copied')
for file1 in label_list:
    src_path = '/content/drive/MyDrive/dataset/labels/train/' + file1
    dst_path = '/content/Yolo_bbox_manipulation/' + file1
    shutil.copy(src_path, dst_path)
    print('Copied')


# In[ ]:


get_ipython().run_line_magic('cd', 'Yolo_bbox_manipulation')


# Zum Rotieren wird  `rotate.py` aus dem erwähnten Repository über die `subprocess` -Bibliothek aufgerufen. Die Bibliothek ermöglicht den Terminalaufruf in einer For-Schleife. Für einen signifikaten Anstieg der Instanzanzahl der Helikopter wird jedes Bild sechsmal rotiert. Bei jeder Rotation wird ein zufälliger Winkel zwischen 1 und 359 Grad genutzt. 

# In[ ]:


counter = 0
import subprocess
img_list = img_list[1:]
# myangle="65"
for x in img_list:
  for i in range(0,6):
    myangle = str(np.random.randint(1,359))
    cmd = "python rotate.py -i "+x+" -a "+myangle
    subprocess.call(cmd, shell=True)  # returns the exit code in unix
    counter += 1
    print('counter:', counter)


# Zum schnelleren Verschieben werden die rotierten Bilder gezippt. Für die Modellerstellung werden sie in die erläuterte Ordnerstruktur übertragen. 

# In[ ]:


get_ipython().system('zip -r /content/drive/MyDrive/heli_train_2nd_Round.zip /content/Yolo_bbox_manipulation')

