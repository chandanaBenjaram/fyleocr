from __future__ import division

import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from skimage import transform
import cv2
from skimage import io, color
from skimage.transform import resize
from skimage.transform import rescale
from skimage import feature
from skimage.transform import probabilistic_hough_line
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import defaultdict
from sklearn import cluster
from sklearn.neighbors import KDTree
from scipy.spatial import distance
import math
import re
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\raj\AppData\Local\Tesseract-OCR\tesseract.exe'
from flask import Flask, request, Response
from flask import json
from flask import jsonify 
import jsonpickle
from PIL import Image
import datefinder
from werkzeug.utils import secure_filename
import logging
from logging import Formatter, FileHandler



basedir = os.path.abspath(os.path.dirname(__file__))

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

format = "%Y-%m-%dT%H:%M:%S"
#now = datetime.datetime.utcnow().strftime(format)

_VERSION = 1  # API version
app = Flask(__name__, static_url_path='/static')

#app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['POST','GET'])
def home():
    file=request.files['file']
    file.save("./static/test.jpeg")
    print("file successfully saved")

    #if file and allowed_file(file.filename):
        #filename = now + '_' +str(current_user) + '_' + file.filename
        #filename = secure_filename(filename)
    #file.save(os.path.join(app.config['UPLOAD_FOLDER']))
    #file_uploaded = True
    #path="./uploads/{}".format(filename)
    #   print ("file was uploaded in {} ".format(path))

    image = cv2.imread("./static/test.jpeg")
    img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    img_rescal = rescale(img_gray,scale=(0.5, 0.5))
    img_blur = cv2.GaussianBlur(img_rescal,(5,5),0)
    edges = feature.canny(img_blur, sigma=1)
    cord = probabilistic_hough_line(edges, threshold=15, line_length=15,line_gap=13)
    b_list= []
    c_list=[]
    for i in range(0 , len(cord)):
      for m in range(0, 2):
       b_list.append(cord[i][m])
    num=2*len(cord)
    for k in range(0,num):
       x0=b_list[k][0]
       y0=b_list[k][1]
       q1_flag=0
       q2_flag=0  
       q3_flag=0
       q4_flag=0
       for j in range(0 ,num):
         if((b_list[j][0]>x0) and (b_list[j][1]<y0)):
            q1_flag =1
         if((b_list[j][0]<x0) and (b_list[j][1]<y0)):
            q2_flag =1
         if((b_list[j][0]<x0) and (b_list[j][1]>y0)):
            q3_flag =1
         if((b_list[j][0]>x0) and (b_list[j][1]>y0)):
            q4_flag =1
         if(((q1_flag) and (q2_flag) and (q3_flag) and (q4_flag)) ==1):
            c_list.append((x0,y0))
    d_list=list(filter(lambda x: x not in c_list, b_list))
    b_list =[]
    b_list = list.copy(d_list)
    d_list.sort(key=lambda y:y[1])
    b_list.sort(key=lambda x:x[0])
    k=5
    num=len(d_list)
    dist_k_list=[]
    dist_list=[]

    for j in range(0 ,num):
      qd=0
      if(j<k):
        index_x_start=0
        index_x_end=j+k
      elif((num-j)<k):
        index_x_start=num-2*k
        index_x_end=num
      else:
        index_x_start=j-k
        index_x_end=j+k
      for i in range(index_x_start,index_x_end):
        dist_k_list.append(abs(b_list[j][0]-b_list[i][0]))
        dist_k_list.sort()
      for i in range(0,k):
        qd=qd+math.pow(dist_k_list[i],2)
        index_d=d_list.index(b_list[j], 0, num)
      if(index_d<k):
        index_y_start=0
        index_y_end=index_d+k
      elif((num-index_d)<k):
        index_y_start=num-2*k
        index_y_end=num
      else:
        index_y_start=index_d-k
        index_y_end=index_d+k
      dist_k_list=[]
      for i in range(index_y_start,index_y_end):
        dist_k_list.append(abs(b_list[j][1]-b_list[i][0]))
        dist_k_list.sort()
      for i in range(0,k):
        qd=qd+math.pow(dist_k_list[i],2)
      qd=0.5*qd
      dist_list.append((int(qd),j))    

    dist_list.sort(key=lambda y:y[0])   #len(dist_list)/4)
    med_distance=int(dist_list[len(dist_list)-int(len(dist_list)/4)][0])
#med_distance=np.median(dist_list,axis=0)
#print(med_distance)
    rem_list=[]
    rem_list=list(filter(lambda x: x[0]>med_distance , dist_list))
#print(rem_list)
#print(b_list)
#print(len(d_list))
    c_list=[]
    for m in range(0 ,len(rem_list)):
      rem_ind=rem_list[m][1]
     #print(rem_ind)
      c_list.append(b_list[rem_ind])
    b_list=list(filter(lambda x: x not in c_list, b_list))

#print(c_list)
#print(b_list)


    q5=0
    a_list=[]
    num=len(b_list)


    for i in range(0, num):
        q5=q5+1
        x0=b_list[i][0]
        y0=b_list[i][1]
        q1=0
        q2=0
        q3=0
        q4=0
        for j in range(0 ,num):
            if((b_list[j][0]>x0) and (b_list[j][1]<y0)):
                q1=q1+math.pow((x0-b_list[j][0]),2)+math.pow((y0-b_list[j][1]),2)
            elif((b_list[j][0]<x0) and (b_list[j][1]<y0)):
                q2=q2+math.pow((x0-b_list[j][0]),2)+math.pow((y0-b_list[j][1]),2)
            elif((b_list[j][0]<x0) and (b_list[j][1]>y0)):
                q3=q3+math.pow((x0-b_list[j][0]),2)+math.pow((y0-b_list[j][1]),2)
            elif((b_list[j][0]>x0) and (b_list[j][1]>y0)):
                q4=q4+math.pow((x0-b_list[j][0]),2)+math.pow((y0-b_list[j][1]),2)
        q1=round(q1*0.1)
        q2=round(q2*0.1)
        q3=round(q3*0.1)
        q4=round(q4*0.1)
        a_list.append(tuple([q1, q2,q3,q4,q5]))
    c_list=[]
#print(a_list)
#print(q5)
    for k in range(0,4):
    #print(k)
       q_max=max(a_list, key = lambda i : i[k])
       q_index=q_max[4]
    #print(q_max)
    #print(q_index)
    #print(a_list[q_index-1])
    #print(b_list[q_index-1])
       c_list.append(b_list[q_index-1])  
    y_max1=max(c_list, key = lambda i : i[1])
    y_min1=min(c_list, key = lambda i : i[1])
    x_min1=min(c_list, key = lambda i : i[0])
    x_max1=max(c_list, key = lambda i : i[0])
#print(y_min1[1],y_max1[1],x_max1[0],x_min1[0])
    y_max=int(2*y_max1[1])
    y_min=int(0.5*y_min1[1])
    x_min=int(0.5*x_min1[0])
    x_max=int(2*x_max1[0])
#print(y_min1[1],y_max1[1],x_max1[0],x_min1[0])
    start_co_rd=(x_min1[0],y_min1[1])
    width=x_max1[0]-x_min1[0]
    height=y_max1[1]-y_min1[1]
    # print(start_co_rd,width,height)
    cropped = img_gray[y_min:y_max, x_min:x_max]
   # cropped=Image.fromarray(cropped.astype(np.uint8))
    text = (pytesseract.image_to_string((cropped)))
    #text = pytesseract.image_to_string(ROI)
    #print(text)
    dat=str(re.search(r'[Date:]*([0-9]{0,2}[.\/-]([0-9]{0,2}|[a-z]{3})[.\/-][0-9]{0,4})',text))
    
    st_list=dat.split()
    date=st_list[(len(st_list)-1)]
    date=(re.sub("'>|l"," ",re.sub("'>'|l"," ",re.sub("match='|l", " ", date))))
    if dat is None:
        matches = list(datefinder.find_dates(text))
        if len(matches) > 0:
            dat = (matches[0])
            date=str(dat.date())
        else:
            date=None
    
        
    #date=(re.search(r'[Date:]*([0-9]{0,2}[.\/-]([0-9]{0,2}|[a-z|A-Z]{3})[.\/-][0-9]{0,4})',text))
    #if date=="None":
    # date=re.search(r'[Date:]*([\]([a-z|A-Z]{2})([0-9]{0,2))[\"\/-][0-9]{2})',text)   
    #return date
    #task["date"]=date
 
    #print(date)
    #return jsonify({"date":date})
    return jsonify({'date':date})


if __name__ == "__main__":
 #app.run()
 app.run(threaded=True, port=5000)

    #print (re.search(r'[Date:]*([0-9]{0,2}[\/-]([0-9]{0,2}|[a-z]{3})[\/-][0-9]{0,4})',text))
   # return sucess
#app.run(host="0.0.0.0", port=5000)

