import numpy as np
import cv2
import random
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import sys
import argparse
import time
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow import device as model_device
import json
import progressbar
from reportlab.pdfgen import canvas
from pdf2docx import parse

from image_preprocessing.scan import *
from image_preprocessing.helper import resize,get_sorted_contours_bounding_box,skew_correction
UNKNOWN = np.NaN

ap = argparse.ArgumentParser()
ap.add_argument('-s','--src',required=True,help = "Relative source path for the image to be scanned.")
ap.add_argument('-m','--models',required=True,help='Relative directory path for the saved models')
args = vars(ap.parse_args())
src_path = os.path.join(os.getcwd(),args['src'])
models_path = os.path.join(os.getcwd(),args['models'])
if(not os.path.isfile(src_path) or not os.path.isdir(models_path)):
    sys.exit('invalid --src or --models')
    
def rf_classify(rf_preds,chars,bar):
    letter_class = []
    probs = []
    for i,c in enumerate(chars):
        tmp = np.reshape(c,((1,)+c.shape))

        if(rf_preds[i]==0):
            
            upper_pred = uppercase.predict(tmp)
            lower_pred = lowercase.predict(tmp)
            idx1,idx2 = np.argmax(upper_pred),np.argmax(lower_pred)
            final_class = chr(int(upper_classes[str(idx1)],16))
            prob = upper_pred[0][idx1]
            # cv2.imshow('c',c)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # print("upper,prob {},{:.2f} lower,prob {},{:.2f}".format(idx1,prob,idx2,lower_pred[0][idx2]))
            if(upper_pred[0][idx1]<lower_pred[0][idx2]):
                final_class = chr(int(lower_classes[str(idx2)],16))
                prob = lower_pred[0][idx2]
        if(rf_preds[i]==1 or prob<0.55):

            preds = nums.predict(tmp)
            idx1 = np.argmax(preds)
            final_class = chr(48+idx1)
            prob = preds[0][idx1]
        if(prob<0.5):
            label = UNKNOWN
        
        # print('letter {}:prob {}'.format(final_class,prob))
        letter_class.append(final_class)
        probs.append(prob)
        bar.update(i)
    return letter_class,probs

def validate_models(models_path):
    models_names = ['nums.h5','lower.h5','upper.h5','rf.sav']
    for entry in os.scandir(models_path):
        if entry.name in models_names:
            models_names.remove(entry.name)
    if len(models_names) != 0:
        sys.exit('Missing models'+str(models_names))


def progress_bar(msg,max_len):
    widgets = ['{} ['.format(msg),
         progressbar.Timer(format= 'elapsed time: %(elapsed)s'),
         '] ',
           progressbar.Bar('='),' (',
           progressbar.ETA(), ') ',
          ]           

    progressbar.streams.flush()
    bar = progressbar.ProgressBar(max_value=max_len,
                                  widgets=widgets).start()
    return bar
def make_results_dir(dir):
    try:
        os.mkdir(dir)
    except OSError:
        print('OCR Results Directory exists in the system already.')

img = cv2.imread(r'C:\Users\Avinash\Desktop\New folder\OCR\test ocr\test 1.jpg')
if(img is None):
    print('Please provide a valid Image')
    sys.exit(src_path+' is not supported')
print('Image Loaded into memory',flush=True)
orig = img.copy()
try:
    img = document_warper(img)
except FourPointException:
    print("Couldn't find the edges of the current doucment.\nTo improve accuracy working on skew correctrion on the present document",flush=True)
    img = skew_correction(img)


if(img.shape[0]>(1200)):
    img = resize(img,height=1200)
img = cv2.fastNlMeansDenoisingColored(img)
gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
processed = cv2.addWeighted(img, 1.5, gaussian, -0.9, 0)

# print(img.shape)
gray = cv2.cvtColor(processed,cv2.COLOR_RGB2GRAY)
edged = edge_detection(processed)
cnts,heirarchy = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts,bounding_box = get_sorted_contours_bounding_box(cnts,method='top-to-bottom')
chars=[]
boxs=[]
bar = progress_bar('Extracting characters',len(cnts))
for i,c in enumerate(cnts):
    x,y,w,h = bounding_box[i]
    if (w>=5 and w<=150) and (h>=15 and h<=150):
        roi = gray[y:y+h,x:x+w]
        thresh = cv2.threshold(roi,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        del roi
        dims = random.randint(45,55)
        thresh = cv2.resize(thresh,(dims,dims),cv2.INTER_CUBIC)
        tH,tW = thresh.shape
        dX = int(max(0,128-tW)/2.0)
        dY = int(max(0,128-tH)/2.0)
        padded = cv2.copyMakeBorder(thresh,top=dY,bottom=dY,right=dX,left=dX,borderType=cv2.BORDER_CONSTANT,value=(255,255,255))
        del thresh
        padded = cv2.resize(padded,(128,128),cv2.INTER_CUBIC)
        padded = padded.astype('float32')/255.
        padded = np.expand_dims(padded,axis=-1)
        boxs.append((x,y,w,h))
        chars.append(padded)
        time.sleep(0.07)
    bar.update(i)
bar.finish()
del bar
chars = np.array(chars,dtype='float32')
validate_models(models_path)
print('Loading neural networks',flush=True)
with model_device('/cpu:0'):
    nums = load_model(os.path.join(models_path,'nums.h5'))
    lowercase = load_model(os.path.join(models_path,'lower.h5'))

with model_device('/gpu:1'):
    uppercase = load_model(os.path.join(models_path,'upper.h5'))


with open(os.path.join(models_path,'rf.sav'),'rb') as f:
    random_forest = pickle.load(f)

with open('classes/lower_classes.json','r') as f:
    lower_classes = json.load(f)
f.close()
with open('classes/upper_classes.json','r') as f:
    upper_classes = json.load(f)
    f.close()



rf_preds=random_forest.predict(chars.reshape((chars.shape[0],chars.shape[1]*chars.shape[2])))


bar = progress_bar('Classifying',chars.shape[0])
labels,probs = rf_classify(rf_preds,chars,bar)
bar.finish()


overview = img.copy()
for (label,(x,y,w,h)) in zip(labels,boxs):
    if(label==UNKNOWN):
        label="?"
    cv2.rectangle(overview,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.putText(overview,label,(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,0),2)
    
    
cv2.imshow("Overview",overview)
cv2.waitKey(0)
cv2.destroyAllWindows()


result_dir = os.path.join(os.getcwd(),r'OCR Results')
make_results_dir(result_dir)

curr_result_dir = os.path.join(result_dir,'results-{}'.format(str(datetime.now()).replace(':','-')[:19]))
try:
    os.mkdir(curr_result_dir)
except OSError:
    print(curr_result_dir+" exists in the system")

pdf_path=curr_result_dir+r'\ocr.pdf'

pdf = canvas.Canvas(pdf_path,bottomup=0,pagesize=(img.shape[1],img.shape[0]))
pdf.setTitle('OCR Results'+str(datetime.now())[:10])
px,py,ph=None,None,None
for label,(x,y,w,h) in zip(labels,boxs):
    if px is None and py is None:
        px,py=x,y
        pw,ph=w,h
    
    if py+ph>y:
        pdf.setFont('Times-Bold',ph)
        pdf.drawString(x,py+ph,label)
    else:
        pdf.setFont('Times-Bold',h)
        pdf.drawString(x,y+h,label)
        px,py,ph=x,y,h
pdf.save()

parse(pdf_path,curr_result_dir+r'\ocr.docx',start=0,end=None)
print('Results path {}'.format(result_dir))
cv2.imwrite(curr_result_dir+r'\overview.png',overview)