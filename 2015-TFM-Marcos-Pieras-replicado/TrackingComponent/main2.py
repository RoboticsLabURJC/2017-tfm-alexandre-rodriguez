import cv2
import numpy as np
import os
import time
import sys
# import classFlow.py
sys.path.insert(0, '/home/marc/repo2/TrackingComponent')
from classFlow import *
import time
import matplotlib.pyplot as plt


# keras
import tensorflow as tf
import cv2

slim = tf.contrib.slim

import sys
#sys.path.append('./SSD-Tensorflow/')


from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization
import threading

# keras

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D
from keras.regularizers import l2
from keras.initializers import Constant
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import regularizers


DETECTION_RATE = 30


PATH_IMAGES = './img1'

listImages = os.listdir( PATH_IMAGES )
listImages.sort()
numFiles = np.shape(listImages)

MAX_FRAME = numFiles[0]
listOfDetections = [i*DETECTION_RATE for i in range(0,1000) if i*DETECTION_RATE <= MAX_FRAME]
NUM_ITERATIONS = np.shape(listOfDetections)[0]


#                                                       siamese netwrk
def create_model(): 
    model = Sequential()
    input_shape = (128,64, 6)

    model.add(Conv2D(20, (3, 3), input_shape=input_shape, kernel_initializer='he_uniform',name='conv1',activation = 'relu',bias_initializer= Constant(value=0.1)))
    model.add(Conv2D(25, (3, 3),name='conv2',kernel_initializer='he_uniform',activation = 'relu',bias_initializer= Constant(value=0.1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(30, (3, 3),name='conv3',kernel_initializer='he_uniform',activation = 'relu',bias_initializer= Constant(value=0.1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(35, (3, 3),name='conv5',kernel_initializer='he_uniform',activation = 'relu',bias_initializer= Constant(value=0.1)))
    model.add(Conv2D(35, (3, 3),name='conv6',kernel_initializer='he_uniform',activation = 'relu',bias_initializer= Constant(value=0.1)))
    model.add(Conv2D(35, (3, 3),name='conv7',kernel_initializer='he_uniform',activation = 'relu',bias_initializer= Constant(value=0.1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(128,kernel_initializer='glorot_uniform',bias_initializer= Constant(value=0.1)))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00005)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['binary_accuracy'])
    return model

#                                                                                               end siamese

modelSiamese = create_model()
weights_path = './main1weights-improvement-64.hdf5'
modelSiamese.load_weights(weights_path)




# Object detector thread

class myThread (threading.Thread):
        def __init__(self,NUM_DETCTIONS):
            threading.Thread.__init__(self)
 
            self.NUM = NUM_DETCTIONS
            self.isess = tf.InteractiveSession()
            net_shape = (300, 300)
            data_format = 'NHWC'
            self.img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
            image_pre, labels_pre, bboxes_pre, self.bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(self.img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
            self.image_4d = tf.expand_dims(image_pre, 0)
            reuse = None
            ssd_net = ssd_vgg_300.SSDNet()
            with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
                self.predictions, self.localisations, _, _ = ssd_net.net(self.image_4d, is_training=False, reuse=reuse)

            #ckpt_filename = './checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
            ckpt = tf.train.get_checkpoint_state('./checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt')
            #ckpt_filename = './checkpoints/ssd_300_vgg.ckpt/ssd_300_vgg.ckpt'
            self.isess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(self.isess, "VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt")
            self.ssd_anchors = ssd_net.anchors(net_shape)
            self.detecction = []
            self.iDetections = []

        def resultadso(self):
            return self.detecction
        def iniResult(self):
            self.detecction = []

        def run(self):
            


            for i in range(0,self.NUM):
 
                startA = time.time()

                image_names = sorted(os.listdir(PATH_IMAGES))

                indexFrame = listOfDetections[i]
                img = cv2.imread(PATH_IMAGES+'/'+image_names[indexFrame],1)

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                select_threshold=0.2
                nms_threshold=.45
                net_shape=(300, 300)


                #def process_image(img, select_threshold=0.2, nms_threshold=.45, net_shape=(300, 300)):

                rimg, rpredictions, rlocalisations, rbbox_img = self.isess.run([self.image_4d, self.predictions, self.localisations, self.bbox_img],feed_dict={self.img_input: img})


                # Get classes and bboxes from the net outputs.
                rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(rpredictions, rlocalisations, self.ssd_anchors,select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)


                idx = np.where(rclasses != 15)[0]
                rclasses = np.delete(rclasses, idx,0)   
                rscores = np.delete(rscores, idx,0)   
                rbboxes = np.delete(rbboxes, idx,0)    

                rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
                rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
                rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
                rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)

                sizeImg = np.shape(img)
                height = sizeImg[0]
                width = sizeImg[1]

                sizeDetection = np.shape(rbboxes)
                
                detecctionNET = np.zeros(sizeDetection)

                ymin = (np.around(rbboxes[:, 0] * height))
                xmin = (np.around(rbboxes[:, 1] * width))
                ymax = (np.around(rbboxes[:, 2] * height))
                xmax = (np.around(rbboxes[:, 3] * width))


                detecctionNET[:,0]=xmin
                detecctionNET[:,1]=ymin
                detecctionNET[:,2]=xmax
                detecctionNET[:,3]=ymax
               
                self.detecction.append(detecctionNET)


                end = time.time()

# Read first image
frame1x = cv2.imread(PATH_IMAGES+'/'+listImages[0])

# Launch object detector threas
thread= myThread(NUM_ITERATIONS)
thread.start()


# Wait for first detection
while(True):

    if thread.resultadso() != []:
        break



initialDetection = thread.resultadso()




mean1 = 118.268211396 
mean2 = 110.23197584
mean3 = 109.777103553
mean4 = 118.316815951
mean5 = 110.257208327
mean6 = 109.849493294




# Declare variables
listLostROI = []
listLostColor =  []
listLostID = []
roi1 = initialDetection[0][:][:]
roi = np.copy(roi1)
numPersonas = np.shape(roi)[0]
MAX_PERSON = 500
colors = np.random.randint(0,255,(MAX_PERSON,3))
randomIdentities = np.random.randint(1,255,(MAX_PERSON,1))
roiID = randomIdentities[:numPersonas]
roiColors = colors[:numPersonas]
colors = np.delete(colors, np.s_[:numPersonas], 0)
randomIdentities = np.delete(randomIdentities, np.s_[:numPersonas], 0)
velocidades = np.zeros((numPersonas,2))
flagVelocidades = 0
detectionestemps = []
detectiones = []
sats = np.zeros([numFiles[0]-1,4])
listOfDetections.append(10000)
indexDetections = 1
results = []



# Start tracking module
for x in range(1,numFiles[0]-1):

    
    startA = time.time()

    if x == 1:
        
        frame1 = frame1x


    # Read first image
    frame2 = cv2.imread(PATH_IMAGES+'/'+listImages[x])
    framePintar = np.copy(frame2)
    endImage = time.time()
    sats[x,0]= endImage-startA
    startData = time.time()
    
    # dataAssociation
        
    if x == listOfDetections[indexDetections]:
        
        # Situation 1

        detecctionBuff = thread.resultadso()
        detecction = detecctionBuff[indexDetections][:][:]
       
        roiAux = []
        roiColorsAux = []
        roiIDaux = []

        lukasCentreX  = roi[:,0]+((roi[:,2]-roi[:,0])/2)
        lukasCentreY  = roi[:,1]+((roi[:,3]-roi[:,1])/2)

        detCentreX  = detecction[:,0]+((detecction[:,2]-detecction[:,0])/2)
        detCentreY  = detecction[:,1]+((detecction[:,3]-detecction[:,1])/2)

        numLK = np.shape(roi)
        for idx in range(0,numLK[0]):

            numD = np.shape(detecction)[0]

            if numD == 0:

                roiAux.append(roi[idx,:])
                roiColorsAux.append(roiColors[idx,:])
                roiIDaux.append(roiID[idx,:])
                continue

            distancia = np.sqrt( (lukasCentreX[idx]-detCentreX)**2 + (lukasCentreY[idx]-detCentreY)**2 )
            

            numD = np.shape(detecction)

            idxMinDistance = np.argmin(distancia)

            if distancia[idxMinDistance] < 40.0:

                roiAux.append(detecction[idxMinDistance,:])
                roiColorsAux.append(roiColors[idx,:])
                roiIDaux.append(roiID[idx,:])

                detCentreX = np.delete(detCentreX,idxMinDistance)
                detCentreY = np.delete(detCentreY,idxMinDistance)
                detecction = np.delete(detecction,idxMinDistance,0)

            else:
                # Situation 2
                roiAux.append(roi[idx,:])
                roiColorsAux.append(roiColors[idx,:])
                roiIDaux.append(roiID[idx,:])
                
        
        # Situation 3
        numDeteccionesSin = np.shape(detecction)
        numLostRois = np.shape(listLostROI)
        matrixResults = np.zeros((1,numLostRois[0]))


        for xs in range(0,numDeteccionesSin[0]):

            # There are not lost trackets to match, then the detections are new trackets

            if  numLostRois[0] == 0:

                roiAux.append(detecction[xs,:])
                roiColorsAux.append(colors[xs,:])
                roiIDaux.append(np.array(randomIdentities[xs,:]))
                randomIdentities = np.delete(randomIdentities, xs, 0)
                colors = np.delete(colors, xs, 0)
                continue

            for iterLost in range(0,numLostRois[0]):       

                imagePos_1 = listLostROI[iterLost]
                res1 = cv2.resize(imagePos_1,(64, 128), interpolation = cv2.INTER_CUBIC)

                imagePos_2 = frame1[int(detecction[xs,1]):int(detecction[xs,3]),int(detecction[xs,0]):int(detecction[xs,2])]
                res2 = cv2.resize(imagePos_2,(64, 128), interpolation = cv2.INTER_CUBIC)

                image = np.concatenate((res1,res2 ), axis=2)

                image[:,:,0] = image[:,:,0]-mean1
                image[:,:,1] = image[:,:,1]-mean2
                image[:,:,2] = image[:,:,2]-mean3
                image[:,:,3] = image[:,:,3]-mean4
                image[:,:,4] = image[:,:,4]-mean5
                image[:,:,5] = image[:,:,5]-mean6


                image = image.astype("float32")
                image /= 255.0
                imageExpand = np.expand_dims(image, axis=0)

                output = modelSiamese.predict(imageExpand)
                matrixResults[0,iterLost]=output[:]

            idxMajorSimiliratiry =  np.argmax(matrixResults)
            # the most similiraity and over a threshold
            if matrixResults[0,idxMajorSimiliratiry]>0.9:
                

                roiAux.append(detecction[xs,:])
                roiColorsAux.append(listLostColor[idxMajorSimiliratiry])
                roiIDaux.append([listLostID[idxMajorSimiliratiry]])
                listLostROI = np.delete(listLostROI, idxMajorSimiliratiry, 0)
                listLostColor = np.delete(listLostColor, idxMajorSimiliratiry, 0)
                listLostID = np.delete(listLostID, idxMajorSimiliratiry, 0)

                numLostRois = np.shape(listLostROI)
                matrixResults = np.zeros((1,numLostRois[0]))

            # no match, it's a new detection
            else:

                roiAux.append(detecction[xs,:])
                roiColorsAux.append(colors[xs,:])
                roiIDaux.append(randomIdentities[xs,:])
                randomIdentities = np.delete(randomIdentities, xs, 0)
                colors = np.delete(colors, xs, 0)


        roi = np.copy(roiAux)
        roiColors = np.copy(roiColorsAux)
        roiID = np.copy(roiIDaux)

        numPersonas = np.shape(roi)[0]
        velocidades = np.zeros((numPersonas,2))
        listLostROI = []
        listLostColor =  []
        listLostID = []
        indexDetections += 1
        flagVelocidades =1
    
    endDAta = time.time()
    sats[x,1]= endDAta-startData
    startLK = time.time()
    listOf = []
    
    # Compute discplacement for each ROI (blob)

    for iPerson in range(0,numPersonas):

        roiFrame1 = frame1[int(roi[iPerson,1]):int(roi[iPerson,3]),int(roi[iPerson,0]):int(roi[iPerson,2])]
        roiFrame2 = frame2[int(roi[iPerson,1]):int(roi[iPerson,3]),int(roi[iPerson,0]):int(roi[iPerson,2])]

        roi[iPerson,0],roi[iPerson,1],roi[iPerson,2],roi[iPerson,3],tracas,dx,dy = lucasKanadeTrackerMedianScaleStatic2PlusOptimized2Deploy(roiFrame1,roiFrame2,roi[iPerson,0],roi[iPerson,1],roi[iPerson,2],roi[iPerson,3])
        

        # modulo velocidad
        if flagVelocidades==1:
            velocidades[iPerson,0]=dx
            velocidades[iPerson,1]=dy
            flagVelocidades = 0

        # X
        if np.absolute(velocidades[iPerson,0])==0.0:

            if np.absolute(velocidades[iPerson,1])==0.0:

                vxPorcTemporal = 0.0
                vyPorcTemporal = 0.0

            else:

                vxPorcTemporal = 0.0
                vyPorcTemporal = np.absolute(dy-velocidades[iPerson,1])/np.absolute(velocidades[iPerson,1])

        
        # Y 
        elif np.absolute(velocidades[iPerson,1])==0.0:

            if np.absolute(velocidades[iPerson,0])==0.0:

                vxPorcTemporal = 0.0
                vyPorcTemporal = 0.0
            else:


                vxPorcTemporal = np.absolute(dx-velocidades[iPerson,0])/np.absolute(velocidades[iPerson,0])
                vyPorcTemporal = 0.0
        else:
            
            vxPorcTemporal = np.absolute(dx-velocidades[iPerson,0])/(np.absolute(velocidades[iPerson,0]))
            vyPorcTemporal = np.absolute(dy-velocidades[iPerson,1])/(np.absolute(velocidades[iPerson,1]))

            
        velocidades[iPerson,0] = dx
        velocidades[iPerson,1] = dy

    
        n = str(x)
        n2 = n.zfill(3)

        aStr = str(iPerson)
        n21 = aStr.zfill(3)
        limitBOTH = 2.5
        limitTemporal = 3.0

        # Check it is correct

        if (vxPorcTemporal >= limitBOTH and vyPorcTemporal >=limitBOTH) or (tracas == 1) or (vyPorcTemporal > limitTemporal) or (vxPorcTemporal > limitTemporal):

        
            if (np.shape(roiFrame1)[0] != 0) and (np.shape(roiFrame1)[1] != 0):


                listLostROI.append(roiFrame1)
                listLostColor.append([roiColors[iPerson,0],roiColors[iPerson,1],roiColors[iPerson,2]])
                listLostID.append(roiID[iPerson][0])



            listOf.append(iPerson)
        else:
            cv2.putText(framePintar,str(roiID[iPerson][0]),(int(roi[iPerson,0]),int(roi[iPerson,1])),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA) 
            cv2.rectangle(framePintar,(int(roi[iPerson,0]),int(roi[iPerson,1])),(int(roi[iPerson,2]),int(roi[iPerson,3])),(int(roiColors[iPerson,0]),int(roiColors[iPerson,1]),int(roiColors[iPerson,2])),3)
            #cv2.rectangle(framePintar,(int(roi[iPerson,0]),int(roi[iPerson,1])),(int(roi[iPerson,2]),int(roi[iPerson,3])),(255,0,255),3)
            results.append([int(x+1),int(roiID[iPerson][0]),int(roi[iPerson,0]+1),int(roi[iPerson,1]+1),int(roi[iPerson,2]-roi[iPerson,0]+1),int(roi[iPerson,3]-roi[iPerson,1]+1),-1,-1,-1,-1])



    endLK = time.time()
    sats[x,2]= endLK-startLK

    startEND = time.time()
    frame1 = np.copy(frame2)

    roi = np.delete(roi, listOf,0)    
    roiColors = np.delete(roiColors, listOf,0) 
    roiID = np.delete(roiID,listOf,0) 

   
    numPersonas = np.shape(roi)[0]  
    endEND = time.time()
    sats[x,3]= endEND-startEND


siz =  range(0,numFiles[0]-1)
print('ds',np.mean(sats[:,0]),np.mean(sats[:,1]),np.mean(sats[:,2]),np.mean(sats[:,0]+sats[:,1]+sats[:,2]))

fig = plt.figure()

plt.bar(siz,sats[:,0]+sats[:,1]+sats[:,2], color='r')
plt.bar(siz,sats[:,0]+sats[:,1], color='b')
plt.bar(siz,sats[:,1], color='y')
plt.show()
																
np.savetxt('/home/marc/Dropbox/tfmDeepLearning/semana8/componente/scoreTest/MOT16-14.txt', results, delimiter=',',fmt="%.5e") 
