import cv2
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import matplotlib

#feature_params = dict( maxCorners = 1000,qualityLevel = 0.1,minDistance = 2,blockSize = 7 )
# Parameters for lucas kanade optical flow
#lk_params = dict( winSize  = (15,15),maxLevel = 1,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#kernel_sharpen_1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

#feature_params = dict( maxCorners = 1000,qualityLevel = 0.1,minDistance = 2,blockSize = 7 )
    #feature_params = dict( maxCorners = 1000,qualityLevel = 0.2,minDistance = 4,blockSize = 7 )

#lk_params = dict( winSize  = (15,15),maxLevel = 4,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def lucasKanadeTrackerMedianScale(roiFrame1,roiFrame2,xmin,ymin,xmax,ymax):
    # INPUT:
	# 		ROI1 in RGB
	#		ROI2 in RGB
	# OUTPUT:
	#		DisplacementX
	#		DisplacementY

	
    #frame1 = cv2.imread(directoryImages+'/'+listImages[x-1])
    #roiFrame1 = frame1[ymin:ymax,xmin:xmax]
    
    
    #feature_params = dict( maxCorners = 100,qualityLevel = 0.1,minDistance = 4,blockSize = 7 )
    #lk_params = dict( winSize  = (15,15),maxLevel = 4,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    #dircX = 0
    #dircY = 0
    trackLost = 0
    
    
    if np.shape(roiFrame1)[0]==0 or np.shape(roiFrame1)[1]==0:
        trackLost = 1
        return xmin,ymin,xmax,ymax,trackLost

    old_frame = cv2.GaussianBlur(roiFrame1,(11,11),0)
    #old_frame = cv2.filter2D(roiFrame1, -1, kernel_sharpen_1)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_RGB2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    #print('point',np.shape(p0)[0])
    #print(p0)
    if p0 is None:

        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        trackLost = 1
        return xmin,ymin,xmax,ymax,trackLost
    else:
        if np.shape(p0)[0]!=0:

            #print('sa',np.shape(p0))
            #frame2 = cv2.imread(directoryImages+'/'+listImages[x])
            #roiFrame2 = frame2[ymin:ymax,xmin:xmax] 
            output_2 = cv2.GaussianBlur(roiFrame2,(11,11),0)
            #output_2 = cv2.filter2D(roiFrame2, -1, kernel_sharpen_1)
            frame_gray = cv2.cvtColor(output_2, cv2.COLOR_RGB2GRAY)
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            good_new = p1[st==1]
            good_old = p0[st==1]
            #print('tam2',np.shape(good_new),np.shape(good_old))

            err = err[[st==1]].flatten()
            indx = np.argsort(err)
            half_indx = indx[:len(indx) // 2]
            good_old = (p0[[st==1]])[half_indx]
            good_new = (p1[[st==1]])[half_indx]
            #print('points',np.shape(half_indx))


            dx = np.median(good_new[:, 0] - good_old[:, 0])
            dy = np.median(good_new[:, 1] - good_old[:, 1])
            #print('m1',dx,dy)
            #print('tms2',np.sum(good_new[:][0]-good_old[:][0]),np.sum(good_new[:][1]-good_old[:][1]))
            i, j = np.triu_indices(len(good_old), k=1)
            #print('numP',np.shape(good_new))

            pdiff0 = good_old[i] - good_old[j]
            pdiff1 = good_new[i] - good_new[j]
            
            p0_dist = np.sum(pdiff0 ** 2, axis=1)
            p1_dist = np.sum(pdiff1 ** 2, axis=1)
            ds = np.median(np.sqrt((p1_dist / (p0_dist + 2**-23))))
            
            

            if np.isnan(dx) or np.isnan(dy) or np.isnan(ds) :
                #print('m1',dx,dy,ds,np.isnan(dx))

                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)
                trackLost = 1
                return xmin,ymin,xmax,ymax,trackLost
            else:
                '''
                ds_factor = 1.5
                ds = (1.0 - ds_factor) + ds_factor * ds;
                dx_scale = (ds - 1.0) * 0.5 * (xmax - xmin + 1)
                dy_scale = (ds - 1.0) * 0.5 * (ymax - ymin + 1)
                '''
                dx_scale = (ds-1.0) * 0.5 * (xmax - xmin + 1)
                dy_scale = (ds-1.0) * 0.5 * (ymax - ymin + 1)


                xmin = int(xmin+dx-dx_scale+0.5)
                ymin = int(ymin+dy-dy_scale+0.5)
                xmax = int(xmax+dx+dx_scale+0.5)
                ymax = int(ymax+dy+dy_scale+0.5)
        else:
            trackLost = 1 
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            return xmin,ymin,xmax,ymax,trackLost
    #print(endA-startA)
    #return xmin,ymin,xmax,ymax,good_new,good_old
    return xmin,ymin,xmax,ymax,trackLost

def lucasKanadeTrackerMedianScale2(roiFrame1,roiFrame2,xmin,ymin,xmax,ymax):
    # INPUT:
    #       ROI1 in RGB
    #       ROI2 in RGB
    # OUTPUT:
    #       DisplacementX
    #       DisplacementY

    
    #frame1 = cv2.imread(directoryImages+'/'+listImages[x-1])
    #roiFrame1 = frame1[ymin:ymax,xmin:xmax]
    
    
    #feature_params = dict( maxCorners = 100,qualityLevel = 0.1,minDistance = 4,blockSize = 7 )
    #lk_params = dict( winSize  = (15,15),maxLevel = 4,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    #dircX = 0
    #dircY = 0
    trackLost = 0
    good_new = []
    good_old = []
    
    if np.shape(roiFrame1)[0]==0 or np.shape(roiFrame1)[1]==0:
        trackLost = 1
        return xmin,ymin,xmax,ymax,good_new,good_old

    old_frame = cv2.GaussianBlur(roiFrame1,(11,11),0)
    #old_frame = cv2.filter2D(roiFrame1, -1, kernel_sharpen_1)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_RGB2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    #print('point',np.shape(p0)[0])
    #print(p0)
    if p0 is None:

        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        trackLost = 1
        return xmin,ymin,xmax,ymax,good_new,good_old
    else:
        if np.shape(p0)[0]!=0:

            #print('sa',np.shape(p0))
            #frame2 = cv2.imread(directoryImages+'/'+listImages[x])
            #roiFrame2 = frame2[ymin:ymax,xmin:xmax] 
            output_2 = cv2.GaussianBlur(roiFrame2,(9,9),0)
            #output_2 = cv2.filter2D(roiFrame2, -1, kernel_sharpen_1)
            frame_gray = cv2.cvtColor(output_2, cv2.COLOR_RGB2GRAY)
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            good_new = p1[st==1]
            good_old = p0[st==1]
            #print('tam2',np.shape(good_new),np.shape(good_old))

            err = err[[st==1]].flatten()
            indx = np.argsort(err)
            half_indx = indx[:len(indx) // 2]
            good_old = (p0[[st==1]])[half_indx]
            good_new = (p1[[st==1]])[half_indx]
            #print('points',np.shape(half_indx))


            dx = np.median(good_new[:, 0] - good_old[:, 0])
            dy = np.median(good_new[:, 1] - good_old[:, 1])
            ll = [good_new[:, 0] - good_old[:, 0],good_new[:, 1] - good_old[:, 1]]
            print(np.array(ll))
            #print('m1',dx,dy)
            #print('tms2',np.sum(good_new[:][0]-good_old[:][0]),np.sum(good_new[:][1]-good_old[:][1]))
            i, j = np.triu_indices(len(good_old), k=1)
            #print('numP',np.shape(good_new))

            pdiff0 = good_old[i] - good_old[j]
            pdiff1 = good_new[i] - good_new[j]
            
            p0_dist = np.sum(pdiff0 ** 2, axis=1)
            p1_dist = np.sum(pdiff1 ** 2, axis=1)
            ds = np.median(np.sqrt((p1_dist / (p0_dist + 2**-23))))
            
            

            if np.isnan(dx) or np.isnan(dy) or np.isnan(ds) :
                #print('m1',dx,dy,ds,np.isnan(dx))

                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)
                trackLost = 1
                return xmin,ymin,xmax,ymax,good_new,good_old
            else:
                '''
                ds_factor = 1.5
                ds = (1.0 - ds_factor) + ds_factor * ds;
                dx_scale = (ds - 1.0) * 0.5 * (xmax - xmin + 1)
                dy_scale = (ds - 1.0) * 0.5 * (ymax - ymin + 1)
                '''
                dx_scale = (ds-1.0) * 0.5 * (xmax - xmin + 1)
                dy_scale = (ds-1.0) * 0.5 * (ymax - ymin + 1)


                xmin = int(xmin+dx-dx_scale+0.5)
                ymin = int(ymin+dy-dy_scale+0.5)
                xmax = int(xmax+dx+dx_scale+0.5)
                ymax = int(ymax+dy+dy_scale+0.5)
        else:
            trackLost = 1 
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            return xmin,ymin,xmax,ymax,good_new,good_old
    #print(endA-startA)
    return xmin,ymin,xmax,ymax,good_new,good_old
    #return xmin,ymin,xmax,ymax,trackLost

def lucasKanadeTrackerMedianScaleGaussian(roiFrame1,roiFrame2,xmin,ymin,xmax,ymax):
    # INPUT:
    #       ROI1 in RGB
    #       ROI2 in RGB
    # OUTPUT:
    #       DisplacementX
    #       DisplacementY

    
    #frame1 = cv2.imread(directoryImages+'/'+listImages[x-1])
    #roiFrame1 = frame1[ymin:ymax,xmin:xmax]
    
    
    #feature_params = dict( maxCorners = 100,qualityLevel = 0.1,minDistance = 4,blockSize = 7 )
    #lk_params = dict( winSize  = (15,15),maxLevel = 4,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    #dircX = 0
    #dircY = 0
    trackLost = 0
    
    good_new = 0
    good_old = 0 
    if np.shape(roiFrame1)[0]==0 or np.shape(roiFrame1)[1]==0:
        trackLost = 1
        return xmin,ymin,xmax,ymax,good_new,good_old

    old_frame = cv2.GaussianBlur(roiFrame1,(11,11),0)
    #old_frame = cv2.filter2D(roiFrame1, -1, kernel_sharpen_1)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_RGB2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    #print('point',np.shape(p0)[0])
    #print(p0)
    if p0 is None:

        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        trackLost = 1
        return xmin,ymin,xmax,ymax,good_new,good_old
    else:
        if np.shape(p0)[0]!=0:

            #print('sa',np.shape(p0))
            #frame2 = cv2.imread(directoryImages+'/'+listImages[x])
            #roiFrame2 = frame2[ymin:ymax,xmin:xmax] 
            output_2 = cv2.GaussianBlur(roiFrame2,(9,9),0)
            #output_2 = cv2.filter2D(roiFrame2, -1, kernel_sharpen_1)
            frame_gray = cv2.cvtColor(output_2, cv2.COLOR_RGB2GRAY)
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            good_new = p1[st==1]
            good_old = p0[st==1]
            #print('tam2',np.shape(good_new),np.shape(good_old))

            err = err[[st==1]].flatten()
            indx = np.argsort(err)
            half_indx = indx[:len(indx) // 2]
            good_old = (p0[[st==1]])[half_indx]
            good_new = (p1[[st==1]])[half_indx]
            #print('points',np.shape(half_indx))

            sigmaX = 0
            sigmaY = 0
            longitudX = (xmax-xmin)
            longitudY = (ymax-ymin)
            
            xcentroide = ((xmax+xmin)/2)-xmin
            ycentroide = ((ymax+ymin)/2)-ymin
            ratioSigmaX = 0.05
            ratioSigmaY = 0.005
            sigmaX = ratioSigmaX*longitudX
            sigmaY = ratioSigmaY*longitudY

            meanX = int(ycentroide)
            meanY = int(xcentroide)
            exponent = np.exp(-(((good_old[:, 0]-meanX)**2)/(2*sigmaX**2))-((good_old[:, 1]-meanY)**2/(2*sigmaY**2)))
            value = (exponent)/(2*np.pi*sigmaX*sigmaY)

            numerator = np.sum(value)
            dx = np.sum(value*(good_new[:, 0] - good_old[:, 0])/numerator)
            dy = np.sum(value*(good_new[:, 1] - good_old[:, 1])/numerator)
            
            #dx = np.median(good_new[:, 0] - good_old[:, 0])
            #dy = np.median(good_new[:, 1] - good_old[:, 1])
            ll = [good_new[:, 0] - good_old[:, 0],good_new[:, 1] - good_old[:, 1]]
            #print(np.array(ll))
            #print('m1',dx,dy)
            #print('tms2',np.sum(good_new[:][0]-good_old[:][0]),np.sum(good_new[:][1]-good_old[:][1]))
            i, j = np.triu_indices(len(good_old), k=1)
            #print('numP',np.shape(good_new))

            pdiff0 = good_old[i] - good_old[j]
            pdiff1 = good_new[i] - good_new[j]
            
            p0_dist = np.sum(pdiff0 ** 2, axis=1)
            p1_dist = np.sum(pdiff1 ** 2, axis=1)
            ds = np.median(np.sqrt((p1_dist / (p0_dist + 2**-23))))
            #print(np.shape(value),np.shape(pdiff1))
            

            if np.isnan(dx) or np.isnan(dy) or np.isnan(ds) :
                #print('m1',dx,dy,ds,np.isnan(dx))

                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)
                trackLost = 1
                return xmin,ymin,xmax,ymax,good_new,good_old
            else:
                '''
                ds_factor = 1.5
                ds = (1.0 - ds_factor) + ds_factor * ds;
                dx_scale = (ds - 1.0) * 0.5 * (xmax - xmin + 1)
                dy_scale = (ds - 1.0) * 0.5 * (ymax - ymin + 1)
                '''
                dx_scale = (ds-1.0) * 0.5 * (xmax - xmin + 1)
                dy_scale = (ds-1.0) * 0.5 * (ymax - ymin + 1)


                xmin = int(xmin+dx-dx_scale+0.5)
                ymin = int(ymin+dy-dy_scale+0.5)
                xmax = int(xmax+dx+dx_scale+0.5)
                ymax = int(ymax+dy+dy_scale+0.5)
        else:
            trackLost = 1 
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            return xmin,ymin,xmax,ymax,good_new,good_old
    #print(endA-startA)
    return xmin,ymin,xmax,ymax,good_new,good_old
#   Given 2 consecutives ROI's, computes shiTomasi points in one and then
#   computes optical flow in the next
#
def lucasKanadeTracker(roiFrame1,roiFrame2):
    # INPUT:
    #       ROI1 in RGB
    #       ROI2 in RGB
    # OUTPUT:
    #       DisplacementX
    #       DisplacementY

    
    #frame1 = cv2.imread(directoryImages+'/'+listImages[x-1])
    #roiFrame1 = frame1[ymin:ymax,xmin:xmax]

    feature_params = dict( maxCorners = 100,qualityLevel = 0.1,minDistance = 4,blockSize = 7 )
    lk_params = dict( winSize  = (15,15),maxLevel = 4,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    kernel_sharpen_1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    dircX = 0
    dircY = 0



    old_frame = cv2.filter2D(roiFrame1, -1, kernel_sharpen_1)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_RGB2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    #frame2 = cv2.imread(directoryImages+'/'+listImages[x])
    #roiFrame2 = frame2[ymin:ymax,xmin:xmax] 
    output_2 = cv2.filter2D(roiFrame2, -1, kernel_sharpen_1)
    frame_gray = cv2.cvtColor(output_2, cv2.COLOR_RGB2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    good_new = p1[st==1]
    good_old = p0[st==1]
    #imageShow = np.copy(frame1)
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        dircX += c-a
        dircY += d-b

    dircX = (dircX)/(i+1)
    dircY = (dircY)/(i+1)
    return dircX,dircY

#	Given 2 consecutives ROI's, computes shiTomasi points in one and then
#	computes optical flow in the next, then computes the median
#
def lucasKanadeTrackerMedian(roiFrame1,roiFrame2):
	# INPUT:
	# 		ROI1 in RGB
	#		ROI2 in RGB
	# OUTPUT:
	#		DisplacementX
	#		DisplacementY

	
    #frame1 = cv2.imread(directoryImages+'/'+listImages[x-1])
    #roiFrame1 = frame1[ymin:ymax,xmin:xmax]

    feature_params = dict( maxCorners = 100,qualityLevel = 0.1,minDistance = 4,blockSize = 7 )
    lk_params = dict( winSize  = (15,15),maxLevel = 4,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    kernel_sharpen_1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

    dircXList = []
    dircYList = []
    dircX = 0
    dircY = 0


    old_frame = cv2.filter2D(roiFrame1, -1, kernel_sharpen_1)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_RGB2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    #frame2 = cv2.imread(directoryImages+'/'+listImages[x])
    #roiFrame2 = frame2[ymin:ymax,xmin:xmax] 
    output_2 = cv2.filter2D(roiFrame2, -1, kernel_sharpen_1)
    frame_gray = cv2.cvtColor(output_2, cv2.COLOR_RGB2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    good_new = p1[st==1]
    good_old = p0[st==1]
    #imageShow = np.copy(frame1)
    for i,(new,old) in enumerate(zip(good_new,good_old)):
    	a,b = new.ravel()
    	c,d = old.ravel()
    	dircXList.append(c-a)
    	dircYList.append(d-b)
    	#dircX += c-a
    	#dircY += d-b

    dircX = np.median(dircXList)
    dircY = np.median(dircYList)
   	#median([1, 3, 5])
    #dircX = (dircX)/(i+1)
    #dircY = (dircY)/(i+1)
    return dircX,dircY

#	Given 2 consecutives ROI's, computes forwardBackwardFlow
#	'Forward-Backward Error: Automatic Detection of Tracking Failures'
#
def lucasKanadeTrackerFB(roiFrame1,roiFrame2):
	# INPUT:
	# 		ROI1 in RGB
	#		ROI2 in RGB
	# OUTPUT:
	#		DisplacementX
	#		DisplacementY

	
    #frame1 = cv2.imread(directoryImages+'/'+listImages[x-1])
    #roiFrame1 = frame1[ymin:ymax,xmin:xmax]

    feature_params = dict( maxCorners = 100,qualityLevel = 0.1,minDistance = 4,blockSize = 7 )
    lk_params = dict( winSize  = (15,15),maxLevel = 4,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    kernel_sharpen_1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    dircX = 0
    dircY = 0
    old_pts=[]
    ### -----------------------------------------------------
    output_1 = cv2.GaussianBlur(roiFrame1,(5,5),0)
    g = cv2.cvtColor(output_1, cv2.COLOR_RGB2GRAY)
    pt = cv2.goodFeaturesToTrack(g, **feature_params)
    
    
    p0 = np.float32(pt).reshape(-1, 1, 2)
    
    
    output_2 = cv2.GaussianBlur(roiFrame2,(5,5),0)
    newg = cv2.cvtColor(output_2, cv2.COLOR_RGB2GRAY)
    p0 = np.float32(pt).reshape(-1, 1, 2)
    p1, st, err = cv2.calcOpticalFlowPyrLK(g, newg, p0,None, **lk_params)
    p0r, st, err = cv2.calcOpticalFlowPyrLK(newg, g, p1,None, **lk_params)
    d = abs(p0-p0r).reshape(-1, 2).max(-1)
    good = d < 1
    new_pts = []
    p0 = p0[good]
    for pts, val in zip(p1, good):
    	if val:

    		new_pts.append([pts[0][0], pts[0][1]])
            

    
    p0 = p0.reshape((np.shape(p0)[0],np.shape(p0)[2]))
    
    #print(new_pts[:,0],new_pts[0,1])
    dircX = np.median(new_pts[:][0] - p0[:][0])
    dircY = np.median(new_pts[:][1] - p0[:][1])
    old_pts = new_pts
    return dircX,dircY


def lucasKanadeTrackerWeighted(roiFrame1,roiFrame2):
    # INPUT:
    #       ROI1 in RGB
    #       ROI2 in RGB
    # OUTPUT:
    #       DisplacementX
    #       DisplacementY

    
    #frame1 = cv2.imread(directoryImages+'/'+listImages[x-1])
    #roiFrame1 = frame1[ymin:ymax,xmin:xmax]

    #feature_params = dict( maxCorners = 100,qualityLevel = 0.1,minDistance = 4,blockSize = 7 )
    #lk_params = dict( winSize  = (15,15),maxLevel = 4,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    dircX = 0
    dircY = 0


    old_frame = cv2.GaussianBlur(roiFrame1,(5,5),0)
    
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_RGB2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    #frame2 = cv2.imread(directoryImages+'/'+listImages[x])
    #roiFrame2 = frame2[ymin:ymax,xmin:xmax]
    output_2 = cv2.GaussianBlur(roiFrame2,(5,5),0) 
        
    frame_gray = cv2.cvtColor(output_2, cv2.COLOR_RGB2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    good_new = p1[st==1]
    good_old = p0[st==1]
    
    centerX =  (np.shape(old_frame)[0]+1)/2
    centerY =  (np.shape(old_frame)[1]+1)/2
    
    distanceCenterX = []
    distanceCenterY = []

    #imageShow = np.copy(frame1)
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        #dircX += c-a
        #dircY += d-b
        distanceCenterX.append((c-centerX)**2)
        distanceCenterY.append((d-centerY)**2)

    disa = 1/(np.sqrt(distanceCenterX+distanceCenterY)+ 2**-23)
    
    weights = disa/np.sum(disa)
    print(np.sum(weights))
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        dircX += (c-a)*weights[i]
        dircY += (d-b)*weights[i]

    print(di)


    #dircX = (dircX)/(i+1)
    #dircY = (dircY)/(i+1)
    return dircX,dircY

#   Given 2 consecutives ROI's, computes shiTomasi points in one and then
#   computes optical flow in the next, then computes the median
#

def lucasKanadeTrackerMedianScaleStatic(roiFrame1,roiFrame2,xmin,ymin,xmax,ymax):
    # INPUT:
    #       ROI1 in RGB
    #       ROI2 in RGB
    # OUTPUT:
    #       DisplacementX
    #       DisplacementY

    
    #frame1 = cv2.imread(directoryImages+'/'+listImages[x-1])
    #roiFrame1 = frame1[ymin:ymax,xmin:xmax]
    
    
    feature_params = dict( maxCorners = 1000,qualityLevel = 0.1,minDistance = 2,blockSize = 7 )
    #feature_params = dict( maxCorners = 1000,qualityLevel = 0.2,minDistance = 4,blockSize = 7 )

    lk_params = dict( winSize  = (15,15),maxLevel = 4,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    #dircX = 0
    #dircY = 0
    trackLost = 0
    good_new = []
    good_old = []
    if np.shape(roiFrame1)[0]==0 or np.shape(roiFrame1)[1]==0:
        trackLost = 1
        return xmin,ymin,xmax,ymax,good_new,good_old

    old_gray = cv2.cvtColor(roiFrame1, cv2.COLOR_RGB2GRAY)
    equ = cv2.equalizeHist(old_gray)
    
    #old_frame = cv2.filter2D(roiFrame1, -1, kernel_sharpen_1)
    #old_gray = cv2.cvtColor(old_frame, cv2.COLOR_RGB2GRAY)
    p0 = cv2.goodFeaturesToTrack(equ, mask = None, **feature_params)

    #print('point',np.shape(p0)[0])
    #print(p0)
    if p0 is None:

        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        trackLost = 1
        return xmin,ymin,xmax,ymax,good_new,good_old
    else:
        if np.shape(p0)[0]!=0:

            #print('sa',np.shape(p0))
            #frame2 = cv2.imread(directoryImages+'/'+listImages[x])
            #roiFrame2 = frame2[ymin:ymax,xmin:xmax]
            frame_gray = cv2.cvtColor(roiFrame2, cv2.COLOR_RGB2GRAY)
            equ2 = cv2.equalizeHist(frame_gray)
            #output_2 = cv2.GaussianBlur(roiFrame2,(9,9),0)

            p1, st, err = cv2.calcOpticalFlowPyrLK(equ, equ2, p0, None, **lk_params)
            good_new = p1[st==1]
            good_old = p0[st==1]
            #print('tam2',np.shape(good_new),np.shape(good_old))

            err = err[[st==1]].flatten()
            indx = np.argsort(err)
            half_indx = indx[:len(indx) // 2]
            good_old = (p0[[st==1]])[half_indx]
            good_new = (p1[[st==1]])[half_indx]
            #print('points',np.shape(half_indx))


            #dx = np.median(good_new[:, 0] - good_old[:, 0])
            #dy = np.median(good_new[:, 1] - good_old[:, 1])
            ll = [good_new[:, 0] - good_old[:, 0],good_new[:, 1] - good_old[:, 1]]
            #print(ll)
            thresHOLD = 1.3
            idxOFstatic0 = np.where(ll[:][0]>thresHOLD) 
            idxOFstatic1 = np.where(ll[:][0]<-thresHOLD)
            idx1Fstatic0 = np.where(ll[:][1]>thresHOLD) 
            idx1Fstatic1 = np.where(ll[:][1]<-thresHOLD)
            #print('.l')
            idxOFstaticX = list(set(idxOFstatic0[0]) | set(idxOFstatic1[0]))
            idxOFstaticY = list(set(idx1Fstatic0[0]) | set(idx1Fstatic1[0]))

            #print(idxOFstaticX,idxOFstaticY)
            
            if np.shape(idxOFstaticX)[0] ==0 and np.shape(idxOFstaticY)[0] ==0:
                
                return xmin,ymin,xmax,ymax,good_new,good_old

            if np.shape(idxOFstaticY)[0] ==0:
                
                dx = np.median(good_new[idxOFstaticX, 0] - good_old[idxOFstaticX, 0])
                dy = 0

                #print('x')

            elif np.shape(idxOFstaticY)[0] ==0:
                
                dx = np.median(good_new[idxOFstaticX, 0] - good_old[idxOFstaticX, 0])
                dy = 0

                #print('y')
            
            #elif np.shape(idxOFstaticX)[0] != 0 and np.shape(idxOFstaticY)[0] !=0:
            else:
                dx = np.median(good_new[idxOFstaticX, 0] - good_old[idxOFstaticX, 0])
                dy = np.median(good_new[idxOFstaticY, 1] - good_old[idxOFstaticY, 1])

                #print('alright')

            idxOFgodd = list(set(idxOFstaticX) | set(idxOFstaticY))
            #print(idxOFgodd)
            
            #dx = np.median(good_new[idxOFstaticX, 0] - good_old[idxOFstaticX, 0])
            #dy = np.median(good_new[idxOFstaticY, 1] - good_old[idxOFstaticY, 1])
            #print('m1',dx,dy)
           

            good_new2 = good_new[idxOFgodd, :]
            good_old2 = good_old[idxOFgodd, :]

            #good_new2 = good_new
            #good_old2 = good_old

            
            i, j = np.triu_indices(len(good_old2), k=1)
            #print('numP',np.shape(good_new))

            pdiff0 = good_old2[i] - good_old2[j]
            pdiff1 = good_new2[i] - good_new2[j]
            
            p0_dist = np.sum(pdiff0 ** 2, axis=1)
            p1_dist = np.sum(pdiff1 ** 2, axis=1)
            ds = np.median(np.sqrt((p1_dist / (p0_dist + 2**-23))))
            
            #print(dx,dy,ds)
            #print('---------')
            if np.isnan(dx) or np.isnan(dy) or np.isnan(ds) :
                #print('m1',dx,dy,ds,np.isnan(dx))

                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)
                trackLost = 1
                return xmin,ymin,xmax,ymax,good_new,good_old

            else:
                '''
                ds_factor = 1.5
                ds = (1.0 - ds_factor) + ds_factor * ds;
                dx_scale = (ds - 1.0) * 0.5 * (xmax - xmin + 1)
                dy_scale = (ds - 1.0) * 0.5 * (ymax - ymin + 1)
                '''
                dx_scale = (ds-1.0) * 0.5 * (xmax - xmin + 1)
                dy_scale = (ds-1.0) * 0.5 * (ymax - ymin + 1)


                xmin = int(xmin+dx-dx_scale+0.5)
                ymin = int(ymin+dy-dy_scale+0.5)
                xmax = int(xmax+dx+dx_scale+0.5)
                ymax = int(ymax+dy+dy_scale+0.5)
        else:
            trackLost = 1 
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            return xmin,ymin,xmax,ymax,good_new,good_old
    #print(endA-startA)
    return xmin,ymin,xmax,ymax,good_new,good_old
    #return xmin,ymin,xmax,ymax,trackLost

def lucasKanadeTrackerMedianScaleStatic2(roiFrame1,roiFrame2,xmin,ymin,xmax,ymax):
    # INPUT:
    #       ROI1 in RGB
    #       ROI2 in RGB
    # OUTPUT:
    #       DisplacementX
    #       DisplacementY

    
    #frame1 = cv2.imread(directoryImages+'/'+listImages[x-1])
    #roiFrame1 = frame1[ymin:ymax,xmin:xmax]
    
    
    feature_params = dict( maxCorners = 1000,qualityLevel = 0.1,minDistance = 2,blockSize = 7 )
    #feature_params = dict( maxCorners = 1000,qualityLevel = 0.2,minDistance = 4,blockSize = 7 )

    lk_params = dict( winSize  = (15,15),maxLevel = 4,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    #dircX = 0
    #dircY = 0
    trackLost = 0
    good_new = []
    good_old = []
    dss = 0
    dss1 = 0

    if np.shape(roiFrame1)[0]==0 or np.shape(roiFrame1)[1]==0:
        trackLost = 1
        return xmin,ymin,xmax,ymax,trackLost,dss,dss1

    old_gray = cv2.cvtColor(roiFrame1, cv2.COLOR_RGB2GRAY)
    equ = cv2.equalizeHist(old_gray)
    

    p0 = cv2.goodFeaturesToTrack(equ, mask = None, **feature_params)

    #print('point',np.shape(p0)[0])
    #print(p0)
    if p0 is None:

        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        trackLost = 1
        return xmin,ymin,xmax,ymax,trackLost,dss,dss1
    else:
        if np.shape(p0)[0]!=0:

            #print('sa',np.shape(p0))
            #frame2 = cv2.imread(directoryImages+'/'+listImages[x])
            #roiFrame2 = frame2[ymin:ymax,xmin:xmax]
            frame_gray = cv2.cvtColor(roiFrame2, cv2.COLOR_RGB2GRAY)
            equ2 = cv2.equalizeHist(frame_gray)
            #output_2 = cv2.GaussianBlur(roiFrame2,(9,9),0)

            p1, st, err = cv2.calcOpticalFlowPyrLK(equ, equ2, p0, None, **lk_params)
            good_new = p1[st==1]
            good_old = p0[st==1]
            #print('tam2',np.shape(good_new),np.shape(good_old))

            err = err[[st==1]].flatten()
            indx = np.argsort(err)
            half_indx = indx[:len(indx) // 2]
            good_old = (p0[[st==1]])[half_indx]
            good_new = (p1[[st==1]])[half_indx]
            #print('points',np.shape(half_indx))


            #dx = np.median(good_new[:, 0] - good_old[:, 0])
            #dy = np.median(good_new[:, 1] - good_old[:, 1])
            ll = [good_new[:, 0] - good_old[:, 0],good_new[:, 1] - good_old[:, 1]]
            #print(ll)
            thresHOLD = 1.3
            idxOFstatic0 = np.where(ll[:][0]>thresHOLD) 
            idxOFstatic1 = np.where(ll[:][0]<-thresHOLD)
            idx1Fstatic0 = np.where(ll[:][1]>thresHOLD) 
            idx1Fstatic1 = np.where(ll[:][1]<-thresHOLD)
            #print('.l')
            idxOFstaticX = list(set(idxOFstatic0[0]) | set(idxOFstatic1[0]))
            idxOFstaticY = list(set(idx1Fstatic0[0]) | set(idx1Fstatic1[0]))

            #print(idxOFstaticX,idxOFstaticY)
            
            if np.shape(idxOFstaticX)[0] ==0 and np.shape(idxOFstaticY)[0] ==0:
                
                return xmin,ymin,xmax,ymax,trackLost,dss,dss1

            if np.shape(idxOFstaticY)[0] ==0:
                
                dx = np.median(good_new[idxOFstaticX, 0] - good_old[idxOFstaticX, 0])
                dy = 0

                #print('x')

            elif np.shape(idxOFstaticY)[0] ==0:
                
                dx = np.median(good_new[idxOFstaticX, 0] - good_old[idxOFstaticX, 0])
                dy = 0

                #print('y')
            
            #elif np.shape(idxOFstaticX)[0] != 0 and np.shape(idxOFstaticY)[0] !=0:
            else:
                dx = np.median(good_new[idxOFstaticX, 0] - good_old[idxOFstaticX, 0])
                dy = np.median(good_new[idxOFstaticY, 1] - good_old[idxOFstaticY, 1])

                #print('alright')

            idxOFgodd = list(set(idxOFstaticX) | set(idxOFstaticY))
            #print(idxOFgodd)
            
            #dx = np.median(good_new[idxOFstaticX, 0] - good_old[idxOFstaticX, 0])
            #dy = np.median(good_new[idxOFstaticY, 1] - good_old[idxOFstaticY, 1])
            #print('m1',dx,dy)
           

            good_new2 = good_new[idxOFgodd, :]
            good_old2 = good_old[idxOFgodd, :]

            #good_new2 = good_new
            #good_old2 = good_old

            
            i, j = np.triu_indices(len(good_old2), k=1)
            #print('numP',np.shape(good_new))

            pdiff0 = good_old2[i] - good_old2[j]
            pdiff1 = good_new2[i] - good_new2[j]
            
            p0_dist = np.sum(pdiff0 ** 2, axis=1)
            p1_dist = np.sum(pdiff1 ** 2, axis=1)
            ds = np.median(np.sqrt((p1_dist / (p0_dist + 2**-23))))
            
            #print(dx,dy,ds)
            #print('---------')
            if np.isnan(dx) or np.isnan(dy) or np.isnan(ds) :
                #print('m1',dx,dy,ds,np.isnan(dx))

                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)
                trackLost = 1
                return xmin,ymin,xmax,ymax,trackLost,dss,dss1

            else:
                '''
                ds_factor = 1.5
                ds = (1.0 - ds_factor) + ds_factor * ds;
                dx_scale = (ds - 1.0) * 0.5 * (xmax - xmin + 1)
                dy_scale = (ds - 1.0) * 0.5 * (ymax - ymin + 1)
                '''
                dx_scale = (ds-1.0) * 0.5 * (xmax - xmin + 1)
                dy_scale = (ds-1.0) * 0.5 * (ymax - ymin + 1)

                #print(dx,dy)
                xmin = int(xmin+dx-dx_scale+0.5)
                ymin = int(ymin+dy-dy_scale+0.5)
                xmax = int(xmax+dx+dx_scale+0.5)
                ymax = int(ymax+dy+dy_scale+0.5)
        else:
            trackLost = 1 
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            return xmin,ymin,xmax,ymax,trackLost,dss,dss1
    #print(endA-startA)
    return xmin,ymin,xmax,ymax,trackLost,dx,dy
    #return xmin,ymin,xmax,ymax,trackLost

def lucasKanadeTrackerMedianScaleStatic2Plus(roiFrame1,roiFrame2,xmin,ymin,xmax,ymax):
    # INPUT:
    #       ROI1 in RGB
    #       ROI2 in RGB
    # OUTPUT:
    #       DisplacementX
    #       DisplacementY

    
    #frame1 = cv2.imread(directoryImages+'/'+listImages[x-1])
    #roiFrame1 = frame1[ymin:ymax,xmin:xmax]
    
    
    feature_params = dict( maxCorners = 1000,qualityLevel = 0.1,minDistance = 2,blockSize = 7 )
    #feature_params = dict( maxCorners = 1000,qualityLevel = 0.2,minDistance = 4,blockSize = 7 )

    lk_params = dict( winSize  = (15,15),maxLevel = 4,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    #dircX = 0
    #dircY = 0
    trackLost = 0
    good_new = []
    good_old = []
    dss = 0
    dss1 = 0

    if np.shape(roiFrame1)[0]==0 or np.shape(roiFrame1)[1]==0:

        trackLost = 1
        return xmin,ymin,xmax,ymax,trackLost,dss,dss1

    old_gray = cv2.cvtColor(roiFrame1, cv2.COLOR_RGB2GRAY)
    equ = cv2.equalizeHist(old_gray)
    
    p0 = cv2.goodFeaturesToTrack(equ, mask = None, **feature_params)

    if p0 is None:

        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        trackLost = 1
        return xmin,ymin,xmax,ymax,trackLost,dss,dss1
    else:
        if np.shape(p0)[0]!=0:


            frame_gray = cv2.cvtColor(roiFrame2, cv2.COLOR_RGB2GRAY)
            equ2 = cv2.equalizeHist(frame_gray)

            p1, st, err = cv2.calcOpticalFlowPyrLK(equ, equ2, p0, None, **lk_params)
            good_new = p1[st==1]
            good_old = p0[st==1]

            err = err[[st==1]].flatten()
            indx = np.argsort(err)
            half_indx = indx[:len(indx) // 2]
            good_old = (p0[[st==1]])[half_indx]
            good_new = (p1[[st==1]])[half_indx]
  
            #ll = [good_new[:, 0] - good_old[:, 0],good_new[:, 1] - good_old[:, 1]]
            ll = good_new-good_old
            thresHOLD = 1.0

            idxX = np.where(np.absolute(ll[:,0])>thresHOLD)
            idxY = np.where(np.absolute(ll[:,1])>thresHOLD)

            numPointsX = np.shape(idxX)[1]
            numPointsY = np.shape(idxY)[1]
            

            if ( numPointsX != 0 ) and (numPointsY != 0):
                
                dx = np.median(ll[idxX[0],0])
                dy = np.median(ll[idxY[0],1])
                
                idxOFgodd = np.union1d(idxX[0],idxY[0])

            elif ( numPointsX == 0 ) and (numPointsY != 0): 

                dx = 0.0
                dy = np.median(ll[idxY[0],1])
                idxOFgodd = idxY[0]

            elif ( numPointsX != 0 ) and (numPointsY == 0): 

                dx = np.median(ll[idxX[0],0])
                dy = 0.0
                idxOFgodd = idxX[0]
                
            else:

                dx = 0.0
                dy = 0.0

                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)
                trackLost = 0
                #print('bu')

                return xmin,ymin,xmax,ymax,trackLost,dx,dy

            good_new2 = good_new[idxOFgodd, :]
            good_old2 = good_old[idxOFgodd, :]
            
            i, j = np.triu_indices(len(good_old2), k=1)

            pdiff0 = good_old2[i] - good_old2[j]
            pdiff1 = good_new2[i] - good_new2[j]
            
            p0_dist = np.sum(pdiff0 ** 2, axis=1)
            p1_dist = np.sum(pdiff1 ** 2, axis=1)

            
            if (np.shape(p0_dist)[0]==0 ) or (np.shape(p1_dist)[0]==0):
                
                xmin = int(xmin+dx)
                ymin = int(ymin+dy)
                xmax = int(xmax+dx)
                ymax = int(ymax+dy)
                trackLost = 0
                return xmin,ymin,xmax,ymax,trackLost,dx,dy

            else:

                '''
                ds_factor = 1.5
                ds = (1.0 - ds_factor) + ds_factor * ds;
                dx_scale = (ds - 1.0) * 0.5 * (xmax - xmin + 1)
                dy_scale = (ds - 1.0) * 0.5 * (ymax - ymin + 1)
                '''
                ds = np.median(np.sqrt((p1_dist / (p0_dist + 2**-23))))
                ds_factor = 0.95
                ds = (1.0 - ds_factor) + ds_factor * ds
                dx_scale = (ds-1.0) * 0.5 * (xmax - xmin + 1)
                dy_scale = (ds-1.0) * 0.5 * (ymax - ymin + 1)
                trackLost = 0
                #print('Normal',dx,dy,ds)
                xmin = int(xmin+dx-dx_scale+0.5)
                ymin = int(ymin+dy-dy_scale+0.5)
                xmax = int(xmax+dx+dx_scale+0.5)
                ymax = int(ymax+dy+dy_scale+0.5)
                return xmin,ymin,xmax,ymax,trackLost,dx,dy
        else:

            trackLost = 1 
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            return xmin,ymin,xmax,ymax,trackLost,dss,dss1
    
    return xmin,ymin,xmax,ymax,trackLost,dx,dy

def lucasKanadeTrackerMedianScaleStatic2PlusOptimized(roiFrame1,roiFrame2,xmin,ymin,xmax,ymax):
    # INPUT:
    #       ROI1 in RGB
    #       ROI2 in RGB
    # OUTPUT:
    #       DisplacementX
    #       DisplacementY

    
    #frame1 = cv2.imread(directoryImages+'/'+listImages[x-1])
    #roiFrame1 = frame1[ymin:ymax,xmin:xmax]
    
    
    feature_params = dict( maxCorners = 1000,qualityLevel = 0.2,minDistance = 2,blockSize = 7 )
    #feature_params = dict( maxCorners = 1000,qualityLevel = 0.2,minDistance = 4,blockSize = 7 )

    lk_params = dict( winSize  = (15,15),maxLevel = 4,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    

    trackLost = 0
    good_new = []
    good_old = []
    dss = 0
    dss1 = 0

  
    pss = [0,0,0,0,0]
    p0 = 0

    if np.shape(roiFrame1)[0]==0 or np.shape(roiFrame1)[1]==0:

        trackLost = 1
        return xmin,ymin,xmax,ymax,trackLost,dss,dss1,pss

    #old_gray = cv2.cvtColor(roiFrame1, cv2.COLOR_RGB2GRAY)
    equ = cv2.equalizeHist(roiFrame1)

    #startB = time.time()  
    p0 = cv2.goodFeaturesToTrack(equ, mask = None, **feature_params)
    #endB = time.time()
    #pss[0]=endB-startB
    if np.shape(p0)[0]!=0:

    	#frame_gray = cv2.cvtColor(roiFrame2, cv2.COLOR_RGB2GRAY)
    	equ2 = cv2.equalizeHist(roiFrame2)
    	#startB = time.time() 
    	p1, st, err = cv2.calcOpticalFlowPyrLK(equ, equ2, p0, None, **lk_params)
    	#endB = time.time()
    	#pss[1]=endB-startB

    	#startB = time.time() 
    	good_new = p1[st==1]
    	good_old = p0[st==1]
    	err = err[[st==1]].flatten()
    	indx = np.argsort(err)
    	half_indx = indx[:len(indx) // 2]
    	#half_indx = indx[:10]
    	good_old = (p0[[st==1]])[half_indx]
    	good_new = (p1[[st==1]])[half_indx]

    	#endB = time.time()
    	#pss[2]=endB-startB

    	#startB = time.time()
    	ll = good_new-good_old
    	thresHOLD = 1.0
    	idxX = np.where(np.absolute(ll[:,0])>thresHOLD)
    	idxY = np.where(np.absolute(ll[:,1])>thresHOLD)
    	numPointsX = np.shape(idxX)[1]
    	numPointsY = np.shape(idxY)[1]


    	#print(idxX[0][:10])
    	#print(idxY[0][:10])
    	if ( numPointsX != 0 ) and (numPointsY != 0):
    		dx = np.median(ll[idxX[0],0])
    		dy = np.median(ll[idxY[0],1])
    		idxOFgodd = np.union1d(idxX[0],idxY[0])
    	elif ( numPointsX == 0 ) and (numPointsY != 0):
    		dx = 0.0
    		dy = np.median(ll[idxY[0],1])
    		idxOFgodd = idxY[0]
    	elif ( numPointsX != 0 ) and (numPointsY == 0):
    		dx = np.median(ll[idxX[0],0])
    		dy = 0.0
    		idxOFgodd = idxX[0]
    	else:
    		dx = 0.0
    		dy = 0.0
    		xmin = int(xmin)
    		ymin = int(ymin)
    		xmax = int(xmax)
    		ymax = int(ymax)
    		trackLost = 0
    		#print('bu')
    		#endB = time.time()
    		return xmin,ymin,xmax,ymax,trackLost,dss,dss1,pss
    	
    	numPois = np.shape(idxOFgodd)[0]
    	good_new2 = good_new[idxOFgodd, :]
    	good_old2 = good_old[idxOFgodd, :]
    	endB = time.time()
    	#pss[3]=endB-startB


    	#startB = time.time()
    	i, j = np.triu_indices(len(good_old2), k=1)
    	pdiff0 = good_old2[i] - good_old2[j]
    	pdiff1 = good_new2[i] - good_new2[j]
    	p0_dist = np.sum(pdiff0 ** 2, axis=1)
    	p1_dist = np.sum(pdiff1 ** 2, axis=1)

    	if (np.shape(p0_dist)[0]!=0 ) or (np.shape(p1_dist)[0]!=0):
    		ds = np.median(np.sqrt((p1_dist / (p0_dist + 2**-23))))
    		ds_factor = 0.95
    		ds = (1.0 - ds_factor) + ds_factor * ds
    		dx_scale = (ds-1.0) * 0.5 * (xmax - xmin + 1)
    		dy_scale = (ds-1.0) * 0.5 * (ymax - ymin + 1)
    		trackLost = 0
    		xmin = int(xmin+dx-dx_scale+0.5)
    		ymin = int(ymin+dy-dy_scale+0.5)
    		xmax = int(xmax+dx+dx_scale+0.5)
    		ymax = int(ymax+dy+dy_scale+0.5)

    		#endB = time.time()
    		#pss[4]=endB-startB
    		return xmin,ymin,xmax,ymax,trackLost,dss,dss1,numPois
    	else:
    		xmin = int(xmin+dx)
    		ymin = int(ymin+dy)
    		xmax = int(xmax+dx)
    		ymax = int(ymax+dy)
    		trackLost = 0
    		return xmin,ymin,xmax,ymax,trackLost,dss,dss1,numPois


    else:

        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        trackLost = 1
        return xmin,ymin,xmax,ymax,trackLost,dss,dss1,pss
 
    
def lucasKanadeTrackerMedianScaleStatic2PlusOptimized2(roiFrame1,roiFrame2,xmin,ymin,xmax,ymax):
    # INPUT:
    #       ROI1 in RGB
    #       ROI2 in RGB
    # OUTPUT:
    #       DisplacementX
    #       DisplacementY

    
    #frame1 = cv2.imread(directoryImages+'/'+listImages[x-1])
    #roiFrame1 = frame1[ymin:ymax,xmin:xmax]
    
    
    feature_params = dict( maxCorners = 20,qualityLevel = 0.2,minDistance = 2,blockSize = 7 )
    #feature_params = dict( maxCorners = 1000,qualityLevel = 0.2,minDistance = 4,blockSize = 7 )

    lk_params = dict( winSize  = (15,15),maxLevel = 4,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    

    trackLost = 0
    good_new = []
    good_old = []
    dss = 0
    dss1 = 0

  
    pss = [0,0,0]
    p0 = 0

    if np.shape(roiFrame1)[0]==0 or np.shape(roiFrame1)[1]==0:

        trackLost = 1
        return xmin,ymin,xmax,ymax,trackLost,dss,dss1,pss

    #startB = time.time() 
    #old_gray = cv2.cvtColor(roiFrame1, cv2.COLOR_RGB2GRAY)
    equ = cv2.equalizeHist(roiFrame1)
    
    p0 = cv2.goodFeaturesToTrack(equ, mask = None, **feature_params)
    #endB = time.time()
    #pss[0]=endB-startB
    

    if np.shape(p0)[0]!=0:

    	#startB = time.time() 
    	#frame_gray = cv2.cvtColor(roiFrame2, cv2.COLOR_RGB2GRAY)
    	equ2 = cv2.equalizeHist(roiFrame2)
    	p1, st, err = cv2.calcOpticalFlowPyrLK(equ, equ2, p0, None, **lk_params)
    	#endB = time.time()
    	#pss[1]=endB-startB
    	
    	#startB = time.time()
    	good_new = p1[st==1]
    	good_old = p0[st==1]
    	err = err[[st==1]].flatten()
    	indx = np.argsort(err)
    	half_indx = indx[:len(indx) // 2]
    	good_old = (p0[[st==1]])[half_indx]
    	good_new = (p1[[st==1]])[half_indx]

    	ll = good_new-good_old
    	thresHOLD = 1.0
    	idxX = np.where(np.absolute(ll[:,0])>thresHOLD)[0]
    	idxY = np.where(np.absolute(ll[:,1])>thresHOLD)[0]

    	
    	numPointsX = np.shape(idxX)[0]
    	numPointsY = np.shape(idxY)[0]
    	
    	if ( numPointsX != 0 ) and (numPointsY != 0):
    		dx = np.median(ll[idxX,0])
    		dy = np.median(ll[idxY,1])
    		idxOFgodd = np.union1d(idxX,idxY)
    	elif ( numPointsX == 0 ) and (numPointsY != 0):
    		dx = 0.0
    		dy = np.median(ll[idxY,1])
    		idxOFgodd = idxY
    	elif ( numPointsX != 0 ) and (numPointsY == 0):
    		dx = np.median(ll[idxX,0])
    		dy = 0.0
    		idxOFgodd = idxX
    	else:
    		dx = 0.0
    		dy = 0.0
    		xmin = int(xmin)
    		ymin = int(ymin)
    		xmax = int(xmax)
    		ymax = int(ymax)
    		trackLost = 1
    		#endB = time.time()
    		#pss[2]=endB-startB

    		return xmin,ymin,xmax,ymax,trackLost,dss,dss1,pss
    	

    	numPois = np.shape(idxOFgodd)[0]
    	good_new2 = good_new[idxOFgodd, :]
    	good_old2 = good_old[idxOFgodd, :]


      	
    	i, j = np.triu_indices(len(good_old2), k=1)
    	pdiff0 = good_old2[i] - good_old2[j]
    	pdiff1 = good_new2[i] - good_new2[j]
    	p0_dist = np.sum(pdiff0 ** 2, axis=1)
    	p1_dist = np.sum(pdiff1 ** 2, axis=1)

    	if (np.shape(p0_dist)[0]!=0 ) or (np.shape(p1_dist)[0]!=0):
    		ds = np.median(np.sqrt((p1_dist / (p0_dist + 2**-23))))
    		ds_factor = 0.95
    		ds = (1.0 - ds_factor) + ds_factor * ds
    		dx_scale = (ds-1.0) * 0.5 * (xmax - xmin + 1)
    		dy_scale = (ds-1.0) * 0.5 * (ymax - ymin + 1)
    		trackLost = 0
    		xmin = int(xmin+dx-dx_scale+0.5)
    		ymin = int(ymin+dy-dy_scale+0.5)
    		xmax = int(xmax+dx+dx_scale+0.5)
    		ymax = int(ymax+dy+dy_scale+0.5)
    		#endB = time.time()
    		#pss[2]=endB-startB
    		return xmin,ymin,xmax,ymax,trackLost,dss,dss1,pss
    	else:
    		xmin = int(xmin+dx)
    		ymin = int(ymin+dy)
    		xmax = int(xmax+dx)
    		ymax = int(ymax+dy)
    		trackLost = 0
    		return xmin,ymin,xmax,ymax,trackLost,dss,dss1,pss


    else:

        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        trackLost = 1
        return xmin,ymin,xmax,ymax,trackLost,dss,dss1,pss


def lucasKanadeTrackerMedianScaleStatic2PlusOptimized2Deploy(roiFrame1,roiFrame2,xmin,ymin,xmax,ymax):
    # INPUT:
    #       ROI1 in RGB
    #       ROI2 in RGB
    # OUTPUT:
    #       DisplacementX
    #       DisplacementY

    
    #frame1 = cv2.imread(directoryImages+'/'+listImages[x-1])
    #roiFrame1 = frame1[ymin:ymax,xmin:xmax]
    
    
    feature_params = dict( maxCorners = 2000,qualityLevel = 0.15,minDistance = 2,blockSize = 7 )
    #feature_params = dict( maxCorners = 1000,qualityLevel = 0.2,minDistance = 4,blockSize = 7 )

    lk_params = dict( winSize  = (15,15),maxLevel = 4,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    

    trackLost = 0
    good_new = []
    good_old = []
    dss = 0
    dss1 = 0

  
    pss = 0
    p0 = 0

    if np.shape(roiFrame1)[0]==0 or np.shape(roiFrame1)[1]==0:

        trackLost = 1
        return xmin,ymin,xmax,ymax,trackLost,dss,dss1

    old_gray = cv2.cvtColor(roiFrame1, cv2.COLOR_RGB2GRAY)
    equ = cv2.equalizeHist(old_gray)
    
      
    p0 = cv2.goodFeaturesToTrack(equ, mask = None, **feature_params)

    if not p0 is None:
    #if np.shape(p0)[0]!=0:

    	frame_gray = cv2.cvtColor(roiFrame2, cv2.COLOR_RGB2GRAY)
    	equ2 = cv2.equalizeHist(frame_gray)
    	#p0 = p0[:40]
    	
    	
    	p1, st, err = cv2.calcOpticalFlowPyrLK(equ, equ2, p0, None, **lk_params)

    	good_new = p1[st==1]
    	good_old = p0[st==1]
    	err = err[[st==1]].flatten()
    	indx = np.argsort(err)
    	half_indx = indx[:len(indx) // 2]
    	good_old = (p0[[st==1]])[half_indx]
    	good_new = (p1[[st==1]])[half_indx]



    	ll = good_new-good_old
    	thresHOLD = 0.8
    	idxX = np.where(np.absolute(ll[:,0])>thresHOLD)[0]
    	idxY = np.where(np.absolute(ll[:,1])>thresHOLD)[0]

    	
    	numPointsX = np.shape(idxX)[0]
    	numPointsY = np.shape(idxY)[0]
    	
    	if ( numPointsX != 0 ) and (numPointsY != 0):
    		dx = np.median(ll[idxX,0])
    		dy = np.median(ll[idxY,1])
    		idxOFgodd = np.union1d(idxX,idxY)
    	elif ( numPointsX == 0 ) and (numPointsY != 0):
    		dx = 0.0
    		dy = np.median(ll[idxY,1])
    		idxOFgodd = idxY
    	elif ( numPointsX != 0 ) and (numPointsY == 0):
    		dx = np.median(ll[idxX,0])
    		dy = 0.0
    		idxOFgodd = idxX
    	else:
    		#dx = 0.0
    		#dy = 0.0
            dx = np.median(ll[:,0])
            dy = np.median(ll[:,1])
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            #trackLost = 1
            return xmin,ymin,xmax,ymax,trackLost,dx,dy
    	

    	numPois = np.shape(idxOFgodd)[0]
    	good_new2 = good_new[idxOFgodd, :]
    	good_old2 = good_old[idxOFgodd, :]


    	i, j = np.triu_indices(len(good_old2), k=1)
    	pdiff0 = good_old2[i] - good_old2[j]
    	pdiff1 = good_new2[i] - good_new2[j]
    	p0_dist = np.sum(pdiff0 ** 2, axis=1)
    	p1_dist = np.sum(pdiff1 ** 2, axis=1)

    	if (np.shape(p0_dist)[0]!=0 ) or (np.shape(p1_dist)[0]!=0):
    		ds = np.median(np.sqrt((p1_dist / (p0_dist + 2**-23))))
    		ds_factor = 0.95
    		ds = (1.0 - ds_factor) + ds_factor * ds
    		dx_scale = (ds-1.0) * 0.5 * (xmax - xmin + 1)
    		dy_scale = (ds-1.0) * 0.5 * (ymax - ymin + 1)
    		trackLost = 0
    		xmin = int(xmin+dx-dx_scale+0.5)
    		ymin = int(ymin+dy-dy_scale+0.5)
    		xmax = int(xmax+dx+dx_scale+0.5)
    		ymax = int(ymax+dy+dy_scale+0.5)

    		return xmin,ymin,xmax,ymax,trackLost,dx,dy
    	else:
    		xmin = int(xmin+dx)
    		ymin = int(ymin+dy)
    		xmax = int(xmax+dx)
    		ymax = int(ymax+dy)
    		trackLost = 0
    		return xmin,ymin,xmax,ymax,trackLost,dx,dy


    else:

        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        trackLost = 1
        return xmin,ymin,xmax,ymax,trackLost,dss,dss1
 
    
def lucasKanadeTrackerMedianScaleStatic2PlusOptimized2DeployPoints(roiFrame1,roiFrame2,xmin,ymin,xmax,ymax):
    # INPUT:
    #       ROI1 in RGB
    #       ROI2 in RGB
    # OUTPUT:
    #       DisplacementX
    #       DisplacementY

    
    #frame1 = cv2.imread(directoryImages+'/'+listImages[x-1])
    #roiFrame1 = frame1[ymin:ymax,xmin:xmax]
    
    
    feature_params = dict( maxCorners = 60,qualityLevel = 0.10,minDistance = 2,blockSize = 7 )
    #feature_params = dict( maxCorners = 1000,qualityLevel = 0.2,minDistance = 4,blockSize = 7 )

    lk_params = dict( winSize  = (15,15),maxLevel = 4,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    

    trackLost = 0
    good_new = []
    good_old = []
    dss = 0
    dss1 = 0
    dsw2 = 0
  
    pss = 0
    p0 = 0

    if np.shape(roiFrame1)[0]==0 or np.shape(roiFrame1)[1]==0:

        trackLost = 1
        return xmin,ymin,xmax,ymax,trackLost,dss,dss1,dsw2

    old_gray = cv2.cvtColor(roiFrame1, cv2.COLOR_RGB2GRAY)
    equ = cv2.equalizeHist(old_gray)
    
      
    p0 = cv2.goodFeaturesToTrack(equ, mask = None, **feature_params)

    if not p0 is None:
    #if np.shape(p0)[0]!=0:

        frame_gray = cv2.cvtColor(roiFrame2, cv2.COLOR_RGB2GRAY)
        equ2 = cv2.equalizeHist(frame_gray)
        #p0 = p0[:40]
        
        
        p1, st, err = cv2.calcOpticalFlowPyrLK(equ, equ2, p0, None, **lk_params)

        good_new = p1[st==1]
        good_old = p0[st==1]
        err = err[[st==1]].flatten()
        indx = np.argsort(err)
        half_indx = indx[:len(indx) // 2]
        good_old = (p0[[st==1]])[half_indx]
        good_new = (p1[[st==1]])[half_indx]



        ll = good_new-good_old
        thresHOLD = 0.8
        idxX = np.where(np.absolute(ll[:,0])>thresHOLD)[0]
        idxY = np.where(np.absolute(ll[:,1])>thresHOLD)[0]

        
        numPointsX = np.shape(idxX)[0]
        numPointsY = np.shape(idxY)[0]
        
        if ( numPointsX != 0 ) and (numPointsY != 0):
            dx = np.median(ll[idxX,0])
            dy = np.median(ll[idxY,1])
            idxOFgodd = np.union1d(idxX,idxY)
        elif ( numPointsX == 0 ) and (numPointsY != 0):
            dx = 0.0
            dy = np.median(ll[idxY,1])
            idxOFgodd = idxY
        elif ( numPointsX != 0 ) and (numPointsY == 0):
            dx = np.median(ll[idxX,0])
            dy = 0.0
            idxOFgodd = idxX
        else:
            #dx = 0.0
            #dy = 0.0
            dx = np.median(ll[:,0])
            dy = np.median(ll[:,1])
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            #trackLost = 1
            return xmin,ymin,xmax,ymax,trackLost,dx,dy,dsw2
        

        numPois = np.shape(idxOFgodd)[0]
        good_new2 = good_new[idxOFgodd, :]
        good_old2 = good_old[idxOFgodd, :]


        i, j = np.triu_indices(len(good_old2), k=1)
        pdiff0 = good_old2[i] - good_old2[j]
        pdiff1 = good_new2[i] - good_new2[j]
        p0_dist = np.sum(pdiff0 ** 2, axis=1)
        p1_dist = np.sum(pdiff1 ** 2, axis=1)

        if (np.shape(p0_dist)[0]!=0 ) or (np.shape(p1_dist)[0]!=0):
            ds = np.median(np.sqrt((p1_dist / (p0_dist + 2**-23))))
            ds_factor = 0.95
            ds = (1.0 - ds_factor) + ds_factor * ds
            dx_scale = (ds-1.0) * 0.5 * (xmax - xmin + 1)
            dy_scale = (ds-1.0) * 0.5 * (ymax - ymin + 1)
            trackLost = 0
            xmin = int(xmin+dx-dx_scale+0.5)
            ymin = int(ymin+dy-dy_scale+0.5)
            xmax = int(xmax+dx+dx_scale+0.5)
            ymax = int(ymax+dy+dy_scale+0.5)

            return xmin,ymin,xmax,ymax,trackLost,dx,dy,good_new2
        else:
            xmin = int(xmin+dx)
            ymin = int(ymin+dy)
            xmax = int(xmax+dx)
            ymax = int(ymax+dy)
            trackLost = 0
            return xmin,ymin,xmax,ymax,trackLost,dx,dy,good_new2


    else:

        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        trackLost = 1
        return xmin,ymin,xmax,ymax,trackLost,dss,dss1,dsw2

def lucasKanadeTrackerMedianScaleStatic2PlusOptimized2DeployPointsMemoria(roiFrame1,roiFrame2,xmin,ymin,xmax,ymax):
    # INPUT:
    #       ROI1 in RGB
    #       ROI2 in RGB
    # OUTPUT:
    #       DisplacementX
    #       DisplacementY

    
    #frame1 = cv2.imread(directoryImages+'/'+listImages[x-1])
    #roiFrame1 = frame1[ymin:ymax,xmin:xmax]
    
    
    feature_params = dict( maxCorners = 600,qualityLevel = 0.1,minDistance = 2,blockSize = 7 )
    #feature_params = dict( maxCorners = 1000,qualityLevel = 0.2,minDistance = 4,blockSize = 7 )

    lk_params = dict( winSize  = (15,15),maxLevel = 4,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    

    trackLost = 0
    good_new = []
    good_old = []
    dss = 0
    dss1 = 0
    dsw2 = 0
  
    pss = 0
    p0 = 0

    if np.shape(roiFrame1)[0]==0 or np.shape(roiFrame1)[1]==0:

        trackLost = 1
        return xmin,ymin,xmax,ymax,trackLost,dss,dss1,dsw2

    old_gray = cv2.cvtColor(roiFrame1, cv2.COLOR_RGB2GRAY)
    equ = cv2.equalizeHist(old_gray)
    
      
    p0 = cv2.goodFeaturesToTrack(equ, mask = None, **feature_params)

    if not p0 is None:
    #if np.shape(p0)[0]!=0:

        frame_gray = cv2.cvtColor(roiFrame2, cv2.COLOR_RGB2GRAY)
        equ2 = cv2.equalizeHist(frame_gray)
        #p0 = p0[:40]
        
        
        p1, st, err = cv2.calcOpticalFlowPyrLK(equ, equ2, p0, None, **lk_params)

        good_new = p1[st==1]
        good_old = p0[st==1]
        err = err[[st==1]].flatten()
        indx = np.argsort(err)
        half_indx = indx[:len(indx) // 2]
        good_old = (p0[[st==1]])[half_indx]
        good_new = (p1[[st==1]])[half_indx]



        ll = good_new-good_old
        thresHOLD = 0.8
        idxX = np.where(np.absolute(ll[:,0])>thresHOLD)[0]
        idxY = np.where(np.absolute(ll[:,1])>thresHOLD)[0]

        idxX2 = np.where(np.absolute(ll[:,0])<thresHOLD)[0]
        idxY2 = np.where(np.absolute(ll[:,1])<thresHOLD)[0]

        
        numPointsX = np.shape(idxX)[0]
        numPointsY = np.shape(idxY)[0]
        
        if ( numPointsX != 0 ) and (numPointsY != 0):
            
            dx = np.median(ll[idxX,0])
            dy = np.median(ll[idxY,1])
            idxOFgodd = np.union1d(idxX,idxY)
            idxOFgodd2 = np.union1d(idxX2,idxY2)

        elif ( numPointsX == 0 ) and (numPointsY != 0):
            dx = 0.0
            dy = np.median(ll[idxY,1])
            idxOFgodd = idxY
        elif ( numPointsX != 0 ) and (numPointsY == 0):
            dx = np.median(ll[idxX,0])
            dy = 0.0
            idxOFgodd = idxX
        else:
            #dx = 0.0
            #dy = 0.0
            dx = np.median(ll[:,0])
            dy = np.median(ll[:,1])
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            #trackLost = 1
            return xmin,ymin,xmax,ymax,trackLost,dx,dy,dsw2
        
        roiFrame1 = cv2.cvtColor(roiFrame1, cv2.COLOR_BGR2RGB)
        numPois = np.shape(idxOFgodd)[0]
        good_new2 = good_new[idxOFgodd, :]
        good_old2 = good_old[idxOFgodd, :]

        bad_old = good_old[idxOFgodd2,:]

        numPoisBAD = np.shape(bad_old)[0]

        for item in range(0,numPois):

            x = good_old2[item,0]
            y = good_old2[item,1]
            cv2.circle(roiFrame1, (x,y), 3, (0,255,0), -1)

        for item in range(0,numPoisBAD):

            x = bad_old[item,0]
            y = bad_old[item,1]
            cv2.circle(roiFrame1, (x,y), 3, (0,0,255), -1)

        #plt.imshow(roiFrame1)
        #plt.show()
        cv2.imwrite('/home/marc/Dropbox/tfmDeepLearning/semana8/imagesMemoria/data/reejctMore.png',roiFrame1)
        
        '''
        fig = plt.figure()
        #a=fig.add_subplot(1,2,1)

        plt.plot(ll[idxX2,0],'ro',markersize=20.0)
        plt.plot(ll[idxX,0],'go',markersize=20.0)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.title('Displacement X',fontsize = 40)
        plt.ylabel('Displacement',fontsize = 30)
        plt.xlabel('Points',fontsize = 30)
        plt.show()
        #plt.axis([0, durada, 0, 1.5])
        '''

        #a=fig.add_subplot(1,2,2)
        plt.plot(ll[idxY2,1],'ro',markersize=20.0)
        plt.plot(ll[idxY,1],'go',markersize=20.0)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.title('Displacement Y',fontsize = 40)
        plt.ylabel('Displacement',fontsize = 30)
        plt.xlabel('Points',fontsize = 30)
        #plt.axis([0, durada, 0, 1.5])
        plt.show()



        i, j = np.triu_indices(len(good_old2), k=1)
        pdiff0 = good_old2[i] - good_old2[j]
        pdiff1 = good_new2[i] - good_new2[j]
        p0_dist = np.sum(pdiff0 ** 2, axis=1)
        p1_dist = np.sum(pdiff1 ** 2, axis=1)

        if (np.shape(p0_dist)[0]!=0 ) or (np.shape(p1_dist)[0]!=0):
            ds = np.median(np.sqrt((p1_dist / (p0_dist + 2**-23))))
            ds_factor = 0.95
            ds = (1.0 - ds_factor) + ds_factor * ds
            dx_scale = (ds-1.0) * 0.5 * (xmax - xmin + 1)
            dy_scale = (ds-1.0) * 0.5 * (ymax - ymin + 1)
            trackLost = 0
            xmin = int(xmin+dx-dx_scale+0.5)
            ymin = int(ymin+dy-dy_scale+0.5)
            xmax = int(xmax+dx+dx_scale+0.5)
            ymax = int(ymax+dy+dy_scale+0.5)

            return xmin,ymin,xmax,ymax,trackLost,dx,dy,good_new2
        else:
            xmin = int(xmin+dx)
            ymin = int(ymin+dy)
            xmax = int(xmax+dx)
            ymax = int(ymax+dy)
            trackLost = 0
            return xmin,ymin,xmax,ymax,trackLost,dx,dy,good_new2


    else:

        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        trackLost = 1
        return xmin,ymin,xmax,ymax,trackLost,dss,dss1,dsw2
 
    
        

def lucasKanadeTrackerMedianScaleStatic3Test(roiFrame1,roiFrame2,xmin,ymin,xmax,ymax,pas):
    # INPUT:
    #       ROI1 in RGB
    #       ROI2 in RGB
    # OUTPUT:
    #       DisplacementX
    #       DisplacementY

    directoSave = '/home/marc/Dropbox/tfmDeepLearning/semana6/mejoraLK/dat3' 
    
    
    feature_params = dict( maxCorners = 1000,qualityLevel = 0.1,minDistance = 2,blockSize = 7 )
    #feature_params = dict( maxCorners = 1000,qualityLevel = 0.2,minDistance = 4,blockSize = 7 )

    lk_params = dict( winSize  = (15,15),maxLevel = 4,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    #dircX = 0
    #dircY = 0
    trackLost = 0
    good_new = []
    good_old = []
    dss = 0
    dss1 = 0
    dss2 = 0

    if np.shape(roiFrame1)[0]==0 or np.shape(roiFrame1)[1]==0:
        trackLost = 1
        return xmin,ymin,xmax,ymax,trackLost,dss,dss1,dss2

    old_gray = cv2.cvtColor(roiFrame1, cv2.COLOR_RGB2GRAY)
    
    equ = cv2.equalizeHist(old_gray)
    
    

    p0 = cv2.goodFeaturesToTrack(equ, mask = None, **feature_params)
    
    if p0 is None:

        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        trackLost = 1
        return xmin,ymin,xmax,ymax,trackLost,dss,dss1,dss2

    else:
        if np.shape(p0)[0]!=0:


            frame_gray = cv2.cvtColor(roiFrame2, cv2.COLOR_RGB2GRAY)
            equ2 = cv2.equalizeHist(frame_gray)

            p1, st, err1 = cv2.calcOpticalFlowPyrLK(equ, equ2, p0, None, **lk_params)
            indx = np.where(st == 1)[0]
            p0 = p0[indx, :]
            p1 = p1[indx, :]
            p0r, st, err = cv2.calcOpticalFlowPyrLK(equ2, equ, p1, None, **lk_params)
            fb_dist = np.abs(p0 - p0r).max(axis=1)
            

            fb_error = (p0 - p0r)**2
            sizeFe = np.shape(fb_error)          
            fb_error = fb_error.reshape((sizeFe[0], 2))
            fb_error = np.sqrt(np.sum(fb_error,axis=1))

            idx_fb_error = np.where(fb_error>1)[0]


            fb_error = np.sort(fb_error)



            goodX = fb_dist[:,0] < 1
            goodY = fb_dist[:,1] < 1

            goodElements =np.logical_and(np.array(goodX) , np.array(goodY))
            goodIx = np.where(goodElements==True)
            good = goodElements[goodIx]

            err2 = err[goodIx[0]]

            good_newPrevious = p1[goodIx[0]]
            good_oldPrevious = p0[goodIx[0]]       
           
            
            indxError = np.argsort(err2,axis=0) 
            half_indx = indxError[:len(indxError) // 2]
            
            durada = np.shape(err2)[0]
            
            good_new = good_newPrevious[half_indx,:]
            good_old = good_oldPrevious[half_indx,:]

            sizeN = np.shape(good_old)
            
            good_new = good_new.reshape((sizeN[0], 2))
            good_old = good_old.reshape((sizeN[0], 2))

            
            
            #plt.plot(fb_error,'ro',markersize=10.0)
            #plt.show()

            #plt.plot(fb_error,'ro',markersize=10.0)
            #plt.show()
            sizeFBerro = np.shape(idx_fb_error)[0]
            #print('points',np.shape(p0),'errorFB',sizeFBerro)
            color = np.random.randint(0,255,(sizeFBerro,3))

            for iNew,new in enumerate(p0[idx_fb_error, :]):
                a,b = new.ravel()
                cv2.circle(roiFrame1,(a,b),3,color[iNew].tolist(),-1)

            for iold,old in enumerate(p0r[idx_fb_error, :]):
                a,b = old.ravel()
                cv2.circle(roiFrame1,(a,b),3,color[iold].tolist(),-1)


            for iold,old in enumerate(p1[idx_fb_error, :]):
                a,b = old.ravel()
                cv2.circle(roiFrame2,(a,b),3,color[iold].tolist(),-1)
            
                      
     

            fig = plt.figure()
            a=fig.add_subplot(1,2,1)
            imgplot = plt.imshow(roiFrame1)
            b=fig.add_subplot(1,2,2)
            imgplot = plt.imshow(roiFrame2)
            a.title.set_text('Frame T')
            b.title.set_text('Frame T+1')
            #plt.show()




            # check errors 
            '''
            newErrro = err1[np.argsort(err1,axis=0)]
            sizeNPrint = np.shape(newErrro)
            newErrro = newErrro.reshape((sizeNPrint[0], 1))
            
            errsa = err2[half_indx]
            errsa = errsa.reshape(sizeN[0],1)
            print(np.shape(errsa),np.shape(newErrro))


            plt.plot(newErrro,'bo',markersize=10.0)
            plt.plot(errsa,'ro',markersize=10.0)
            
            
            plt.xlabel('Number of element',fontsize=40)
            plt.ylabel('Error',fontsize=40)
            plt.show()
            
            '''
            '''
            plt.plot(err,'ro',markersize=10.0)
            #plt.plot(err[half_indx],'bo',markersize=10.0)
            plt.axis([0, durada, 0, 100])
            plt.xlabel('Number of element',fontsize=40)
            plt.ylabel('Error',fontsize=40)
            plt.show()
            '''
            
            '''
            plt.plot(err2,'ro',markersize=10.0)
            #plt.plot(err[half_indx],'bo',markersize=10.0)
            plt.axis([0, durada, 0, 100])
            plt.xlabel('Number of element',fontsize=40)
            plt.ylabel('Error',fontsize=40)
            plt.show()
            '''
            '''
            color = np.random.randint(0,255,(100,3))
            for iNew,new in enumerate(good_new):
                a,b = new.ravel()
                cv2.circle(roiFrame2,(a,b),5,color[iNew].tolist(),-1)

            for iold,old in enumerate(good_old):
                a,b = old.ravel()
                cv2.circle(roiFrame1,(a,b),5,color[iold].tolist(),-1)
        

            fig = plt.figure()
            a=fig.add_subplot(1,2,1)
            imgplot = plt.imshow(roiFrame1)
            a=fig.add_subplot(1,2,2)
            imgplot = plt.imshow(roiFrame2)
            #plt.show()
            idas += 1
            n = str(idas)
            n2 = n.zfill(3)
            plt.savefig(directoSave+'/'+n2+'.jpg')
            '''


            # Calculo puntos estaticos
            #ll = [good_new[:, 0] - good_old[:, 0],good_new[:, 1] - good_old[:, 1]]
            ll = good_new-good_old


            #dx = np.median(good_new[:,0]-good_old[:,0])
            #dy = np.median(good_new[:,1]-good_old[:,1])


            thresHOLD = 1.0

            idxX = np.where(np.absolute(ll[:,0])>thresHOLD)
            idxY = np.where(np.absolute(ll[:,1])>thresHOLD)
            #print(idxX[0])

            numPointsX = np.shape(idxX)[1]
            numPointsY = np.shape(idxY)[1]

            

            if ( numPointsX != 0 ) and (numPointsY != 0):
                
                dx = np.median(ll[idxX[0],0])
                dy = np.median(ll[idxY[0],1])
                
                idxOFgodd = np.union1d(idxX[0],idxY[0])

            elif ( numPointsX == 0 ) and (numPointsY != 0): 

                dx = 0.0
                dy = np.median(ll[idxY[0],1])
                idxOFgodd = idxY[0]

            elif ( numPointsX != 0 ) and (numPointsY == 0): 

                dx = np.median(ll[idxX[0],0])
                dy = 0.0
                idxOFgodd = idxX[0]
            else:

                dx = 0.0
                dy = 0.0

                trackLost = 1
                return xmin,ymin,xmax,ymax,trackLost,dss,dss1
            
           
            intersa = np.intersect1d(idxOFgodd,idx_fb_error)

            sizeFBerro = np.shape(idx_fb_error)[0]
            print('points',np.shape(p0)[0],'halfPoints',np.shape(half_indx)[0],'errorFB',sizeFBerro,'inter',np.shape(intersa),'nonStatic',np.shape(idxOFgodd)[0])

            
            '''
            fig = plt.figure()
            a=fig.add_subplot(2,2,1)
            plt.plot(ll[:,0],'bo',markersize=10.0)
            a=fig.add_subplot(2,2,2)
            plt.plot(ll[idxOFgodd,0],'ro',markersize=10.0)
            a=fig.add_subplot(2,2,3)
            plt.plot(ll[:,1],'bo',markersize=10.0)
            a=fig.add_subplot(2,2,4)
            plt.plot(ll[idxOFgodd,1],'ro',markersize=10.0)
            plt.show()
            '''


            '''
            
            for iNew,new in enumerate(good_new):
                a,b = new.ravel()
                cv2.circle(roiFrame2,(a,b),2,(255,0,0),-1)

            for iold,old in enumerate(good_new[idxOFgodd,:]):
                a,b = old.ravel()
                cv2.circle(roiFrame2,(a,b),2,(0,255,0),-1)

            #imgplot = plt.imshow(roiFrame2)
            n = str(pas)
            n2 = n.zfill(3)
            #plt.savefig(directoSave+'/'+n2+'.jpg')
            '''
            # fin analizar

            # matching
            
            sizeGoodPoints = np.shape(idxOFgodd)[0]

            #print('goods',sizeGoodPoints)
            


            # draw points correspondences
            '''
            color = np.random.randint(0,255,(sizeGoodPoints,3))
            for iNew,new in enumerate(good_new[idxOFgodd, :]):
                a,b = new.ravel()
                cv2.circle(roiFrame2,(a,b),3,color[iNew].tolist(),-1)

            for iold,old in enumerate(good_old[idxOFgodd, :]):
                a,b = old.ravel()
                cv2.circle(roiFrame1,(a,b),3,color[iold].tolist(),-1)
            
            fig = plt.figure()

            a=fig.add_subplot(1,2,1)
            imgplot = plt.imshow(roiFrame1)
            a=fig.add_subplot(1,2,2)
            imgplot = plt.imshow(roiFrame2)

            line = matplotlib.lines.Line2D((good_old[idxOFgodd, 0],good_new[idxOFgodd, 0]),(good_old[idxOFgodd, 1],good_new[idxOFgodd, 1]),
                               transform=fig.transFigure)
            fig.lines = line        


            #n = str(pas)
            #n2 = n.zfill(3)
            #plt.savefig(directoSave+'/'+n2+'.jpg')
            
            plt.show()
            '''

            # draw points with lines correspondences
            # http://stackoverflow.com/questions/17543359/drawing-lines-between-two-plots-in-matplotlib
            '''
            good_new2 = good_new[idxOFgodd, :]
            good_old2 = good_old[idxOFgodd, :]

            color = np.random.randint(0,255,(sizeGoodPoints,3))

            fig = plt.figure()
            transFigure = fig.transFigure.inverted()

            for iNew,new in enumerate(good_new2):
                
                cv2.circle(roiFrame2,(good_new2[iNew,0],good_new2[iNew,1]),3,color[iNew].tolist(),-1)
                cv2.circle(roiFrame1,(good_old2[iNew,0],good_old2[iNew,1]),3,color[iNew].tolist(),-1)
                
                #line = matplotlib.lines.Line2D((good_old2[iNew, 0],good_new2[iNew, 0]),(good_old2[iNew, 1],good_new2[iNew, 1]),markeredgewidth=60,markeredgecolor=color[iNew])
                #fig.lines = line, 
            

            a = fig.add_subplot(1,2,1)
            imgplot = plt.imshow(roiFrame1)
            b = fig.add_subplot(1,2,2)
            imgplot = plt.imshow(roiFrame2)


            for iNew,new in enumerate(good_new2):

                coord1 = transFigure.transform(a.transData.transform([good_old2[iNew, 0],good_old2[iNew, 1]]))
                coord2 = transFigure.transform(b.transData.transform([good_new2[iNew, 0],good_new2[iNew, 1]]))
                print(iNew)

                line = matplotlib.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),
                               transform=fig.transFigure)
                #line = matplotlib.lines.Line2D((int(good_old2[iNew, 0]),int(good_new2[iNew, 0])),(int(good_old2[iNew, 1]),int(good_new2[iNew, 1])),markeredgewidth=60,markeredgecolor=color[iNew])
                fig.lines = line, 
            

            #line = matplotlib.lines.Line2D((good_old[idxOFgodd, 0],good_new[idxOFgodd, 0]),(good_old[idxOFgodd, 1],good_new[idxOFgodd, 1]))
            #fig.lines = line        


            #n = str(pas)
            #n2 = n.zfill(3)
            #plt.savefig(directoSave+'/'+n2+'.jpg')
            
            plt.show()
            '''
            







            good_new2 = good_new[idxOFgodd, :]
            good_old2 = good_old[idxOFgodd, :]
            


            
            i, j = np.triu_indices(len(good_old2), k=1)
            #print('numP',np.shape(good_new))

            pdiff0 = good_old2[i] - good_old2[j]
            pdiff1 = good_new2[i] - good_new2[j]
            
            p0_dist = np.sum(pdiff0 ** 2, axis=1)
            p1_dist = np.sum(pdiff1 ** 2, axis=1)
            ds = np.median(np.sqrt((p1_dist / (p0_dist + 2**-23))))
            
            #print(dx,dy,ds)
            #print('---------')
            if np.isnan(dx) or np.isnan(dy) or np.isnan(ds) :
                #print('m1',dx,dy,ds,np.isnan(dx))
                print('nan')
                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)
                trackLost = 1
                return xmin,ymin,xmax,ymax,trackLost,dss,dss1,dss2

            else:
                '''
                ds_factor = 1.5
                ds = (1.0 - ds_factor) + ds_factor * ds;
                dx_scale = (ds - 1.0) * 0.5 * (xmax - xmin + 1)
                dy_scale = (ds - 1.0) * 0.5 * (ymax - ymin + 1)
                '''
                dx_scale = (ds-1.0) * 0.5 * (xmax - xmin + 1)
                dy_scale = (ds-1.0) * 0.5 * (ymax - ymin + 1)
                print('ok')
                
                xmin = int(xmin+dx-dx_scale+0.5)
                ymin = int(ymin+dy-dy_scale+0.5)
                xmax = int(xmax+dx+dx_scale+0.5)
                ymax = int(ymax+dy+dy_scale+0.5)
        else:
            trackLost = 1 
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            return xmin,ymin,xmax,ymax,trackLost,dss,dss1,dss2
    #print(endA-startA)
    return xmin,ymin,xmax,ymax,trackLost,dx,dy,ds
    #return xmin,ymin,xmax,ymax,trackLost
def lucasKanadeTrackerMedianScaleStatic3Deploy(roiFrame1,roiFrame2,xmin,ymin,xmax,ymax,pas):
    # INPUT:
    #       ROI1 in RGB
    #       ROI2 in RGB
    # OUTPUT:
    #       DisplacementX
    #       DisplacementY

    #directoSave = '/home/marc/Dropbox/tfmDeepLearning/semana6/mejoraLK/dat3' 
    
    
    feature_params = dict( maxCorners = 1000,qualityLevel = 0.1,minDistance = 2,blockSize = 7 )
    #feature_params = dict( maxCorners = 1000,qualityLevel = 0.2,minDistance = 4,blockSize = 7 )

    lk_params = dict( winSize  = (15,15),maxLevel = 4,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    #dircX = 0
    #dircY = 0
    trackLost = 0
    good_new = []
    good_old = []
    dss = 0
    dss1 = 0

    if np.shape(roiFrame1)[0]==0 or np.shape(roiFrame1)[1]==0:
        trackLost = 1
        
        return xmin,ymin,xmax,ymax,trackLost,dss,dss1

    old_gray = cv2.cvtColor(roiFrame1, cv2.COLOR_RGB2GRAY)
    
    equ = cv2.equalizeHist(old_gray)
    
    

    p0 = cv2.goodFeaturesToTrack(equ, mask = None, **feature_params)
    
    if not p0 is None:

        frame_gray = cv2.cvtColor(roiFrame2, cv2.COLOR_RGB2GRAY)
        equ2 = cv2.equalizeHist(frame_gray)

        p1, st, err1 = cv2.calcOpticalFlowPyrLK(equ, equ2, p0, None, **lk_params)
        indx = np.where(st == 1)[0]
        p0 = p0[indx, :]
        p1 = p1[indx, :]
        p0r, st, err = cv2.calcOpticalFlowPyrLK(equ2, equ, p1, None, **lk_params)
        fb_dist = np.abs(p0 - p0r).max(axis=1)
        good2 = fb_dist < 1.0
        good =np.logical_and(np.array(good2[:,0]) , np.array(good2[:,1]))
        
        if err is None:
            trackLost = 1
        
            return xmin,ymin,xmax,ymax,trackLost,dss,dss1


        err = err[good].flatten()
        indx = np.argsort(err)
        half_indx = indx[:len(indx) // 2]
        good_old = (p0[good])[half_indx]
        good_new = (p1[good])[half_indx]

        ll = good_new-good_old
        
        ll = ll.reshape(np.shape(ll)[0],2)

        
        thresHOLD = 0.8

        idxX = np.where(np.absolute(ll[:,0])>thresHOLD)
        idxY = np.where(np.absolute(ll[:,1])>thresHOLD)
        #print(idxX[0])

        numPointsX = np.shape(idxX)[1]
        numPointsY = np.shape(idxY)[1]

        

        if ( numPointsX != 0 ) and (numPointsY != 0):
            
            dx = np.median(ll[idxX[0],0])
            dy = np.median(ll[idxY[0],1])
            
            idxOFgodd = np.union1d(idxX[0],idxY[0])

        elif ( numPointsX == 0 ) and (numPointsY != 0): 

            dx = 0.0
            dy = np.median(ll[idxY[0],1])
            idxOFgodd = idxY[0]

        elif ( numPointsX != 0 ) and (numPointsY == 0): 

            dx = np.median(ll[idxX[0],0])
            dy = 0.0
            idxOFgodd = idxX[0]
            
        else:

            #dx = 0.0
            #dy = 0.0   

            dx = np.median(ll[:,0])
            dy = np.median(ll[:,1])
            
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            trackLost = 0
            #print('bu')

            return xmin,ymin,xmax,ymax,trackLost,dx,dy
        

        good_new2 = good_new[idxOFgodd, :]
        good_old2 = good_old[idxOFgodd, :]
                    
        i, j = np.triu_indices(len(good_old2), k=1)
        
        pdiff0 = good_old2[i] - good_old2[j]
        pdiff1 = good_new2[i] - good_new2[j]
        
        p0_dist = np.sum(pdiff0 ** 2, axis=1)
        p1_dist = np.sum(pdiff1 ** 2, axis=1)
        
        if (np.shape(p0_dist)[0]!=0 ) or (np.shape(p1_dist)[0]!=0):
            ds = np.median(np.sqrt((p1_dist / (p0_dist + 2**-23))))
            ds_factor = 0.95
            ds = (1.0 - ds_factor) + ds_factor * ds
            dx_scale = (ds-1.0) * 0.5 * (xmax - xmin + 1)
            dy_scale = (ds-1.0) * 0.5 * (ymax - ymin + 1)
            trackLost = 0
            xmin = int(xmin+dx-dx_scale+0.5)
            ymin = int(ymin+dy-dy_scale+0.5)
            xmax = int(xmax+dx+dx_scale+0.5)
            ymax = int(ymax+dy+dy_scale+0.5)

            return xmin,ymin,xmax,ymax,trackLost,dx,dy

        else:

            xmin = int(xmin+dx)
            ymin = int(ymin+dy)
            xmax = int(xmax+dx)
            ymax = int(ymax+dy)
            trackLost = 0
            return xmin,ymin,xmax,ymax,trackLost,dx,dy


    else:
        
        trackLost = 1 
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        return xmin,ymin,xmax,ymax,trackLost,dss,dss1
    #print(endA-startA)
    return xmin,ymin,xmax,ymax,trackLost,dx,dy
    #return xmin,ymin,xmax,ymax,trackLost
