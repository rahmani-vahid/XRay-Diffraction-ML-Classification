
from sklearn.cluster import MiniBatchKMeans
import cv2 as cv
import globals
import numpy as np
import os
import pickle

#-------------------------------------------------------------------

def SIFT():
    
    SIFT = cv.xfeatures2d.SIFT_create(nfeatures = 0,
                                      nOctaveLayers = 3,
                                      contrastThreshold = 0.04,
                                      edgeThreshold = 10,
                                      sigma = 1.6)

    return SIFT

#-------------------------------------------------------------------
def FAST():
    fast = cv.FastFeatureDetector_create()
    return fast    

#-------------------------------------------------------------------
def SURF():
    SURF = cv.xfeatures2d.SURF_create(hessianThreshold = 500,
                                      nOctaves = 4,
                                      nOctaveLayers = 3,
                                      extended = True,
                                      upright = False)

    return SURF

#-------------------------------------------------------------------
def KAZE():
    KAZE = cv.KAZE_create(extended = False,
                          upright = False,
                          threshold = 0.001,
                          nOctaves = 4,
                          nOctaveLayers = 4,
                          diffusivity = cv.KAZE_DIFF_PM_G2)

    return KAZE

#-------------------------------------------------------------------
def BRIEF():
    BRIEF = cv.xfeatures2d.BriefDescriptorExtractor_create(bytes = 16,
                                                           use_orientation = False)

    return BRIEF

#-------------------------------------------------------------------
def ORB():
    ORB = cv.ORB_create(nfeatures = 1000,
                        scaleFactor = 1.2,
                        nlevels = 2,
                        edgeThreshold = 2,
                        firstLevel = 0,
                        WTA_K = 2,
                        scoreType = 0,
                        patchSize = 2,
                        fastThreshold = 20)

    return ORB

#-------------------------------------------------------------------
def BRISK():
    BRISK = cv.BRISK_create(thresh = 30,
                            octaves = 3,
                            patternScale = 1.0)

    return BRISK

#-------------------------------------------------------------------
def AKAZE():
    AKAZE = cv.AKAZE_create(descriptor_type = cv.AKAZE_DESCRIPTOR_MLDB,
                            descriptor_size = 0,
                            descriptor_channels = 3,
                            threshold = 0.001,
                            nOctaves = 4,
                            nOctaveLayers = 4,
                            diffusivity = cv.KAZE_DIFF_PM_G2)

    return AKAZE

#-------------------------------------------------------------------
def FREAK():
    FREAK = cv.xfeatures2d.FREAK_create(orientationNormalized = True,
                                        scaleNormalized = True,
                                        patternScale = 22.0,
                                        nOctaves = 4)

    return FREAK
#-------------------------------------------------------------------
def printDetector():
    print('Detector selected: %s\n' % globals.detector, file = globals.file)

#-------------------------------------------------------------------
def printDescriptor():
    print('Descriptor selected: %s\n' % globals.descriptor, file = globals.file)

#-------------------------------------------------------------------

def features(image):
    keypoints = globals.detector.detect(image, None)
    keypoints, descriptors = globals.descriptor.compute(image, keypoints)
    
    # index = []

    # for point in keypoints:
    #     temp = (point.pt,
    #            point.size,
    #             point.angle,
    #             point.response,
    #             point.octave, 
    #             point.class_id)
    
    #     index.append(temp)
    # # File name
    # filename = "Features/keypoints.pickle"
    # filename2 = "Features/descriptors.pickle"
    # pickle_out = open(filename, "wb")
    # pickle.dump(index, pickle_out)
    # pickle_out.close()    
    
    # pickle_out2 = open(filename2, "wb")
    # pickle.dump(descriptors, pickle_out2)
    # pickle_out2.close()    
    # if os.path.exists(filename):
    #     os.remove(filename)
    # file = open(filename, "wb")
    # file.write(pickle.dumps(index))
    # file.close()    
    #print(len(keypoints))
    return keypoints, descriptors
#-------------------------------------------------------------------

def extractFeatures(crop_list, featureFunc):
    descriptors = {}
    for i in range(0, len(crop_list)):
        kp, desc = featureFunc(crop_list[i])
        # print('----')
        # print(len(desc))
        # print(len(kp))
        if len(kp)==0:
            desc= np.random.randint(0, 5, size=(1000, 32),dtype=np.uint8)
        if type(desc) != type(None):
            descriptors[i] = desc
    return descriptors

#-------------------------------------------------------------------
def groupAllFeatures(descriptors_hash):
    all_descriptors = []
    
    for subject, crops in descriptors_hash.items():
        for idx, desc in crops.items():
            all_descriptors.extend(desc)
            
    return all_descriptors

#-------------------------------------------------------------------
def trainKMeans(descriptors):
    initial_size = 1 * globals.K
    batch_size = int(len(descriptors) / 3) 
    kmeans = MiniBatchKMeans(n_clusters = globals.K,
                             batch_size = batch_size,
                             init_size = initial_size,
                             verbose = 0).fit(descriptors)
    
    return kmeans

#-------------------------------------------------------------------
def generateHistograms(descriptors, kmeans_centers):
    histograms = []
    
    for i, desc in descriptors.items(): 
        preds = kmeans_centers.predict(desc)
        hist, bin_edges = np.histogram(a = preds,
                                       bins = range(0, globals.K))
        histograms.append(hist)
    
    return histograms