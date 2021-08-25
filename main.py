import sys
import warnings
import itertools
import threading
import save_figures
import save_models
import sys
import time
import numpy as np
import datasets
import features
import globals
from timer import Timer
import argparse
import cv2 as cv
import svm
import os
import outputs
import parameters
import pickle
import multilayer
import nb
import rf




if not sys.warnoptions:
    warnings.simplefilter("ignore")

proc="p1"
done = False
def animate():
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if done:
            break
        sys.stdout.write('\rprocessing --> ' + proc+' '+c + ':')
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\rDone!     ')

#-------------------------------------------------------------------
globals.initialize()
#-------------------------------------------------------------------

message = '''main.py [-h]

             --MLmodel      {SVM,MLP,RF,NB}
             --detector     {SIFT, SURF, KAZE, ORB, BRISK, AKAZE, BRIEF}
             --descriptor   {SIFT, SURF, KAZE, BRIEF, ORB, BRISK, AKAZE, FREAK, BREAF}
             --dataset      { DiffraNet, AGIPD}'''

#-------------------------------------------------------------------

parser = argparse.ArgumentParser(usage = message)

#-------------------------------------------------------------------
parser.add_argument('--MLmodel',
                    action = 'store',
                    choices = ['SVM','MLP','RF','NB'],
                    required = True,
                    metavar = '',
                    dest = 'MLmodel',
                    help = 'select the ML model to be used in this experiment')
#---------------------------------------------------------------------                    
parser.add_argument('--detector',
                    action = 'store',
                    choices = ['SIFT', 'SURF', 'KAZE', 'ORB', 'BRISK', 'AKAZE','BRIEF'],
                    required = True,
                    metavar = '',
                    dest = 'detector',
                    help = 'select the detector to be used in this experiment')

#-------------------------------------------------------------------
parser.add_argument('--descriptor',
                    action = 'store',
                    choices = ['SIFT', 'SURF', 'KAZE', 'BRIEF', 'ORB', 'BRISK', 'AKAZE', 'FREAK','BRIEF'],
                    required = True,
                    metavar = '',
                    dest = 'descriptor',
                    help = 'select the descriptor to be used in this experiment')

#-------------------------------------------------------------------
parser.add_argument('--dataset',
                    action = 'store',
                    choices = ['DiffraNet','AGIPD' ],

                    required = True,
                    metavar = '',
                    dest = 'dataset',
                    help = 'select the visual dataset to be used in this experiment')

#------------------------------------------------------------------- 
arguments = parser.parse_args()
filename = 'Outputs/Datasets/%s/%s-outputs.txt' % (arguments.dataset, arguments.descriptor)
outputs.openFile(filename = filename)
#-------------------------------------------------------------------

if arguments.detector == 'SIFT':
    globals.detector = features.SIFT()

elif arguments.detector == 'SURF':
    globals.detector = features.SURF()

elif arguments.detector == 'KAZE':
    globals.detector = features.SIFT()

elif arguments.detector == 'ORB':
    globals.detector = features.ORB()

elif arguments.detector == 'BRISK':
    globals.detector = features.BRISK()

elif arguments.detector == 'AKAZE':
    globals.detector = features.AKAZE()

features.printDetector()

#-------------------------------------------------------------------

if arguments.descriptor == 'SIFT':
    globals.descriptor = features.SIFT()

elif arguments.descriptor == 'SURF':
    globals.descriptor = features.SURF()

elif arguments.descriptor == 'KAZE':
    globals.descriptor = features.SIFT()

elif arguments.descriptor == 'BRIEF':
    globals.descriptor = features.BRIEF()

elif arguments.descriptor == 'ORB':
    globals.descriptor = features.ORB()

elif arguments.descriptor == 'BRISK':
    globals.descriptor = features.BRISK()

elif arguments.descriptor == 'AKAZE':
    globals.descriptor = features.AKAZE()

elif arguments.descriptor == 'FREAK':
    globals.descriptor = features.FREAK()

features.printDescriptor()
#-------------------------------------------------------------------

if arguments.dataset == 'DiffraNet':
    globals.data_train_images, globals.data_test_images = datasets.DiffraNet()

elif arguments.dataset == 'AGIPD':
    globals.data_train_images, globals.data_test_images = datasets.JAFFE()

#-------------------------------------------------------------------
datasets.printTrainingPath()
datasets.printTestPath()
datasets.printNumberOfClasses()
globals.K = globals.num_classes * 5
#-------------------------------------------------------------------
print('\n\n---------------------Extract Features-----------------------------\n')

all = {}
print('Extract Features\n', file = globals.file)

proc="Features Extraction"
t = threading.Thread(target=animate)
t.start()

object = sorted(os.listdir(globals.data_train_images))    
i, total = 0, len(object)

with Timer() as timer:
    for subject in object:
        i += 1
        print('Processing the subdirectory named:', subject, '\t[', i , '/', total, ']', file = globals.file)

        # Read in cropped data
        crop_names = os.listdir(os.path.join(globals.data_train_images, subject))
        crop_names = list(map(lambda x: os.path.join(globals.data_train_images, subject, x), crop_names)) 
        crops = [cv.imread(x , cv.IMREAD_GRAYSCALE) for x in crop_names]

        # Get Features
        desc = features.extractFeatures(crops, features.features)
        print('Number of extracted features for class ' ,i,' :' , len(desc))
        # np.savetxt("foo.txt", desc, delimiter=",")
        all[subject] = desc
        st=str(i)
        filename2 = "Features/AllDesc_Class"+st+".pickle"
        # if os.path.exists(filename):
        #     os.remove(filename)
        pickle_out = open(filename2, "wb")
        pickle.dump(all[subject], pickle_out)
        pickle_out.close()    
    
        print('Extracted', arguments.descriptor, '\n', file = globals.file)

time.sleep(10)
done = True

print('Time:', timer, '\n', file = globals.file)


print('\n\n---------------------Create BOVFs -----------------------------\n')



matrix = features.groupAllFeatures(all)
done-False
proc='Create BOVFs'
t = threading.Thread(target=animate)
t.start()
print('\nTotal number of features (all classes): ',len(matrix))
kmeans = None
print('Create Bag of Visual Features\n', file = globals.file)

# Train K-Means
print('Training', arguments.descriptor, 'K-Means\n', file = globals.file)

with Timer() as timer:
    kmeans = features.trainKMeans(matrix)
    
time.sleep(10)
done = True
print('Time:', timer, '\n', file = globals.file)
# print('Done!\n')


print('\n\n---------------------Prepare Training Data -----------------------------\n')
#print('Prepare Training Data\n')
done=False
proc='Training Data'

t = threading.Thread(target=animate)
t.start()
# Training Data
globals.train_feature_vec = [[], []]

print(arguments.descriptor, 'Training Data\n', file = globals.file)
object = sorted(os.listdir(globals.data_train_images))

i, total = 0, len(object)

with Timer() as timer:
    for subject in object:
        i += 1
        print('Processing the subdirectory named:', subject, '\t[', i , '/', total, ']\n', file = globals.file)

        # Get Features
        histograms = features.generateHistograms(all[subject], kmeans)
        globals.train_feature_vec[0].extend(histograms)
        globals.train_feature_vec[1].extend([subject] * len(histograms))
    
print('Time:', timer, '\n', file = globals.file)
#print('Done!\n')
time.sleep(10)
done = True
print('Done!\n')


print('\n\n---------------------Prepare Testing Data -----------------------------\n')
#print('\nPrepare Testing Data\n')

done=False
proc='Testing Data'
t = threading.Thread(target=animate)
t.start()
# Testing Data
globals.test_feature_vec = [[], []]

print(arguments.descriptor, 'Testing Data\n', file = globals.file)
object = sorted(os.listdir(globals.data_test_images))

i, total = 0, len(object)

with Timer() as timer:
    for subject in object:
        i += 1
        print('Processing the subdirectory named:', subject, '\t[', i , '/', total, ']\n', file = globals.file)
      
        crop_names = os.listdir(os.path.join(globals.data_test_images, subject))
        crop_names = list(map(lambda x: os.path.join(globals.data_test_images, subject, x), crop_names)) 
        crops = [cv.imread(x , cv.IMREAD_GRAYSCALE) for x in crop_names ]
      
        # Get Features
        desc = features.extractFeatures(crops, features.features)
      
        # Get Histograms
        histograms = features.generateHistograms(desc, kmeans)
        globals.test_feature_vec[0].extend(histograms)
        globals.test_feature_vec[1].extend([subject] * len(histograms))
    
print('Time:', timer, '\n', file = globals.file)
time.sleep(10)
done = True
print('Done!\n')

############################################################################################
if arguments.MLmodel == 'SVM':
    print('\n\n--------------------- Training SVM -----------------------------\n')
    done=False
    proc='Training SVM'
    t = threading.Thread(target=animate)
    t.start()   

# Train SVM Model
    print('Training %s SVM Models\n' % arguments.descriptor, file = globals.file)

    SVM = svm.train(gama = 0.001,
                descriptor_name = arguments.descriptor,
                model_name = 'SVM')

    print('Done!\n')
    time.sleep(10)
    done = True

    print('\n\n--------------------- Testing SVM -----------------------------\n')

    done=False
    proc='Testing SVM'
    t = threading.Thread(target=animate)
    t.start() 
# Test SVM Model
    print('Testing %s SVM Model\n' % arguments.descriptor, file = globals.file)

    SVM_predict = svm.test(model = SVM,
                       descriptor_name = arguments.descriptor,
                       model_name = 'SVM')

    print('Done!\n')
    time.sleep(10)
    done = True
 
    print('\n\n--------------------- Creating Reports -----------------------------\n')

    done=False
    proc='Creating Reports'
    t = threading.Thread(target=animate)
    t.start()   

    print('\nClassification Report\n')
    print('\nClassification Report\n', file = globals.file)
 
 
# SVM Model 
    svm.classificationReport(model = SVM,
                         predict = SVM_predict,
                         descriptor_name = arguments.descriptor,
                         model_name = 'SVM')

    #print('Done!\n')
    print('\nSaving Confusion Matrix\n')

# Save Confusion Matrix figure SVM Model
    save_figures.confusionMatrix(predict = SVM_predict,
                             descriptor_name = arguments.descriptor,
                             dataset_name = arguments.dataset,
                             model_name = 'SVM')

    #print('Done!\n')
    print('\nSaving Bag of Visual Features\n')
#-------------------------------------------------------------------

# Save Bag of Visual Features 
    save_models.saveBoVF(kmeans = kmeans,
                     descriptor_name = arguments.descriptor,
			         dataset_name = arguments.dataset)

    #print('Done!\n')
    print('\nSaving Descriptors\n')
#-------------------------------------------------------------------

# Save Training Descriptors
    save_models.saveDescriptors(feature_vector = globals.train_feature_vec,
                            descriptor_name = arguments.descriptor,
			                dataset_name = arguments.dataset,
                            flag = 'train')
#-------------------------------------------------------------------
# Save Test Descriptors
    save_models.saveDescriptors(feature_vector = globals.test_feature_vec,
                            descriptor_name = arguments.descriptor,
			                dataset_name = arguments.dataset,
                            flag = 'test')
    #print('Done!\n')
    print('\nSaving SVM Models\n')
#-------------------------------------------------------------------
# SVM Model
    save_models.saveSVM(model = SVM,
                    descriptor_name = arguments.descriptor,
			        dataset_name = arguments.dataset,
                    model_name = 'SVM')

    #print('Done!\n')

    time.sleep(10)
    done = True
    
    outputs.closeFile()
    print('\n\n*************** Finished! **********************')
 

############################################################################################
elif arguments.MLmodel == 'RF':
    print('\n\n--------------------- Training RF -----------------------------\n')
    done=False
    proc='Training RF'
    t = threading.Thread(target=animate)
    t.start()   

# Train RF Model
    print('Training %s RF Models\n' % arguments.descriptor, file = globals.file)

    RF = rf.train(n_estimators=100,
                descriptor_name = arguments.descriptor,
                model_name = 'RF')

    print('Done!\n')
    time.sleep(10)
    done = True

    print('\n\n--------------------- Testing RF -----------------------------\n')

    done=False
    proc='Testing RF'
    t = threading.Thread(target=animate)
    t.start() 
# Test RF Model
    print('Testing %s RF Model\n' % arguments.descriptor, file = globals.file)

    RF_predict = rf.test(model = RF,
                       descriptor_name = arguments.descriptor,
                       model_name = 'RF')

    print('Done!\n')
    time.sleep(10)
    done = True
 
    print('\n\n--------------------- Creating Reports -----------------------------\n')

    done=False
    proc='Creating Reports'
    t = threading.Thread(target=animate)
    t.start()   

    print('\nClassification Report\n')
    print('\nClassification Report\n', file = globals.file)
 
 
# RF Model 
    rf.classificationReport(model = RF,
                         predict = RF_predict,
                         descriptor_name = arguments.descriptor,
                         model_name = 'RF')

    #print('Done!\n')
    print('\nSaving Confusion Matrix\n')

# Save Confusion Matrix figure RF Model
    save_figures.confusionMatrix(predict = RF_predict,
                             descriptor_name = arguments.descriptor,
                             dataset_name = arguments.dataset,
                             model_name = 'RF')

    #print('Done!\n')
    print('\nSaving Bag of Visual Features\n')
#-------------------------------------------------------------------

# Save Bag of Visual Features 
    save_models.saveBoVF(kmeans = kmeans,
                     descriptor_name = arguments.descriptor,
			         dataset_name = arguments.dataset)

    #print('Done!\n')
    print('\nSaving Descriptors\n')
#-------------------------------------------------------------------

# Save Training Descriptors
    save_models.saveDescriptors(feature_vector = globals.train_feature_vec,
                            descriptor_name = arguments.descriptor,
			                dataset_name = arguments.dataset,
                            flag = 'train')
#-------------------------------------------------------------------
# Save Test Descriptors
    save_models.saveDescriptors(feature_vector = globals.test_feature_vec,
                            descriptor_name = arguments.descriptor,
			                dataset_name = arguments.dataset,
                            flag = 'test')
    #print('Done!\n')
    print('\nSaving RF Models\n')
#-------------------------------------------------------------------
# SVM Model
    save_models.saveRF(model = RF,
                    descriptor_name = arguments.descriptor,
			        dataset_name = arguments.dataset,
                    model_name = 'RF')

    #print('Done!\n')

    time.sleep(10)
    done = True
    
    outputs.closeFile()
    print('\n\n*************** Finished! **********************')
 
############################################################################################
elif arguments.MLmodel == 'NB':
    print('\n\n--------------------- Training NB -----------------------------\n')
    done=False
    proc='Training NB'
    t = threading.Thread(target=animate)
    t.start()   

# Train NB Model
    print('Training %s NB Models\n' % arguments.descriptor, file = globals.file)

    NB = nb.train(descriptor_name = arguments.descriptor,
                model_name = 'NB')

    print('Done!\n')
    time.sleep(10)
    done = True

    print('\n\n--------------------- Testing NB -----------------------------\n')

    done=False
    proc='Testing NB'
    t = threading.Thread(target=animate)
    t.start() 
# Test RF Model
    print('Testing %s NB Model\n' % arguments.descriptor, file = globals.file)

    NB_predict = nb.test(model = NB,
                       descriptor_name = arguments.descriptor,
                       model_name = 'NB')

    print('Done!\n')
    time.sleep(10)
    done = True
 
    print('\n\n--------------------- Creating Reports -----------------------------\n')

    done=False
    proc='Creating Reports'
    t = threading.Thread(target=animate)
    t.start()   

    print('\nClassification Report\n')
    print('\nClassification Report\n', file = globals.file)
 
 
# RF Model 
    nb.classificationReport(model = NB,
                         predict = NB_predict,
                         descriptor_name = arguments.descriptor,
                         model_name = 'NB')

    #print('Done!\n')
    print('\nSaving Confusion Matrix\n')

# Save Confusion Matrix figure NB Model
    save_figures.confusionMatrix(predict = NB_predict,
                             descriptor_name = arguments.descriptor,
                             dataset_name = arguments.dataset,
                             model_name = 'NB')

    #print('Done!\n')
    print('\nSaving Bag of Visual Features\n')
#-------------------------------------------------------------------

# Save Bag of Visual Features 
    save_models.saveBoVF(kmeans = kmeans,
                     descriptor_name = arguments.descriptor,
			         dataset_name = arguments.dataset)

    #print('Done!\n')
    print('\nSaving Descriptors\n')
#-------------------------------------------------------------------

# Save Training Descriptors
    save_models.saveDescriptors(feature_vector = globals.train_feature_vec,
                            descriptor_name = arguments.descriptor,
			                dataset_name = arguments.dataset,
                            flag = 'train')
#-------------------------------------------------------------------
# Save Test Descriptors
    save_models.saveDescriptors(feature_vector = globals.test_feature_vec,
                            descriptor_name = arguments.descriptor,
			                dataset_name = arguments.dataset,
                            flag = 'test')
    #print('Done!\n')
    print('\nSaving NB Models\n')
#-------------------------------------------------------------------
# NB Model
    save_models.saveNB(model = NB,
                    descriptor_name = arguments.descriptor,
			        dataset_name = arguments.dataset,
                    model_name = 'NB')

    #print('Done!\n')

    time.sleep(10)
    done = True
    
    outputs.closeFile()
    print('\n\n*************** Finished! **********************')
 ####################################################################################
elif arguments.MLmodel == 'MLP':
    print('\n\n--------------------- Training MLP -----------------------------\n')

    #print('Training Multilayer Perceptron Model\n')
    done=False
    proc='Training MLP 1 '
    t = threading.Thread(target=animate)
    t.start()  

    # Train MLP Model
    print('Training %s MLP Models\n' % arguments.descriptor, file = globals.file)

# MLP Model 1
    MLP1 = multilayer.train(solver = 'sgd',
                        activation=parameters.activation,
                        power_t=parameters.power_t,
                        momentum = 0,
                        hidden_layer_sizes = parameters.hidden_layer,
                        alpha = parameters.alpha,
                        learning_rate = 'constant',
                        learning_rate_init = parameters.learning_rate,
                        random_state = parameters.seed,
                        max_iter = parameters.max_iterations,
                        beta_1=parameters.beta_1,
                        beta_2=parameters.beta_2,
                        n_iter_no_change=parameters.n_iter_no_change,
                        descriptor_name = arguments.descriptor,
                        model_name = 'MLP1')
    time.sleep(10)
    done = True   
    print('Done!\n')

    done=False
    proc='Training MLP 2 '
    t = threading.Thread(target=animate)
    t.start() 
# MLP Model 2
    MLP2 = multilayer.train(solver = 'sgd',
                        activation=parameters.activation,
                        power_t=parameters.power_t,
                        momentum = 0.3,
                        hidden_layer_sizes = parameters.hidden_layer,
                        alpha = parameters.alpha,
                        learning_rate = 'constant',
                        learning_rate_init = parameters.learning_rate,
                        random_state = parameters.seed,
                        max_iter = parameters.max_iterations,
                        beta_1=parameters.beta_1,
                        beta_2=parameters.beta_2,
                        n_iter_no_change=parameters.n_iter_no_change,
                        descriptor_name = arguments.descriptor,
                        model_name = 'MLP2')
    time.sleep(10)
    done = True
    print('Done!\n')

    done=False
    proc='Training MLP 3 '
    t = threading.Thread(target=animate)
    t.start()    
# MLP Model 3
    MLP3 = multilayer.train(solver = 'sgd',
                        activation=parameters.activation,
                        power_t=parameters.power_t,
                        momentum = 0.6,
                        hidden_layer_sizes = parameters.hidden_layer,
                        alpha = parameters.alpha,
                        learning_rate = 'constant',
                        learning_rate_init = parameters.learning_rate,
                        random_state = parameters.seed,
                        max_iter = parameters.max_iterations,
                        beta_1=parameters.beta_1,
                        beta_2=parameters.beta_2,
                        n_iter_no_change=parameters.n_iter_no_change,
                        descriptor_name = arguments.descriptor,
                        model_name = 'MLP3')
    time.sleep(10)
    done = True
    print('Done!\n')

    done=False
    proc='Training MLP 4 '
    t = threading.Thread(target=animate)
    t.start()  
# MLP Model 4
    MLP4 = multilayer.train(solver = 'sgd',
                        activation=parameters.activation,
                        power_t=parameters.power_t,
                        momentum = 0.9,
                        hidden_layer_sizes = parameters.hidden_layer,
                        alpha = parameters.alpha,
                        learning_rate = 'constant',
                        learning_rate_init = parameters.learning_rate,
                        random_state = parameters.seed,
                        max_iter = parameters.max_iterations,
                        beta_1=parameters.beta_1,
                        beta_2=parameters.beta_2,
                        n_iter_no_change=parameters.n_iter_no_change,
                        descriptor_name = arguments.descriptor,
                        model_name = 'MLP4')
    time.sleep(10)
    done = True
    print('Done!\n')

    done=False
    proc='Training MLP 5 '
    t = threading.Thread(target=animate)
    t.start()  
# MLP Model 5
    MLP5 = multilayer.train(solver = 'sgd',
                        activation=parameters.activation,
                        power_t=parameters.power_t,
                        momentum = 0.9,
                        hidden_layer_sizes = parameters.hidden_layer,
                        alpha = parameters.alpha,
                        learning_rate = 'adaptive',
                        learning_rate_init = parameters.learning_rate,
                        random_state = parameters.seed,
                        max_iter = parameters.max_iterations,
                        beta_1=parameters.beta_1,
                        beta_2=parameters.beta_2,
                        n_iter_no_change=parameters.n_iter_no_change,
                        descriptor_name = arguments.descriptor,
                        model_name = 'MLP5')
    time.sleep(10)
    done = True
    print('Done!\n')

    done=False
    proc='Training MLP 6 '
    t = threading.Thread(target=animate)
    t.start()  
# MLP Model 6
    MLP6 = multilayer.train(solver = 'adam',
                        activation=parameters.activation,
                        power_t=parameters.power_t,
                        momentum = 0.9,
                        hidden_layer_sizes = parameters.hidden_layer,
                        alpha = parameters.alpha,
                        learning_rate = 'constant',
                        learning_rate_init = parameters.learning_rate,
                        random_state = parameters.seed,
                        max_iter = parameters.max_iterations,
                        beta_1=parameters.beta_1,
                        beta_2=parameters.beta_2,
                        n_iter_no_change=parameters.n_iter_no_change,
                        descriptor_name = arguments.descriptor,
                        model_name = 'MLP6')

    #print('Done!\n')
    time.sleep(10)
    done = True 
    print('Done!\n')

    
    print('\n\n--------------------- Saving Training Loss -----------------------------\n')

    #print('\nSaving Training Loss\n')

    done=False
    proc='Save Training Loss'
    t = threading.Thread(target=animate)
    t.start()   

# Save Training Loss Figure
    save_figures.saveTrainingLoss(model1 = MLP1,
                              model2 = MLP2,
                              model3 = MLP3,
                              model4 = MLP4,
                              model5 = MLP5,
                              model6 = MLP6,
                              descriptor_name = arguments.descriptor,
                              dataset_name = arguments.dataset)

# Print
    
    time.sleep(10)
    done = True
    print('Done!\n')


    print('\n\n--------------------- Testing MLP -----------------------------\n')

# Test MLP Model
    print('Testing %s MLP Model\n' % arguments.descriptor, file = globals.file)

# MLP Model 1

    done=False
    proc='Testning MLP 1'
    t = threading.Thread(target=animate)
    t.start()

    MLP1_predict = multilayer.test(model = MLP1,
    
                             #Activation= ‘relu’, returns f(x) = max(0, x)
                               solver = 'sgd',
                               momentum = 0,
                               learning_rate = 'constant',
                               descriptor_name = arguments.descriptor,
                               model_name = 'MLP1')
    time.sleep(10)
    done = True 
    done=False    
    print('Done!\n')

    proc='Testning MLP 2'
    t = threading.Thread(target=animate)
    t.start()  
# MLP Model 2
    MLP2_predict = multilayer.test(model = MLP2,
                               solver = 'sgd',
                               momentum = 0.3,
                               learning_rate = 'constant',
                               descriptor_name = arguments.descriptor,
                               model_name = 'MLP2')
    time.sleep(10)
    done = True 
    done=False
    print('Done!\n')

    proc='Testning MLP 3'
    t = threading.Thread(target=animate)
    t.start() 
# MLP Model 3
    MLP3_predict = multilayer.test(model = MLP3,
                               solver = 'sgd',
                               momentum = 0.6,
                               learning_rate = 'constant',
                               descriptor_name = arguments.descriptor,
                               model_name = 'MLP3')
    time.sleep(10)
    done = True 
    done=False
    print('Done!\n')

    proc='Testning MLP 4'
    t = threading.Thread(target=animate)
    t.start() 
# MLP Model 4
    MLP4_predict = multilayer.test(model = MLP4,
                               solver = 'sgd',
                               momentum = 0.9,
                               learning_rate = 'constant',
                               descriptor_name = arguments.descriptor,
                               model_name = 'MLP4')
    time.sleep(10)
    done = True 
    done=False
    print('Done!\n')
  
    proc='Testning MLP 5'
    t = threading.Thread(target=animate)
    t.start() 
# MLP Model 5
    MLP5_predict = multilayer.test(model = MLP5,
                               solver = 'sgd',
                               momentum = 0.9,
                               learning_rate = 'adaptative',
                               descriptor_name = arguments.descriptor,
                               model_name = 'MLP5')
    time.sleep(10)
    done = True 
    done=False
    print('Done!\n')

    proc='Testning MLP 6'
    t = threading.Thread(target=animate)
    t.start() 
# MLP Model 6
    MLP6_predict = multilayer.test(model = MLP6,
                               solver = 'adam',
                               momentum = 0.9,
                               learning_rate = 'constant',
                               descriptor_name = arguments.descriptor,
                               model_name = 'MLP6')

    time.sleep(10)
    done = True
    print('Done!\n')
    print('\n\n---------------------Classification Report-----------------------------\n')

    #print('Classification Report\n')
    print('Classification Report\n', file = globals.file)

# MLP Model 1

    done=False
    proc='Classification Report'
    t = threading.Thread(target=animate)
    t.start()

    multilayer.classificationReport(model = MLP1,
                                predict = MLP1_predict,
                                descriptor_name = arguments.descriptor,
                                model_name = 'MLP1')

# MLP Model 2
    multilayer.classificationReport(model = MLP2,
                                predict = MLP2_predict,
                                descriptor_name = arguments.descriptor,
                                model_name = 'MLP2')

# MLP Model 3
    multilayer.classificationReport(model = MLP3,
                                predict = MLP3_predict,
                                descriptor_name = arguments.descriptor,
                                model_name = 'MLP3')

# MLP Model 4
    multilayer.classificationReport(model = MLP4,
                                predict = MLP4_predict,
                                descriptor_name = arguments.descriptor,
                                model_name = 'MLP4')

# MLP Model 5
    multilayer.classificationReport(model = MLP5,
                                predict = MLP5_predict,
                                descriptor_name = arguments.descriptor,
                                model_name = 'MLP5')

# MLP Model 6
    multilayer.classificationReport(model = MLP6,
                                predict = MLP6_predict,
                                descriptor_name = arguments.descriptor,
                                model_name = 'MLP6')

    time.sleep(10)
    done = True
       
    print('Done!\n')

    print('\n\n---------------------Saving Confusion Matrix-----------------------------\n')

    done=False
    proc='Saving Confusion Matrix'
    t = threading.Thread(target=animate)
    t.start()   
# Save Confusion Matrix figure MLP Model 1
    save_figures.confusionMatrix(predict = MLP1_predict,
                             descriptor_name = arguments.descriptor,
                             dataset_name = arguments.dataset,
                             model_name = 'MLP1')

# Save Confusion Matrix figure MLP Model 2
    save_figures.confusionMatrix(predict = MLP2_predict,
                             descriptor_name = arguments.descriptor,
                             dataset_name = arguments.dataset,
                             model_name = 'MLP2')

# Save Confusion Matrix figure MLP Model 3
    save_figures.confusionMatrix(predict = MLP3_predict,
                             descriptor_name = arguments.descriptor,
                             dataset_name = arguments.dataset,
                             model_name = 'MLP3')

# Save Confusion Matrix figure MLP Model 4
    save_figures.confusionMatrix(predict = MLP4_predict,
                             descriptor_name = arguments.descriptor,
                             dataset_name = arguments.dataset,
                             model_name = 'MLP4')

# Save Confusion Matrix figure MLP Model 5
    save_figures.confusionMatrix(predict = MLP5_predict,
                             descriptor_name = arguments.descriptor,
                             dataset_name = arguments.dataset,
                             model_name = 'MLP5')

# Save Confusion Matrix figure MLP Model 6
    save_figures.confusionMatrix(predict = MLP6_predict,
                             descriptor_name = arguments.descriptor,
                             dataset_name = arguments.dataset,
                             model_name = 'MLP6')

    time.sleep(10)
    done = True 
    print('Done!\n')

    print('\n\n---------------------Saving Weights-----------------------------\n')

    done=False
    proc='Saving Weights'
    t = threading.Thread(target=animate)
    t.start()  

# Save Weights figure MLP Model 1
    save_figures.weights(model = MLP1,
                     descriptor_name = arguments.descriptor,
                     dataset_name = arguments.dataset,
                     model_name = 'MLP1')

# Save Weights figure MLP Model 2
    save_figures.weights(model = MLP2,
                     descriptor_name = arguments.descriptor,
                     dataset_name = arguments.dataset,
                     model_name = 'MLP2')

# Save Weights figure MLP Model 3
    save_figures.weights(model = MLP3,
                     descriptor_name = arguments.descriptor,
                     dataset_name = arguments.dataset,
                     model_name = 'MLP3')

# Save Weights figure MLP Model 4
    save_figures.weights(model = MLP4,
                     descriptor_name = arguments.descriptor,
                     dataset_name = arguments.dataset,
                     model_name = 'MLP4')

# Save Weights figure MLP Model 5
    save_figures.weights(model = MLP5,
                     descriptor_name = arguments.descriptor,
                     dataset_name = arguments.dataset,
                     model_name = 'MLP5')

# Save Weights figure MLP Model 6
    save_figures.weights(model = MLP6,
                     descriptor_name = arguments.descriptor,
                     dataset_name = arguments.dataset,
                     model_name = 'MLP6')

    time.sleep(10)
    done = True 
    print('Done!\n')
    print('\n\n---------------------Saving BOVFs-----------------------------\n')

    done=False
    proc='Saving BOVFs'
    t = threading.Thread(target=animate)
    t.start() 

# Save Bag of Visual Features 
    save_models.saveBoVF(kmeans = kmeans,
                     descriptor_name = arguments.descriptor,
			         dataset_name = arguments.dataset)

    time.sleep(10)
    done = True 
    print('Done!\n')



    print('\n\n---------------------Saving Descriptors-----------------------------\n')
    done=False
    proc='Saving Descriptors'
    t = threading.Thread(target=animate)
    t.start() 
# Save Training Descriptors
    save_models.saveDescriptors(feature_vector = globals.train_feature_vec,
                            descriptor_name = arguments.descriptor,
			                dataset_name = arguments.dataset,
                            flag = 'train')

# Save Test Descriptors
    save_models.saveDescriptors(feature_vector = globals.test_feature_vec,
                            descriptor_name = arguments.descriptor,
			                dataset_name = arguments.dataset,
                            flag = 'test')

    time.sleep(10)
    done = True
    print('Done!\n')
 
    
    print('\n\n---------------------Saving MLP Models-----------------------------\n')
    done=False
    proc='Saving MLP Models'
    t = threading.Thread(target=animate)
    t.start()    
  

# MLP Model 1
    save_models.saveMLP(model = MLP1,
                    descriptor_name = arguments.descriptor,
			        dataset_name = arguments.dataset,
                    model_name = 'MLP1')

# MLP Model 2
    save_models.saveMLP(model = MLP2,
                    descriptor_name = arguments.descriptor,
			        dataset_name = arguments.dataset,
                    model_name = 'MLP2')

# MLP Model 3
    save_models.saveMLP(model = MLP3,
                    descriptor_name = arguments.descriptor,
			        dataset_name = arguments.dataset,
                    model_name = 'MLP3')

# MLP Model 4
    save_models.saveMLP(model = MLP4,
                    descriptor_name = arguments.descriptor,
			        dataset_name = arguments.dataset,
                    model_name = 'MLP4')

# MLP Model 5
    save_models.saveMLP(model = MLP5,
                    descriptor_name = arguments.descriptor,
			        dataset_name = arguments.dataset,
                    model_name = 'MLP5')

# MLP Model 6
    save_models.saveMLP(model = MLP6,
                    descriptor_name = arguments.descriptor,
			        dataset_name = arguments.dataset,
                    model_name = 'MLP6')

    time.sleep(10)
    done = True
    print('Done!\n')
 
    outputs.closeFile()
    print('\n\n\n*************** Finished! **********************')
  
##############################################################################