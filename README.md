

##X-ray diffraction pattern Classification based on Hand-crafted feature extractors and Support Vector Machine (SVM) or Multi-layer Perceptron (MLP) classifiers##



-For run the code:

              python main.py --MLmodel SVM --detector ORB --descriptor ORB --dataset DiffraNet

-You can run the code with different feature extractor and descriptor algorithms and then classify images (based on extracted features) with SVM or MLP classifier.             

             --MLmodel      {SVM,MLP}
             --detector     {SIFT, SURF, KAZE, ORB, BRISK, AKAZE, BRIEF}
             --descriptor   {SIFT, SURF, KAZE, BRIEF, ORB, BRISK, AKAZE, FREAK, BREAF}
             --dataset      { DiffraNet, AGIPD}


- For MLP, six different configurations (Solver, momentum) have been implemented.
- All the extracted features for each class are saved separately in the folder "/Features/" as ".pickle" file.
- You can load and read easily the Pickle files and see the features or train them to your ML model. 
- All the confused matrixed are saved in the folder "Figures/Dataset/DiffraNet" as ".png" file.
- Some basic evaluations like precision, recall, f1-score are saved in the folder "Output/Datasets/DiffraNet/". 

NOTE: there are some bugs in the evaluation part that I will fix in the following days.

  