
## Usage

python main.py --MLmodel <Mlmodel> --detector <detector> --descriptor <descriptor> --dataset <dataset>


------------------------------------------------------------------------------------------
| Arguments     | Info                                                                    |
| ------------- | ----------------------------------------------------------------------- |
| `MLmodel`	| Specify Support Vector Machine (SVM) or Multi Layer Perceptron (MLP)   |                                              |
| `--detector`  | Specify detectors: SIFT, SURF, KAZE, ORB, BRISK, AKAZE, BRIEF                      |
| `--descriptor`| Specify descriptors: SIFT, SURF, KAZE, BRIEF, ORB, BRISK, AKAZE or FREAK             |
| `--dataset `  | Specify DiffraNet or AGIPD                                              |


-------------------------------------------------------------------------------------------
## Examples


python main.py --MLmodel SVM --detector ORB --descriptor ORB --dataset DiffraNet

#### Extract ORB Descriptor on Diffranet datasets and classify imageeg with SVM 
 