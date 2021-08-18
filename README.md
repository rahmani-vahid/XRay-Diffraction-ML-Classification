
## Usage

python main.py --MLmodel <Mlmodel> --detector <detector> --descriptor <descriptor> --dataset <dataset>


------------------------------------------------------------------------------------------
| Arguments     | Info                                                                    |
| ------------- | ----------------------------------------------------------------------- |
| `MLmodel`	| Specify Sumpport Vector Machine (SVM) or Multi Layer Perceptron (MLP)   |                                              |
| `--detector`  | Specify SIFT, SURF, KAZE, ORB, BRISK, AKAZE, BRIEF                      |
| `--descriptor`| Specify SIFT, SURF, KAZE, BRIEF, ORB, BRISK, AKAZE or FREAK             |
| `--dataset `  | Specify DiffraNet of AGIPD                                              |


-------------------------------------------------------------------------------------------
## Examples


python main.py --MLmodel SVM --detector ORB --descriptor ORB --dataset DiffraNe

#### Extract ORB Descriptor on Diffranet datasets and classify imageeg with SVM 
 