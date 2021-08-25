
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from timer import Timer
import globals


def train(n_estimators,
          descriptor_name,
          model_name):

    with Timer() as timer:
        # RF Model
        clf=RandomForestClassifier(n_estimators=100)

        clf.fit(globals.train_feature_vec[0],
                globals.train_feature_vec[1])
        
        print('%s %s\n' % (descriptor_name, model_name), file = globals.file)

        print('Training Set Evaluation\n', file = globals.file)
        
        print('Train score: %.2f\n' % (clf.score(globals.train_feature_vec[0],
                                                 globals.train_feature_vec[1])),
                                                 file = globals.file)

        return clf

    print('Time:', timer, '\n', file = globals.file)

#-------------------------------------------------------------------

def test(model,
         descriptor_name,
         model_name):

    with Timer() as timer:
        # RF Model
        print('%s %s\n' % (descriptor_name, model_name), file = globals.file)

        print('Testing Set Evaluation\n', file = globals.file)

        RF_predict = model.predict(globals.test_feature_vec[0])

        print('Test score: %.2f\n' % (accuracy_score(globals.test_feature_vec[1],
                                                     RF_predict)),
                                                     file = globals.file)

        return RF_predict

    print('Time:', timer, '\n', file = globals.file)

#-------------------------------------------------------------------

def classificationReport(model,
                         predict,
                         descriptor_name,
                         model_name):
    #  RF Model 
    print('%s %s\n' % (descriptor_name, model_name), file = globals.file)

    # Classification report for Classifier
    print("Classification report for Classifier: \n\n%s: \n\n %s" % (model,
                                                                     classification_report(globals.test_feature_vec[1],
                                                                                           predict)),
                                                                                           file = globals.file)



                                                       