
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from timer import Timer
import globals


def train(descriptor_name,
          model_name):

    with Timer() as timer:
        # NB Model
        model = GaussianNB()
        model.fit(globals.train_feature_vec[0],
                globals.train_feature_vec[1])
        
        print('%s %s\n' % (descriptor_name, model_name), file = globals.file)

        print('Training Set Evaluation\n', file = globals.file)
        
        print('Train score: %.2f\n' % (model.score(globals.train_feature_vec[0],
                                                 globals.train_feature_vec[1])),
                                                 file = globals.file)

        return model

    print('Time:', timer, '\n', file = globals.file)

#-------------------------------------------------------------------

def test(model,
         descriptor_name,
         model_name):

    with Timer() as timer:
        # NB Model
        print('%s %s\n' % (descriptor_name, model_name), file = globals.file)

        print('Testing Set Evaluation\n', file = globals.file)

        NB_predict = model.predict(globals.test_feature_vec[0])

        print('Test score: %.2f\n' % (accuracy_score(globals.test_feature_vec[1],
                                                     NB_predict)),
                                                     file = globals.file)

        return NB_predict

    print('Time:', timer, '\n', file = globals.file)

#-------------------------------------------------------------------

def classificationReport(model,
                         predict,
                         descriptor_name,
                         model_name):
    #  NB Model 
    print('%s %s\n' % (descriptor_name, model_name), file = globals.file)

    # Classification report for Classifier
    print("Classification report for Classifier: \n\n%s: \n\n %s" % (model,
                                                                     classification_report(globals.test_feature_vec[1],
                                                                                           predict)),
                                                                                           file = globals.file)



                                                       