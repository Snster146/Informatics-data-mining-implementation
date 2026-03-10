# Task 1 [15 points out of 40] Naïve Bayes classifier

# Your first task is to implement the naïve Bayes classifier on your own. This involves calculating all the necessary
# probabilities from the provided data, and using them to make predictions for new unseen records.
# The required version is the one covered in the module. Implementing other naïve Bayes approaches (e.g. Gaussian) or
# using further modifications that do not correspond to the technique practiced in the module will lead to
# significant loss of points.
# The template contains a range of functions you must implement and use appropriately for this task. The template also
# uses a range of functions implemented by the module leader to support you in this task.

import pandas as pd


class NaiveBayes:

    # This function simply initializes an instance of NaiveBayes class. The constructor takes at input:
    # - class_info - pair that contains the name of the class column and its permitted values
    # - feature_info - dictionary that states attribute names and their permitted values

    def __init__(self, class_info: tuple[str, list[str]], feature_info: dict[str, list[str]]):
        self.class_info = class_info
        self.feature_info = feature_info

        # dictionary to store class labels along with class probabilites 
        self.class_probabilities={}
        # dictionary to store each record assocaited with each class label
        self.class_dataframes={}
        
        self.class_occurence={}
        
        self.attribute_probabilities={}

        self.trainingTookPlace=False

        # You can add further variables/attributes/etc. here
        
    # This function trains the model, aka calculates all the necessary probabilities that a naive Bayes model needs.
    # How you store the computed probabilities internally is up to you - you may want to extend the init function.
    # For the purpose of this task, numerical values are treated just like categorical ones. Any new training
    # should purge old data.
    # At input, train_model takes:
    # - training_data - a pandas DataFrame that contains all the attribute values and class value for a given entry
    def train_model(self, training_data: pd.DataFrame):
        

        for classlabel in self.class_info[1]:
            #retreives the number of records associated with a given class label
            self.class_occurence[classlabel]=training_data["Class"].value_counts().get(classlabel,0)

            # stores the class proability of a given class as a value associated with a class label in a dictionary  
            self.class_probabilities[classlabel]=(float((self.class_occurence[classlabel]/training_data.shape)[0]))
            # stores each record associated with a giveen class label in a dictionary
            self.class_dataframes[classlabel]=training_data[training_data["Class"]==classlabel]

        
        for classlabel,dataframe in self.class_dataframes.items():
            for attributeLabel,attributeValue in self.feature_info.items():    
                
                try:
                    self.attribute_probabilities[classlabel][attributeLabel]=[]
                except:
                    self.attribute_probabilities[classlabel]={attributeLabel:[]}

                for value in attributeValue:
                    curr_attr_prob=float(dataframe[attributeLabel].value_counts().get(value,0)/dataframe.shape[0])
                    # attribute_probabilities[attributeLabel].append(float(dataframe[attributeLabel].value_counts().get(value,0)/dataframe.shape[0]))
                    self.attribute_probabilities[classlabel][attributeLabel].append(curr_attr_prob)
  
        self.trainingTookPlace=True
        
    # This function predicts the classes for entries in the training_data and produces an extended data frame.
    # At input, it takes:
    # - training_data - a pandas DataFrame that contains all the attribute values and class value for a given entry
    # The function outputs:
    # classified_data - a pandas DataFrame which expands the training_data by adding the "PredictedClass" column
    #                   that for every entry states the class value predicted for that entry. In case of ties,
    #                   the chosen class is the one that appears earlier alphabetically.
    def predict(self, testing_data: pd.DataFrame) -> pd.DataFrame:
        classified_data = None
        testDataProbabilities={}

        for classLabel, classProb in self.class_probabilities.items():
            testDataProbabilities[classLabel]=[]

            for attributeLabel,attributeValue in self.feature_info.items():  
                for value in attributeValue:  
                    attributeValueIndex=attributeValue.index(value)

                    if(testing_data[attributeLabel].value_counts().get(value,0)>0):
                        testDataProbabilities[classLabel].append(self.attribute_probabilities[classLabel][attributeLabel][attributeValueIndex])
            
        for classLabel, attr_prob in testDataProbabilities.items():
            classProb=1

            attr_prob.append(self.class_probabilities[classLabel])

            for prob in attr_prob:
                classProb=classProb*prob
            self.class_probabilities[classLabel]=classProb

        predictedClass=max(self.class_probabilities,key=self.class_probabilities.get)

        classified_data=testing_data
        classified_data["PredictedClass"]=predictedClass

        return classified_data

    # The function returns the probability of a given class value. You can assume
    # that this function simply retrieves the desired probability after training rather than
    # recomputes them from scratch. A value of 0 should be returned if no training took place.
    # At input, it takes:
    # - class_value - the class value for which we want to calculate the probability
    # The function outputs:
    # - probability - float representing the probability of the given class value
    def retrieve_class_probability(self, class_value: str) -> float:
        if (not(self.trainingTookPlace) or (class_value not in self.class_probabilities.keys()) ):
            return 0;
        else:
            return self.class_probabilities[class_value]

    # The function returns the conditional probably of a feature value assuming a given class value. You can assume
    # that this function simply retrieves the desired probability after training rather than
    # recomputes them from scratch. A value of 0 should be returned if no training took place.
    # At input, it takes:
    # - class_value - the class value on which the feature_value is conditional
    # - feature_name - the name of the feature we want to calculate for
    # - feature_value - the feature value we want to calculate the conditional probability for
    # The function outputs:
    # - probability - float representing the calculated conditional probability
    #
    def retrieve_conditional_probability(self, class_value: str, feature_name: str, feature_value: str) -> float:
# checks if training hasn't taken place or a given feature value isnt associated with a given feature name -> return 0
        if  (not(self.trainingTookPlace)or (feature_value not in self.feature_info[feature_name])):
            return 0;
        else:
# retrieves the index of the feature value in the array associated with the feature name key in the feature info dictionary  
            featureValueIndex=self.feature_info[feature_name].index(feature_value)
# retrieves the probabilities associated with the feature values of a given feature name given a class value 
            featureProbabilities=self.attribute_probabilities[class_value][feature_name]
# retreives the feature value probability through using the index of the given feature value in the array associated with a feature name 
            featureValueProbability=featureProbabilities[featureValueIndex]

            return featureValueProbability
