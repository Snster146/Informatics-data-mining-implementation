# Task 2 [10 points out of 40] Classifier evaluation
# This task focuses on evaluating the naïve Bayes classifier from Task 1. On your own, implement binary precision,
# recall and f-measure, as well as their macro and weighted counterparts.
# You are also asked to implement the multiclass version of accuracy, and its weighted counterpart. You
# need to follow the formulas covered in the module. Remember to be mindful of edge cases (the approach for handling
# them is explained in lecture slides).
# Please note that this template also contains empty functions pertaining to
# creating a confusion matrix and calculating TPs, FPs and FNs based on it. These will be implemented during the
# practicals, with the code to be released later. They are not a part of the marking criteria.

import pandas as pd


# This function computes the confusion matrix based on the provided series of actual and predicted classes.
# The returned data frame must contain appropriate column and row names, and be filled with integers.
# The columns correspond to actual classes and rows to predicted classes, in the sense that the i-th row
# is the row representing how often entries actually belonging to some class, were predicted as the i-th class value;
# the i-th column represents how often entries predicted as some other class, actually belonged to the i-th class.
#
# At input, function takes:
# - actual_class, predicted_class - series of class values representing actual and predicted classes of some dataset.
#                                   NOT guaranteed to contain all possible class values from the classification schema.
# - class_values - all possible values of the class from which actual_class and predicted_class were drawn.
#
# As output, it produces:
# - matrix : a data frame representing the confusion matrix computed based on the offered series of actual
#            and predicted classes. The data frame must contain appropriate column and row names, and be
#            filled with integers.

def confusion_matrix(actual_class: pd.Series, predicted_class: pd.Series, class_values: list[str]) -> pd.DataFrame:
    matrix = pd.DataFrame(0, index=class_values, columns=class_values)

    for i in range(len(actual_class)):
        try:
            actual = actual_class.iloc[i]
            predicted = predicted_class.iloc[i]
        except:
            print("Dataset contains a class not in the classification scheme!")
        value = matrix.loc[predicted, actual]
        matrix.loc[predicted, actual] = value + 1
    return matrix


# These functions compute per-class true positives and false positives/negatives based on the provided confusion matrix.
#
# As input, these functions take:
# - matrix - a data frame representing the confusion matrix computed based on the offered series of actual
#            and predicted classes. See confusion_matrix function for description.
#
# As output, these functions produce:
# - tps/fps/fns - dictionaries that for every class value in the classification scheme (corresponding to names of
#                 all rows and/or all columns in the matrix) return the true positive, false positive or
#                 false negative values for that class.

def compute_TPs(matrix: pd.DataFrame) -> dict[str, int]:
    class_vals = matrix.columns
    tps = {}
    for class_v in class_vals:
    # TPs is simply about retrieving values from the diagonal
    # .loc retrieves values via names, not indices
        tps[class_v] = matrix.loc[class_v, class_v]
    return tps


def compute_FPs(matrix: pd.DataFrame) -> dict[str, int]:
    class_vals = matrix.columns
    fps = {}
    for class_1 in class_vals:
        sum_vals = 0
        for class_2 in class_vals:
        # For FPs, we need to add up values predicted to belong to that class, but not belonging to it in reality
            if class_1 != class_2:
                sum_vals += matrix.loc[class_1, class_2]
        fps[class_1] = sum_vals
    return fps


def compute_FNs(matrix: pd.DataFrame) -> dict[str, int]:
    class_vals = matrix.columns
    fns = {}
    for class_1 in class_vals:
        sum_vals = 0
        # For FNs, we need to add up values not predicted to belong to that class, but belonging to it in reality
        for class_2 in class_vals:
            if class_1 != class_2:
                sum_vals += matrix.loc[class_2, class_1]
        fns[class_1] = sum_vals
    return fns


# These functions compute the binary measures based on the provided values. Not all measures use all the values.
# Do not remove the unused variables from the function pattern.
# At input, the functions take:
# - tp, fp, fn : the single values of true positives, false positive and negatives
#
# As output, they produce:
# - binary precision/recall/f-measure - appropriate evaluation measure created using the binary approach.

def compute_binary_precision(tp: int, fp: int, fn: int) -> float:
    return tp/(tp+fp)


def compute_binary_recall(tp: int, fp: int, fn: int) -> float:
    return tp/(tp+fn)


def compute_binary_f_measure(tp: int, fp: int, fn: int) -> float:
    p=compute_binary_precision(tp,fp,fn)
    r=compute_binary_recall(tp,fp,fn)
    return 2*p*r/(p+r)
# These functions compute the macro precision, macro recall, macro f-measure, based on the offered confusion matrix.
# You are expected to use appropriate binary counterparts when needed (binary recall for macro recall, binary precision
# for macro precision, binary f-measure for macro f-measure) and the functions for computing tps/fps/fns as needed.
#
# As input, these functions take:
# - matrix - a data frame representing the confusion matrix computed based on the offered series of actual
#            and predicted classes. See confusion_matrix function for description.
# As output, they produce:
# - macro precision/recall/f-measure - appropriate evaluation measures created using the macro average approach.

def compute_macro_precision(matrix: pd.DataFrame) -> float:
    attr_FNs=compute_FNs(matrix)
    attr_FPs=compute_FPs(matrix)
    attr_TPs=compute_TPs(matrix)

    binaryPrecisions=[]
    attributeNames=list(matrix.columns)
    for attr_name in attributeNames:
        currAttrFN=attr_FNs[attr_name]
        currAttrFP=attr_FPs[attr_name]
        currAttrTP=attr_TPs[attr_name]
        attrPrecision=float(compute_binary_precision(fp=currAttrFP,fn=currAttrFN,tp=currAttrTP))
        binaryPrecisions.append(attrPrecision)

    
    return sum(binaryPrecisions)/len(binaryPrecisions)


def compute_macro_recall(matrix: pd.DataFrame) -> float:
    attr_FNs=compute_FNs(matrix)
    attr_FPs=compute_FPs(matrix)
    attr_TPs=compute_TPs(matrix)

    binaryRecalls=[]
    attributeNames=list(matrix.columns)
    for attr_name in attributeNames:
        currAttrFN=attr_FNs[attr_name]
        currAttrFP=attr_FPs[attr_name]
        currAttrTP=attr_TPs[attr_name]
        attrRecall=float(compute_binary_recall(fp=currAttrFP,fn=currAttrFN,tp=currAttrTP))
        binaryRecalls.append(attrRecall)

    
    return sum(binaryRecalls)/len(binaryRecalls)

    


def compute_macro_f_measure(matrix: pd.DataFrame) -> float:
    attr_FNs=compute_FNs(matrix)
    attr_FPs=compute_FPs(matrix)
    attr_TPs=compute_TPs(matrix)

    binaryFMeasures=[]
    attributeNames=list(matrix.columns)
    for attr_name in attributeNames:
        currAttrFN=attr_FNs[attr_name]
        currAttrFP=attr_FPs[attr_name]
        currAttrTP=attr_TPs[attr_name]
        attrFMeasure=float(compute_binary_f_measure(fp=currAttrFP,fn=currAttrFN,tp=currAttrTP))
        binaryFMeasures.append(attrFMeasure)

    
    return sum(binaryFMeasures)/len(binaryFMeasures)


# These functions compute the weighted precision, macro recall, macro f-measure, based on the offered confusion matrix.
# You are expected to use appropriate binary counterparts when needed (binary recall for weighted recall,
# binary precision for weighted precision, binary f-measure for weighted f-measure) and the functions
# for computing tps/fps/fns as needed.
#
# As input, these functions take:
# - matrix - a data frame representing the confusion matrix computed based on the offered series of actual
#            and predicted classes. See confusion_matrix function for description.
# As output, they produce:
# - weighted precision/recall/f-measure - appropriate evaluation measures created using the weighted average approach.

def compute_weighted_precision(matrix: pd.DataFrame) -> float:
#stores dictionary of attribute labels associated with FNs, FPs and Tps respectively  
    attr_FNs=compute_FNs(matrix)
    attr_FPs=compute_FPs(matrix)
    attr_TPs=compute_TPs(matrix)

# variable to store weighted precision which will be incremented whilst iterating through every attribute
    weighted_precision=0

    attributeLabels=list(matrix.columns)
# dictionary to store attribute labels and their associated number of occurences
    attributeOccurences={}
    # dictionary to store attribute labels and their associated precisions (P=TP/TP+FP)
    attributePrecisions={}
    

    for attributeLable in attributeLabels:
        # occurence is calculated as TP + FN, 
        attrOccurence=int(attr_TPs[attributeLable] +attr_FNs[attributeLable])
        # store occurence of current attribute in a dictionary
        attributeOccurences[attributeLable]=attrOccurence
        # compute the binary precision of the current attribute 
        curr_attr_precision=compute_binary_precision(fp=int(attr_FPs[attributeLable]),tp=int(attr_TPs[attributeLable]),fn=int(attr_FNs[attributeLable]))
        # store binary precision of current attribute in dictionary
        attributePrecisions[attributeLable]=curr_attr_precision
# this loop calculates the numerator of the equation for weighted precision by summing 
# the product of each attribute precision and its occurence
    for attributeLabel , value in attributePrecisions.items(): 
        weighted_precision+=attributeOccurences[attributeLabel]*value
# stores total number of occurences , which will be used as the denomenator for weighted precision 
    numOccurences=(sum(list(attributeOccurences.values())))
# calculates weighted precision as the sum of the product of each attributes precision and its occurence 
# divided by the total number of occurences
    weighted_precision=weighted_precision/numOccurences

    return weighted_precision
    


def compute_weighted_recall(matrix: pd.DataFrame) -> float:
# stores dictionary of attribute labels along with associated FNs,FPs and TPs respectively
    attr_FNs=compute_FNs(matrix)
    attr_FPs=compute_FPs(matrix)
    attr_TPs=compute_TPs(matrix)
# variable to store weighted recall which will be incremented iteratively once the recall of individual attributes has been computed
    weighted_recall=0

    attributeLabels=list(matrix.columns)
# dictionary to store attribute labels and their associated number of occurences
    attributeOccurences={}
    # dictionary to store attribute labels and their associated recall values (R=TP/TP+FN)
    attributeRecalls={}

    for attributeLable in attributeLabels:
        # occurence is calculated as TP + FN, 
        attrOccurence=int(attr_TPs[attributeLable] +attr_FNs[attributeLable])
        # store occurence of current attribute in a dictionary
        attributeOccurences[attributeLable]=attrOccurence
        # compute the binary recall of the current attribute 
        curr_attr_recall=compute_binary_recall(fp=int(attr_FPs[attributeLable]),tp=int(attr_TPs[attributeLable]),fn=int(attr_FNs[attributeLable]))
        # store binary recall of current attribute in dictionary
        attributeRecalls[attributeLable]=curr_attr_recall

    # this loop calculates the numerator of the equation for weighted recall by summing 
# the product of each attributes recall value and its occurence
    for attributeLabel , value in attributeRecalls.items(): 
        weighted_recall+=attributeOccurences[attributeLabel]*value
# stores total number of occurences , which will be used as the denomenator for weighted recall 
    numOccurences=(sum(list(attributeOccurences.values())))
# calculates weighted recall as the sum of the product of each attributes recall and its occurence 
# divided by the total number of occurences
    weighted_recall=weighted_recall/numOccurences

    return weighted_recall



def compute_weighted_f_measure(matrix: pd.DataFrame) -> float:
    # stores dictionary of attribute labels along with associated FNs,FPs and TPs respectively
    attr_FNs=compute_FNs(matrix)
    attr_FPs=compute_FPs(matrix)
    attr_TPs=compute_TPs(matrix)
# variable to store weighted f-measure which will be incremented iteratively once the f-measure of individual attributes has been computed
    weighted_fmeasure=0

    attributeLabels=list(matrix.columns)
# dictionary to store attribute labels and their associated number of occurences
    attributeOccurences={}
    # dictionary to store attribute labels and their associated fmeasure values (F=2.P.R/P+R)
    attributeFmeasure={}

    for attributeLable in attributeLabels:
        # occurence is calculated as TP + FN, 
        attrOccurence=int(attr_TPs[attributeLable] +attr_FNs[attributeLable])
        # store occurence of current attribute in a dictionary
        attributeOccurences[attributeLable]=attrOccurence
        # compute the binary fmeasure of the current attribute 
        curr_attr_recall=compute_binary_f_measure(fp=int(attr_FPs[attributeLable]),tp=int(attr_TPs[attributeLable]),fn=int(attr_FNs[attributeLable]))
        # store binary f-measure of current attribute in dictionary
        attributeFmeasure[attributeLable]=curr_attr_recall

    # this loop calculates the numerator of the equation for weighted f-measure by summing 
# the product of each attributes f-measure value and its occurence
    for attributeLabel , value in attributeFmeasure.items(): 
        weighted_fmeasure+=attributeOccurences[attributeLabel]*value
# stores total number of occurences , which will be used as the denomenator for weighted f-measure 
    numOccurences=(sum(list(attributeOccurences.values())))
# calculates weighted f-measure as the sum of the product of each attributes recall and its occurence 
# divided by the total number of occurences
    weighted_fmeasure=weighted_fmeasure/numOccurences

    return weighted_fmeasure
    


# These functions compute the standard and balanced multiclass accuracies based on the offered confusion matrix.
# You are expected to use appropriately select and use the functions defined previously.
#
# As input, these functions take:
# - matrix - a data frame representing the confusion matrix computed based on the offered series of actual
#            and predicted classes. See confusion_matrix function for description.
# As output, they produce:
# - standard/balanced multiclass accuracy - appropriate evaluation measures created using the
#                                           standard/balanced approach.

def compute_standard_accuracy(matrix: pd.DataFrame) -> float:
# stores dictionary of attribute labels along with associated FNs,FPs and TPs respectively
    attr_FNs=[int(x) for x in list(compute_FNs(matrix).values())]
    attr_TPs=[int(x) for x in list(compute_TPs(matrix).values())]

    tps_occurence=sum(attr_TPs)
    fns_occurence=sum(attr_FNs)
    # occurence is calculated as TP + FN, 
    total_occurence=tps_occurence+fns_occurence

    return tps_occurence/total_occurence

def compute_balanced_accuracy(matrix: pd.DataFrame) -> float:
    # retreivce all fn , tp and fp values and store them in respective arrays 
    attr_FNs=[int(x) for x in list(compute_FNs(matrix).values())]
    attr_TPs=[int(x) for x in list(compute_TPs(matrix).values())]
    attr_FPs=[int(x) for x in list(compute_FPs(matrix).values())]
    # define number of occurences as the sum of each element in each array for tp,fp and fn respectively
    tp_occurence=sum(attr_TPs)
    fp_occurence=sum(attr_FPs)
    fn_occurence=sum(attr_FNs)

    tp_fn_occurence=tp_occurence+fn_occurence

    tn_occurence=tp_occurence+fp_occurence+fn_occurence-tp_fn_occurence

    correct_predictions=(tp_occurence+tn_occurence)
    all_predictions=(tp_occurence+tn_occurence+fp_occurence+fn_occurence)

    return correct_predictions/all_predictions


# In this function you are expected to compute precision, recall, f-measure and accuracy of your classifier using
# the macro average approach.
# At input, the function takes:
# - actual_class - a pandas Series of actual class values
# - predicted_class - a pandas Series of predicted class values
# - class_values - a list of all possible class values (actual and predicted classes are not guaranteed to be complete)
# - confusion_func - function to be invoked to compute the confusion matrix
# Function outputs:
# - computed measures - a dictionary of measures, explicitly listing 'macro_precision', 'macro_recall',
#                       'macro_f_measure', 'weighted_precision', 'weighted_recall', 'weighted_f_measure',
#                       'standard_accuracy' and 'balanced_accuracy'

def evaluate_classification(actual_class: pd.Series, predicted_class: pd.Series, class_values: list[str],
                            confusion_func=confusion_matrix) \
        -> dict[str, float]:
    # Have fun with the computations!
    macro_precision = -1.0
    macro_recall = -1.0
    macro_f_measure = -1.0

    weighted_precision = -1.0
    weighted_recall = -1.0
    weighted_f_measure = -1.0

    standard_accuracy = -1.0
    balanced_accuracy = -1.0
    # once ready, we return the values
    return {'macro_precision': macro_precision, 'macro_recall': macro_recall, 'macro_f_measure': macro_f_measure,
            'weighted_precision': weighted_precision, 'weighted_recall': weighted_recall,
            'weighted_f_measure': weighted_f_measure, 'standard_accuracy': standard_accuracy,
            'balanced_accuracy': balanced_accuracy}
