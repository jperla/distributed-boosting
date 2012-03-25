#!/usr/bin/env python
import csv
import copy
import math
import random
from itertools import izip

import numpy as np

def is_probability_distribution(d):
    if not all([i >= 0 for i in d]):
        return False
    if abs(1.0 - sum(d)) > 0.0000000001:
        return False
    return True

class WeakLearner(object):
    def train(self, distribution, data):
        """Accepts probability distribution (list of reals that sums to 1).
            Also accepts data which is list of rows (lists of features).

            Returns a dictionary which describes a classifier.
        """
        assert is_probability_distribution(distribution)

    def classify(self, params, data):
        """Accepts a dictionary that describes a classifier.
            Also accepts a list of rows (lists of features).
            Return a numpy array of classifications (+1, -1, or 0).
        """
        pass

class StumpLearner(WeakLearner):
    def train(self, distribution, data):
        """Tries a random row, and tries a random value"""
        WeakLearner.train(self, distribution, data)

        num_random_classifiers = 100
        random_classifiers = [self.build_random_classifier(data)
                                    for i in xrange(num_random_classifiers)]
        best = list(sorted([self.correct_classifier(distribution, c, data)
                                            for c in random_classifiers]))[0]
        return best[1]

    def correct_classifier(self, distribution, classifier, data):
        """Accepts a distribution, 
                dictionary describing classifier,
                and data.
            Returns 2-tuple of (error, new classifier) 
                    (opposite of original).
        """
        # training error
        c = copy.deepcopy(classifier)
        predicted = self.classify(c, data)
        error = calculate_error(distribution, predicted, data)
        if error > 0.5:
            c['multiplier'] = -1.0
            error = 1.0 - error
        return error, c


    def build_random_classifier(self, data):
        """Accepts data, 
            makes a random classifiers that matches a value on a random row.
            Returns classifier parameters (dictionary).
        """
        f = len(data[0]) - 1
        r = random.randint(0, f-1) #random row
        values = list(set([d[r] for d in data]))
        v = random.sample(values, 1)[0] # random value
        return {'row': r, 'value': v, 'multiplier': 1}

    def classify(self, params, data):
        labels = []

        row = params['row']
        value = params['value']
        multiplier = params['multiplier']
        
        for d in data:
            l = (1.0 if d[row] == value else -1.0)
            labels.append(l * multiplier)
        return np.array(labels)

def calculate_normalizer(error):
    return 2 * math.sqrt(error * (1.0 - error))

def calculate_alpha(error):
    return 0.5 * math.log((1.0 - error) / error)

def calculate_error(distribution, predicted, training_data):
    """Accept predicted labels (list of +1 and -1),
        and training data list of rows (list of features).
        Returns error rate of predictions versus data.
    """
    assert is_probability_distribution(distribution)
    assert len(predicted) == len(training_data)
    assert len(distribution) == len(training_data)
    assert all([t[-1] in [-1, 1] for t in training_data])
    assert all([t in [-1, 0, 1] for t in predicted])

    training_labels = [t[-1] for t in training_data]
    labels = np.array([(1.0 if p != t else 0.0)
                        for p,t in izip(predicted, training_labels)])

    return np.sum(distribution * labels)


def boost(D1, weak_learner, training_data, stopping_criterion):
    """Accepts an initial probability distribution D1.
        Also accepts a weak learner object.
        Finally, accepts a method stopping_criterion that 
            accepts round t and error
            and returns true or false about whether it should stop.

        Returns a final hypothesis.
    """
    assert is_probability_distribution(D1)

    D = np.array(D1)

    final_hypothesis = []
    
    t = 0
    while True:
        t += 1
        # learn and predict
        classifier = weak_learner.train(D, training_data)
        predicted = weak_learner.classify(classifier, training_data)

        # calculate epsilon_t, alpha_t, and Z_t
        error = calculate_error(D, predicted, training_data)
        assert error <= 0.5
        alpha = calculate_alpha(error)
        normalizer = calculate_normalizer(error)

        #update D
        y = np.array([d[-1] for d in training_data])
        D = D * np.exp(-1 * alpha * y * np.array(predicted)) / normalizer
        print 'D', np.sum(D), 'E:', error
        #import pdb; pdb.set_trace()

        #TODO: jperla: why doesn't this normalize correctly?
        assert abs(np.sum(D) - 1.0) < 0.001

        #D = D / np.sum(D)
        assert is_probability_distribution(D)

        final_hypothesis.append((alpha, classifier))

        if stopping_criterion(t, error):
            break

    return final_hypothesis

def combined_classify(weak_learner, classifier, data):
    """Accepts a weak learner object, 
        and a classifier 
            (list of 2-tuples of weights and weak classifier descriptions)
        Returns a classification of data.
    """
    return np.sum([(a * weak_learner.classify(c, data)) 
                                    for a,c in classifier], axis=0)


def read_adult_data(filename):
    """Reads in the adult dataset type.
        Returns list of rows (list of string features).
            Last element of each row is +1 or -1 classification.
    """
    reader = csv.reader(open(filename, 'r'))
    data = []
    for row in reader:
        row = [r.strip('\r\n ') for r in row]
        row[-1] = (1 if row[-1].startswith('>50K') else -1)
        data.append(row)
    return data

def uniform_distribution(m):
    return np.ones(m) / m

if __name__=='__main__':
    #load data
    training_data = read_adult_data('adult.data')
    test_data = read_adult_data('adult.test')

    # set up arguments
    m = len(training_data)
    D1 = uniform_distribution(m)
    stump_learner = StumpLearner()
    stopper = lambda t,e: t > 1000 #e == 0

    #import pdb; pdb.set_trace()

    # boost
    combined_classifier = boost(D1, stump_learner, training_data, stopper)

    # training error
    Fx = combined_classify(stump_learner, 
                                  combined_classifier, 
                                  training_data)
    assert len(Fx) == len(training_data)
    predicted = np.sign(Fx)
    training_error = calculate_error(D1, predicted, training_data)
    print training_error

    # test error
    Fx = combined_classify(stump_learner, 
                           combined_classifier, 
                           test_data)
    predicted = np.sign(Fx)
    m = len(predicted)
    test_error = calculate_error(uniform_distribution(m), predicted, test_data)
    print test_error


