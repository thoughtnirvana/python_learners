#!/usr/bin/env python
"""
Generic learners. This module defines the learners we use and the data models.
"""
from __future__ import print_function, division
from utils import *
import random, cPickle, itertools, math, binascii, types, os
import sqlite3, targets
from orm.models import Config
import svm, linearutil
from collections import defaultdict
from math import tanh
import pdb

class PickleModelSaver(object):
    """
    Serializes/loads the model in pickle format.
    >>> foo = 'bar'
    >>> p = PickleModelSaver()
    >>> p.save_model(foo, '/tmp/foo.model')
    >>> restored = p.load_model('/tmp/foo.model')
    >>> restored == foo
    True
    >>> os.remove('/tmp/foo.model')
    """

    @staticmethod
    def save_model(model, name):
        """
        Dumps the model in pickle format.
        """
        if isinstance(name, file):
            cPickle.dump(model, name)
        else:
            with open(name, 'w') as dump_file:
                cPickle.dump(model, dump_file)

    @staticmethod
    def load_model(name):
        """
        Loads the model from a serialized file.
        """
        if isinstance(name, file):
            return cPickle.load(name)
        else:
            with open(name) as dump_file:
                return cPickle.load(dump_file)


class GenericDataSet(object):
    """
    Data set interface for a machine learning problem.  It can have following attributes:
    d.examples    A list of examples.  Each one is a list of attribute values.
    d.name        Name of the data set. Generally for output display only.
                  But if the training data set(d.examples) isn't specified, it's used
                  to formulate the name of the model from which the dataset would be loaded.
    d.targets     Set of target attributes.
    Sub-classes can add specialized attributes.
    >>> ex = [['screwed', 'are', 'sad'], ['am', 'happy', 'happy']]
    >>> ds = GenericDataSet(ex)
    >>> len(ds.examples)
    2
    >>> ds.targets
    set(['happy', 'sad'])
    >>> ds.add_example(['foo', 'bar', 'happy'])
    >>> len(ds.examples)
    3
    >>> ds.save_dataset('/tmp/test.ds')
    >>> rds = GenericDataSet(name='/tmp/test.ds')
    >>> rds.examples == ds.examples
    True
    >>> os.remove('/tmp/test.ds')
    """
    def __init__(self, examples=None,  name='', target=-1,
                 loader=PickleModelSaver()):
        """
        Accepts any of DataSet's fields.  Examples can
        also be a string or file from which to parse examples using parse_csv.
        """
        update(self, loader=loader, name=name, target=target)
        # Initialize .examples from string or list or data directory
        if isinstance(examples, str):
            self.examples = parse_csv(examples)
        elif examples is None:
            self.examples = self.load_dataset(name)
        else:
            self.examples = examples
        self.targets = set(e[self.target] for e in self.examples)

    def add_example(self, example):
        """Add an example to the list of examples"""
        self.examples.append(example)
        self.targets.add(example[self.target])

    def __repr__(self):
        return '<%s (%s): %d examples' % (
            self.__class__.__name__, self.name, len(self.examples))

    def save_dataset(self, name=''):
        """
        Serializes the examples from the serialized model.
        """
        name = name or (name(self) + '.model')
        self.loader.save_model(self.examples, name)

    def load_dataset(self, name=''):
        """
        Loads the model from a serialized file.
        """
        name = name or (name(self) + '.model')
        return self.loader.load_model(name)



class SimpleDataSet(GenericDataSet):
    """
    A data set for a machine learning problem.  It has the following fields:
    d.examples    A list of examples.  Each one is a list of attribute values.
    d.name        Name of the data set. Generally for output display only.
                  But if the training data set(d.examples) isn't specified, it's used
                  to formulate the name of the model from which the dataset would be loaded.
    >>> ex = [['screwed', 'are', 'sad'], ['am', 'happy', 'happy']]
    >>> ds = SimpleDataSet(ex)
    >>> len(ds.examples)
    2
    >>> ds.targets
    set(['happy', 'sad'])
    >>> ds.add_example(['foo', 'bar', 'happy'])
    >>> len(ds.examples)
    3
    >>> ds.save_dataset('/tmp/test.ds')
    >>> rds = GenericDataSet(name='/tmp/test.ds')
    >>> rds.examples == ds.examples
    True
    >>> os.remove('/tmp/test.ds')
    """
    # There is no specialized functionality.
    pass


class TextDataSet(GenericDataSet):
    """
    DataSet sutiable for text classification. If strings are passed to it, it can tokenize them into
    individual words.
    It takes following attributes.
    d.examples    A list of examples. Either a list of attributes of text to be tokenized.
    d.name        Name of the data set. Generally for output display only.
                  But if the training data set(d.examples) isn't specified, it's used
                  to formulate the name of the model from which the dataset would be loaded.
    >>> ex = [["I am happy", "happy"], ["I am sad", "sad"]]
    >>> ds = TextDataSet(ex)
    >>> ds.examples
    [['am', 'happy', 'happy'], ['am', 'sad', 'sad']]
    """
    def __init__(self, examples=None, name='', tokenize=True):
        if tokenize:
            examples = map(break_tokens, examples)
        super(TextDataSet, self).__init__(examples, name)


def break_tokens(example):
    """
    If the example is of the form (text, category), break text into individual words.
    Returns (word1, word2, word3...wordn, category)
    >>> break_tokens(["I am happy:)", "happy"])
    ['am', ':)', 'happy', 'happy']
    >>> break_tokens(["I am sad:)", "happy"])
    ['am', ':)', 'sad', 'happy']
    >>> break_tokens(["I", "am", "happy",  "happy"])
    ['I', 'am', 'happy', 'happy']
    >>> break_tokens(["I", "am", "happy:)",  "happy"])
    ['I', 'am', 'happy:)', 'happy']
    >>> break_tokens(["I", "am", "happy"])
    ['I', 'am', 'happy']
    """
    if len(example) == 2 and isinstance(example[0], str):
        text = example[0]; result = example[1]
        example = list(itertools.chain(web_tokenize(text), [result]))
    return example


def parse_csv(text, delim=','):
    r"""
    text is a string consisting of lines, each line has comma-delimited
    fields.  Convert this into a list of lists.  Blank lines are skipped.
    Fields that look like numbers are converted to numbers.
    The delim defaults to ',' but '\t' and None are also reasonable values.
    >>> parse_csv('1, 2, 3 \n 0, 2, na')
    [[1, 2, 3], [0, 2, 'na']]
    """
    lines = [line for line in text.splitlines() if line.strip() is not '']
    return [map(num_or_str, line.split(delim)) for line in lines]


class Learner(object):
    """
    A Learner, or Learning Algorithm, can be trained with a dataset,
    and then asked to predict the target attribute of an example.
    """

    def train(self, dataset):
        abstract

    def predict(self, example, prob=False):
        abstract


class NaiveBayesTextLearner(Learner):
    """
    Uses Bayesian probability to predict the outcome.
    Modified for spam/sentiment classification.
    """

    # Constants to be used as keys in the model.
    TARGET_ATTR = 0
    TARGETS = 2
    TARGET_COUNT = 4
    WORD_COUNT = 8

    def __init__(self, loader=PickleModelSaver()):
        """
        Sets the default attributes.
        >>> nb = NaiveBayesTextLearner()
        """
        super(NaiveBayesTextLearner, self).__init__()
        update(self, loader=loader, _N=defaultdict(dict))
        self._N[self.TARGET_ATTR] = -1
        self._N[self.TARGETS] = set()

    def train(self, dataset):
        """Trains the classifier.Just count the target/val occurences.
        Count how many times each value of each attribute occurs.
        For large models which can't fit in memory, use update_model instead.
        Example:
            nb = NaiveBayesTextLearner()
            for example in examples:
                self.update_model(example)
        If examples is a generator, this would have only one record in memory at a time.
        """
        self._N[self.TARGET_ATTR] = dataset.target
        # Record counts for all words falling in a particular category as well as total.
        self._N[self.TARGETS].update(dataset.targets)
        for example in dataset.examples:
            self.update_model(example)

    def update_model(self, example):
        """
        Updates the count model with the given example.
        >>> nb = NaiveBayesTextLearner()
        >>> nb.update_model([':)', 'excited', 'great', 'happy'])
        >>> nb._N[nb.TARGETS]
        set(['happy'])
        >>> nb._N[nb.TARGET_ATTR]
        -1
        >>> nb._N[nb.TARGET_COUNT]
        1
        >>> nb.update_model([':(', 'sucks', 'screwed', 'sad'])
        >>> nb._N[nb.TARGETS]
        set(['sad', 'happy'])
        >>> nb._N[nb.TARGET_COUNT]
        2
        >>> nb.update_model([':(', 'sucks', 'screwed', 'sad'])
        >>> nb._N[nb.TARGET_COUNT]
        3
        >>> nb._N[nb.TARGETS]
        set(['sad', 'happy'])
        >>> nb._N[nb.TARGET_COUNT]
        3
        >>> nb._N['happy'][nb.TARGET_COUNT]
        1
        >>> nb._N['happy'][nb.WORD_COUNT]
        3
        """
        N = self._N
        target = example[ N[self.TARGET_ATTR] ]
        N[self.TARGETS].add(target)
        N[self.TARGET_COUNT] = N.get(self.TARGET_COUNT, 0) + 1
        # Count for this target.
        N[target][self.TARGET_COUNT] = N[target].get(self.TARGET_COUNT, 0) + 1
        target_idx = N[self.TARGET_ATTR]
        for word in remove_index(target_idx, example):
            # Count for this particular word in this category.
            N[target][word] = N[target].get(word, 0) + 1
            # Total words in this category.
            N[target][self.WORD_COUNT] = N[target].get(self.WORD_COUNT, 0) + 1

    def add_examples(self, examples):
        """
        Adds examples to the model.
        >>> nb = NaiveBayesTextLearner()
        >>> examples = [[':)', 'excited', 'happy'],
        ... [':(', 'sad', 'blues', 'sad']]
        >>> nb.add_examples(examples)
        >>> nb._N['happy'][nb.WORD_COUNT]
        2
        >>> nb._N['happy'][nb.TARGET_COUNT]
        1
        >>> nb._N['sad'][nb.TARGET_COUNT]
        1
        >>> nb._N[nb.TARGET_COUNT]
        2
        >>> nb._N[nb.TARGETS]
        set(['sad', 'happy'])
        """
        for example in examples:
            self.update_model(example)

    def N(self, targetval, attr, default=0):
        """
        Return the count in the training data of this combination.
        >>> nb = NaiveBayesTextLearner()
        >>> nb.update_model([':)', 'excited', 'great', 'happy'])
        >>> nb.update_model([':(', 'sucks', 'screwed', 'sad'])
        >>> nb.update_model([':(', 'sucks', 'screwed', 'sad'])
        >>> nb.update_model([':)', 'excited', 'great', 'happy'])
        >>> nb.N('happy', 'great')
        2
        >>> nb.N('happy', 'graet')
        0
        >>> nb.N('happy', nb.TARGET_COUNT)
        2
        >>> nb.N('happy', nb.WORD_COUNT)
        6
        """
        try:
            return self._N[targetval][attr]
        except KeyError:
            return default

    def P(self, targetval, example):
        """
        Probability that given a document, it falls in a particular category.
        """
        # Bayes' theorem
        # P(A/B) = P(B/A)P(A) / (P(B/A)P(A) + P(B/A')P(A') + ...)
        # numerator = P(B/A)P(A)
        # denominator = P(B/A)P(A) + P(B/A')P(A') + ...
        numerator = self._get_bayes_numerator(targetval, example)
        denominator = self._get_bayes_denominator(example)
        return pow(10, numerator - denominator)

    def _word_prob(self, targetval, word):
        """
        Calculates the probability given word falls in a particular target.
        >>> nb = NaiveBayesTextLearner()
        >>> examples = [[':)', 'excited', 'happy'],
        ... [':(', 'sad', 'blues', 'sad']]
        >>> nb.add_examples(examples)
        >>> nb._word_prob('happy', 'unseen')
        0.33333333333333331
        >>> nb._word_prob('sad', 'sad')
        0.5
        >>> nb._word_prob('happy', 'excited')
        0.66666666666666663
        """
        # Smoothing. Compensate for unknown words by adding 1 to both numerator
        # and denominator.
        word_cat_count = self.N(targetval, word) + 1.0
        word_count = self.N(targetval, self.WORD_COUNT) + 1.0
        return  word_cat_count / word_count

    def _get_bayes_numerator(self, targetval, example):
        """
        Returns the numerator as per the Bayes' theorem.
        >>> nb = NaiveBayesTextLearner()
        >>> examples = [[':)', 'excited', 'happy'],
        ... [':(', 'sad', 'blues', 'sad']]
        >>> nb.add_examples(examples)
        >>> nb.N('happy', nb.TARGET_COUNT)
        1
        >>> nb._N[nb.TARGET_COUNT]
        2
        >>> nb.N('sad', nb.TARGET_COUNT)
        1
        >>> example = ["I", "am", "excited"]
        >>> nb._get_bayes_numerator('happy', example)
        -1.4313637641589876
        >>> nb._get_bayes_numerator('sad', example)
        -2.1072099696478683
        """
        # For calculations, the product can become too small for floating point.
        # Take the log of it to avoid underflow.
        # a * b becomes log(a * b) becomes log a + log b.
        # Bayes' theorem.
        # P(A/B) = P(B/A)P(A) / P(B)
        # This method returns the numerator i.e P(B/A)P(A)
        # where A is targetval and B is example.
        target_prob = math.log10(self.N(targetval, self.TARGET_COUNT) / self._N[self.TARGET_COUNT])
        doc_target_prob = sum(math.log10(p) for p in
                                  (self._word_prob(targetval, e) for e in example))
        return doc_target_prob + target_prob

    def _get_bayes_denominator(self, example):
        """
        Returns the denominator as per the Bayes' theorem.
        >>> nb = NaiveBayesTextLearner()
        >>> examples = [[':)', 'excited', 'happy'],
        ... [':(', 'sad', 'blues', 'sad']]
        >>> nb.add_examples(examples)
        >>> example = ["I", "am", "excited"]
        >>> nb._get_bayes_denominator(example)
        -1.3482420356365643
        """
        # Bayes' theorem
        # P(A/B) = P(B/A)P(A) / (P(B/A)P(A) + P(B/A')P(A') + ...)
        # numerator = P(B/A)P(A)
        # denominator = P(B/A)P(A) + P(B/A')P(A') + ...
        # NOTE: We can't take a log here. log(a+b) doesn't expand to anything.
        # So we first calculate (a + b + c...) and then return its log.
        denominator = 0.0
        for targetval in self._N[self.TARGETS]:
            target_prob = self.N(targetval, self.TARGET_COUNT) / self._N[self.TARGET_COUNT]
            doc_target_prob = product(self._word_prob(targetval, e) for e in example)
            denominator += target_prob * doc_target_prob
        return math.log10(denominator)

    def predict(self, example, prob=False):
        """
        Predict the target value for example. Consider each possible value,
        choose the most likely, by looking at each attribute independently.
        >>> nb = NaiveBayesTextLearner()
        >>> examples = [[':)', 'excited', 'happy'],
        ... [':(', 'sad', 'blues', 'sad']]
        >>> nb.add_examples(examples)
        >>> example = ["I", "am", "excited"]
        >>> nb.predict(example)
        'happy'
        >>> nb.predict(example, True)
        (0.82580645161290289, 'happy')
        """
        possible_values = self._N[self.TARGETS]
        prediction = max((self.P(targetval, example), targetval)
                         for targetval in possible_values)
        # Return either (class, prob) or just class.
        if not prob: res = prediction[1]
        else:
            # For consistency's sake, all learners should have the result
            # first followed by probability - (class, prob)
            res = (prediction[1], prediction[0])
        return res

    def save_model(self, name=''):
        """
        Serializes the examples from the serialized model.
        >>> nb = NaiveBayesTextLearner()
        >>> examples = [[':)', 'excited', 'happy'],
        ... [':(', 'sad', 'blues', 'sad']]
        >>> nb.add_examples(examples)
        >>> example = ["I", "am", "excited"]
        >>> nb.predict(example)
        'happy'
        >>> nb.predict(example, True)
        (0.82580645161290289, 'happy')
        >>> nb.save_model(name='/tmp/bayes.model')
        >>> rnb = NaiveBayesTextLearner()
        >>> rnb.load_model(name='/tmp/bayes.model')
        >>> rnb.predict(example, True)
        (0.82580645161290289, 'happy')
        >>> os.remove('/tmp/bayes.model')
        """
        name = name or (name(self) + '.model')
        self.loader.save_model(self._N, name)

    def load_model(self, name=''):
        """
        Loads the model from a serialized file.
        """
        name = name or (name(self) + '.model')
        self._N.update(self.loader.load_model(name))


class SentimentClassifier(NaiveBayesTextLearner):
    """
    Does sentiment classification. If the difference in the probabilities for the given
    targets are below the given threshold, return a default.
    For example, if (0.4, good) and (0.39, bad) are respective probabilities and the threshold
    is 0.1, classify it as default.
    """
    def __init__(self, threshold=0.1, default='neutral'):
        super(SentimentClassifier, self).__init__()
        update(self, threshold=threshold, default=default)

    def predict(self, example, prob=False):
        """
        Predicts the target value for example. Consider each possible value,
        choose the most likely, by looking at each attribute independently.
        """
        possible_values = self._N[self.TARGETS]
        prob_gen = ((self.P(targetval, example), targetval)
                        for targetval in possible_values)
        prediction = sorted(prob_gen, reverse=True)
        # Check if the best class is more probable than the second best as per the threshold.
        if (prediction[0][0] - prediction[1][0]) > self.threshold:
            # The class with highest probability.
            prediction = prediction[0]
        else:
            # Else the default value.
            prediction = ('-1', self.default)
        # Return either (class, prob) or just class.
        if not prob: res = prediction[1]
        else:
            # For consistency's sake, all learners should have the result
            # first followed by probability - (class, prob)
            res = (prediction[1], prediction[0])
        return res


class DigitizingLearner(Learner):
    """
    Generic base class for classifier which expects numbers as features
    and results. To be extended by concrete implementations.
    """

    def __init__(self, lower=-1, upper=1):
        """
        Initialize the max and min later used for scaling input.
        """
        super(DigitizingLearner, self).__init__()
        update(self, lower=lower, upper=upper)

    def train(self, dataset):
        """
        Trains the svm classifier. Converts words to real numbers for training
        as SVM expects only numbers.
        """
        target = dataset.target
        # Calculate the crc32 for the given words. SVM works only for numbers.
        self.results = []
        self.observations = []
        # Break training data into results and observations and convert them to real numbers.
        max_val = float('-inf'); min_val = float('inf')
        for item in dataset.examples:
            # Convert result to num. Save num->result for showing predictions.
            r = targets.target_to_num[ item[target].strip() ]
            self.results.append(r)
            items_to_num = [binascii.crc32( u_to_ascii(v) )
                                for v in remove_index(target, item)]
            # Set the max and min.
            for num in items_to_num:
                if num > max_val: max_val = num
                if num < min_val: min_val = num
            self.observations.append(items_to_num)
        update(self, max_val=max_val, min_val=min_val)
        self.observations = self.scale_data(self.observations)

    def scale_data(self, observations):
        """
        Scales the observations to normalize very large and small numbers.
        """
        max_min_diff = self.max_val - self.min_val
        # Subtract the min from each item and divide the difference by max-min
        # to normalize all values to fall between lower and upper.
        def process_item(item):
            if item == self.min_val:
                return self.lower
            elif item == self.max_val:
                return self.upper
            else:
                return self.lower + (self.upper - self.lower) *\
                        (item - self.min_val) / max_min_diff
        scaled_data = [[process_item(item) for item in row]
                            for row in observations]
        return scaled_data

    def save_model(self, file_name):
        """
        Saves the svm model to the given file.
        """
        key = 'model.' + os.path.basename(file_name)
        # Load the model properties required for scaling
        # the input.
        Config(key=key+'.max_val', value=self.max_val).save()
        Config(key=key+'.min_val', value=self.min_val).save()
        Config(key=key+'.lower', value=self.lower).save()
        Config(key=key+'.upper', value=self.upper).save()

    def load_model(self, file_name):
        """
        Loads the svm model from the given file.
        """
        key = 'model.' + os.path.basename(file_name)
        # Load the model properties required for scaling
        # the input.
        self.max_val = float( Config.objects.get(key=key+'.max_val').value )
        self.min_val = float( Config.objects.get(key=key+'.min_val').value )
        self.lower = float( Config.objects.get(key=key+'.lower').value )
        self.upper = float( Config.objects.get(key=key+'.upper').value )


class SvmLearner(DigitizingLearner):
    """
    Uses SVM(Support Vector Machines) for predictions.
    """

    def train(self, dataset):
        """
        Trains the svm classifier. Converts words to real numbers for training
        as SVM expects only numbers.
        """
        super(SvmLearner, self).train(dataset)
        prob  = svm.svm_problem(self.results, self.observations)
        param = svm.svm_parameter(kernel_type=svm.LINEAR, C=10, probability=1)
        self.model = svm.svm_model(prob, param)

    def predict(self, example, prob=False):
        """
        Predict the target value for example. Consider each possible value,
        choose the most likely, by looking at each attribute independently.
        """
        # Convert words to numbers and normalize them for prediction.
        # Algorithm is a bit inefficient but considering small number of
        # features in example, it should be reasonably fast.
        example = [binascii.crc32( u_to_ascii(e) ) for e in example]
        example = self.scale_data([example])[0]
        # Predict on scaled data.
        prediction = self.model.predict_probability(example)
        # Convert predicted number back to corresponding label.
        res = []
        res = max(((targets.num_to_target[k], v)
                        for k, v in prediction[1].items()),
                  key=lambda item: item[1])
        if not prob: res = res[0]
        return res

    def save_model(self, file_name=''):
        """
        Saves the svm model to the given file.
        """
        file_name = file_name or (name(self) + '.model')
        self.model.save(file_name)
        super(SvmLearner, self).save_model(file_name)

    def load_model(self, file_name=''):
        """
        Loads the svm model from the given file.
        """
        file_name = file_name or (name(self) + '.model')
        self.model = svm.svm_model(file_name)
        super(SvmLearner, self).load_model(file_name)


class LinearLearner(DigitizingLearner):
    """
    Uses linear methods(logistic regressions, regular regressions et al. defined in
    liblinear) for predictions.
    """

    def train(self, dataset):
        """
        Trains the svm classifier. Converts words to real numbers for training
        as SVM expects only numbers.
        """
        super(LinearLearner, self).train(dataset)
        prob  = linearutil.problem(self.results, self.observations)
        # TODO: I am not sure about various regression models.
        # I chose "-s 0" because it's a regularized logistic regression
        # and supports probability prediction. Need to read on various
        # regression models and decide upon the best.
        param = linearutil.parameter("-s 0")
        self.model = linearutil.train(prob, param)

    def predict(self, example, prob=False):
        """
        Predict the target value for example. Consider each possible value,
        choose the most likely, by looking at each attribute independently.
        """
        # Convert words to numbers and normalize them for prediction.
        # Algorithm is a bit inefficient but considering small number of
        # features in example, it should be reasonably fast.
        example = [binascii.crc32( u_to_ascii(e) ) for e in example]
        example = self.scale_data([example])[0]
        # Predict on scaled data.
        _, _, p_prob = linearutil.predict([], [example], self.model, "-b 1")
        prediction = max(
                    zip(self.model.get_labels(), p_prob[0]),
                    key=lambda item:item[1]
                    )
        # Convert predicted number back to corresponding label.
        res = (targets.num_to_target[ prediction[0] ], prediction[1])
        if not prob: res = res[0]
        return res

    def save_model(self, file_name=''):
        """
        Saves the svm model to the given file.
        """
        file_name = file_name or (name(self) + '.model')
        linearutil.save_model(file_name, self.model)
        super(LinearLearner, self).save_model(file_name)

    def load_model(self, file_name=''):
        """
        Loads the svm model from the given file.
        """
        file_name = file_name or (name(self) + '.model')
        self.model = linearutil.load_model(file_name)
        super(LinearLearner, self).load_model(file_name)


class NeuralNetLearner(Learner):
    """
    A multi-layer perceptron neural net implementation.
    All the freely available neural net implementations have fixed
    number of inputs which won't work with the sentiment classification
    where the features(words) are variable for every tweet/blog.

    For ease of implementation, only one hidden layers is created. This
    suffices for our use case. More general Nerual Net implementations are
    avaialable on pypi.
    """

    # Symbolic constants for table names.
    INPUT = 'input_hidden'
    OUTPUT = 'hidden_output'

    def __init__(self, db_name, create_layers=False, target=-1):
        """
        Initializes the db to be used for persisting the neural net.
        """
        self.db_name = db_name
        self.con = sqlite3.connect(db_name)
        self.targets = set()
        self.target = target
        # Either create the db, or load given db for inference.
        if create_layers: self.create_layers()
        else: self.load_model(db_name)

    def __del__(self):
        self.con.close()

    def create_layers(self):
        """
        Creates the input, hidden and output layers.
        """
        # Create tables.
        self.con.execute('create table hidden(key)')
        self.con.execute('create table %s(from_id, to_id, strength)' % self.INPUT)
        self.con.execute('create table %s(from_id, to_id, strength)' % self.OUTPUT)
        # Create indices.
        self.con.execute('create index hidden_key on hidden(key)')
        self.con.execute('create index input_key on %s(from_id, to_id)' % self.INPUT)
        self.con.execute('create index output_key on %s(from_id, to_id)' % self.OUTPUT)
        self.con.commit()

    def get_strength(self, from_id, to_id, layer):
        """
        Returns the strength between two given ids. Returns a default
        value if there are no existing connections.
        For links from input to hidden layers, the default is -0.2 - new
        features contribute negatively.
        For links from hidden to output, the default is 0.
        """
        res = self.con.execute('select strength from %s where from_id=? and to_id=?' %
                                layer, (from_id, to_id)).fetchone()
        default = {self.INPUT: -0.2, self.OUTPUT: 0}
        if res is None: res = default[layer]
        else: res = res[0]
        return res

    def set_strength(self, from_id, to_id, layer, strength):
        """
        Sets the strength. If link exists, updates it; else
        creates new link.
        """
        res = self.con.execute('select rowid from %s where from_id=? and to_id=?' %
                layer, (from_id, to_id)).fetchone()
        if res is None:
            # No link. Create new link.
            self.con.execute('insert into %s (from_id, to_id, strength) values (?, ?, ?)' %
                layer, (from_id, to_id, strength))
        else:
            # Existing link. Update strength.
            rowid = res[0]
            self.con.execute('update %s set strength=? where rowid=?' %
            layer, (strength, rowid))

    def generate_hidden(self, words, results, commit=True):
        """
        Generates the hidden nodes where 'words' is the input and 'result'
        is the classification.
        """
        # Check if we already created a node for this set of words
        key = '_'.join(sorted(str(wi) for wi in words))
        res = self.con.execute(
            'select rowid from hidden where key=?', (key,)).fetchone()
        # If not, create it
        if res is None:
            cur = self.con.execute(
                'insert into hidden (key) values (?)', (key,))
            hidden_id = cur.lastrowid
            # Put in some default weights
            for word in words:
                self.set_strength(word, hidden_id, self.INPUT, 1.0 / len(words))
            for r in results:
                # NOTE: What should be the default weight? 0.1 is just a random guess.
                self.set_strength(hidden_id, r, self.OUTPUT, 0.1)
            if commit: self.con.commit()

    def get_hidden_ids(self, words):
        """
        Returns the hidden ids activated by given words(features).
        """
        res = set()
        for word in words:
            cur = self.con.execute(
                'select to_id from input_hidden where from_id=?', (word,))
            for row in cur: res.add(row[0])
        return res

    def setup_network(self, words, results):
        """
        Constructs the relevant network.
        """
        self.words = words
        self.hidden_ids = self.get_hidden_ids(words)
        self.results = results
        # Node outputs.
        # Input activation. A feature is there, or it's not.
        # 1 indicates presence.
        self.ai = dict((word, 1.0) for word in words)
        # Hidden activation. Sigmoid activation based on inputs.
        # Actual value calculated later in self.feed_forward.
        self.ah = dict((hidden, 1.0) for hidden in self.hidden_ids)
        # Output activation. Sigmoid activation based on hidden inputs.
        # Actual value calculated later in self.feed_forward.
        self.ao = dict((r, 1.0) for r in results)
        # Create weights matrix.
        self.wi = dict((word,
                            dict((hidden, self.get_strength(word, hidden, self.INPUT))
                                    for hidden in self.hidden_ids))
                            for word in self.words)
        self.wo = dict((hidden,
                            dict((r, self.get_strength(hidden, r, self.OUTPUT))
                                    for r in self.results))
                            for hidden in self.hidden_ids)

    def feed_forward(self):
        """
        Takes a list of inputs, pushes them through the network and
        returns the output of output nodes.
        """
        # Hidden activations.
        for hidden in self.hidden_ids:
            sum = 0.0
            for word in self.words:
                sum += self.ai[word] * self.wi[word][hidden]
            # Sigmoid activation.
            self.ah[hidden] = tanh(sum)
        # Output activations.
        for r in self.results:
            sum = 0.0
            for hidden in self.hidden_ids:
                sum += self.ah[hidden] * self.wo[hidden][r]
            # Sigmoid activation.
            self.ao[r] = tanh(sum)
        # Return output of output nodes.
        return self.ao.copy()

    def predict(self, example, prob=False):
        """
        Sets up the network for given words and results.
        Returns the output after running feedforward.
        """
        results = self.targets
        self.setup_network(example, results)
        res = max(self.feed_forward().items(), key=lambda item: item[1])
        if not res[1]: return None
        if not prob: res = res[0]
        return res

    def back_propagate(self, targets, N=0.5):
        """
        Based on the selected targets, update the input weights.
        """
        # Calculate errors for output.
        output_deltas = dict((r, 0.0) for r in self.results)
        for r in self.results:
            error = targets.get(r, 0.0) - self.ao[r]
            output_deltas[r] = dtanh(self.ao[r]) * error
        # Calculate errors for hidden layer.
        hidden_deltas = dict((hidden, 0.0) for hidden in self.hidden_ids)
        for hidden in self.hidden_ids:
            error = 0.0
            for r in self.results:
                error += output_deltas[r] * self.wo[hidden][r]
            hidden_deltas[hidden] = dtanh(self.ah[hidden]) * error
        # Update output weights.
        for hidden in self.hidden_ids:
            for r in self.results:
                change = output_deltas[r] * self.ah[hidden]
                self.wo[hidden][r] = self.wo[hidden][r] + N * change
        # Update input weights.
        for word in self.words:
            for hidden in self.hidden_ids:
                change = hidden_deltas[hidden] * self.ai[word]
                self.wi[word][hidden] = self.wi[word][hidden] + N * change

    def train_net(self, features, result):
        """
        Re-classify a given example. Backpropagate the error deltas.
        """
        # Setup the network and feed forward.
        self.setup_network(features, self.targets)
        self.feed_forward()
        # Backpropagation is used to incorporate feedbacks. 
        # If the user marks something belonging to cat2 when
        # the categorization is cat1, the changes are backpropagated.
        self.back_propagate({result: 1.0})
        self.update_database()

    def train(self, dataset):
        """
        Trains the neural net for the given dataset.
        """
        self.target = getattr(dataset, 'target', -1)
        for example in dataset.examples:
            self.update_model(example, False)
        # Commit the model. SQLite performs better in transactions
        # compared to individual commits.
        self.con.commit()

    def update_model(self, example, commit=True):
        """
        Adds given example to the neural net.
        """
        words = remove_index(self.target, example)
        result = example[self.target]
        self.targets.add(result)
        # Generate hidden nodes if not already there.
        # If the commit is False, the onus lies on the 
        # application code to commit when done.
        self.generate_hidden(words, [result], commit)

    def update_database(self, commit=True):
        """
        Update the database with the new weights.
        """
        for word in self.words:
            for hidden in self.hidden_ids:
                self.set_strength(word, hidden,
                                  self.INPUT, self.wi[word][hidden])
        for hidden in self.hidden_ids:
            for r in self.results:
                self.set_strength(hidden, r,
                                 self.OUTPUT, self.wo[hidden][r])
        if commit: self.con.commit()

    def save_model(self, db_name=''):
        """
        Commit wrapper.
        """
        # Check if the correct model is being saved and commit.
        if db_name and not os.path.samefile(db_name, self.db_name):
            raise ValueError("Model in use is {0}. Tried to save {1}".format(
                                                                        self.db_name,
                                                                        file_name))
        self.con.commit()

    def load_model(self, db_name):
        """
        Loads given db as inference model.
        Initializes targets.
        """
        # Check if a different db is to be loaded.
        if not os.path.samefile(db_name, self.db_name):
            self.con = sqlite3.connect(db_name)
            self.db_file = db_file
        # Populate targets.
        targets = self.targets
        cur = self.con.execute(
            'select distinct to_id from %s' % self.OUTPUT)
        for row in cur: targets.add( row[0] )


class EnsembleLearner(Learner):
    """
    Given a list of learning algorithms, have them vote.
    """

    def __init__(self, learners=[], weights=[], estimate=weighted_mean):
        if learners and not weights: weights = [1] * len(learners)
        update(self, learners=learners, weights=weights, estimate=estimate)

    def train(self, dataset):
        for learner in self.learners:
            learner.train(dataset)

    def predict(self, example):
        return self.estimate((w, learner.predict(example)) for (w, learner)
                            in zip(self.weights, self.learners))


def test(learner, dataset, examples=None, verbose=0):
    """
    Return the proportion of the examples that are correctly predicted.
    Assumes the learner has already been trained.
    """
    if examples == None: examples = dataset.examples
    if len(examples) == 0: return 0.0
    right = 0.0
    for example in examples:
        desired = example[dataset.target]
        output = learner.predict(remove_index(dataset.target, example))
        if output == desired:
            right += 1
            if verbose >= 2:
                print('   OK: got %s for %s' % (desired, example))
        elif verbose:
            print('WRONG: got %s, expected %s for %s' % (
               output, desired, example))
    return right / len(examples)


def train_and_test(learner, dataset, start, end):
    """
    Reserve dataset.examples[start:end] for test; train on the remainder.
    Return the proportion of examples correct on the test examples.
    """
    examples = dataset.examples
    try:
        dataset.examples = examples[:start] + examples[end:]
        learner.dataset = dataset
        learner.train(dataset)
        return test(learner, dataset, examples[start:end])
    finally:
        dataset.examples = examples


def cross_validation(learner, dataset, k=10, trials=1):
    """
    Do k-fold cross_validate and return their mean.
    That is, keep out 1/k of the examples for testing on each of k runs.
    Shuffle the examples first; If trials>1, average over several shuffles.
    """
    if k == None:
        k = len(dataset.examples)
    if trials > 1:
        return mean([cross_validation(learner, dataset, k, trials=1)
                     for t in range(trials)])
    else:
        n = len(dataset.examples)
        random.shuffle(dataset.examples)
        return mean([train_and_test(learner, dataset, i*(n//k), (i+1)*(n//k))
                     for i in range(k)])


def leave1out(learner, dataset):
    "Leave one out cross-validation over the dataset."
    return cross_validation(learner, dataset, k=len(dataset.examples))

# Artificial, generated  examples.

def Majority(k, n):
    """
    Return a DataSet with n k-bit examples of the majority problem:
    k random bits followed by a 1 if more than half the bits are 1, else 0.
    """
    examples = []
    for i in range(n):
        bits = [random.choice([0, 1]) for i in range(k)]
        bits.append(sum(bits) > k//2)
        examples.append(bits)
    return SimpleDataSet(name="majority", examples=examples)

def Parity(k, n, name="parity"):
    """
    Return a DataSet with n k-bit examples of the parity problem:
    k random bits followed by a 1 if an odd number of bits are 1, else 0.
    """
    examples = []
    for i in range(n):
        bits = [random.choice([0, 1]) for i in range(k)]
        bits.append(sum(bits) % 2)
        examples.append(bits)
    return SimpleDataSet(name=name, examples=examples)

def Xor(n):
    "Return a DataSet with n examples of 2-input xor."
    return Parity(2, n, name="xor")


def compare(algorithms=[NaiveBayesTextLearner,],
            datasets=[Majority(7, 100), Parity(7, 100), Xor(100)],
            k=10, trials=1):
    """
    Compare various learners on various datasets using cross-validation.
    Print results as a table.
    """
    print_table([[a.__name__.replace('Learner', '')] +
                 [cross_validation(a(), d, k, trials) for d in datasets]
                 for a in algorithms],
                header=[''] + [d.name[0:7] for d in datasets])
