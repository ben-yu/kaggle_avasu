import pandas as pd
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, Perceptron
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.externals import joblib
from docopt import docopt
import sys


def main(model_type):
    train_reader = pd.read_csv('./sample_data/train.csv', dtype=unicode, chunksize=20)

    model_map = {
        'sgd': SGDClassifier,
        'multi': MultinomialNB,
        'bern': BernoulliNB,
        'prcpt': Perceptron,
        'pa': PassiveAggressiveClassifier
    }

    classifier = model_map[model_type]()

    vectorizer = DV(sparse=False)
    for chunk in train_reader:
        train_data = chunk.T.to_dict().values()
        vectorizer.fit(train_data)

    for chunk in train_reader:
        train_data = chunk.T.to_dict().values()
        labels = chunk['click'].T.to_dict().values()
        vec_train = vectorizer.transform(train_data)
        classifier.partial_fit(vec_train, labels, classes=['0', '1'])

    joblib.dump(classifier, 'models/online_model_{0}.pkl'.format(model_type))


if __name__ == '__main__':
    usage = '''Train an online model to predict Avasu Ad CTR
        Usage:
        %(program_name)s --model=<m>
        %(program_name)s (-h | --help)
        Options:
        -h --help                  Show this screen
        --model=<m>                Train either a 'MultinomialNB', 'BernoulliNB', 'Perceptron', 'SGD', or 'PassiveAggressive'
        ''' % {'program_name': sys.argv[0]}

    arguments = docopt(usage)

    main(
        arguments.get('--model', 'SGD')
    )
