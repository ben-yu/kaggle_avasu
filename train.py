import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.svm import SVC
from docopt import docopt
from sklearn.externals import joblib
import sys


def main(model_type):
    tp = pd.read_csv('./sample_data/train.csv', dtype=unicode, iterator=True, chunksize=1000)
    train_df = pd.concat(list(tp), ignore_index=True)

    model_map = {
        'rfc': RandomForestClassifier,
        'svc': SVC
    }

    classifier = model_map[model_type]()
    train_data = train_df.T.to_dict().values()
    labels = train_df['click'].values

    vectorizer = DV(sparse=False)
    vec_train = vectorizer.fit_transform(train_data)

    classifier.fit(vec_train, labels)
    joblib.dump(classifier, 'models/model_{0}.pkl'.format(model_type))


if __name__ == '__main__':
    usage = '''Train a model to predict Avasu Ad CTR
        Usage:
        %(program_name)s --model=<m>
        %(program_name)s (-h | --help)
        Options:
        -h --help                  Show this screen
        --submission_number=<s>    Submission number
        --model=<m>                 Type of model to train
        ''' % {'program_name': sys.argv[0]}

    arguments = docopt(usage)

    main(
        arguments.get('--model', 'rfc')
    )
