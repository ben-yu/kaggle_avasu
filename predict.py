import pandas as pd
from sklearn.feature_extraction import DictVectorizer as DV
from docopt import docopt
import sys
from lib.util import write_submission
from sklearn.externals import joblib

def main(submission_number, model_name):
    tp1 = pd.read_csv('./data/test',dtype=unicode, iterator=True, chunksize=1000)
    test_df = pd.concat(list(tp1), ignore_index=True)

    classifier = joblib.load('./models/{0}.pkl'.format(model_name))

    vectorizer = DV(sparse=False)
    vec_test = vectorizer.transform(test_df)

    click_probs = [prob[0] for prob in classifier.predict_proba(vec_test)]
    write_submission(submission_number,test_df['id'].values,click_probs)


if __name__ == '__main__':
    usage = '''Generate Predictions from a model
        Usage:
        %(program_name)s --submission_num=<s> --model_name=<m>
        %(program_name)s (-h | --help)
        Options:
        -h --help                  Show this screen
        --submission_number=<s>    Submission number
        --model_name=<p>           Name of picked model
        ''' % {'program_name': sys.argv[0]}

    arguments = docopt(usage)

    main(
        arguments.get('--submission_number', 0),
        arguments.get('--model_name', 'online_model')
    )
