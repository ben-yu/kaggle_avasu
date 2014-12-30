import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import cross_val_score
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn import svm

def write_submission(submission_number, ids, probs):
    f = open("./submission{0}.csv".format(submission_number),'w')
    f.write("id,click\n")
    for result in zip(ids,probs):
        f.write("{0},{1}\n".format(result[0],round(result[1],4)))
    f.close()

def main():
    tp = pd.read_csv('./data/train',dtype=unicode, iterator=True, chunksize=1000)
    train_df = concat(list(tp), ignore_index=True)

    tp1 = pd.read_csv('./data/test',dtype=unicode, iterator=True, chunksize=1000)
    test_df = concat(list(tp1), ignore_index=True)

    #id,hour,C1,banner_pos,site_id,site_domain,site_category,app_id,app_domain,app_category,device_id,device_ip,device_model,device_type,device_conn_type,C14,C15,C16,C17,C18,C19,C20,C21
    #10000174058809263569,14103100,1005,0,235ba823,f6ebf28e,f028772b,ecad2386,7801e8d9,07d7df22,a99f214a,69f45779,0eb711ec,1,0,8330,320,50,761,3,175,100075,23

    rfc = RandomForestClassifier()

    columns = ['hour','C1','banner_pos','site_id','site_domain','site_category','app_id','app_domain',
            'app_category','device_id','device_ip','device_model','device_type','device_conn_type',
            'C14','C15','C16','C17','C18','C19','C20','C21']
    labels = train_df['click'].values
    features = train_df[list(columns)].T.to_dict().values()

    vectorizer = DV( sparse = False )
    vec_train = vectorizer.fit_transform(features)
    vec_test = vectorizer.transform(test_df[list(columns)].T.to_dict().values())

    rfc.fit(vec_train, labels)
    click_probs = [prob[0] for prob in rfc.predict_proba(vec_test)]
    write_submission(0,test_df['id'].values,click_probs)

    # clf = svm.SVC()
    # et_score = cross_val_score(clf, vec_train, labels).mean()
    # print("SVM: {0}".format(et_score))

    # nb = GaussianNB()
    # et_score = cross_val_score(nb, vec_train, labels).mean()
    # print("Naive Bayes: {0}".format(et_score))


if __name__ == '__main__':
    main()
