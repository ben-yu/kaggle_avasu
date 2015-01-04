def clean_df(df):
    pass

def write_submission(submission_number, ids, probs):
    f = open("./submission{0}.csv".format(submission_number),'w')
    f.write("id,click\n")
    for result in zip(ids,probs):
        f.write("{0},{1}\n".format(result[0],round(result[1],4)))
    f.close()

