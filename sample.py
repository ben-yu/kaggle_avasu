from csv import DictReader
import csv

train = 'data/train.csv'


x_names = ["id","click","hour","C1","banner_pos","site_id","site_domain","site_category","app_id","app_domain","app_category","device_id","device_ip","device_model","device_type","device_conn_type","C14","C15","C16","C17","C18","C19","C20","C21"]

train_file = open(train,'rb')
holdout = 40

sample = open("sample.csv", 'w')
sample.write(next(train_file)) # write first row

for t, row in enumerate(train_file):
    if (holdout and t % holdout == 0):
        sample.write(row)