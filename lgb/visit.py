# coding:utf-8
import pandas as pd
import os
import numpy as np

TRAIN_DIR = r'../train_visit'


def get_list(path):
    if not os.path.exists(path):
        print("{} is not exists".format(path))
        exit(-1)

    file_names = os.listdir(path)
    return file_names

def decode_txt(file_names,path = TRAIN_DIR):
    '''
    AreaID_CategoryID
    000001_001.txt，表示该文件记录区域为000001的用户到访行为，该区域的功能类别为居住区。文件格式为：
    USERID \t day_a&hour_x|hour_y|..., day_b&hour_x|hour_z|...
    例如：aff296a485010219 \t 20190129&21|22,20190218&19|20|21  表示用户aff296a485010219在2019年01月29日的21点、22点， 2019年02月18日的19点、20点和21点到访过该区域测试集


    制作两张表：
    第一张对应AreaID与用户ID的访问数的关系：visit_record
    第二张表格对应AreaID与Label之间的关系：area_classfication
    :param file_names:文件名
    :path:Visit Dir
    '''
    visit_record = []
    area_classfication = []

    visit_columns = ['AreaID','UserID','Day'] + [str(i) for i in range(24)]
    classfier_columns = ['AreaID','CategoryID']

    for name in file_names:
        AreaID = name[:6]
        CategoryID = name[7:10]
        area_classfication.append([AreaID,CategoryID])

        with open(os.path.join(path,name),'r') as f:
            lines = f.readlines()
        f.close()
        for line in lines:
            line = line.strip()
            UserID,Records = line.split('\t')
            Records = Records.split(',')
            for records in Records:
                day,hours = records.split('&')
                hours = hours.split('|')
                hour_record = [0 for i in range(24)]
                for hour in hours:
                    hour_record[int(hour)] +=1

                visit_record.append([AreaID,UserID,day] + hour_record)

    visit_record = pd.DataFrame(visit_record,columns=visit_columns)
    visit_record.to_csv('visit_record.csv',index=False)

    area_classfication = pd.DataFrame(area_classfication,columns=classfier_columns)
    area_classfication.to_csv('area_calssfication.csv',index=False)

if __name__ == '__main__':
    filenames = get_list(TRAIN_DIR)
    decode_txt(filenames)





