import csv

def WriteDictToCSV(csv_file,csv_columns,dict_data):
    
    with open('history.csv','wb') as csvfile:
        w = csv.writer(csvfile)
        w.writerows(dict_data.items())

    return