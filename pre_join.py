
# coding: utf-8

# Pre-processing of Kickstarter_2020-02-13T03_20_04_893Z
#Join csv files



import numpy as np
import json
import glob
import csv

header_already_written = False

with open("../../data/joined.csv", "wt") as f_out:
    csv_writer = csv.writer(f_out)

    for filename in glob.glob("../../data/Kickstarter_2020-02-13T03_20_04_893Z/*.csv"):
        print(filename)
    
        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
    
            for row in csv_reader:
                if row[0] == 'backers_count' and not header_already_written: 
                    csv_writer.writerow(row)
                    header_already_written = True
                    continue # skip header
            
                #print(row)
            
                for indice_col in (2, 7, 22, 25, 27, 35):
                    string = row[indice_col]
                    try:
                        structure = json.loads(string)
                    except json.JSONDecodeError:
                        structure = None
                        break
                
                if structure:
                    csv_writer.writerow(row)
                
                    
                        