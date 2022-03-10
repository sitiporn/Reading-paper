import csv


a_file = open("sample.csv","w")
a_dict = {"0":1,"1":1}

writer = csv.writer(a_file)

for k,v in a_dict.items():
    writer.writerow([k,v])


a_file.close()
