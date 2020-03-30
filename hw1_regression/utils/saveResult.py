import csv

def import_csv(result, PATH):
    with open(PATH, mode='w', newline='') as submit_file:
        csv_writer = csv.writer(submit_file)
        header = ['id', 'value']
        print(header)
        csv_writer.writerow(header)
        for i in range(240):
            row = ['id_' + str(i), result[i][0]]
            csv_writer.writerow(row)
            print(row)