import csv

#write a program that converts a large txt file of data into a csv file
#There are 6.4 Million Rows of data
input_file_path = "./Data/Gowalla_totalCheckins.txt"
output_file_path = "./Output/gowalla_checkins.csv"

# Headers
headers = ['user', 'time', 'latitude', 'longitude', 'locationID']

with open(input_file_path, 'r') as txt_file, open(output_file_path,'w',newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(headers)

    for line in txt_file:

        columns = line.strip().split()

        csv_writer.writerow(columns)

print(f'Conversion complete. CSV file created at: {output_file_path}')

    