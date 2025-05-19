# handle_data.py
# TrainNetwork
# Author: [John Miller]
# Description: This is the file used for taking in datasets with
# different input features, and then normalizing them such that they will be of use to the
# Date: [1/15/2025]

import csv

def normalize_time(time:str):
    hour, minute = map(int, time.split(":"))
    total_minutes = (hour * 60) + minute
    normalized = total_minutes / 1440  # maps 0:00 → 0.0, 12:00 → 0.5, 23:59 → ~0.999
    shifted = (normalized + 0.5) % 1.0  # shifts so 12:00 AM becomes 0.5
    return shifted


def physical_activity_level(data):
    if data == "low":
        return 0
    elif data == "medium":
        return 0.5
    elif data == "high":
        return 1

def sleep_data() -> list:
    data = []
    with open("SecondaryData.csv") as csvfile:
        reader = csv.reader(csvfile)
        row_index = 0
        for row in enumerate(reader):
            if row_index != 0:
                row_data = row[1]
                output = int(row_data[3]) / 10                       # grab the output label because that's how we will adjust the model
                input = []
                for i in range(1, len(row_data)):
                    if i != 3:
                        if i == 4 or i == 5:
                            input.append(normalize_time(row_data[i]))
                        elif i == 6:
                            input.append(int(row_data[i]) / 11000)
                        elif i == 8:
                            input.append(physical_activity_level(row_data[i]))
                        elif i == 10:
                            input.append(int(row_data[10] == "yes"))

                new_observation = [input, output]
                print(new_observation)
                data.append(new_observation)
            row_index += 1
    return data

    
def sleep_data_three_observations() -> list:
    data = []
    with open("SecondaryData.csv") as csvfile:
        reader = csv.reader(csvfile)
        row_index = 0
        for row in enumerate(reader):
            if row_index != 0:
                row_data = row[1]
                output = int(row_data[3]) / 10                       # grab the output label because that's how we will adjust the model
                input = []
                for i in range(1, len(row_data)):
                    if i != 3:
                        if i == 4 or i == 5:
                            input.append(normalize_time(row_data[i]))   # append the bed/wakeup time
                        elif i == 6:
                            input.append(int(row_data[i]) / 11000)  # append the normalized steps
                new_observation = [input, output]
                data.append(new_observation)
            row_index += 1
    return data




    
    
    
        





