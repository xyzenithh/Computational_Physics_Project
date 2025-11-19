# quality of life function as I'm too lazy to copy and past the code line above every time I start a new file
import numpy as np 

def load_data():
    data = []
    flux = []
    with open('data.txt', 'r') as f:
        # stripping blank lines from data and insert into a full list
        data_rows = [data_row.strip() for data_row in f if data_row.strip()]

        # initialise headers where each block starts
        header_data = None
        header_flux = None

        # making case insensitive and setting range for list loop
        for i, data_row in enumerate(data_rows):
            lower = data_row.lower()
            if lower.startswith("data to fit"):
                header_data = i
            elif lower.startswith("unoscillated flux"):
                header_flux = i

        # iterate through numbers inbetween the headers and add to list
        for data_row in data_rows[header_data + 1 : header_flux]:
            data.append(int(data_row))

        # as above but for flux
        for data_row in data_rows[header_flux + 1 :]:
            flux.append(float(data_row))
        
    return np.array(data), np.array(flux)

