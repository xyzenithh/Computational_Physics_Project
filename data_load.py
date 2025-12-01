# quality of life function as I'm too lazy to copy and past the code line above every time I start a new file

# these functions must be kept, as defining a global value for a variable can mean that the function cheats/breaks
# and therefore will not give us a correct value. This can also be worked around using differing variable names but
# easiest to do this


import numpy as np 

def load_data():
    data = []
    flux = []
    with open('data.txt', 'r') as f:
        """ 
        Load Data Function

        Loads in given text file, reads it and then separates according to header labels

        For this given data set we should see a shape of (200)
        """
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

def pmumu(theta, dm2):
    """
    Survival Probability Function

    The probability a neutrino will be observed as a muon neutrino
    If you plot prob_mumu without the '1-' then you get the oscillation probability

    theta23 = mixing angle --> amplitude of neutrino oscillation probability
    dm2 = (change in mass of neutrino)^2 in eV^2
    E_centres = middle of energy bin,  used here to avoid any issues when encountering a 0
    """
    E = np.linspace(0, 10, 200) 
    dE = 10.0 / 200.0                              
    E_centre = E + dE*0.5                   
    L = 295                   
    E = np.array(E_centre)
    arg = 1.267 * dm2 * L / E
    prob_mumu = 1 - (np.sin(2*theta))**2*(np.sin(arg))**2
    return np.array(prob_mumu)

def lam_i(theta, dm2):
    """ 
    Function to Calculate and Return lambda_i, the expected no. of events in each energy bin

    flux = unoscillated prediction data
    """
    
    data, flux = load_data()
    prob = pmumu(theta, dm2)
    return flux * prob

def nll(theta, dm2):
    """ 
    Negative Log-Liklihood Function using equation given in Project1 File

    m_i = data (counts)     (from equation)
    lambda_i = flux * prob_mumu (model counts)
    """
    
    data, flux = load_data()
    lam = lam_i(theta, dm2)
    return 2.0 * np.sum(lam - data * np.log(lam))
