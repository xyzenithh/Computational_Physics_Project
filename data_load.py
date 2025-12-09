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

def parabolic(f, x0, x1, x2, tol, max_iter=300):
    """
    1D parabolic minimiser 

    f = NLL function for 1 var
    x0,x1,x2 = initial points
    tol = convergence critera

    Returns minimum of x and minimum of NLL at x_min
    """
    f0 = f(x0)
    f1 = f(x1)
    f2 = f(x2)

    for i in range(max_iter):
        x3 = 0.5*((x2**2 - x1**2) * f0 + (x0**2 - x2**2) * f1 + (x1**2 - x0**2) * f2)/((x2 - x1) * f0 + (x0 - x2) * f1 + (x1 - x0) * f2)
        f3 = f(x3)

        # check convergence criteria and then input values 
        if abs(x3 - x1) < tol:
            x_mins = np.array([x0, x1, x2, x3])
            f_mins = np.array([f0, f1, f2, f3])
            min_val = np.argmin(f_mins)
            return x_mins[min_val], f_mins[min_val]

        # keep best 3 points with most minimised f
        x_mins = np.array([x0, x1, x2, x3])
        f_mins = np.array([f0, f1, f2, f3])

        # Sort by x so the bracket stays ordered
        sort = np.argsort(x_mins)
        x_mins = x_mins[sort]
        f_mins = f_mins[sort]

        # replace initial guesses with newly minimised approximations
        x0, x1, x2 = x_mins[0], x_mins[1], x_mins[2]
        f0, f1, f2 = f_mins[0], f_mins[1], f_mins[2]

    # if not converged at max iterations then just return best value
    x_mins = np.array([x0, x1, x2])
    f_mins = np.array([f0, f1, f2])
    min_val = np.argmin(f_mins)
    return x_mins[min_val], f_mins[min_val]

def nll_dm2(dm2):
    theta_approx = 0.6751499971144296
    return nll(theta_approx, dm2)

def nll_theta(theta):
    dm2_approx = 0.0024473468435815774
    return nll(theta, dm2_approx)