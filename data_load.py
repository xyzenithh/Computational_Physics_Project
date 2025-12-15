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
    E = np.linspace(0, 10, 200, endpoint = False) 
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

    Returns minimum of x and minimum of NLL at dm2_min
    """
    f0 = f(x0)
    f1 = f(x1)
    f2 = f(x2)

    x3_vals = []
    f3_vals = []

    # best so far
    x_best = x1
    f_best = f1

    # loop through iterations, break if denominator is too small/0, append trial vals for plotting
    for i in range(max_iter):
        denom = (x2 - x1) * f0 + (x0 - x2) * f1 + (x1 - x0) * f2
        if np.abs(denom) < 1e-14:
            break

        x3 = 0.5 * ((x2**2 - x1**2) * f0+ (x0**2 - x2**2) * f1+ (x1**2 - x0**2) * f2 ) / denom

        # keep x3 inside the current bracket to avoid wild jumps
        x_lo = np.min([x0, x1, x2])
        x_hi = np.max([x0, x1, x2])
        if (not np.isfinite(x3)) or (x3 <= x_lo) or (x3 >= x_hi):
            x3 = 0.5 * (x_lo + x_hi)

        f3 = f(x3)
        x3_vals.append(x3)
        f3_vals.append(f3)

        # update best value 
        if f3 < f_best:
            f_best = f3
            x_best = x3

        # convergence criteria check (stop when x3 stops moving)
        if len(x3_vals) > 1 and np.abs(x3_vals[-1] - x3_vals[-2]) < tol:
            break

        # store values
        x_all = np.array([x0, x1, x2, x3])
        f_all = np.array([f0, f1, f2, f3])

        # keep best 3 points by NLL, then sort by x so the bracket stays ordered
        keep = np.argsort(f_all)[:3]
        x_keep = x_all[keep]
        f_keep = f_all[keep]

        order = np.argsort(x_keep)
        x0, x1, x2 = x_keep[order]
        f0, f1, f2 = f_keep[order]

    # if no appends anything, return best of the initial three points
    if len(x3_vals) == 0:
        x_init = np.array([x0, x1, x2])
        f_init = np.array([f0, f1, f2])
        j = np.argmin(f_init)
        return x_init[j], f_init[j], np.array(x3_vals), np.array(f3_vals)

    return x_best, f_best, np.array(x3_vals), np.array(f3_vals)

def nll_dm2(dm2):

    theta_approx = 0.6759913232677012
    return nll(theta_approx, dm2)

def nll_theta(theta):
    dm2_approx = 0.002490764417552603
    return nll(theta, dm2_approx)

def deltaNLL(f, range_min, range_max, points=1000):
    """
    f = NLL
    range_min/max = scan range
    points =  number of grid points


    returns minimum parameter, alongside values where deltaNLL = +-1, and the associated errors
    """

    # range for scanning
    param_range = np.linspace(range_min, range_max, points)
    param_vals = np.array([f(x) for x in param_range])
    # minimum
    index_min = np.argmin(param_vals)
    param_min = param_range[index_min]
    nll_min  = param_vals[index_min]
    target = nll_min + 1.0  

    # upward scan
    i = index_min
    while i < points and param_vals[i] < target:
        i += 1
    if i == points:
        x_plus = param_range[-1]
    else:
        x1, y1 = param_range[i-1], param_vals[i-1]
        x2, y2 = param_range[i],   param_vals[i]
        x_plus = x1 + (target - y1) / (y2 - y1) * (x2 - x1)
    print(f'NLL upwards scan value:{param_vals[i]}')
    # downward scan
    i = index_min
    while i >= 0 and param_vals[i] < target:
        i -= 1

    if i < 0:
        x_minus = param_range[0]
    else:
        x1, y1 = param_range[i+1], param_vals[i+1]
        x2, y2 = param_range[i],   param_vals[i]
        x_minus = x1 + (target - y1) / (y2 - y1) * (x2 - x1)

    print(f'NLL downwards scan value:{param_vals[i]}')
    err_plus  = x_plus  - param_min
    err_minus = param_min  - x_minus

    return param_min, x_minus, x_plus, err_minus, err_plus

def curv_method(f, min, step_size = 1e-3):
    """ 
    Calculating second derivative to then calculate sigma of the given function

    f = NLL function for 1 variable
    min = minimum value of parameter for given function

    Returns sigma
    """

    f0 = f(min - step_size)
    f1 = f(min)
    f2 = f(min + step_size)

    deriv2 = (f2 - 2 * f1 + f0)/step_size**2
    return np.sqrt(2/deriv2)