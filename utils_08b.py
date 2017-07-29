import numpy as np
import emcee
import pdb

################################################################################
################### COLOR TRANSFORMATIONS FOR USNO-B1.0 data ###################
################################################################################
# TODO convert magnitudes to fluxes, average fluxes, then convert back to
# magnitudes
def USNOB_V(magDict):
    '''Computes the V-band magnitude for a USNO-B1.0 star based emulsion
    magnitude values. Returns a tuple containing (Vmag, sigma_Vmag).
    '''
    # Grab the magnitudes needed for this transformation
    Omag = magDict['O']
    Emag = magDict['E']
    Jmag = magDict['J']
    Fmag = magDict['F']

    # Compute the USNO-B1.0 colors
    O_E    = Omag - Emag
    J_F    = Jmag - Fmag
    sig_OE = np.sqrt(2*(0.25**2))
    sig_JF = np.sqrt(2*(0.25**2))

    # Compute the value and uncertainty of (V - O) in the relation...
    # 1) (V - O) = a1 + a2*(O - E)
    a1  = 0.0342438
    a2  = -0.618338
    V_O = a1 + a2*(O_E)

    # The variance-covariance matrix for the relation
    # [[sig_m^2,         rho*sig_m*sig_b],
    #  [rho*sig_m*sig_b, sig_b^2        ]]
    # [[ 0.94850228, -1.20158126],
    #  [-1.20158126,  2.42877908]]*(1e-3)

    # Do the error-propagation by hand...
    sig_m2  = (1e-3)*0.94850228
    sig_b2  = (1e-3)*2.42877908
    rhosmsb = (1e-3)*(-1.20158126)
    sig_VO  = np.sqrt(sig_m2*(O_E**2) + sig_b2 + 2*rhosmsb*O_E + (a2*sig_OE)**2)

    # Compute the V-band magnitude by adding the O-mag and propagate uncertainty
    # (assume all USNO-B1.0 have an uncertainty of 0.25 magnitudes)
    Vmag1     = V_O + Omag
    sig_Vmag1 = np.sqrt(sig_VO**2 + 0.25**2)

    # Compute the value and uncertainty of (V - J) in the relation...
    # 2) (V - J) = a1 + a2*(J - F)
    a1  = -0.027482
    a2  = -0.51196
    V_J = a1 + a2*(J_F)

    # The variance-covariance matrix for the relation
    # [[sig_m^2,         rho*sig_m*sig_b],
    #  [rho*sig_m*sig_b, sig_b^2        ]]
    # [[ 1.00622543, -1.19021477],
    #  [-1.19021477,  1.33947329]]*(1e-3)
    sig_m2  = (1e-3)*1.00622543
    sig_b2  = (1e-3)*1.33947329
    rhosmsb = (1e-3)*(-1.19021477)
    sig_VJ  = np.sqrt(sig_m2*(J_F**2) + sig_b2 + 2*rhosmsb*J_F + (a2*sig_JF)**2)

    # Compute the V-band magnitude by adding the O-mag and propagate uncertainty
    # (assume all USNO-B1.0 have an uncertainty of 0.25 magnitudes)
    Vmag2     = V_J + Jmag
    sig_Vmag2 = np.sqrt(sig_VJ**2 + 0.25**2)

    # Compute an weighted average for the V-band magnitude
    wts1     = 1.0/sig_Vmag1**2.0
    wts2     = 1.0/sig_Vmag2**2.0
    Vmag     = (wts1*Vmag1 + wts2*Vmag2)/(wts1 + wts2)
    sig_Vmag = np.sqrt(1.0/(wts1 + wts2))

    # Take care of inaccurate measures of color from Vmag1 or Vmag2.
    # Find where there is a missing USNO-B1.0 magnitude
    badInds1 = np.where(np.logical_or(Omag == 0, Emag == 0))
    badInds2 = np.where(np.logical_or(Jmag == 0, Fmag == 0))

    # Replace the final magnitude and uncertainty estimates with the true value.
    Vmag[badInds1] = Vmag2[badInds1]
    Vmag[badInds2] = Vmag1[badInds2]
    sig_Vmag[badInds1] = sig_Vmag2[badInds1]
    sig_Vmag[badInds2] = sig_Vmag1[badInds2]

    # Return the final values
    return (Vmag, sig_Vmag)

def USNOB_R(magDict):
    '''Computes the R-band magnitude for a USNO-B1.0 star based emulsion
    magnitude values. Returns a tuple containing (Rmag, sigma_Rmag).
    '''
    # Grab the magnitudes needed for this transformation
    Omag = magDict['O']
    Emag = magDict['E']
    Jmag = magDict['J']
    Fmag = magDict['F']

    # Compute the USNO-B1.0 colors
    O_E    = Omag - Emag
    J_F    = Jmag - Fmag
    sig_OE = np.sqrt(2*(0.25**2))
    sig_JF = np.sqrt(2*(0.25**2))

    # Compute the value and uncertainty of (R - O) in the relation...
    # 3) (R - O) = a1 + a2*(O - E)
    a1 = -0.044736
    a2 = -0.55756
    R_O = a1 + a2*(O_E)

    # The variance-covariance matrix for the relation
    # [[sig_m^2,         rho*sig_m*sig_b],
    #  [rho*sig_m*sig_b, sig_b^2        ]]
    # [[ 1.00622543 -1.19021477]
    #  [-1.19021477  1.33947329]]*(1e-3)

    # Do the error-propagation by hand...
    sig_m2  = (1e-3)*1.00622543
    sig_b2  = (1e-3)*1.33947329
    rhosmsb = (1e-3)*(-1.19021477)
    sig_RO  = np.sqrt(sig_m2*(O_E**2) + sig_b2 + 2*rhosmsb*O_E + (a2*sig_OE)**2)

    # Compute the V-band magnitude by adding the O-mag and propagate uncertainty
    # (assume all USNO-B1.0 have an uncertainty of 0.25 magnitudes)
    Rmag1     = R_O + Omag
    sig_Rmag1 = np.sqrt(sig_RO**2 + 0.25**2)

    # Compute the value and uncertainty of (R - J) in the relation...
    # 4) (R - J) = a1 + a2*(J - F)
    a1 = -0.027676
    a2 = -0.51192
    R_J = a1 + a2*(J_F)

    # The variance-covariance matrix for the relation
    # [[sig_m^2,         rho*sig_m*sig_b],
    #  [rho*sig_m*sig_b, sig_b^2        ]]
    # [[ 0.99551818 -1.17736189]
    #  [-1.17736189  2.17796975]]*(1e-3)
    sig_m2  = (1e-3)*0.99551818
    sig_b2  = (1e-3)*2.17796975
    rhosmsb = (1e-3)*(-1.17736189)
    sig_RJ  = np.sqrt(sig_m2*(J_F**2) + sig_b2 + 2*rhosmsb*J_F + (a2*sig_JF)**2)

    # Compute the V-band magnitude by adding the O-mag and propagate uncertainty
    # (assume all USNO-B1.0 have an uncertainty of 0.25 magnitudes)
    Rmag2     = R_J + Jmag
    sig_Rmag2 = np.sqrt(sig_RJ**2 + 0.25**2)

    # Compute an weighted average for the R-band magnitude
    wts1     = 1.0/sig_Rmag1**2.0
    wts2     = 1.0/sig_Rmag2**2.0
    Rmag     = (wts1*Rmag1 + wts2*Rmag2)/(wts1 + wts2)
    sig_Rmag = np.sqrt(1.0/(wts1 + wts2))

    # Take care of inaccurate measures of color from Rmag1 or Rmag2.
    # Find where there is a missing USNO-B1.0 magnitude
    badInds1 = np.where(np.logical_or(Omag == 0, Emag == 0))
    badInds2 = np.where(np.logical_or(Jmag == 0, Fmag == 0))

    # Replace the final magnitude and uncertainty estimates with the true value.
    Rmag[badInds1] = Rmag2[badInds1]
    Rmag[badInds2] = Rmag1[badInds2]
    sig_Rmag[badInds1] = sig_Rmag2[badInds1]
    sig_Rmag[badInds2] = sig_Rmag1[badInds2]

    # Return the final values
    return (Rmag, sig_Rmag)

def USNOB_VR(magDict):
    '''Computes the V-R  color for a USNO-B1.0 star based emulsion magnitude
    values. Returns a tuple containing (V-R, sigma_V-R).
    '''
    # Grab the magnitudes needed for this transformation
    Omag = magDict['O']
    Emag = magDict['E']
    Jmag = magDict['J']
    Fmag = magDict['F']

    # Compute the USNO-B1.0 colors
    O_E    = Omag - Emag
    J_F    = Jmag - Fmag
    sig_OE = np.sqrt(2*(0.25**2))
    sig_JF = np.sqrt(2*(0.25**2))

    # Compute the value and uncertainty of (V - R) in the relation...
    # 5) (V - R) = a1 + a2*(O - E)
    a1   = 0.141308
    a2   = 0.247982
    V_R1 = a1 + a2*(O_E)

    # The variance-covariance matrix for the relation
    # [[sig_m^2,         rho*sig_m*sig_b],
    #  [rho*sig_m*sig_b, sig_b^2        ]]
    # [[ 0.04324588 -0.0537035 ]
    #  [-0.0537035   0.13787215]]*(1e-3)

    # Do the error-propagation by hand...
    sig_m2  = (1e-3)*0.04324588
    sig_b2  = (1e-3)*0.13787215
    rhosmsb = (1e-3)*(-0.0537035)
    sig_VR1 = np.sqrt(sig_m2*(O_E**2) + sig_b2 + 2*rhosmsb*O_E + (a2*sig_OE)**2)

    # Compute the value and uncertainty of (V - R) in the relation...
    # 6) (V - R) = a1 + a2*(J - F)
    a1   = 0.153232
    a2   = 0.273522
    V_R2 = a1 + a2*(J_F)

    # The variance-covariance matrix for the relation
    # [[sig_m^2,         rho*sig_m*sig_b],
    #  [rho*sig_m*sig_b, sig_b^2        ]]
    # [[ 0.06020978 -0.06448749]
    #  [-0.06448749  0.15663887]]*(1e-3)
    sig_m2  = (1e-3)*0.06020978
    sig_b2  = (1e-3)*0.15663887
    rhosmsb = (1e-3)*(-0.06448749)
    sig_VR2  = np.sqrt(sig_m2*(J_F**2) + sig_b2 + 2*rhosmsb*J_F + (a2*sig_JF)**2)

    # Compute an weighted average for the V-R color
    wts1   = 1.0/sig_VR1**2.0
    wts2   = 1.0/sig_VR2**2.0
    V_R    = (wts1*V_R1 + wts2*V_R2)/(wts1 + wts2)
    sig_VR = np.sqrt(1.0/(wts1 + wts2))

    # Take care of inaccurate measures of color from Vmag1 or Vmag2.
    # Find where there is a missing USNO-B1.0 magnitude
    badInds1 = np.where(np.logical_or(Omag == 0, Emag == 0))
    badInds2 = np.where(np.logical_or(Jmag == 0, Fmag == 0))

    # Replace the final magnitude and uncertainty estimates with the true value.
    V_R[badInds1] = V_R2[badInds1]
    V_R[badInds2] = V_R1[badInds2]
    sig_VR[badInds1] = sig_VR2[badInds1]
    sig_VR[badInds2] = sig_VR1[badInds2]

    # Return the final values
    return (V_R, sig_VR)


#******************************************************************************
# Define a photometric transform to compute R-band magnitudes
#******************************************************************************
def APASS_V(magDict):
    '''
    Computes the V-band magnitude from APASS V and B-V magnitudes.

    In this case, the most accurate estimate *IS* the APASS V-band magnitude.
    '''
    Vmag     = np.array(magDict['Vmag'])
    sig_Vmag = np.array(magDict['e_Vmag'])

    return (Vmag, sig_Vmag)

def APASS_R(magDict):
    '''Computes the R-band magnitude from APASS sloan magnitudes'''
    # 1) BGrab the magnitudes needed for this transformation
    g_mag   = np.array(magDict['g_mag'])
    s_g_mag = np.array(magDict['e_g_mag'])
    r_mag   = np.array(magDict['r_mag'])
    s_r_mag = np.array(magDict['e_r_mag'])

    # 2) Compute the sloan bands color
    ga_ra     = g_mag - r_mag
    sig_ga_ra = np.sqrt(s_g_mag**2 + s_r_mag**2)

    # 3) Compute the value and uncertainty of (Rl - r_a) in the relation...
    # (Rl - r_a) = a1 + a2*(g_a - r_a)
    a1    = -0.143307088
    a2    = -1.112865077
    Rl_ra = a1 + a2*(ga_ra)

    # The variance-covariance matrix for the relation
    # [[sig_m^2,         rho*sig_m*sig_b],
    #  [rho*sig_m*sig_b, sig_b^2        ]]
    # [[ 3.93e-5 -1.97e-5]
    #  [-1.97e-5  1.79e-5]]

    # Do the error-propagation for (Rl - r_a) by hand...
    sig_m2    = 3.93489486155285e-5
    sig_b2    = 1.79279902874194e-5
    rhosmsb   = -1.96807984258731e-5
    sig_Rlra  = np.sqrt(sig_m2*(ga_ra**2) + sig_b2 + 2*rhosmsb*ga_ra + (a2*sig_ga_ra)**2)

    # Compute the R-band magnitude by adding the r_mag and propagate uncertainty
    Rmag     = Rl_ra + r_mag
    sig_Rmag = np.sqrt(sig_Rlra**2 + s_r_mag**2)

    # #### OK, here is where I need to compute the R-band magnitude from the
    # #### OTHER side using (V - r) ...
    # #### That is possible but probably not necessary.

    # Return the final values
    return (Rmag, sig_Rmag)

def APASS_VR(magDict):
    """Computes the V-R magnitude from APASS catalog entries"""
    # Compute the APASS V-mag
    Vmag, s_Vmag = APASS_V(magDict)

    # Compute the APASS R-mag
    Rmag, s_Rmag = APASS_R(magDict)

    # Compute the APASS V-R color
    VRmag     = Vmag - Rmag
    sig_VRmag = np.sqrt(s_Vmag**2 + s_Rmag**2)

    return (VRmag, sig_VRmag)

################################################################################
############## DEFINE A FUNCTION TO PERFORM THE BASIC, LINEAR FIT ##############
################################################################################
# Define two functions for swapping between (sigma_x, sigma_y, theta) and
# (sigma_x, sigma_y, rhoxy)
def convert_angle_to_covariance(sx, sy, theta):
    # Define the rotation matrix using theta
    rotation_matrix = np.matrix([[np.cos(theta), -np.sin(theta)],
                                 [np.sin(theta),  np.cos(theta)]])

    # Build the eigen value matrix
    lamda_matrix = np.matrix(np.diag([sx, sy]))

    # Construct the covariance matrix
    cov_matrix = rotation_matrix*lamda_matrix*lamda_matrix*rotation_matrix.I

    # Extract the variance and covariances
    sx1, sy1 = np.sqrt(cov_matrix.diagonal().A1)
    rhoxy    = cov_matrix[0,1]/(sx1*sy1)

    return sx1, sy1, rhoxy

def convert_covariance_to_angle(sx, sy, rhoxy):
    # build the covariance matrix from sx, sy, and rhoxy
    cov_matrix = build_cov_matrix(sx, sy, rhoxy)

    # Extract the eigen values and eigen vectors
    eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
    x_stddev, y_stddev = np.sqrt(eig_vals)

    # Convert the eigen vector into an angle
    y_vec = eig_vecs[:, 0]
    theta = np.arctan2(y_vec[1,0], y_vec[0,0])

    # Make sure the sign of rhoxy is the same as the sign on theta
    if np.sign(rhoxy) != np.sign(theta):
        if np.sign(theta) < 0:
            theta += np.pi
        if np.sign(theta) > 0:
            theta -= np.pi

    return x_stddev, y_stddev, theta

def MCMCfunc(data, bounds, n_walkers=100, n_burn_in_steps=250, n_steps=1000):
    '''This function takes provided log-likelihood and log-prior functions and
    performs an MCMC sampling of the log-posterior-probability distribution.

    parameters:

    data   -- This must be a tuple containing the values (x, y, sx, sy). The x
              and y data ought to be arrays, but the sx and/or sy values can be
              scalars if the relative magnitudes of the uncertainties are
              unknown.

    bounds -- This must be an array-like iterable containing one two-element set
              of boundaries for each parameter. This will be used by the prior
              function.
    '''
    # Simply start the parameters in the center of the bounds
    params_guess = np.array([np.mean(b) for b in bounds])
    n_dim        = len(params_guess)

    # Define a prior function for the parameters
    def ln_prior(params):
        # We'll just put reasonable uniform priors on all the parameters.
        if not all(b[0] < v < b[1] for v, b in zip(params, bounds)):
            return -np.inf
        return 0

    # The "foreground" linear likelihood:
    def ln_like_fg(params, x, y, sx, sy):
        theta, b_perp = params

        # The following terms allow us to skip over the matrix algebra in the Loop
        # below. These are the absolute value of the components of a unit vector
        # pointed pependicular to the line.
        sinT = np.sin(theta)
        cosT = np.cos(theta)

        # This code can be worked out from some matrix algebra, but it is much
        # faster to avoid going back and forth between matrices, so the explicit
        # form is used here

        # Comput the projected distance of this point from the line
        Di  = cosT*y - sinT*x - b_perp

        # Compute the projected variance of the data
        Si2 = (sinT*sx)**2 + (cosT*sy)**2

        # Compute the final likelihood
        out = -0.5*(((Di**2)/Si2) + np.log(Si2))

        # Return the sum of the ln_likelihoods (product in linear space)
        return out

    # # The "background" outlier likelihood:
    # # (For a complete, generalized, covariance matrix treatment, see the file
    # # "HBL_exer_14_with_outlier_covariance.py")
    # def ln_like_bg(params, x, y, sx, sy):
    #     theta, b_perp, Mx, Pb, lnVx, My, lnVy = params
    #
    #     sinT = np.sin(theta)
    #     cosT = np.cos(theta)
    #
    #     # Build the orthogonal-to-line unit vector (and its transpose)
    #     v_vecT = np.matrix([-sinT, cosT])
    #     v_vec  = v_vecT.T
    #
    #     # In parameter terms, "Vx" is the actual variance along the x-axis
    #     # *NOT* the variance along the eigen-vector. Using this assumption, and
    #     # forcing the outlier distribution tilt-angle to be equal to the main
    #     # distribution tilt angle, we get...
    #     sxOut, syOut, rhoxy = convert_angle_to_covariance(
    #         np.exp(lnVx), np.exp(lnVy), theta)
    #
    #     # Compute the elements of the convolved covariance matrix
    #     # # The old way...
    #     # out_cov_matrix = build_cov_matrix(sxOut, syOut, rhoxy)
    #     # data_cov_matrix = np.array([build_cov_matrix(np.sqrt(2*0.25), syi, 0) for syi in sy])
    #     # testList = []
    #     # for data_cov_mat1 in data_cov_matrix:
    #     #     conv_cov_matrix = out_cov_matrix + data_cov_mat1
    #     #     testList.append((v_vecT.dot(conv_cov_matrix).dot(v_vec))[0,0])
    #     # Si2test = np.array(testList)
    #
    #     # The fast way...
    #     conv_cov_matrix11 = sxOut**2 + sx**2
    #     conv_cov_matrix12 = rhoxy*sxOut*syOut
    #     conv_cov_matrix22 = syOut**2 + sy**2
    #
    #     # Now get the projection of the covariance matrix along the direction
    #     # orthogonal to the line
    #     Si2 = (-sinT*(-sinT*conv_cov_matrix11 + cosT*conv_cov_matrix12) +
    #            cosT*(-sinT*conv_cov_matrix12 + cosT*conv_cov_matrix22))
    #
    #     # Compute the distances between the data and the line can do it all at once.
    #     Di = cosT*y - sinT*x - b_perp
    #
    #     out = -0.5*(((Di**2)/Si2) + np.log(Si2))
    #
    #     if np.sum(np.isnan(out)) > 0: pdb.set_trace()
    #
    #     # Return the ln_likelihood of the background model given these-data points
    #     # Equally likely for ALL data-points
    #     return out

    # # Define a likelihood function for the parameters and data
    # def ln_like(params, x, y, sx, sy):
    #     # Unpack the parameters
    #     _, _, Pb, Mx, lnVx, My, lnVy = params
    #
    #     # Compute the vector of foreground likelihoods and include the Pb prior.
    #     natLogLike_fg = ln_like_fg(params, x, y, sx, sy)
    #     arg1 = natLogLike_fg + np.log(1.0 - Pb)
    #
    #     # Compute the vector of background likelihoods and include the Pb prior.
    #     natLogLike_bg = ln_like_bg(params, x, y, sx, sy)
    #     arg2 = natLogLike_bg + np.log(Pb)
    #
    #     # Combine these using log-add-exp for numerical stability.
    #     natLogLike = np.sum(np.logaddexp(arg1, arg2))
    #
    #     # Include a second output as "blobs"
    #     return natLogLike, (arg1, arg2)

    # Define a likelihood function for the parameters and data
    def ln_like(params, x, y, sx, sy):

        # Include a second output as "blobs"
        return np.sum(ln_like_fg(params, x, y, sx, sy))

    # Now we need to actually APPLY Baye's rule to construct the log-proability
    # function. This is simply the log of the product of the prior and
    # likelihood functions, or the sum of the log of the prior and the log of
    # the liklihood function. This function will only take the two dictionaries
    # provided

    # First, test if the ln_like function is returning blobs and set the boolean
    # value to signal the ln_probability function to also use these blobs
    global blobsBool, iteration, lastPercent, burn_in
    test_lnLike = ln_like(params_guess, *data)
    usingBlobs  = isinstance(test_lnLike, tuple)

    def ln_probability(params, *args):
        global blobsBool

        ########################################################################
        # USE THIS CODE TO PRINT PROGRESS UPDATES
        ########################################################################
        global iteration, lastPercent, burn_in

        # Increment the iteration each time the function is called
        iteration  +=1

        if burn_in:
            # If this is being called by a burn-in run, then compute the
            # percentage of the burn in completed
            thisPercent = np.int(100.0*iteration/
                (n_walkers*n_burn_in_steps + n_walkers -1))
        else:
            # If this is being called by a production run, then compute the
            # percentage of the production completed
            thisPercent = np.int(100.0*iteration/
                (n_walkers*n_steps + n_walkers -1))

        # Updating every call dramatically slows down the run, so test if new
        # on-screen text is even required
        if thisPercent > lastPercent:
            print("Sampler progress: {0:3d}%".format(thisPercent), end='\r')
            lastPercent = thisPercent
        ########################################################################

        # Call the prior function, check that it doesn't return -inf then create
        # the log-probability function
        natLogPrior = ln_prior(params)
        if not np.isfinite(natLogPrior):
            # If there is not a finite probability of the sampled parameter,
            # then simple return the most negative possible value

            if usingBlobs:
                return -np.inf, None
            else:
                return -np.inf
        else:
            # Otherwise return Ln(posterior) = Ln(prior) + Ln(Likelihood) + C
            # (We can ignore the "C" value because it's just a scaling constant)
            natLogLikelihood = ln_like(params, *args)

            # Test for any "blobs" output from the ln_like function
            if usingBlobs:
                natLogLikelihood, blob = natLogLikelihood

                # Combine the ln_like and ln_prior outputs to get ln_posterior.
                natLogPostProb = natLogPrior + natLogLikelihood

                return natLogPostProb, blob
            else:
                # Combine the ln_like and ln_prior outputs to get ln_posterior.
                natLogPostProb = natLogPrior + natLogLikelihood

                return natLogPostProb

    # Almost there! Now we must initialize our walkers. Remember that emcee uses
    # a bunch of walkers, and we define their starting distribution. If you have
    # an idea of where your best-fit parameters will be, you can start the
    # walkers in a small Gaussian bundle around that value (as I am doing).
    # Otherwise, you can start them evenly across your parameter space (that is
    # limited by the priors). This will require more walkers and more steps.

    # Setup the initial positions of the random walkers for the MCMC sampling
    p0 = np.array(params_guess)
    p0 = [p0 + 1e-5*np.random.randn(n_dim) for k in range(n_walkers)]

    #Finally, you're ready to set up and run the emcee sampler
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, ln_probability,
        args=data)

    # Run the burn-in
    print('Running burn-in...')
    burn_in = True
    iteration = -1
    lastPercent = -1
    output = sampler.run_mcmc(p0, n_burn_in_steps)
    if len(output) > 3:
        pos, prob, state, blobs = output
    else:
        pos, prob, state = output
    print('')

    print("Running production...")
    sampler.reset()
    burn_in = False
    iteration = -1
    lastPercent = -1
    output = sampler.run_mcmc(pos, n_steps, rstate0=state)
    if len(output) > 3:
        pos, prob, state, blobs = output
    else:
        pos, prob, state = output
    print('')

    return sampler
