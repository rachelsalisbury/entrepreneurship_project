# -*- coding: utf-8 -*-
"""
@author: rsalisbury
File: V2_intercept.py
Started: March 26, '21

Goals for this version:
- Expanding the intercept model to included sophistication and the monopoly market structure.
- Parameter testing to decide on a base parameterization and a difference between the monopolist and the start-up within the model.

Changes:

"""



# %% Imports
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['figure.max_open_warning'] = 0
import matplotlib.pyplot as plt
import seaborn as sns

import multiprocessing as mp
#print ("Number of processors: ", mp.cpu_count()) # 8 processors

'''important for file path'''
version_no = '2_int_'


# %% Define functions
def olsfunction(x, y):
    '''
    Returns coefficients and adjusted r-squared for OLS regression in variable results.
    Results is a dictionary that contains coefficients as the first term and adjusted r-square, adjrsq, as the second term.
    '''
    results={}
    A=np.c_[np.ones(len(y)),x] # generate matrix containing column of 1's (for intercept term) appended to X matrix
    olsresults = np.linalg.lstsq(A, y, rcond = -100) # find coefficient vector that minimizes squared residuals
    results['coefficients']=olsresults[0] # store coefficient vector in 'results' list
    ss_res=0 # initialize sum of squared residuals
    for row in range(len(y)):
        ss_res+=(y[row]-sum(A[row,:]*olsresults[0]))**2 # find sum of squared residuals
    df_reg=len(y)-int(x.shape[1])-1 # save degrees of freedom of regression
    df_tot = len(y)-1 # total degrees of freedom
    adjrsq=1-((ss_res/df_reg)/((y.size*y.var())/df_tot)) # find adjusted r squared
    results['adjrsq']=adjrsq # store adjusted r q=squared value in results
    return results

def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

class Firm:
    '''
    A firm performs an OLS regression on previous period prices and quanities to estimate demand.
    PERIOD 0: The firm reflects on prices and quanitites it observed in the pre-history and calibrates based on its history to decide its initial betas. Since the history is deterministically derived from the true betas,
    a properly functioning firm will arrive at the true betas when it performs an OLS regression on its pre-history.
    The firm then chooses its initial Q based on these initial betas and the shocks to demand in that period.
    PERIOD 1: The firm re-calibrates its conception of demand based on what it has observed in the previous period.
    In period 0, there were shocks to demand, so quantities were not chosen deterministically by the true model as they were
    in the pre-history. As a result, the firm's will shift their estimates of the betas when they attempt to estimate the
    true betas by OLS regression on what happened in period 0 and in the pre-history. In period 1, the firm saves its
    period 1 betas, which are the first betas to be adjusted, and the quantities it chooses based on those betas.
    If both firms are correct in their mental models and initial betas, then they will continue to estimate the correct betas.
    If a firm is sophisticated and knows its rival firm's mental models, then it will also continue to estimate the correct betas.
    PERIODS 2-99: The firm will continue to re-calibrate its estimates of the betas based on past periods. In any period n,
    the nth period betas are estimated at the beginning of the period, and then the nth period Q is chosen.

    Each firm has:
    - mental model (mental_model), a list that indicates which variables the firm takes into account when it calibrates
      its betas. The list is binary, 0 indicating that a variable is not taken into account and 1 indicating that it is,
      and the list is in the order [intercept, beta_1, beta_2, beta_3]. For example, [1,1,0,1] indicates that the firm
      does not take beta_2 into account.
    - initial betas (init_b, or weights_mm), a list containing the values of the betas, also called the beta magnitudes
      or the weights of the mental model.
    - competitor model (comp_model, or compmod), a string that indicates the firm's conception of its competiton,
      which is either self-reflective (the firm believes that its rival is the same as itself) or sophisticated
      (the firm knows its rivals mental model and initial betas), which are denoted 'sr' and 'so', respectively.
    - cost: right now, all firms face the same global cost.
    - market position (market_position, or market_pos): indicates whether a firm believes that it is a monopolist
      ('monop') or a duopolist ('duop').
    - perceived rival Q (perceived_Q): [FILL IN]
    - Several variables used for keeping track of capacity. [FILL IN]

    '''

    def __init__(self, mental_model, init_betas, competitor_model,
                 global_cost, market_position):
        self.mm = mental_model
        self.init_b = init_betas
        self.comp_model = competitor_model
        self.cost = global_cost
        self.market_position = market_position
        #For sophisticated firms only:
        self.perceivedQ = {}
        #Capacity tracking variables:
        self.cap = 0
        self.cap_posdemand = 0
        self.cap_maxedout = 0
        self.Q_lessthan_zero = 0
        self.choose_Q_zero = 0
        self.choose_Q_max = 0


    def gen_init_history(self, init_hist_length, prehistmean, prehistsd,
                         prehistmean_Q, prehistsd_Q, directory, filebegin, runsim_it, firm_num):
        '''
        - Generates initial history, which consists of the prices and quantities the firm observes
          prior to the simulation beginning in period 0.
        - Takes list of parameters to determine the mean and distribution of the quantities and the X shocks to demand,
          which are both randomly generated on a normal distribution.
        - Returns a list H of length init_hist_length, containing the price and X shocks to demand.
        - The firm does not actually make any 'decisions', like adjusting its betas or choosing its quantity,
          during the pre-history. X shocks determine Q, which determines price, and X shocks and prices are recorded.
          The firm will use this record of the pre-history in period 0 to estimate its period 0 betas and then choose
          period 0 Q.
        - Explanation of why all pre-history (q,p) chosen will fall exactly on the true demand curve:
          Q is not chosen by the firm, but is randomly generated based on parameters. The price is chosen
          deterministically, i.e., at whatever quantity is selected, the price will fall on the linear demand curve
          at that quantity. These prices and quantities fall exactly on the firm's true demand curve,
          since the demand curve is decided by the firm's initial betas. Where the (q,p) points land along
          the demand curve is determined by the variance and mean of the random X shocks to demand and of
          the randomly generated Q. When firms calibrated based on this pre-history, their post-calibration betas
          will be the same as their initial betas, since the initial history process is deterministic and all
          points land exactly on the demand curve determined by the firm's initial betas.
        '''

        weights_mm = self.init_b

        '''
        Distribution params for the pre-history:
        - Takes means and covariances for pre-history X shocks as parameters.
        - No covariance between any of the X shocks to demand, but covariance matrix is necessary to use
          np.random.multivariate_normal to generate the X shocks.
        '''
        pre_hist_cov, pre_hist_covx1x2, pre_hist_covx1x3, pre_hist_covx2x3 = [0,0,0,0]
        pre_hist_mean = prehistmean
        pre_hist_means = [prehistmean, prehistmean, prehistmean]
        pre_hist_sd = prehistsd
        pre_hist_mean_Q = prehistmean_Q
        pre_hist_sd_Q = prehistsd_Q
        covar_matrix = [[pre_hist_sd**2, pre_hist_covx1x2, pre_hist_covx1x3],
                        [pre_hist_covx1x2, pre_hist_sd**2, pre_hist_covx2x3],
                        [pre_hist_covx1x3, pre_hist_covx2x3, pre_hist_sd**2]]
        runsim_it = runsim_it #Added in v2_5 to use to save each of the iterations' pre hist graphs uniquely
        firm_num= firm_num

        Q = {}
        H = []
        X_shocks = []
        hist_len = init_hist_length
        for i in range(hist_len):

            '''random draws for X shocks'''
            #Generates list of randoms of length N using the X shock means and covariances given:
            X = np.random.multivariate_normal(pre_hist_means, covar_matrix)

            '''ADDED IN V2_5: If x is negative, x = 0. Same change was made in simulation draws of xs.'''

            while (X[0] < 0) or (X[1] < 0) or (X[2] < 0):
                X = np.random.multivariate_normal(pre_hist_means, covar_matrix)

            # Select first three random numbers to be x1, x2, x3
            x1 = X[0] #random shock x1
            x2 = X[1] #random shock x2
            x3 = X[2] #random shock x3

            '''random draws for quantities'''
            Q[i] = np.random.normal(pre_hist_mean_Q, pre_hist_sd_Q)
            values = [x1, x2, Q[i]] #CHANGED IN INTERCEPT MODEL

            ###INTERCEPT MODEL PRICE EQUATION###
            p = weights_mm[0] + (weights_mm[1] * values[0]) + (weights_mm[2] * values[1]) + (weights_mm[3] * values[2])

            #NON-intercept model equation:
            #p = weights_mm[0] + (weights_mm[1] * values[0]) + (weights_mm[2] * values[1]) + (weights_mm[3] * values[2])
            #p = weights_mm[0] + ((weights_mm[1] * x1 + weights_mm[3] * x3) * Q[i])

            ##added v2_6: checking if demand is flat or upward sloping
            if weights_mm[3] >= 0:
                print('demand not downward sloping in period 0 in iteration ' + str(runsim_it))
                output_file.write('demand not downward sloping in period 0 in iteration ' + str(runsim_it))

            #changed v2_6: If price is negative, redraw the quantity. (previously price was set to 0 if price was negative.)
            while p <= 0:
                Q[i] = np.random.normal(pre_hist_mean_Q, pre_hist_sd_Q)
                values = [x1, x2, Q[i]] ###CHANGED IN THE INTERCEPT MODEL###
                ###INTERCEPT MODEL PRICE EQUATION###
                p = weights_mm[0] + (weights_mm[1] * values[0]) + (weights_mm[2] * values[1]) + (weights_mm[3] * values[2])


            #H is a list which contains a list [price, x1*Q, x2*Q, x3*Q] for each period.
            H.append([p]+values)
            X_shocks.append(X)

        return H

    def calibrate(self, H):
        '''
        The firm re-calibrates its perception of the beta weights.
        It uses OLS regression to fit the prices and quantities it has seen to its mental model; the
        betas resulting from the OLS regression are its new weights.
        The beta weights are adjusted so the firm's perceived demand curve best fits the history.
        Inputs:
            H: prices and values x1*Q, x2*Q, x3*Q from previous periods
            MM: MM for firm 1 is [p, x1*Q, x3*Q]; for firm 2, [p, x1*Q, x3*Q]
        Returns weights, adj_r2
        '''
        MM = self.mm
        H = np.array(H) ##INTERCEPT MODEL H now contains [x1, x2, Q[i]]
        indexes = [0]

        '''
        - This excludes variables not included in firm's mental model.
        - This index is then used to select only the columns related to the relevant betas from the history.
          For example, if [1,1,0,1] is the firm's mental model, then this will save [0,1,3] to the list indexes,
          since those are the positions of the ones. H_firm will then contain the price (at index 0), x1*Q values
          (at index 1) that correspond to beta_1, and x3*Q values (at index 3) that correspond to beta_3.
        - indexes should also contain 0, since this is the index for price in the history.
        - CHANGED IN V2_5: indexes is initialized to contain 0, so that it still works if mm = [0, 1, 1, 1], for example.
          How it was written before, price was only selected because the intercept was included in the mental model,
          and since the intercept and price both happen to be indexed at zero, price was selected.
          The selection was changed so that 0 is included regardless of the mental models, and any non-zero index is then
          added to the indexes list.

        '''
        for j in range(len(MM)):
            if MM[j] == 1:
                if j == 0:
                    pass
                else:
                    indexes.append(j)
        H_firm = H[:, indexes]

        #Store price and whichever explanatory variables are stored H_firm (x1*Q, x2*Q, and/or x3*Q) separately:
        vals = H_firm[: ,1:]
        price = H_firm[: ,0]
        ##Perform OLS on values and price
        output = olsfunction(vals, price)
        ##Store estimated coefficients of demand:
        weights = output['coefficients']
        adj_r2 = output['adjrsq']

        for j in range(len(MM)):
            if MM[j] == 0:
                weights = np.insert(weights, j, 0)

        return weights, adj_r2



    def chooseQ(self, periods, currentP, xvars, max_q, mean, sd, trick_params, market_type,
                monop_firm, so_firm, other_firm_q, mult_cap):
        '''
        Chooses Q for the current period using firm's mental model and x shocks.
        Returns the quantity Q.

        In the trick version, Q will be changed in this function according to the trick parameters.

        V2_5: Changed chooseQ in this version to be identical to the chooseQ in v1_17_4_trick_master_OrderofOps.
        - chooseQ used to take trick_period_list, trick_size, and trick_inc as params; it now takes the list trick_params.
        - Making this change entails changing the inputs for chooseQ where chooseQ is used.
        - Making this change entails changing the parameters in run_simulation, since v2_4 didn't have a lot of the
          trick-related parameters. Need to add these and just set to 0 since this version doesn't deal with tricks.
        - Changed the sophisticated firm case for downward sloping demand to have the trick option; should be skipped
          over in the consolidated base version since trick_type = 'none'
        '''
        #Firm characteristics
        mm = self.mm
        c = self.cost
        weights = self.weights
        m_pos = self.market_position
        max_q = max_q
        x1, x2, x3 = xvars[0], xvars[1], xvars[2]
        trick_type, ta, tamount_space, tperiod_space, trick_len, trick_freq = *trick_params, #for some reason you need , after starred expression for assignment
        period = currentP
        market_type = market_type
        so_firm = so_firm
        is_soph = self.sophistication
        oppQ = other_firm_q

        #Set Q to 0 to initialize Q.
        Q = 0

        '''
        Upward or Flat Demand curve:
        - If the firm perceives the demand curve to be either flat or upward sloping, then it will want to
          produce infinitely.
        - The firm cannot produce infinitely, since we've assigned it a maximum quantity. When the firm wants
          to produce infinitely, it instead produces the max quantity, max_q.
        - Expected price:
            - The monopolist estimates the price based when the industry quantity is max_q, since it believes it is the
              only firm in the market.
            - The duopolist estimates the price when the industry quantity is max_q * 2, since it believes the other firm
              will produce the same amount as itself.
        - Then the firm decides whether producing at max_q would be profitable at the expected price:
            - if no, it produces nothing.
            - if yes, it produces max_q.
                - It is possible for the monopolist to produce a multiple of the maximum quantity, mult_cap * max_q,
                  but this is not used in the basic consolidated version (i.e., mult_cap = 1).
        - [FILL IN] Capacity testing info and how this works.
        - [FILL IN] Do I need to edit this section so the soph. person estimates price based on its knowledge of the other firm's Q?
                    Or would the other firm also necessarily choose max_q in this situation?


        Downward Sloping Demand Curve:
        - If the intercept, weights[0], is negative, then Q = 0. This is to make sure the demand curve is well behaved.
        - If the demand curve is well behaved, then there are three cases.
            1. The firm is a monopolist.
               It will choose the monopoly quantity based only off of its own Q.
            2. The firm is a sophisticated duopolist.
               It will choose the duopoly quantity based off of its own Q and its opponents true Q, oppQ.
               It should not be true that is_soph == 1 and m_pos == monop, since parameters shouldn't be set this way.
               The monopolist is not sophisticated by definition; if it were sophisticated, then it would realize that
               it is not a monopolist.
            3. The firm is a self-reflective/unsophisticated duopolist.
               It will choose the duopoly quantity, assuming that its opponent chooses the same thing as itself.
               In other words, it assumes that the total industry quantity is twice its own Q.
               This is the Cournot Quantity where Q(i) = Q(-i).

        - Boundary conditions: makes sure that the quantity is non-negative and doesn't exceed the quantity cap, max_q.
            - This should be outside of the other if statements about the slope of demand.
        - [FILL IN] Capacity testing info and how this works.
        '''

        #Upward sloping or flat:
        ###INTERCEPT MODEL: CAN ONLY BE UPWARD SLOPING OR FLAT IF THE IND_BETA TERM IS >= 0###
        if (weights[3] * mm[3]) >= 0:
            print('upward slanting slope in chooseQ')
            print(str(weights[3]) + ' is the slope of the demand curve for firm with mental model ' + str(mm))
            if m_pos == 'monop':
                #monopoly considers only their own max_q, since they believe they are alone in the market.
                ###INTERCEPT MODEL: p = alpha_0 + idio_alpha_1 * x1 * mm[1] + idio_alpha_2 * x2 * mm[2] + ind_beta * Q###
                p_exp = weights[0] + (mm[1]*weights[1]*x1) + (mm[2]*weights[2]*x2) + (mm[3]*weights[3]*max_q)
            else:
                #INTERCEPT MODEL#
                p_exp = weights[0] + (mm[1]*weights[1]*x1) + (mm[2]*weights[2]*x2) + (mm[3]*weights[3]*2*max_q)
                #print('expected price: ' + str(p_exp))
            profit_exp = (p_exp - c) * max_q
            if profit_exp < 0:
                Q = 0
            else:
                if market_type == 'monop_duop':
                    Q = max_q*mult_cap
                    self.cap += 1
                    self.cap_posdemand += 1
                else:
                    Q = max_q
                    self.cap += 1
                    self.cap_posdemand += 1

        #Downward sloping:
        else:
            if weights[0] <0:
                print('negative intercept detected in chooseQ')
                Q = 0
            else:
                if m_pos == 'monop':
                    #Case 1 - Monopolist
                    #Not changed for intercept model yet:
                    Q = (2*c - c - weights[0])/(2*(weights[1]*x1*mm[1] + weights[2]*mm[2]*x2 + mm[3]*weights[3]*x3))
                elif (self.comp_model == 'so') and (is_soph == 1):
                    #Case 2 - Sophisticated duopolist
                    #not changed for intercept model yet:
                    Q = (c - weights[0] - (oppQ*(weights[1]*x1*mm[1] + weights[2]*mm[2]*x2 + mm[3]*weights[3]*x3))) / (2 * (weights[1]*x1*mm[1] + weights[2]*mm[2]*x2 + mm[3]*weights[3]*x3))
                else:
                    #Case 3 - Self-reflective duopolist
                    ###INTERCEPT MODEL:
                    Q = (c - (weights[0] + weights[1]*x1*mm[1] + weights[2]*mm[2]*x2))/(3*(weights[3]))
                    #print(str(Q) + ' in period ' + str(currentP) + ' for firm with weights ' + str(weights))
                    #Q = (2*c - c - weights[0])/(3*(mm[3]*weights[3]))
                    #Slope model: Q = (2*c - c - weights[0])/(3*(weights[1]*x1*mm[1] + weights[2]*mm[2]*x2 + mm[3]*weights[3]*x3))
            p_exp = weights[0] + (mm[1]*weights[1]*x1) + (mm[2]*weights[2]*x2) + (mm[3]*weights[3]*2*Q)
            profit_exp = (p_exp - c) * Q
        #Boundary conditions: Quantity
        if Q < 0:
            Q = 0
            self.Q_lessthan_zero += 1
        if Q > max_q:
            if market_type == 'monop_duop':
                Q = max_q*mult_cap
                self.cap += 1
                self.cap_maxedout += 1
            else:
                Q = max
                self.cap += 1
                self.cap_maxedout += 1

        return Q, p_exp, profit_exp

def run_iteration(firmnum, unknown, compmod, market_type, truebetas, init_betas,
                   mental_models, prehistlen, dist_params, cost, periods, periods_examind,
                   max_q, mult_cap, iterations, trick_params, monop_firm, so_firm,
                   figure_style, directory, filebegin, param_num, Q, p_exp, profit_exp, ideal_Q, firm_profits, firm_betas,
                   truevalues, pricevalues, init_weights, dfs, capvalues, capacity, trackers,
                   iter):

        #Bringing in variables
        param_num, figure_style, filebegin, directory = param_num, figure_style, filebegin, directory
        mental_models, init_betas, truebetas = mental_models, init_betas, truebetas
        c, max_q, periods, firmnum, iterations  = cost, max_q, periods, firmnum, iterations
        prehistmean, prehistsd, mean, sd, prehistmean_Q, prehistsd_Q = dist_params[0], dist_params[1], dist_params[2], dist_params[3], dist_params[4], dist_params[5]
        market_type, comp_model = market_type, compmod
        trick_params = trick_params #changed in v2_5; consistent with v1_17_4.
        monop_firm, so_firm = monop_firm, so_firm

        tperiod_space = trick_params[3]

        profit25 = trackers[0]
        profit50 = trackers[1]
        profit75 = trackers[2]
        profit100 = trackers[3]
        profit1000 = trackers[4]
        x_zero = trackers[5]
        x1_2 = trackers[6]
        x1_3 = trackers[7]
        x1_4 = trackers[8]
        x2_2 = trackers[9]
        x2_3 = trackers[10]
        x2_4 = trackers[11]
        x3_2 = trackers[12]
        x3_3 = trackers[13]
        x3_4 = trackers[14]
        upward_sloping_per = trackers[15]
        upward_sloping = trackers[16]
        price_1000 = trackers[17]
        price_150 = trackers[18]
        price_125 = trackers[19]
        price_100 = trackers[20]
        price_85 = trackers[21]
        price_70 = trackers[22]

        #Not sure why this format isn't working.
        #profit25, profit50, profit75, profit100, profit1000, x_zero, x1_2, x1_3, x1_4, x2_2, x2_3, x2_4,
        #x3_2, x3_3, x3_4, upward_sloping_per, upward_sloping, price_1000, price_150, price_125, price_100,
        #price_85, price_70 = *trackers,


        covx1x2, covx1x3, covx2x3 = [0,0,0]
        mean = mean
        means = [mean, mean, mean]
        sd = sd
        covar_matrix = [[sd**2, covx1x2, covx1x3],
                        [covx1x2, sd**2, covx2x3],
                        [covx1x3, covx2x3, sd**2]]

        runsim_it = iter #added v2_5 to use as an index to unique the pre-hist graphs

        #Model description: If the market type is the monop_duop model, then firm i (which is determined by
        #parameters) is the monopoly firm. This is indicated by the firm's market position, either monop or duop. If the
        #competitive model is 'sr', then both firms are self-reflective. If the competitive model is 'so' and the market
        #model is monop_duop, then the duopoly is the sophisticated firm. [If the competitive model is 'so' and the market
        #model is base duop, then the correct firm is the sophisticated firm.-- this is incomplete and different from
        #previous version].
        #Note: This begins with period 1; in period 0, betas are calibrated based on the pre-history but no Q's are chosen.

        '''
        Creating firm objects.
        - Initialize empty dictionary of firms.
        - For each firm:
            - Assign a mental model and initial betas to the firm based on the firmnum index i and create Firms[i],
              an instant of the Firm class.
            - Assign the firm's market position based on monop_firm. Also saves the monopoly firm under the variable 'monopoly'
            - Firms[i].sophistication = 1 if the firm is sophisticated, and 0 otherwise. Saves the firms as 'soph' and 'unsoph' respectively.
            - Generates a pre-history for the firm
        - [FILL IN]. Info about how the initial weights are saved.
        '''

        Firms = {}
        for i in range(firmnum):
            mm = mental_models[i]
            b = init_betas[i]
            Firms[i] = Firm(mm, b, compmod, c, market_type)

            if (market_type == 'monop_duop') and (i == monop_firm):
                Firms[i].market_position = 'monop'
                monopoly = Firms[i] #this variable is currently not used
            else:
                Firms[i].market_position = 'duop'

            if i != so_firm:
                unsoph = Firms[i] #label the unsophisticated firm for later use; this is used for the sophisticated firm to calibrate the unsophisticated firm's betas.
            else:
                soph = Firms[i] #this variable is current not used

            if i == so_firm:
                Firms[i].sophistication = 1
            else:
                Firms[i].sophistication = 0

            Firms[i].H = Firms[i].gen_init_history(prehistlen, prehistmean, prehistsd,
                                                   prehistmean_Q, prehistsd_Q, directory, filebegin, runsim_it, i)
            Firms[i].ideal_H = Firms[i].H[:]
            Firms[i].expectations = [] #initializes the expectations as an array.
            #print('Firm ' + str(i) + ' prehistory: ')
            #print(Firms[i].H)

            #Saving the firm's initial weights: How is this being used? Might be able to get rid of this once the order
            #of ops is fixed:

            #print("initial betas for this firm: " + str(init_betas))
            Firms[i].weights = Firms[i].calibrate(Firms[i].H)[0] ##check to make sure this is properly updating if theres issues!

            weights = Firms[i].weights
            #print("weights after first calibration: " + str(weights))
            #weights[0] = round(weights[0], 2)

            init_weights[i, iter] = Firms[i].weights

            Firms[i].firmvals = [] #initiate a list where the firm will store the history of the interaction

            #for debugging sr_base init intercept value issue:
            #comparing weights (which were arrived at by calibrating) to the init_betas (or true betas)
            #create a list of the true betas as floats to compare to the calibrated weights (which are floats)
            init_b = np.array(init_betas[i], dtype=np.float32)
            for weight in weights:
                if weight not in init_b:
                    #print("Not matching in iteration " + str(iter) + " for firm " + str(i))
                    output_file.write("\n Not matching in iteration " + str(iter) + " for firm " + str(i))
                    #print('weights ' + str(weights))
                    output_file.write('\n weights ' + str(weights))
                    #print('init_b floats: ' + str(init_b))
                    output_file.write('\n init_b floats: ' + str(init_b))
                    #if one beta doesn't match, break since you don't need to check if the other betas match.
                    break
                    ### CHANGED V2_5 run "J17_rounded2": if weight is not in init_b, round up for positive numbers, round down for negative numbers.
                    #if weight < 0:
                    #    weight = math.floor(weight)
                    #if weight > 0:
                    #    weight = math.ceil(weight)
                    #print('rounded weight: ' + str(weight))
                    #print('history for wrong betas: ' + str(Firms[i].H))
                    #raise ValueError("ERROR WITH PERIOD 0 BETAS NOT MATCHING THE INITIAL BETAS")
            #print('weights after loop: ' + str(weights))
        '''
        [FILL IN] if there is a sophisticated firm, the soph firm saves its opponents initial weights and mental model.
        '''
        if comp_model == 'so':
            if so_firm == 1:
                #If the sophisticated firm is firm 2 (or Firms[1]), then it saves firm 1's info.
                Firms[1].weights_opp = Firms[0].weights
                Firms[1].mm_opp = Firms[0].mm
            else:
                #If the sophisticated firm is firm 1 (or Firms[0]), as it normally is, then it saves firm 2's info.
                Firms[0].weights_opp = Firms[1].weights
                Firms[0].mm_opp = Firms[1].mm





        '''
        [FILL IN] Come back and check that this is consistent with what is happening after the period 0 change has happened;
        also try to sort out what the firm 'perceives' and what it does not 'perceive'
        After firms and their pre-histories have been created, it now runs the in-time simulation:
        '''
        for t in range(periods):

            #There are random shocks to demand in every period. Each x corresponds to a beta that can
            #be considered by the firm to influence demand (is an explanatory variable in the demand fcn).

            X = np.random.multivariate_normal(means, covar_matrix)


            ##ADDED IN V2_5 TO ELIMINATE PEAKS, since negative x values could cause positive slopes and extreme profits.

            while (X[0] < 0) or (X[1] < 0) or (X[2] < 0):
                X = np.random.multivariate_normal(means, covar_matrix)

            x1 = X[0]
            x2 = X[1]
            x3 = X[2]
            xvars = [x1,x2,x3]



            if x1 > 2:
                x1_2 += 1
            if x1 > 3:
                x1_3 += 1
            if x1 > 4:
                x1_4 += 1
            if x2 > 2:
                x2_2 += 1
            if x2 > 3:
                x2_3 += 1
            if x2 > 4:
                x2_4 += 1
            if x3 > 2:
                x3_2 += 1
            if x3 > 3:
                x3_3 += 1
            if x3 > 4:
                x3_4 += 1

            ##INTERCEPT MODEL: only upward sloping if the one beta param is positive##
            if truebetas[3] >= 0:
                upward_sloping += 1
                #print('\n true demand is upward sloping in' + str(t))
                output_file.write('\n true demand is upward sloping in' + str(t))

            #Each firm chooses a quantity, perceives its competitor's quantity, and then saves firmvals (i.e. components
            #of demand, X*Q_perceived) based on its the quantity it thinks was produced in the whole market.
            for i in range(firmnum):

                '''A sophisticated firm perceives what the other firm's Q will be before choosing its own, similar to
                   Stackelberg competition model.

                   Changed in v2_5: sophisticated firm now uses chooseQ() to estimate its opponent's quantity.
                   '''
                if (comp_model == 'so') & (i == so_firm):
                    weights_opp = Firms[i].weights_opp
                    mm_opp = Firms[i].mm_opp
                    #the sophisticated firm estimates the quantity of the other firm correctly,
                    #using that firm's weights on the betas:

                    if i == 1:
                        neg_i = 0
                    else:
                        neg_i = 1

                    fake_oppQ = 0 ##Added just because chooseQ takes this, needs to be filled in, but won't be used here.
                    oppQ_est, opp_p_exp, opp_prof_exp = Firms[neg_i].chooseQ(periods, t, xvars, max_q, mean, sd, trick_params,
                                           market_type, monop_firm, so_firm, fake_oppQ, mult_cap)
                    #oppQ_est, ideal_oppQ_est = Firms[neg_i].chooseQ(periods, t, xvars, max_q, mean, sd, trick_params,
                    #                       market_type, monop_firm, so_firm, fake_oppQ, mult_cap)

                else:

                    '''This is the case where the firm doesn't know what its competitor's Q is.
                       I'm setting oppQ_est > max_q since this won't happen if oppQ is actually estimated.
                       There is no significance to choosing this Q specifically, other than that it is > max_q
                       and an impossible quantity for oppQ_est in the case where the firm is sophisticated.
                       So when the firm goes to choose Q, oppQ_est > max_q is a sign that it is not sophisticated,
                       indicating something is wrong with the code if this oppQ_est is used.
                       Need to assign something to oppQ_est so it can go into chooseQ as a parameter.
                       '''

                    oppQ_est = max_q * mult_cap * 100
                    if comp_model == 'so' and i == so_firm:
                        print('error line 260 run sim')


                ##ADDED V2_5: Track when the firm perceives demand to be upward sloping
                ##INTERCEPT MODEL: only upward sloping if the industry beta is upward
                if weights[3] >= 0:
                    upward_sloping_per += 1
                    #print('firm ' + str(i) + ' perceived upward or flat sloping before choosing Q in period ' + str(t))
                    #print(str(weights[1]) + ', ' + str(weights[2]) + ', ' + str(weights[3]))
                    output_file.write('\n firm ' + str(i) + ' perceived upward or flat sloping before choosing Q in period ' + str(t))
                    output_file.write('\n' + str(weights[1]) + ', ' + str(weights[2]) + ', ' + str(weights[3]))
                #Firm i chooses a quantity for this period t.

                Q[t,i], p_exp[t,i], profit_exp[t,i] = Firms[i].chooseQ(periods, t, xvars, max_q, mean, sd, trick_params,
                                           market_type, monop_firm, so_firm, oppQ_est, mult_cap)

                if (i == so_firm) and (comp_model == 'so') and (trick_type == 'custom_percent') and (ta != 1000) and (t in tperiod_space):
                    #percentage model.
                    Q[t,i] = Q[t,i] * ta
                elif (i == so_firm) and (comp_model == 'so') and (trick_type != 'none') and (ta != 1000) and (t in tperiod_space):
                    #if the sophisticated firm is going to play a trick, save the chosen Q as the ideal Q and set the Q to the trick amount
                    #ideal_Q[t,i] = Q[t,i] #MIGHT BE ABLE / HAVE TO COMMENT THIS OUT!
                    Q[t,i] = ta
                elif (i == so_firm) and (comp_model == 'so') and (trick_type == 'old_paper_test') and (ta != 1000):
                    #SPECIFICALLY FOR TRYING TO RECREATE OLD PAPER GRAPHS!
                    Q[t,i] = 1.0

                #if (i == so_firm) and (comp_model == 'so') and (trick_type != 'none') and (ta != 1000) and (Q[t,i] != ta):
                    #print('trick firm not choosing trick quantity in period ' + str(t))
                    #output_file.write('\n trick firm not choosing trick quantity in period ' + str(t))

                ##V2_5: changed so chooseQ takes trick_params instead of trick_period_list, trick_size, and trick_inc to be consistent with changes made to the chooseQ function.

                '''Firm i perceives its competitor's quantity. This part is different depending on whether the firm
                is sophisticated or self-reflective. If it is sophisticated, it will perceive the competitor's
                quantity correctly (so it's firmvals will be true):'''


                #If this firm is the sophisticated firm in a model with sophistication.
                if (comp_model == 'so') & (i == so_firm):
                    #Uses its estimate of its competitor and its own quantity to estimate the total quantity in the market:
                    est_Q = oppQ_est + Q[t,i]
                    vals = [x1, x2, est_Q] ##INTERCEPT MODEL CHANGE!
                    Firms[i].firmvals.append(vals) #these firm vals should be true!
                    Firms[i].perceivedQ[t] = (est_Q, oppQ_est, Q[t,i])
                    #print('1. Firm ' + str(i) + ' period ' + str(t) + ' firm vals length ' + str(len(Firms[i].firmvals)))
                    #print('1. Firm ' + str(i) + ' period ' + str(t) + ' firm vals ' + str(Firms[i].firmvals))
                else:
                    #If the firm believes its a monop, then it thinks market Q is its own Q
                    if Firms[i].market_position == 'monop':
                        Firms[i].Qperceived = Q[t,i]
                    #If the firm is a self-reflective duopoly, then it thinks the other firm played the same
                    #quantity as itself, and believes the market quantity is twice its own Q
                    else:
                        Firms[i].Qperceived = Q[t,i] * 2

                    #Add values to the firm's history of what it thinks happened in this interaction
                    Qper = Firms[i].Qperceived
                    vals = [x1, x2, Qper] ##INTERCEPT MODEL CHANGE!
                    Firms[i].firmvals.append(vals) #Store firm i's perceived values of X*Q
                    #print('2. Firm ' + str(i) + ' period ' + str(t) + ' firm vals length ' + str(len(Firms[i].firmvals)))
                    #print('2. Firm ' + str(i) + ' period ' + str(t) + ' firm vals ' + str(Firms[i].firmvals))


            #Calculate total actual Q per period:
            Qtotal = np.sum(Q[t,:])

            #Fill dictionary with true values per period:
            truevalues[t] = [x1, x2, Qtotal] ###INTERCEPT MODEL CHANGE!


            #calculate price per period and save in pricevalues array

            ###INTERCEPT MODEL: the true price is determined by x1 and idio_alpha_1, not x2 and idio_alpha_2
            p = truebetas[0] + (truebetas[1] * x1) + (truebetas[3] * Qtotal)

            #Non-intercept model price equation: p = np.inner(truebetas, [1] + truevalues[t])
            #if p < 0:
            #    p = 0
            #    output_file.write('\n firm ' + str(i) + ' perceived price < 0 ' + str(t))
            if p > 1000:
                price_1000 += 1
            if p > 150:
                #p = 150
                price_150 += 1
            if p > 125:
                price_125 +=1
            if p > 100:
                price_100 +=1
                #p = 100
            if p > 85:
                price_85 +=1
            pricevalues[t] = p

            '''Create history of explanatory vars, prices, profits.'''
            '''Calibrate to fit OLS to what has happened in past periods.'''
            '''Self-reflective only looks at its own history.'''
            '''Sophisticated looks at combined history of all firms'''
            comb_hist = []

            for i in range(firmnum):
                #print('3. Firm ' + str(i) + ' period ' + str(t) + ' firm val this period ' + str(len(Firms[i].firmvals[t])))
                Firms[i].H.append([p] + Firms[i].firmvals[t])
                #print([p] + Firms[i].firmvals[t])
                #print('4. Firm ' + str(i) + ' period ' + str(t) + ' firm hist length ' + str(len(Firms[i].H)))
                Firms[i].profit = (p-c)*Q[t,i]
                Firms[i].expectations.append([p_exp[t,i], profit_exp[t,i]]) #expectations is a list of arrays, like firm history
                firm_profits[t,i] = Firms[i].profit
                ''' #ideal_Q version:
                if (i == so_firm) and (comp_model == 'so'):
                    #!!DELETED FROM THE IF STATEMENT:  and (trick_type != 'none') and (ta != 1000)
                    #For a sophisticated firm playing a trick, the calibration needs to happen based on the ideal_Q, not the trick Q that is actually chosen.
                    #1. Calculate the total Q perceived by the soph firm (which would have been correct) if the sophisticated firm had played its ideal Q instead of the trick Q:
                    ideal_Qtotal = oppQ_est + ideal_Q[t,i]
                    #2. Calculate what the vals would have been with the ideal Q:
                    ideal_truevals = [x1*ideal_Qtotal, x2*ideal_Qtotal, x3*ideal_Qtotal]
                    ideal_firmvals = ideal_truevals #since the firm is sophisticated (might consider calculating these separately to make sure)
                    #3. Calculate the ideal p that would have resulted from these values.
                    ideal_p = np.inner(truebetas, [1] + ideal_truevals)
                    #if ideal_p < 0:
                    #    ideal_p = 0
                    #If the firm is playing a trick, it also needs to compile an ideal history.
                    #print('5. Firm ' + str(i) + ': ideal hist length before appending: ' + str(len(Firms[i].ideal_H)))
                    #print('ideal_firmvals: ' + str(ideal_firmvals))
                    Firms[i].ideal_H.append([ideal_p] + ideal_firmvals)
                '''
                comb_hist.append(Firms[i].H)

            ###ALSO SAVE FIRM EXPECTATIONS###:


            for i in range(firmnum):

                '''#ideal Q version:
                if (i == so_firm) and (comp_model == 'so'):
                    ##DELETED FROM THE IF STATEMENT:  and (trick_type != 'none') and (ta != 1000)
                    #If sophisticated and plays a trick, calibrate on the ideal history.
                    weights, adjr2 = Firm.calibrate(Firms[i], H = Firms[i].ideal_H)
                    #print('Firm ' + str(i) + ': ideal hist: ')
                    #print(Firms[i].ideal_H)
                else:
                    #All firms calibrate based on own history:
                    weights, adjr2 = Firm.calibrate(Firms[i], H = Firms[i].H)
                '''
                weights, adjr2 = Firm.calibrate(Firms[i], H = Firms[i].H)
                firm_betas[t,i] = weights
                Firms[i].weights = weights

                #For sophisticated duopoly, also calibrates monopoly's model:
                if (comp_model == 'so') & (i == so_firm):
                    Firms[i].weights_opp, adjr2_opp = Firm.calibrate(unsoph, H = unsoph.H)

        return Firms



def run_simulation(firmnum, unknown, compmod, market_type, truebetas, init_betas,
                   mental_models, prehistlen, dist_params, cost, periods, periods_examind,
                   max_q, mult_cap, iterations, trick_params, monop_firm, so_firm,
                   figure_style, directory, filebegin, param_num):

    '''
    INTERCEPT MODEL CHANGES:
    - changed the structure of init_betas and truebetas, with betas now referring to
    all of the parameters, both alpha and betas.
    -
    '''

    #Bringing in variables
    param_num, figure_style, filebegin, directory = param_num, figure_style, filebegin, directory
    mental_models, init_betas, truebetas = mental_models, init_betas, truebetas
    c, max_q, periods, firmnum, iterations  = cost, max_q, periods, firmnum, iterations
    prehistmean, prehistsd, mean, sd, prehistmean_Q, prehistsd_Q = dist_params[0], dist_params[1], dist_params[2], dist_params[3], dist_params[4], dist_params[5]
    market_type, comp_model = market_type, compmod
    trick_params = trick_params #changed in v2_5; consistent with v1_17_4.
    monop_firm, so_firm = monop_firm, so_firm

    #Create directory if it does not already exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    #Distribution Parameters
    covx1x2, covx1x3, covx2x3 = [0,0,0]
    mean = mean
    means = [mean, mean, mean]
    sd = sd
    covar_matrix = [[sd**2, covx1x2, covx1x3],
                    [covx1x2, sd**2, covx2x3],
                    [covx1x3, covx2x3, sd**2]]

    #Have firms compete in each period (assumes unobserved Q!)
    Q = np.empty((periods, firmnum))
    p_exp = np.empty((periods, firmnum))
    profit_exp = np.empty((periods, firmnum))
    ideal_Q = np.empty((periods, firmnum))
    firm_profits = np.empty((periods, firmnum))
    firm_betas = {}
    truevalues = {}
    pricevalues = {}
    init_weights = {}

    #Create a master list of dataframes to hold the dataframe created by each iteration.
    dfs = []

    #Create list for capacity testing:
    capvalues = []
    capacity = dict()

    '''
    Outlier profit and x-shock tracking:
    - Notice that these are counting instances of high profits across ALL iterations and that there is no averaging.
    '''
    profit25 = 0
    profit50 = 0
    profit75 = 0
    profit100 = 0
    profit1000 = 0
    x_zero = 0
    x1_2 = 0
    x1_3 = 0
    x1_4 = 0
    x2_2 = 0
    x2_3 = 0
    x2_4 = 0
    x3_2 = 0
    x3_3 = 0
    x3_4 = 0


    upward_sloping_per = 0
    upward_sloping = 0
    price_1000 = 0
    price_150 = 0
    price_125 = 0
    price_100 = 0
    price_85 = 0
    price_70 = 0

    trackers = [profit25, profit50, profit75, profit100, profit1000, x_zero, x1_2, x1_3, x1_4, x2_2, x2_3, x2_4,
    x3_2, x3_3, x3_4, upward_sloping_per, upward_sloping, price_1000, price_150, price_125, price_100, price_85, price_70]

    '''SIMULATION OF COMPETITIVE INTERACTION'''
    for iter in range(iterations):

        Firms = run_iteration(firmnum, unknown, compmod, market_type, truebetas, init_betas,
                   mental_models, prehistlen, dist_params, cost, periods, periods_examind,
                   max_q, mult_cap, iterations, trick_params, monop_firm, so_firm,
                   figure_style, directory, filebegin, param_num, Q, p_exp, profit_exp, ideal_Q, firm_profits, firm_betas,
                   truevalues, pricevalues, init_weights, dfs, capvalues, capacity, trackers,
                   iter)



        '''Saving simulation data to csv.'''
        ##BELOW APPLIES TO ONLY N = 2 FIRM SIMULATIONS! DOES NOT SAVE DATA PROPERLY FOR N=/=2
        #Make a firm history dataframe
        #Add firm-specific data to one df
        data = []
        for n in range(firmnum):
            firmhist = np.array(Firms[n].H[prehistlen:]) #excludes the prehistory
            for l in range(len(firmhist)):
                data.append(firmhist[l])
            #print('firm no ' + str(n))
            #print(len(data))
        firmhist_df = pd.DataFrame(data, columns = ['price', 'x1*Q', 'x2*Q', 'x3*Q'])
        del firmhist_df['price']
        #split into 2 so they can be concatenated side-by-side; before this, the two firm histories are stacked on top of
        #one another.
        firmhist_df1 = firmhist_df.iloc[:periods,:]
        #print('firmhist_df1:')
        #print(firmhist_df1)
        firmhist_df2 = firmhist_df.iloc[periods:,:]
        firmhist_df2.index = firmhist_df2.index - periods #corrects index
        #print('firmhist_df2:')
        #print(firmhist_df2)

        #make firm expectations dataframe:
        exp_data = {}
        exp_dict = {}
        for n in range(firmnum):
            exp_data[n] = []
            firmexp = np.array(Firms[n].expectations)
            for l in range(len(firmexp)):
                exp_data[n].append(firmexp[l])
            exp_dict[n] = pd.DataFrame(exp_data[n], columns = ['exp_price', 'exp_profit'])
        exp_df = pd.concat([exp_dict[0], exp_dict[1]], axis=1, sort = True)
        exp_df.columns = ['exp_price_1', 'exp_profit_1', 'exp_price_2', 'exp_profit_2']

        #make firm profit dataframe:
        firmprof_df = pd.DataFrame(firm_profits, columns = ['prof1', 'prof2'])
        #make quantity dataframe:
        Q_df = pd.DataFrame(Q, columns = ['Q1', 'Q2'])
        Q_df['total_Q'] = Q_df.Q1 + Q_df.Q2

        #make price dataframe (Don't need this..same as first column of firmhist_df):
        price_df = pd.DataFrame(pricevalues, index = [0]).transpose()
        price_df.columns = ['price'] # # # 100 rows of 24 vars each iteration...

        #dataframe of firm betas... want to separate firm 1 and firm 2 using multi-index...
        firmb_df = pd.DataFrame.from_dict(firm_betas)
        firmb_df= firmb_df.transpose()
        firmb_df = firmb_df.reset_index(level=[1])
        firmb_df.columns = ['firm', 'int', 'a1', 'a2', 'b']
        #separate into betas by firm, so can be concatenated side-by-side.
        firmb_df1 = firmb_df[firmb_df['firm']==0]
        firmb_df1.columns = ['firm', 'f1_int', 'f1_a1', 'f1_a2', 'f1_b']
        del(firmb_df1['firm'])
        firmb_df2 = firmb_df[firmb_df['firm']==1]
        firmb_df2.columns = ['firm', 'f2_int', 'f2_a1', 'f2_a2', 'f2_b']
        del(firmb_df2['firm'])

        #dataframe of truevales (a dictionary)
        truevals_df = pd.DataFrame.from_dict(truevalues).transpose()
        truevals_df.columns = ['true_x1Q', 'true_x2Q', 'true_x3Q']

        ''' ADDED IN VERSION 13 TO TRACK SOPH MISTAKES: add perceived Q to the csv.
            Adds only the sophisticated firm perceptions '''
        ''' ADDED IN V14: fixed this section so it also runs when not sophisticated without
            throwing error for sophfirmperQ not existing.'''
        #print(Firms[so_firm].perceivedQ)
        #print(type(Firms[so_firm].perceivedQ))
        if comp_model == 'so':
            sophfirmperQ = pd.DataFrame.from_dict(Firms[so_firm].perceivedQ, orient='index')
            sophfirmperQ.columns = ['F1_perQ', 'F1_peroppQ', 'F1_perownQ']
            sim_data = pd.concat([price_df,Q_df,sophfirmperQ, truevals_df,firmprof_df,firmb_df1,firmb_df2,firmhist_df1,firmhist_df2, exp_df], axis=1, sort = True)
        #Storing data in dataframe for each iteration:
        else:
            sim_data = pd.concat([price_df,Q_df,truevals_df,firmprof_df,firmb_df1,firmb_df2,firmhist_df1,firmhist_df2, exp_df], axis=1, sort = True)
        dfs.append(sim_data)

        ###############################################################
        # Price v Q Graphs for each iteration's in-simulation history #
        ###############################################################

        '''
        plot_destination = os.path.join(directory, 'Qrunplots')
        if not os.path.exists(plot_destination):
            os.makedirs(plot_destination)

        print(sim_data.columns)
        #print("sim data index:")
        #print(sim_data.index.tolist())
        sim_data['period'] = sim_data.index
        #print("sim data period:")
        #print(sim_data['period'])
        print(sim_data.total_Q)

        #runplotdf = pd.concat([Q_df, ])
        quant_fig = sns.regplot(x = sim_data.total_Q, y = sim_data.price, data = sim_data, ci = None, color = 'blue', label = 'totalQ')
        sns.regplot(x = sim_data.Q1, y = sim_data.price, data = sim_data, ci = None, color = 'red', label = 'Q1')
        quant_fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=2)
        sim_data['total_Q_rounded'] = round(sim_data.total_Q, 2)
        sim_data['price_rounded'] = round(sim_data.price, 2)
        sim_data['Q1_rounded'] = round(sim_data.Q1, 2)
        for pt_zip in zip(sim_data.total_Q_rounded, sim_data.price_rounded, sim_data.period, sim_data.Q1_rounded):
            print(pt_zip)
            quant_fig.annotate(str(pt_zip[2]), xy = (pt_zip[0], pt_zip[1]), fontsize = 7, color = 'black')
            quant_fig.annotate(str(pt_zip[2]), xy = (pt_zip[3], pt_zip[1]), fontsize = 7, color = 'dimgrey')
        #sns.regplot(x = sim_data.Q, y = sim_data.price, data = sim_data, ci = None, color = 'green')
        #quant_fig.set(xlim = (0, 5), ylim = (20, 115))
        plt.savefig(os.path.join(directory, 'Qrunplots', 'Qrunplot' + str(filebegin) + '_iteration' + str(iter) + '.png'), bbox_inches='tight')
        plt.close()
        plt.cla()
        plt.clf()
        '''

        '''Totaling the per-iteration capacity (Method 1)'''
        total_zero1, total_zero2, total_cap1, total_cap2 = (0,0,0,0)
        for j in range(len(Q_df)):
            if Q_df.iloc[j]['Q1'] == 0:
                total_zero1 +=1
            if Q_df.iloc[j]['Q2'] == 0:
                total_zero2 +=1
            if Q_df.iloc[j]['Q1'] == max_q:
                total_cap1 +=1
            if Q_df.iloc[j]['Q2'] == max_q:
                total_cap2 +=1
        cap_test = np.array([total_zero1, total_zero2, total_cap1, total_cap2])
        cap_test = (cap_test/iterations)
        cap_test_df = pd.DataFrame(cap_test)
        cap_test_df = cap_test_df.T
        cap_test_df.columns = ['zero1%','zero2%','cap1%','cap2%']
        capvalues.append(cap_test_df)
        '''Totaling per-iteration capacity (Method 2)'''
        for n in range(firmnum):
            capacity[iter, n] = (Firms[n].cap, Firms[n].cap_posdemand, Firms[n].cap_maxedout)

    '''Two methods for getting capacity percentages, to check them against each other bc I was having issues.'''
    #print("Print capacity dictionary: ", capacity)

    #Method 1
    total_capvals = np.zeros_like(cap_test_df)
    total_capvals_df = pd.DataFrame(total_capvals)
    for c in capvalues:
        total_capvals_df.columns = cap_test_df.columns
        total_capvals_df.index = cap_test_df.index
        total_capvals_df = c.add(total_capvals_df, fill_value = 0)
    #print(total_capvals_df)


    #Method 2
    zerototal0, zerototal1, captotal0, captotal1  = (0,0,0,0)
    cap_posdemand1, cap_posdemand0, cap_maxedout1, cap_maxedout0 = (0,0,0,0)
    #print("Trying to figure out capacity index.. ", capacity[0,0][0])
    for i in range(iterations):
        ##Counts every time capacity was hit, in every period and in every iterations
        captotal0 += capacity[(i,0)][0]
        cap_posdemand0 += capacity[(i,0)][1]
        cap_maxedout0 += capacity[(i,0)][2]

        captotal1 += capacity[(i,1)][0]
        cap_posdemand1 += capacity[(i,1)][1]
        cap_maxedout1 += capacity[(i,1)][2]
    cap_list = (captotal0, cap_posdemand0, cap_maxedout0, captotal1, cap_posdemand1, cap_maxedout1)
    cap_list_av = np.divide(cap_list,(periods*iterations))*100 #EXPRESSED AS PERCENTAGE!
    #print(cap_list_av)
    #capav0 = (captotal0/(periods*iterations))*100 #The average # of periods the firm hits capacity, expressed as percentage.
    #capav1 = (captotal1/(periods*iterations))*100

    initw_df = pd.DataFrame(init_weights).T
    init_totals = np.zeros((2,4))
    for i in range(firmnum):
        for j in range(iterations):
            init_totals[i] = init_totals[i] + initw_df.loc[i,j]
        init_totals[i] = init_totals[i]/iterations
    inits1 = pd.DataFrame(init_totals[0]).T
    inits2 = pd.DataFrame(init_totals[1]).T
    initav_df = pd.concat([inits1,inits2],axis = 1, sort = True)
    initav_df.columns = ['f1_int', 'f1_a1', 'f1_a2', 'f1_b', 'f2_int', 'f2_a1', 'f2_a2', 'f2_b'] #Names changed for intercept model
    initav_df['period'] = 0
    initial_weights = initav_df
    print("Initial weights: " + str(initial_weights))

    totals = np.zeros_like(sim_data)
    totals_df = pd.DataFrame(totals)

    for df in dfs:
        totals_df.columns = sim_data.columns
        totals_df.index = sim_data.index
        totals_df = df.add(totals_df, fill_value = 0)
    av_data = totals_df/iterations

    '''Save a text file with the parameters and capacity values for this run.'''
    info_file = open(os.path.join(directory,'run_info_' + str(round(initial_weights.f1_a1[0], 2)) + '.txt'), 'w+')
    info_file.write("Capacity Info.")
    info_file.write("\n Firm 1's capacity info: " +
                    "\n  % of periods hitting capacity, unrounded: " + str(cap_list_av[0]) + "%" +
                    "\n  % hitting cap bc perceives positive demand, unrounded: " + str(cap_list_av[1]) + "%" +
                    "\n  % hitting cap bc chooses q>max_q, unrounded: " + str(cap_list_av[2]) + "%" +
                    "\n % hitting cap (not rounded)" + str(cap_list_av[2]) +
                    "\n  number of times choosing Q < 0: " + str(Firms[0].Q_lessthan_zero))
    info_file.write("\n Firm 2's capacity info: " +
                    "\n  % of periods hitting capacity, total: " + str(cap_list_av[3]) + "%" +
                    "\n  % hitting cap bc perceives positive demand: " + str(cap_list_av[4]) + "%" +
                    "\n  % hitting cap bc chooses q>max_q: " + str(cap_list_av[5]) + "%" +
                    "\n  number of times choosing Q < 0: " + str(Firms[1].Q_lessthan_zero))
    info_file.write("\n Count of instances of high profits over all iterations & periods:" +
                    "\n # profits >= 25: " + str(profit25) +
                    "\n # profits >= 100: " + str(profit100) +
                    "\n # profits >= 1000: " + str(profit1000))
    info_file.write("\n Count of instances of outlier x-shocks over all iterations & periods:" +
                    "\n x is negative: " + str(x_zero) +
                    "\n # x1 > 2: " + str(x1_2) +
                    "\n # x1 > 3: " + str(x1_3) +
                    "\n # x1 > 4:" + str(x1_4) +
                    "\n # x2 > 2: " + str(x2_2) +
                    "\n # x2 > 3: " + str(x2_3) +
                    "\n # x2 > 4:" + str(x2_4) +
                    "\n # x3 > 2: " + str(x3_2) +
                    "\n # x3 > 3: " + str(x3_3) +
                    "\n # x3 > 4:" + str(x3_4))
    info_file.write("\n Count of instances where demand is upward sloping:" +
                    "\n Perceived: " + str(upward_sloping_per) +
                    "\n Actual: " + str(upward_sloping) +
                    "\n Price greater than 1000: " + str(price_1000) +
                    "\n Price greater than 150: " + str(price_150) +
                    "\n Price greater than 100: " + str(price_100) +
                    "\n Price greater than 85: " + str(price_85) +
                    "\n Price greater than 70: " + str(price_70))
    # "\n capav1: " + str(round(capav1,2)) + "%")
    info_file.write("\nParameter Info.")
    info_file.write("\n iterations: " + str(iterations) +
                   "\n initial betas: " + str(init_betas) +
                   "\n mental models: " + str(mental_models) +
                   "\n phx_var: " + str(prehistsd) +
                   "\n phq_var: " + str(prehistsd_Q) +
                   "\n x_var:" + str(x_var) +
                   "\n max_q: " + str(max_q) +
                   "\n periods: " + str(periods) +
                   "\n competitor model: " + str(comp_model) +
                   "\n market type: " + str(market_type) +
                   "\n trick parameters (trick_type, ta, tamount_space, tperiod_space, trick_len, trick_freq): " +
                    str(trick_params) +
                   "\n monopoly firm: " + str(monop_firm) +
                   "\n sophisticated firm: " + str(so_firm) +
                   "\n cap multiplier: " + str(mult_cap)
                   )


    av_data['period'] = sim_data.index + 1
    av_data.index = sim_data.index
    av_df = pd.DataFrame.from_dict(av_data) #unnecessary step?

    '''Storing run information in csv - returns csv name'''
    av_df.to_csv(os.path.join(directory,str(filebegin) + '_prior1=' + str(init_betas[0]) + '_prior2=' + str(init_betas[1]) + '_'
                  + 'phlen' + str(prehistlen) + 'phxvar' + str(prehistsd) + 'sim_data.csv'), header = True)
    total_capvals_df.to_csv(os.path.join(directory,str(filebegin) + '_prior1=' + str(init_betas[0]) + '_prior2=' + str(init_betas[1]) + '_' + str(trick_params[1])
                  +  'captest_results.csv'), header = True)
    csv_name = (str(filebegin) + '_prior1=' + str(init_betas[0]) + '_prior2=' + str(init_betas[1]) + '_'
                  + 'phlen' + str(prehistlen) + 'phxvar' + str(prehistsd) + 'sim_data.csv')

    return csv_name, initial_weights

def graph_label(figure, df, graph_content, graph_colors, label_params):

    figure, df = figure, df

    ##Rounding data and finding averages
    f1_data_round = round(graph_content[0], 2)
    f2_data_round = round(graph_content[1], 2)
    av_f2_data = np.mean(f2_data_round)
    av_f1_data = np.mean(f1_data_round)

    F1_inc, F2_inc, zero_inc, zero_horiz = label_params[0], label_params[1], label_params[2], label_params[3]
    f1_edgecolor, f2_edgecolor, f1_boxcolor, f2_boxcolor = graph_colors[0], graph_colors[1], graph_colors[2], graph_colors[3]

    #Loop through zipped points to graph.#
    #Firm 1#
    for pt_zip in zip(df.period, f1_data_round):
        period = pt_zip[0]
        if av_f1_data < av_f2_data:
            #if f1 is smaller, adjust it up
            if period % 25 == 0  and period != 0 and period !=100:
                figure.annotate(str(pt_zip[1]), xy=(pt_zip[0], pt_zip[1] + F1_inc), fontsize = 8,  bbox=dict(boxstyle="round4,pad=.5", edgecolor = f1_edgecolor, fc=f1_boxcolor))
            if period == 0:
                figure.annotate(str(pt_zip[1]), xy=(pt_zip[0] + 8, pt_zip[1] + zero_inc), fontsize = 8,  bbox=dict(boxstyle="round4,pad=.5", edgecolor = f1_edgecolor, fc=f1_boxcolor))
            if period == 100:
                figure.annotate(str(pt_zip[1]), xy=(pt_zip[0] - 5, pt_zip[1] + F1_inc), fontsize = 8,  bbox=dict(boxstyle="round4,pad=.5", edgecolor = f1_edgecolor, fc=f1_boxcolor))
        if av_f1_data > av_f2_data:
            #if f1 is larger, adjust it down
            if period % 25 == 0  and period != 0 and period !=100:
                figure.annotate(str(pt_zip[1]), xy=(pt_zip[0], pt_zip[1] - F1_inc), fontsize = 8,  bbox=dict(boxstyle="round4,pad=.5", edgecolor = f1_edgecolor, fc=f1_boxcolor))
            if period == 0:
                figure.annotate(str(pt_zip[1]), xy=(pt_zip[0] + 8, pt_zip[1] - zero_inc), fontsize = 8,  bbox=dict(boxstyle="round4,pad=.5", edgecolor = f1_edgecolor, fc=f1_boxcolor))
            if period == 100:
                figure.annotate(str(pt_zip[1]), xy=(pt_zip[0] - 5, pt_zip[1] - F1_inc), fontsize = 8,  bbox=dict(boxstyle="round4,pad=.5", edgecolor = f1_edgecolor, fc=f1_boxcolor))

    #Firm 2#
    for pt_zip in zip(df.period, f2_data_round):
        period = pt_zip[0]
        if av_f2_data > av_f1_data:
            #if f2 is larger, adjust it down
            if period % 25 == 0 and period != 0 and period != 100:
                figure.annotate(str(pt_zip[1]), xy=(pt_zip[0], pt_zip[1] - F2_inc), fontsize = 8,  bbox=dict(boxstyle="round4,pad=.5", edgecolor = f2_edgecolor, fc=f2_boxcolor))
            if period == 100:
                figure.annotate(str(pt_zip[1]), xy=(pt_zip[0] - 5, pt_zip[1] - F2_inc), fontsize = 8,  bbox=dict(boxstyle="round4,pad=.5", edgecolor = f2_edgecolor, fc=f2_boxcolor))
        if av_f2_data < av_f1_data:
            #if f2 is smaller, adjust it up
            if period % 25 == 0 and period != 0 and period != 100:
                figure.annotate(str(pt_zip[1]), xy=(pt_zip[0], pt_zip[1] + F2_inc), fontsize = 8,  bbox=dict(boxstyle="round4,pad=.5", edgecolor = f2_edgecolor, fc=f2_boxcolor))
            if period == 100:
                figure.annotate(str(pt_zip[1]), xy=(pt_zip[0] - 5, pt_zip[1] + F2_inc), fontsize = 8,  bbox=dict(boxstyle="round4,pad=.5", edgecolor = f2_edgecolor, fc=f2_boxcolor))

def graph_sns(csv_name, initial_weights, firmnum, unknown, compmod, market_type, truebetas, init_betas,
                   mental_models, prehistlen, dist_params, cost, periods, periods_examind, max_q, mult_cap, iterations,
                   trick_params, monop_firm, so_firm,
                   figure_style, directory, filebegin, param_num):
    '''Time series graphs of simulation... All print separately. This function currently doesn't use figure_style parameter'''

    '''
    Changed in v2_8:
    - Created graphing_params list for putting into the annotate function.
    '''

    graphing_params = [csv_name, initial_weights, firmnum, unknown, compmod, market_type, truebetas, init_betas,
                   mental_models, prehistlen, dist_params, cost, periods, periods_examind, max_q, mult_cap, iterations,
                   trick_params, monop_firm, so_firm,
                   figure_style, directory, filebegin, param_num]

    comp_model = compmod
    figure_style = figure_style
    directory = directory
    mm = mental_models
    truebetas=truebetas
    b = truebetas[1] #INTERCEPT MODEL: THIS NOW INDICATES THE CORRECT IDIO ALPHA VALUE
    periods = periods
    iw_df = initial_weights
    inc_alpha = float(iw_df['f2_a2'])
    inc_alpha = round(inc_alpha, 2)
    trick_params = trick_params

    #bring in data saved in csv:
    df = pd.read_csv(os.path.join(directory, csv_name))

    #Make a beta data frame to include period 0
    dfbeta = df.loc[:, ['period', 'f1_int', 'f1_a1', 'f1_a2', 'f1_b', 'f2_int', 'f2_a1', 'f2_a2', 'f2_b', 'exp_price_1', 'exp_price_2', 'exp_profit_1', 'exp_profit_2']]
    dfbeta = pd.concat([dfbeta, iw_df], axis = 0, sort = True).sort_values(by=['period']).reset_index()

    sns.set_style('white')
    sns.set_context('paper', font_scale=2)

    ##bbox variables:
    f1_edgecolor = 'black'
    f2_edgecolor = 'grey'
    f1_boxcolor = '1.0'
    f2_boxcolor = '0.8'
    graph_colors = [f1_edgecolor, f2_edgecolor, f1_boxcolor, f2_boxcolor]

    #####################################################
    # Plot 1 - idiosyncratic intercept estimate by firm #
    #####################################################

    ## Plotting lines. ##
    #Graph beta1 for firm 1.#
    graph_content = {}
    if mm[0] == [1,1,0,1]:
        fig1 = sns.lineplot(x = 'period', y = 'f1_a1', data = dfbeta, label = 'Firm 1', color = 'black')
        graph_content[0] = dfbeta.f1_a1
    elif mm[0] == [1,0,1,1]:
        fig1 = sns.lineplot(x = 'period', y = 'f1_a2', data = dfbeta, label = 'Firm 1', color = 'black')
        graph_content[0] = dfbeta.f1_a2
    if mm[1] == [1,1,0,1]:
        sns.lineplot(x = 'period', y = 'f2_a1', data = dfbeta, label = 'Firm 2', color = 'grey')
        f2_beta = dfbeta.f2_a1
        graph_content[1] = dfbeta.f2_a1
    elif mm[1] == [1,0,1,1]:
        sns.lineplot(x = 'period', y = 'f2_a2', data = dfbeta, label = 'Firm 2', color = 'grey')
        f2_beta = dfbeta.f2_a2
        graph_content[1] = dfbeta.f2_a2

    #Set the label params depending on model type.#
    label_params = [] #order of label_params: F1_inc, F2_inc, zero_inc, zero_horiz
    if compmod == 'so':
        #sophisticated
        if market_type == 'base_duop':
            #Sophisticated base_duop
            label_params = [0.02, 0.045, 0.02, 0]
        if market_type == 'monop_duop':
            #Sophisticated monop_duop
            label_params = [0.10, 0.015, 0.10, 0]
    elif compmod == 'sr':
        if market_type == 'base_duop':
            #Self-reflective base_duop
            label_params = [0.25, 0.04, 0, 0]
        if market_type == 'monop_duop':
            #Self-reflective monop_duop
            label_params = [0.25, 0.25, 0, 0]

    #graphing_label function takes: figure name, all graphing params, and then label_params list
    graph_label(fig1, dfbeta, graph_content, graph_colors, label_params)

    fig1.set(ylabel='Idiosyncratic variable')
    fig1.lines[1].set_linestyle("--")
    fig1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=2)
    plt.savefig(os.path.join(directory, 'plot1_idio' + str(filebegin) + 'alpha_is_' + str(inc_alpha) + 'seaborn_plt1_3' + '.png'), bbox_inches='tight', dpi = 600)
    plt.close()
    plt.cla()
    plt.clf()


    ###############################################
    # Plot 2 - industry beta estimate by firm #
    ###############################################

    ##Plotting Lines##
    # Plot beta3 for both firms. #
    fig2 = sns.lineplot(x = 'period', y = 'f1_b', data = dfbeta, label = 'Firm 1', color = 'black')
    sns.lineplot(x = 'period', y = 'f2_b', data = dfbeta, label = 'Firm 2', color = 'grey')
    graph_content = {0: dfbeta.f1_b, 1: dfbeta.f2_b}

    #Set different positional variables depending on model type.#
    label_params = []
    if compmod == 'so':
        #sophisticated
        if market_type == 'base_duop':
            label_params = [0.05, 0.05, 0.05, 8]
        if market_type == 'monop_duop':
            label_params = [0.25, 0.25, 0.25, 8]
    elif compmod == 'sr':
        if market_type == 'base_duop':
            label_params = [0.15, 0.15, 0, 5]
        if market_type == 'monop_duop':
            label_params = [0.3, 0.35, 0, 5]

    graph_label(fig2, dfbeta, graph_content, graph_colors, label_params)

    fig2.set(ylabel='Industry variable')
    fig2.lines[1].set_linestyle("--")
    fig2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=2)
    plt.savefig(os.path.join(directory, 'plot2_ind_' + str(filebegin) + 'idio_alpha_' + str(inc_alpha) + 'seaborn_plt1_3' + '.png'), bbox_inches='tight', dpi = 600)
    plt.close()
    plt.cla()
    plt.clf()

    #######################################
    # Plot 3 - intercept estimate by firm #
    #######################################

    ##Plotting Lines##
    # Plot intercept for both firms. #
    fig3 = sns.lineplot(x = 'period', y = 'f1_int', data = dfbeta, label = 'Firm 1', color = 'black')
    sns.lineplot(x = 'period', y = 'f2_int', data = dfbeta, label = 'Firm 2', color = 'grey')
    graph_content = {0: dfbeta.f1_int, 1: dfbeta.f2_int}

    #Set different positional variables depending on model type.#
    if compmod == 'so':
        #sophisticated
        if market_type == 'base_duop':
            label_params = [0.05, 0.05, 0.05, 5]
        if market_type == 'monop_duop':
            label_params = [1.5, 1.5, 1.5, 8]
    elif compmod == 'sr':
        if market_type == 'base_duop':
            label_params = [0.15, 0.15, 0, 5]
        if market_type == 'monop_duop':
            label_params = [0.25, 0.25, 0.25, 8]

    graph_label(fig3, dfbeta, graph_content, graph_colors, label_params)

    fig3.set(ylabel='Intercept')
    fig3.lines[1].set_linestyle("--")
    fig3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=2)
    plt.savefig(os.path.join(directory, 'plot3_int_' + str(filebegin) + 'idio_alpha_' + str(inc_alpha) + 'seaborn_plt1_3' + '.png'), bbox_inches='tight', dpi = 600)
    plt.close()
    plt.cla()
    plt.clf()

    #############################
    # Plot 4 - Quantity by firm #
    #############################

    fig4 = sns.lineplot(x = 'period', y = 'Q1', data = df, label = 'Firm 1', color = 'black')
    sns.lineplot(x = 'period', y = 'Q2', data = df, label = 'Firm 2', color = 'grey')
    graph_content = {0: df.Q1, 1: df.Q2}

    biggestQ = max(max(df.Q1), max(df.Q2))
    smallestQ = min(min(df.Q1), min(df.Q2))
    fig4.set_ylim([smallestQ - 0.4, biggestQ + 0.4])

    label_params = []
    if compmod == 'so':
        #sophisticated
        if market_type == 'base_duop':
            label_params = [0.15, 0.15, 0.05, 5]
        if market_type == 'monop_duop':
            label_params = [1.5, 1.5, 1.5, 8]
    elif compmod == 'sr':
        if market_type == 'base_duop':
            label_params = [0.25, 0.25, 0, 5]
        if market_type == 'monop_duop':
            label_params = [0.25, 0.25, 0.25, 8]

    graph_label(fig4, df, graph_content, graph_colors, label_params)

    fig4.set(ylabel='Quantity')
    fig4.lines[1].set_linestyle("--")
    fig4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=2)
    plt.savefig(os.path.join(directory, 'plot4_q_' + str(filebegin) + 'idio_alpha_' + str(inc_alpha) + 'seaborn_plt1_3' + '.png'), bbox_inches='tight', dpi = 600)
    plt.close()
    plt.cla()
    plt.clf()

    ##################
    # Plot 5 - Price #
    ##################
    fig5 = sns.lineplot(x = 'period', y = 'price', data = df, label = 'price', color = 'black')
    fig5.set(ylabel='Price')
    fig5.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=1)

    fig5.set_ylim([min(df.price) - 0.5, max(df.price) + 0.5])

    price_round = df.price.round(2)
    mean_p = np.mean(price_round)
    for pt_zip in zip(df.period, price_round):
        period = pt_zip[0]
        if (period == 1 or period % 25 == 0) and period != 100:
            if pt_zip[1] > mean_p:
                #Adjust up if the point is above the mean
                fig5.annotate(str(pt_zip[1]), xy = (pt_zip[0], pt_zip[1] + 0.2), fontsize = 8, bbox=dict(boxstyle="round4,pad=.5", edgecolor = 'grey', fc='1.0'))
            if pt_zip[1] < mean_p:
                #Adjust down if the point is below the mean
                fig5.annotate(str(pt_zip[1]), xy = (pt_zip[0], pt_zip[1] - 0.2), fontsize = 8, bbox=dict(boxstyle="round4,pad=.5", edgecolor = 'grey', fc='1.0'))
        if period == 100:
            if pt_zip[1] > mean_p:
                fig5.annotate(str(pt_zip[1]), xy = (pt_zip[0] - 5, pt_zip[1] + 0.2), fontsize = 8, bbox=dict(boxstyle="round4,pad=.5", edgecolor = 'grey', fc='1.0'))
            if pt_zip[1] < mean_p:
                fig5.annotate(str(pt_zip[1]), xy = (pt_zip[0] - 5, pt_zip[1] - 0.2), fontsize = 8, bbox=dict(boxstyle="round4,pad=.5", edgecolor = 'grey', fc='1.0'))
    plt.savefig(os.path.join(directory, 'plot5_p_' + str(filebegin) + 'idio_alpha_' + str(inc_alpha) + 'seaborn_plt1_3' + '.png'), bbox_inches='tight', dpi = 600)
    plt.close()
    plt.cla()
    plt.clf()


    ############################
    # Plot 5a - Expected Price #
    ############################

    fig5a = sns.lineplot(x = 'period', y = 'exp_price_1', data = df, label = 'exp. price, F1', color = 'black')
    sns.lineplot(x = 'period', y = 'exp_price_2', data = df, label = 'exp. price, F2', color = 'grey')
    sns.lineplot(x = 'period', y = 'price', data = df, label = 'actual price', color = 'red')
    fig5a.set(ylabel='Price vs. expectations')
    fig5a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=1)

    fig5a.set_ylim([min(df.price) - 0.5, max(df.price) + 0.5])

    '''
    price_round = df.price.round(2)
    mean_p = np.mean(price_round)
    for pt_zip in zip(df.period, price_round):
        period = pt_zip[0]
        if (period == 1 or period % 25 == 0) and period != 100:
            if pt_zip[1] > mean_p:
                #Adjust up if the point is above the mean
                fig5.annotate(str(pt_zip[1]), xy = (pt_zip[0], pt_zip[1] + 0.2), fontsize = 8, bbox=dict(boxstyle="round4,pad=.5", edgecolor = 'grey', fc='1.0'))
            if pt_zip[1] < mean_p:
                #Adjust down if the point is below the mean
                fig5.annotate(str(pt_zip[1]), xy = (pt_zip[0], pt_zip[1] - 0.2), fontsize = 8, bbox=dict(boxstyle="round4,pad=.5", edgecolor = 'grey', fc='1.0'))
        if period == 100:
            if pt_zip[1] > mean_p:
                fig5.annotate(str(pt_zip[1]), xy = (pt_zip[0] - 5, pt_zip[1] + 0.2), fontsize = 8, bbox=dict(boxstyle="round4,pad=.5", edgecolor = 'grey', fc='1.0'))
            if pt_zip[1] < mean_p:
                fig5.annotate(str(pt_zip[1]), xy = (pt_zip[0] - 5, pt_zip[1] - 0.2), fontsize = 8, bbox=dict(boxstyle="round4,pad=.5", edgecolor = 'grey', fc='1.0'))
    '''
    plt.savefig(os.path.join(directory, 'plot5a_pexp1_' + str(filebegin) + 'idio_alpha_' + str(inc_alpha) + 'seaborn_plt1_3' + '.png'), bbox_inches='tight', dpi = 600)
    plt.close()
    plt.cla()
    plt.clf()


    ###########################
    # Plot 6 - Profit by firm #
    ###########################
    fig6 = sns.lineplot(x = 'period', y = 'prof1', data = df, label = 'Firm 1', color = 'black')
    sns.lineplot(x = 'period', y = 'prof2', data = df, label = 'Firm 2', color = 'grey')
    fig6.set(ylabel='Profit')
    fig6.lines[1].set_linestyle("--")
    fig6.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=1)

    biggestprof = max(max(df.prof1), max(df.prof2))
    smallestprof = min(min(df.prof1), min(df.prof2))
    fig6.set_ylim([smallestprof - 3, biggestprof + 3])

    prof1_round = df.prof1.round(2)
    for pt_zip in zip(df.period, prof1_round):
        period = pt_zip[0]
        if pt_zip[1] > df.prof2[period-1]:
            #If F1 profits are higher, annotate them higher
            if (period == 1 or period % 25 == 0) and period != 100:
                fig6.annotate(str(pt_zip[1]), xy=(pt_zip[0], biggestprof + 1), fontsize = 8, bbox=dict(boxstyle="round4,pad=.5", edgecolor = f1_edgecolor, fc=f1_boxcolor))
            if period == 100:
                fig6.annotate(str(pt_zip[1]), xy=(pt_zip[0] - 5, biggestprof + 1), fontsize = 8, bbox=dict(boxstyle="round4,pad=.5", edgecolor = f1_edgecolor, fc=f1_boxcolor))
        else:
            #If F1 profits are lower, annotate them lower
            if (period == 1 or period % 25 == 0) and period != 100:
                fig6.annotate(str(pt_zip[1]), xy=(pt_zip[0], smallestprof - 1), fontsize = 8, bbox=dict(boxstyle="round4,pad=.5", edgecolor = f1_edgecolor, fc=f1_boxcolor))
            if period == 100:
                fig6.annotate(str(pt_zip[1]), xy=(pt_zip[0] - 5, smallestprof - 1), fontsize = 8,  bbox=dict(boxstyle="round4,pad=.5", edgecolor = f1_edgecolor, fc=f1_boxcolor))
    prof2_round = df.prof2.round(2)
    for pt_zip in zip(df.period, prof2_round):
        period = pt_zip[0]
        if pt_zip[1] > df.prof1[period-1]:
            #If F2 profits are higher, annotate them higher
            if (period == 1 or period % 25 == 0) and period != 100:
                fig6.annotate(str(pt_zip[1]), xy=(pt_zip[0], biggestprof + 1), fontsize = 8,  bbox=dict(boxstyle="round4,pad=.5", edgecolor = f2_edgecolor, fc=f2_boxcolor))
            if period == 100:
                fig6.annotate(str(pt_zip[1]), xy=(pt_zip[0] - 5, biggestprof + 1), fontsize = 8,  bbox=dict(boxstyle="round4,pad=.5", edgecolor = f2_edgecolor, fc=f2_boxcolor))
        else:
            #If F2 profits are lower, annotate them lower
            if (period == 1 or period % 25 == 0) and period != 100:
                fig6.annotate(str(pt_zip[1]), xy=(period, smallestprof - 1), fontsize = 8,  bbox=dict(boxstyle="round4,pad=.5", edgecolor = f2_edgecolor, fc=f2_boxcolor))
            if period == 100:
                fig6.annotate(str(pt_zip[1]), xy=(period - 5, smallestprof - 1), fontsize = 8,  bbox=dict(boxstyle="round4,pad=.5", edgecolor = f2_edgecolor, fc=f2_boxcolor))
    plt.savefig(os.path.join(directory, 'plot6_profit_' + str(filebegin) + 'idio_alpha_' + str(inc_alpha) + 'seaborn_plt1_3' + '.png'), bbox_inches='tight', dpi = 600)
    plt.close()
    plt.cla()
    plt.clf()

    ##############################
    # Plot 7 - Cumulative Profit #
    ##############################

    ##Plots 7a and 7b - cumulative profit by firm, on separate plots
    fig7a, ax7a = plt.subplots(1,dpi=200)
    df['cumprof1'] = df['prof1'].cumsum()
    fig7a = sns.lineplot(x = 'period', y = 'cumprof1', data = df, label = 'F1 ',
                        ax = ax7a, ci = None, linewidth = 0.5, color = 'black')
    fig7a.annotate(str(round(df.cumprof1[periods-1], 1)), xy=(periods - 3, df.cumprof1[periods-1] - 75), fontsize = 8,  bbox=dict(boxstyle="round4,pad=.5", edgecolor = f1_edgecolor, fc=f1_boxcolor))
    ax7a.grid(linestyle = '-', linewidth = 0.3)
    ax7a.spines['top'].set_visible(False)
    ax7a.spines['right'].set_visible(False)
    ax7a.spines['bottom'].set_visible(False)
    ax7a.spines['left'].set_visible(False)
    plt.ylabel('Cumulative Profits')
    plt.xlabel('period')
    ax7a.set_title('Cumulative Profits F1')
    ax7a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
    plt.savefig(os.path.join(directory, 'plot7a_cumprofitF1_' + str(filebegin) + '.png'), bbox_inches='tight')
    plt.clf()

    '''

        fig7b, ax7b = plt.subplots(1,dpi=200)
        inc = 0
        for ta in tamount_space:
            df = master_df.loc[ta,model]
            df['cumprof2'] = df['prof2'].cumsum()
            if ta == 1000:
                trick_label = 'No trick'
                col = 'red'
            else:
                trick_label = 'trick: ' + str(round(ta,2))
                col = 'black'
            mark = marker_styles[inc%(len(marker_styles))]
            fig7b =  sns.lineplot(x = 'period', y = 'cumprof2', data = df, label = 'F2 '+ trick_label,
                        ax = ax7b, ci = None, linewidth = 0.5, color = col, marker = mark,
                         markers = True, ms = 5, markevery = marker_places)
            ax7b.grid(linestyle = '-', linewidth = 0.3)
            ax7b.spines['top'].set_visible(False)
            ax7b.spines['right'].set_visible(False)
            ax7b.spines['bottom'].set_visible(False)
            ax7b.spines['left'].set_visible(False)
            inc += 1
        plt.ylabel('Cumulative Profits')
        plt.xlabel('period')
        ax7b.set_title('Cumulative Profits F2 - Trick: '+ trick_type + ', ' + model)
        ax7b.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
        plt.savefig(os.path.join(directory, 'plot7b_cumprofitF2_' + str(filebegin) + '.png'), bbox_inches='tight')
        plt.clf()



        ##Plot 8 - Non-cumulative advantage of Firm 1 (profit difference)
        fig8, ax8 = plt.subplots(1,dpi=200)
        inc = 0
        for ta in tamount_space:
            df = master_df.loc[ta,model]
            df['soph_adv'] = df['prof1'] - df['prof2']
            if ta == 1000:
                trick_label = 'No trick'
                col = 'red'
            else:
                trick_label = 'trick: ' + str(round(ta,2))
                col = 'black'
            mark = marker_styles[inc%(len(marker_styles))]
            fig8 =  sns.lineplot(x = 'period', y = 'soph_adv', data = df, label = trick_label,
                        ax = ax8, ci = None, linewidth = 0.5, color = col, marker = mark,
                         markers = True, ms = 5, markevery = marker_places)
            sns.lineplot(x = 'period', y = 1, data = df, ax=ax8, color = 'yellow', ci = None, linewidth = 0.4)
            ax8.grid(linestyle = '-', linewidth = 0.3)
            ax8.spines['top'].set_visible(False)
            ax8.spines['right'].set_visible(False)
            ax8.spines['bottom'].set_visible(False)
            ax8.spines['left'].set_visible(False)
            inc += 1
        plt.ylabel('Sophisticated Advantage, non-cumulative')
        plt.xlabel('period')
        ax8.set_title('Absolute Profit F1 Advantage - Trick: '+ trick_type + ', ' + model)
        ax8.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
        plt.savefig(os.path.join(directory, 'plot8_f1adv_' + str(filebegin) + '.png'), bbox_inches='tight')
        plt.clf()


        ##Plot 9 - Cumulative advantage of Firm 1 (profit difference)
        fig9, ax9 = plt.subplots(1,dpi=200)
        inc = 0
        for ta in tamount_space:
            df = master_df.loc[ta,model]
            df['cumprof1'] = df['prof1'].cumsum()
            df['cumprof2'] = df['prof2'].cumsum()
            df['cum_soph_adv'] = df['cumprof1'] - df['cumprof2']
            if ta == 1000:
                trick_label = 'No trick'
                col = 'red'
            else:
                trick_label = 'trick: ' + str(round(ta,2))
                col = 'black'
            mark = marker_styles[inc%(len(marker_styles))]
            fig9 =  sns.lineplot(x = 'period', y = 'cum_soph_adv', data = df, label = 'F1 '+ trick_label,
                        ax = ax9, ci = None, linewidth = 0.5, color = col, marker = mark,
                         markers = True, ms = 5, markevery = marker_places)
            ax9.grid(linestyle = '-', linewidth = 0.3)
            ax9.spines['top'].set_visible(False)
            ax9.spines['right'].set_visible(False)
            ax9.spines['bottom'].set_visible(False)
            ax9.spines['left'].set_visible(False)
            inc += 1
        plt.ylabel('Sophisticated Advantage, cumulative')
        plt.xlabel('period')
        ax9.set_title('Cumulative Profit F1 Advantage - Trick: '+ trick_type + ', ' + model)
        ax9.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
        plt.savefig(os.path.join(directory, 'plot9_cumf1adv_' + str(filebegin) + '.png'), bbox_inches='tight')
        plt.clf()

    '''

def two_graph_overlap(index_1, index_2, csv_name_dict, iw, param_list, model_list):
    '''
    in progress: two_graph_overlap: generalized graph_correct_incorrect to put any two graph types on top of one another.
    graph_correct_incorrect: produces an overlap graph of the both correct and both incorrect parameterizations.
    - Takes csv_name, initial_weights, and the parameters for the both_correct and both_incorrect models.
    - Takes the entire param_list as an input and sorts through it to find the right models within the function.
    - iw is a dictionary of initial_weights of each graph.
    '''
    param_list = param_list

    #Selects the correct csv names from csv_name_dict.
    graph1_params = param_list[index_1]
    graph2_params= param_list[index_2]
    graph1_dir = graph1_params[19]
    graph2_dir = graph2_params[19]
    graph1_model_name = model_list[index_1]
    graph2_model_name = model_list[index_2]
    for name in [graph1_model_name, graph2_model_name]:
        if name == 'both_correct':
            name = 'Both correct'
    graph1_csv_name = csv_name_dict[index_1 + 1]
    graph2_csv_name = csv_name_dict[index_2 + 1]

    initial_weights1 = iw[1]
    initial_weights2 = iw[2]
    graph1_df = pd.read_csv(os.path.join(graph1_dir, graph1_csv_name))
    graph2_df = pd.read_csv(os.path.join(graph2_dir, graph2_csv_name))
    graph1_dfbeta = graph1_df.loc[:, ['period', 'f1_int', 'f1_b1', 'f1_b2', 'f1_b3', 'f2_int', 'f2_b1', 'f2_b2', 'f2_b3']]
    graph1_dfbeta = pd.concat([graph1_dfbeta, initial_weights1], axis = 0, sort = True).sort_values(by=['period']).reset_index()
    graph2_dfbeta = graph2_df.loc[:, ['period', 'f1_int', 'f1_b1', 'f1_b2', 'f1_b3', 'f2_int', 'f2_b1', 'f2_b2', 'f2_b3']]
    graph2_dfbeta = pd.concat([graph2_dfbeta, initial_weights2], axis = 0, sort = True).sort_values(by=['period']).reset_index()
    #How the dfbeta is constructed:
    #iw_df = initial_weights[idio_beta,model]
    #dfbeta = df.loc[:, ['period', 'f1_int', 'f1_b1', 'f1_b2', 'f1_b3', 'f2_int', 'f2_b1', 'f2_b2', 'f2_b3']]
    #dfbeta = pd.concat([dfbeta, iw_df], axis = 0, sort = True).sort_values(by=['period']).reset_index()

    #Create a folder for the graphs: puts it inside the folder of graph 1 model.
    if not os.path.exists(os.path.join(graph1_dir, 'compare')):
        os.makedirs(os.path.join(graph1_dir, 'compare'))

    #Begin graphing.
    sns.set_style('white')
    sns.set_context('paper', font_scale=2)

    #################################################
    # Plot 1 - idiosyncratic slope estimate by firm #
    #################################################

    #Picking the relevant F1 beta for each model.
    if graph1_model_name == 'both_incorrect':
        g1_f1_beta = graph1_dfbeta.f1_b2
    else:
        g1_f1_beta = graph1_dfbeta.f1_b1
    if graph2_model_name == 'both_incorrect':
        g2_f1_beta = graph2_dfbeta.f1_b2
    else:
        g2_f1_beta = graph2_dfbeta.f1_b1

    #Picking the revelevant F2 beta for each model
    if (graph1_model_name == 'both_correct') or (graph1_model_name == 'sr_monop_duop') or (graph1_model_name == 'so_monop_duop'):
        g1_f2_beta = graph1_dfbeta.f2_b1
    else:
        g1_f2_beta = graph1_dfbeta.f2_b2
    if graph2_model_name == 'both_correct' or graph2_model_name == 'sr_monop_duop' or graph2_model_name == 'so_monop_duop':
        g2_f2_beta = graph2_dfbeta.f2_b1
    else:
        g2_f2_beta = graph2_dfbeta.f2_b2

    graph1_color = 'black'
    graph2_color= 'grey'


    ## Plotting lines. ##
    gci_fig1 = sns.lineplot(x = 'period', y = g1_f1_beta, data = graph1_dfbeta, label = 'Firm 1 - ' + graph1_model_name, color = graph1_color) #F1 beta in graph1
    sns.lineplot(x = 'period', y = g1_f2_beta, data = graph1_dfbeta, label = 'Firm 2 - ' + graph1_model_name, color = graph1_color) #F2 beta in graph1
    sns.lineplot(x = 'period', y = g2_f1_beta, data = graph2_dfbeta, label = 'Firm 1 - ' + graph2_model_name, color = graph2_color) #F1 beta in graph2
    sns.lineplot(x = 'period', y = g2_f2_beta, data = graph2_dfbeta, label = 'Firm 2 - ' + graph2_model_name, color = graph2_color) #F2 beta in graph2
    #Set firm 2 lines to be dashed lines:
    gci_fig1.lines[1].set_linestyle("--")
    gci_fig1.lines[3].set_linestyle("--")

    gci_fig1.set(ylabel='Idiosyncratic variable')
    gci_fig1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=2)
    plt.savefig(os.path.join(graph1_dir, 'compare', 'overlap_idio_' + graph1_model_name + '_' + graph2_model_name + '.png'), bbox_inches='tight', dpi=600)
    plt.close()
    plt.cla()
    plt.clf()

    ###############################################
    # Plot 2 - industry variable estimate by firm #
    ###############################################

    ##Plotting Lines##
    # Plot beta3 for both firms. #
    gci_fig2 = sns.lineplot(x = 'period', y = 'f1_b3', data = graph1_dfbeta, label = 'Firm 1 -  ' + graph1_model_name, color = graph1_color) #F1 correct
    sns.lineplot(x = 'period', y = 'f2_b3', data = graph1_dfbeta, label = 'Firm 2 -  ' + graph1_model_name, color = graph1_color) #F2 correct
    sns.lineplot(x = 'period', y = 'f1_b3', data = graph2_dfbeta, label = 'Firm 1 -  ' + graph2_model_name, color = graph2_color) #F1 incorrect
    sns.lineplot(x = 'period', y = 'f2_b3', data = graph2_dfbeta, label = 'Firm 2 -  ' + graph2_model_name, color = graph2_color) #F2 incorrect
    #Set firm 2 lines to be dashed lines:
    gci_fig2.lines[1].set_linestyle("--")
    gci_fig2.lines[3].set_linestyle("--")

    gci_fig2.set(ylabel='Industry variable')
    gci_fig2.lines[1].set_linestyle("--")
    gci_fig2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=2)
    plt.savefig(os.path.join(graph1_dir, 'compare', 'overlap_ind_' + graph1_model_name + '_' + graph2_model_name + '.png'), bbox_inches='tight', dpi = 600)
    plt.close()
    plt.cla()
    plt.clf()

    #######################################
    # Plot 3 - intercept estimate by firm #
    #######################################

    ##Plotting Lines##
    # Plot intercept for both firms. #
    gci_fig3 = sns.lineplot(x = 'period', y = 'f1_int', data = graph1_dfbeta, label = 'Firm 1 -  ' + graph1_model_name, color = graph1_color)
    sns.lineplot(x = 'period', y = 'f2_int', data = graph1_dfbeta, label = 'Firm 2 -  ' + graph1_model_name, color = graph1_color)
    sns.lineplot(x = 'period', y = 'f1_int', data = graph2_dfbeta, label = 'Firm 1 -  ' + graph2_model_name, color = graph2_color)
    sns.lineplot(x = 'period', y = 'f2_int', data = graph2_dfbeta, label = 'Firm 2 -  ' + graph2_model_name, color = graph2_color)
    #Set firm 2 lines to be dashed lines:
    gci_fig3.lines[1].set_linestyle("--")
    gci_fig3.lines[3].set_linestyle("--")

    gci_fig3.set(ylabel='Intercept')
    gci_fig3.lines[1].set_linestyle("--")
    gci_fig3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=2)
    plt.savefig(os.path.join(graph1_dir, 'compare', 'overlap_int_' + graph1_model_name + '_' + graph2_model_name + '.png'), bbox_inches='tight', dpi = 600)
    plt.close()
    plt.cla()
    plt.clf()


    #############################
    # Plot 4 - Quantity by firm #
    #############################

    gci_fig4 = sns.lineplot(x = 'period', y = 'Q1', data = graph1_df, label = 'Firm 1 -  ' + graph1_model_name, color = graph1_color)
    sns.lineplot(x = 'period', y = 'Q2', data = graph1_df, label = 'Firm 2 -  ' + graph1_model_name, color = graph1_color)
    sns.lineplot(x = 'period', y = 'Q1', data = graph2_df, label = 'Firm 1 -  ' + graph2_model_name, color = graph2_color)
    sns.lineplot(x = 'period', y = 'Q2', data = graph2_df, label = 'Firm 2 -  ' + graph2_model_name, color = graph2_color)
    #Set firm 2 lines to be dashed lines:
    gci_fig4.lines[1].set_linestyle("--")
    gci_fig4.lines[3].set_linestyle("--")

    gci_fig4.set(ylabel='Quantity')
    gci_fig4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=2)
    plt.savefig(os.path.join(graph1_dir, 'compare', 'overlap_Q_' + graph1_model_name + '_' + graph2_model_name + '.png'), bbox_inches='tight', dpi = 600)
    plt.close()
    plt.cla()
    plt.clf()

    ##################
    # Plot 5 - Price #
    ##################
    gci_fig5 = sns.lineplot(x = 'period', y = 'price', data = graph1_df, label = 'price -  ' + graph1_model_name, color = graph1_color)
    sns.lineplot(x = 'period', y = 'price', data = graph2_df, label = 'price -  ' + graph2_model_name, color = graph2_color)
    gci_fig5.set(ylabel='Price')
    gci_fig5.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=1)
    #No need to do dashed lines since only two lines are plotted
    plt.savefig(os.path.join(graph1_dir, 'compare', 'overlap_price_' + graph1_model_name + '_' + graph2_model_name + '.png'), bbox_inches='tight', dpi = 600)
    plt.close()
    plt.cla()
    plt.clf()

    ###########################
    ##Plot 6 - Profit by Firm #
    ###########################
    gci_fig6 = sns.lineplot(x = 'period', y = 'prof1', data = graph1_df, label = 'Firm 1 -  ' + graph1_model_name, color = graph1_color)
    sns.lineplot(x = 'period', y = 'prof2', data = graph1_df, label = 'Firm 2 -  ' + graph1_model_name, color = graph1_color)
    sns.lineplot(x = 'period', y = 'prof1', data = graph2_df, label = 'Firm 1 -  ' + graph2_model_name, color = graph2_color)
    sns.lineplot(x = 'period', y = 'prof2', data = graph2_df, label = 'Firm 2 -  ' + graph2_model_name, color = graph2_color)
    #Set firm 2 lines to be dashed lines:
    gci_fig6.lines[1].set_linestyle("--")
    gci_fig6.lines[3].set_linestyle("--")

    gci_fig6.set(ylabel='Profit')
    gci_fig6.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=1)
    plt.savefig(os.path.join(graph1_dir, 'compare', 'overlap_profit_' + graph1_model_name + '_' + graph2_model_name + '.png'), bbox_inches='tight', dpi = 600)
    plt.close()
    plt.cla()
    plt.clf()

    ##############################
    ##Plot 7 - Cumulative Profit #
    ##############################
    fig7a, ax7a = plt.subplots(1,dpi=200)
    graph1_df['cumprof1'] = graph1_df['prof1'].cumsum()
    graph2_df['cumprof1'] = graph2_df['prof1'].cumsum()

    fig7a = sns.lineplot(x = 'period', y = 'cumprof1', data = graph1_df, label = 'Firm 1 -  ' + graph1_model_name,
                        ax = ax7a, ci = None, linewidth = 0.5, color = 'black')
    sns.lineplot(x = 'period', y = 'cumprof1', data = graph2_df, label = 'Firm 1 -  ' + graph2_model_name,
                        ax = ax7a, ci = None, linewidth = 0.5, color = 'red')
    fig7a.annotate(str(round(graph1_df.cumprof1[periods-1], 1)), xy=(periods - 3, graph1_df.cumprof1[periods-1] - 100), fontsize = 8,  bbox=dict(boxstyle="round4,pad=.5", edgecolor = 'black', ))
    fig7a.annotate(str(round(graph2_df.cumprof1[periods-1], 1)), xy=(periods - 3, graph2_df.cumprof1[periods-1] - 250), fontsize = 8,  bbox=dict(boxstyle="round4,pad=.5", edgecolor = 'red'))

    ax7a.grid(linestyle = '-', linewidth = 0.3)
    ax7a.spines['top'].set_visible(False)
    ax7a.spines['right'].set_visible(False)
    ax7a.spines['bottom'].set_visible(False)
    ax7a.spines['left'].set_visible(False)
    plt.ylabel('Cumulative Profits')
    plt.xlabel('period')
    ax7a.set_title('Cumulative Profits F1: ' + graph1_model_name + ' vs. ' + graph2_model_name)
    ax7a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)

    plt.savefig(os.path.join(graph1_dir, 'compare', 'overlap_cumprofitF1_' + graph1_model_name + '_' + graph2_model_name +  '.png'), bbox_inches='tight', dpi = 600)
    plt.clf()

def three_graph_overlap(index_1, index_2, index_3, csv_name_dict, iw, param_list, model_list):
    '''
    in progress: two_graph_overlap: generalized graph_correct_incorrect to put any two graph types on top of one another.
    graph_correct_incorrect: produces an overlap graph of the both correct and both incorrect parameterizations.
    - Takes csv_name, initial_weights, and the parameters for the both_correct and both_incorrect models.
    - Takes the entire param_list as an input and sorts through it to find the right models within the function.
    - iw is a dictionary of initial_weights of each graph.
    '''
    param_list = param_list

    #Selects the correct csv names from csv_name_dict.
    graph1_params = param_list[index_1]
    graph2_params= param_list[index_2]
    graph3_params = param_list[index_3]
    graph1_dir = graph1_params[19]
    graph2_dir = graph2_params[19]
    graph3_dir = graph3_params[19]
    graph1_model_name = model_list[index_1]
    graph2_model_name = model_list[index_2]
    graph3_model_name = model_list[index_3]
    for name in [graph1_model_name, graph2_model_name, graph3_model_name]:
        print(name)
        if name == 'sr_base_duop':
            name = 'self-reflective'
    graph1_csv_name = csv_name_dict[index_1 + 1]
    graph2_csv_name = csv_name_dict[index_2 + 1]
    graph3_csv_name = csv_name_dict[index_3 + 1]

    initial_weights1 = iw[1]
    initial_weights2 = iw[2]
    initial_weights3 = iw[3]
    graph1_df = pd.read_csv(os.path.join(graph1_dir, graph1_csv_name))
    graph2_df = pd.read_csv(os.path.join(graph2_dir, graph2_csv_name))
    graph3_df = pd.read_csv(os.path.join(graph3_dir, graph3_csv_name))
    graph1_dfbeta = graph1_df.loc[:, ['period', 'f1_int', 'f1_b1', 'f1_b2', 'f1_b3', 'f2_int', 'f2_b1', 'f2_b2', 'f2_b3']]
    graph1_dfbeta = pd.concat([graph1_dfbeta, initial_weights1], axis = 0, sort = True).sort_values(by=['period']).reset_index()
    graph2_dfbeta = graph2_df.loc[:, ['period', 'f1_int', 'f1_b1', 'f1_b2', 'f1_b3', 'f2_int', 'f2_b1', 'f2_b2', 'f2_b3']]
    graph2_dfbeta = pd.concat([graph2_dfbeta, initial_weights2], axis = 0, sort = True).sort_values(by=['period']).reset_index()
    graph3_dfbeta = graph3_df.loc[:, ['period', 'f1_int', 'f1_b1', 'f1_b2', 'f1_b3', 'f2_int', 'f2_b1', 'f2_b2', 'f2_b3']]
    graph3_dfbeta = pd.concat([graph3_dfbeta, initial_weights3], axis = 0, sort = True).sort_values(by=['period']).reset_index()


    #Create a folder for the graphs: puts it inside the folder of graph 1 model.
    if not os.path.exists(os.path.join(graph1_dir, 'compare_3')):
        os.makedirs(os.path.join(graph1_dir, 'compare_3'))

    #Begin graphing.
    sns.set_style('white')
    sns.set_context('paper', font_scale=2)
    marker_styles = ['^', 'o', 'D', '*', '', 'X']
    marker_places = range(0, 100, 10)
    graph1_color, graph2_color, graph3_color = 'black', 'darkgrey', 'grey'

    ###########################
    ##    Profit Firm 1      ##
    ###########################
    gci_fig6a = sns.lineplot(x = 'period', y = 'prof1', data = graph1_df, label = 'Firm 1 -  self-reflective', color = graph1_color, marker = marker_styles[0],
                                markers = True, ms = 7, markevery = marker_places)
    sns.lineplot(x = 'period', y = 'prof1', data = graph2_df, label = 'Firm 1 -  sophistication', color = graph2_color, marker = marker_styles[1],
                                markers = True, ms = 7, markevery = marker_places)
    sns.lineplot(x = 'period', y = 'prof1', data = graph3_df, label = 'Firm 1 -  manipulation', color = graph3_color, marker = marker_styles[2],
                                markers = True, ms = 7, markevery = marker_places)

    gci_fig6a.set(ylabel='Firm 1 Profit')
    gci_fig6a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=1)

    #annotation:
    for g1_zipped in zip(graph1_df.period, graph1_df.prof1):
        if g1_zipped[0] in range(0, 100, 20) or g1_zipped[0] in (1, 99):
            gci_fig6a.annotate(str(round(g1_zipped[1], 1)), xy = (g1_zipped[0], g1_zipped[1] - 0.3), fontsize = 9, bbox=dict(boxstyle="round4,pad=.2", edgecolor = graph1_color, fc='white'))
    for g2_zipped in zip(graph2_df.period, graph2_df.prof1):
        if g2_zipped[0] in range(0, 100, 20) or g2_zipped[0] in (1, 99):
            gci_fig6a.annotate(str(round(g2_zipped[1], 1)), xy = (g2_zipped[0], g2_zipped[1]+ 0.2), fontsize = 9, bbox=dict(boxstyle="round4,pad=.2", edgecolor = graph2_color, fc='white'))
    for g3_zipped in zip(graph3_df.period, graph3_df.prof1):
        if g3_zipped[0] in range(0, 100, 20) or g3_zipped[0] in (1, 99):
            gci_fig6a.annotate(str(round(g3_zipped[1], 1)), xy = (g3_zipped[0], g3_zipped[1] + 0.2), fontsize = 9, bbox=dict(boxstyle="round4,pad=.2", edgecolor = graph3_color, fc='white'))

    plt.savefig(os.path.join(graph1_dir, 'compare_3', 'F1profit_' + graph1_model_name + '_' + graph2_model_name + '_' + graph3_model_name + '.png'), bbox_inches='tight', dpi = 600)
    plt.close()
    plt.cla()
    plt.clf()

    ###########################
    ##    Profit Firm 2      ##
    ###########################

    gci_fig6b = sns.lineplot(x = 'period', y = 'prof2', data = graph1_df, label = 'Firm 2 -  self-reflective', color = graph1_color, marker = marker_styles[0],
                                markers = True, ms = 7, markevery = marker_places)
    sns.lineplot(x = 'period', y = 'prof2', data = graph2_df, label = 'Firm 2 -  sophistication', color = graph2_color, marker = marker_styles[1],
                                markers = True, ms = 7, markevery = marker_places)
    sns.lineplot(x = 'period', y = 'prof2', data = graph3_df, label = 'Firm 2 -  manipulation', color = graph3_color, marker = marker_styles[2],
                                markers = True, ms = 7, markevery = marker_places)

    gci_fig6b.set(ylabel='Firm 2 Profit')
    gci_fig6b.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=1)

    #annotation:
    for g1_zipped in zip(graph1_df.period, graph1_df.prof2):
        if g1_zipped[0] in range(0, 100, 20) or g1_zipped[0] in (1, 99):
            gci_fig6b.annotate(str(round(g1_zipped[1], 1)), xy = (g1_zipped[0], g1_zipped[1] + 1), fontsize = 9, bbox=dict(boxstyle="round4,pad=.2", edgecolor = graph1_color, fc='white'))
    for g2_zipped in zip(graph2_df.period, graph2_df.prof2):
        if g2_zipped[0] in range(0, 100, 20) or g2_zipped[0] in (1, 99):
            gci_fig6b.annotate(str(round(g2_zipped[1], 1)), xy = (g2_zipped[0], g2_zipped[1] + 0.75), fontsize = 9, bbox=dict(boxstyle="round4,pad=.2", edgecolor = graph2_color, fc='white'))
    for g3_zipped in zip(graph3_df.period, graph3_df.prof2):
        if g3_zipped[0] in range(0, 100, 20) or g3_zipped[0] in (1, 99):
            gci_fig6b.annotate(str(round(g3_zipped[1], 1)), xy = (g3_zipped[0], g3_zipped[1] - 0.5), fontsize = 9, bbox=dict(boxstyle="round4,pad=.2", edgecolor = graph3_color, fc='white'))

    plt.savefig(os.path.join(graph1_dir, 'compare_3', 'F2profit_' + graph1_model_name + '_' + graph2_model_name + '_' + graph3_model_name + '.png'), bbox_inches='tight', dpi = 600)
    plt.close()
    plt.cla()
    plt.clf()

def graph_sns_overlap(df_dict, option_list, model_list, directory, dir_list, initial_weights, sweep_type):
    '''
    THIS IS ADAPTED FROM v1_13_8 which deals with comparing constant Q tricks to no trick versions.
    Changes for version v1_14_4:
    - Deleted trick-specific parts of graph, like making 'no trick' red and 'trick_label' to replace with normal coloring and labels
    - replaced tamount_space in function parameters to incorrect_set_list
    - Changed for loops of each graph to be over the incorrect_set, not over trick amounts

    Description:
    Graphing all of the time series plots from graph_sns so that all of the information for multiple levels of beta
    are visible on the same graph for comparison.

    Inputs: dataframe dictionary, tamount_space, model list, directory, all params from original runs.
    parameters for other graphing fcns: (dataframes, tamount_space, model_list, directory)

    '''
    #Brining in dataframes and variables
    dfs, option_list, model_list, directory, dir_list, sweep_type = df_dict, option_list, model_list, directory, dir_list, sweep_type
    master_df = pd.concat(dfs, sort = True).astype(float)

    if sweep_type == 'idio_sweep_fixed':
        var_type = 'idio alpha'
    elif sweep_type == 'idio_inc_sweep':
        var_type = 'inc idio alpha'
    elif sweep_type == 'b':
        var_type = 'industry beta'
    elif sweep_type == 'int_sweep_fixed':
        var_type = 'intercept'
    elif sweep_type == 'phlen_sweep':
        var_type = 'pre-history length'
    elif sweep_type == 'entrepreneur_idio_alpha':
        var_type= 'entrant\'s idio alpha'
    else:
        print('invalid sweep type')

    #Creating list and indices for graphing
    pos_inc = 1
    sns.set(style = 'white')

    for model in model_list:
        graph_directory, params = dir_list[model][0], dir_list[model][1]
        [firmnum, unknown, compmod, market_type, truebetas, init_betas, mm, prehistlen,
        dist_params, cost, periods, periods_examind, max_q, mult_cap, iterations,
        trick_params, monop_firm, so_firm,
        figure_style, directory, filebegin, param_num] = params

        marker_styles = ['^', 'o', 'D', '*', '', 'X']
        marker_places = [0,10,20,30,40,50,60,70,80,90,100]
        colors = ['black', 'grey']

        f1_edgecolor = 'black'
        f2_edgecolor = 'grey'
        f1_boxcolor = '1.0'
        f2_boxcolor = '0.8'
        graph_colors = [f1_edgecolor, f2_edgecolor, f1_boxcolor, f2_boxcolor]

        ##############################
        ##Plot 1 - Idio beta by Firm #
        ##############################
        fig1, ax1 = plt.subplots(1,dpi=200)
        inc = 0
        for option in option_list:
            print(str(var_type) + ': ' + str(option))
            df=master_df.loc[option,model]
            iw_df = initial_weights[option,model]
            dfbeta = df.loc[:, ['period', 'f1_int', 'f1_a1', 'f1_a2', 'f1_b', 'f2_int', 'f2_a1', 'f2_a2', 'f2_b']]
            dfbeta = pd.concat([dfbeta, iw_df], axis = 0, sort = True).sort_values(by=['period']).reset_index()

            #run_fold = 'option_' + str(option)
            option_label = str(var_type) + ': ' + str(option)
            mark = marker_styles[inc%(len(marker_styles))]

            if mm[0] == [1,1,0,1]:
                fig1 = sns.lineplot(x = 'period', y = 'f1_a1', data = dfbeta, label = 'F1, ' + option_label,
                                ax=ax1, ci = None, linewidth=0.4, color = 'black', marker = mark,
                                markers = True, ms = 5, markevery = marker_places)
            else:
                fig1 = sns.lineplot(x = 'period', y = 'f1_a2', data = dfbeta, label = 'F1, ' + option_label,
                                ax=ax1, ci = None, linewidth=0.4, color = 'black', marker = mark,
                                markers = True, ms = 5, markevery = marker_places)

            if mm[1] == [1,1,0,1]:
                sns.lineplot(x = 'period', y = 'f2_a1', data = dfbeta, label = 'F2, ' + option_label,
                             dashes=True, ax=ax1, ci = None, linewidth=0.4, color = 'dimgrey', marker = mark,
                             markers = True, ms = 5, markevery = marker_places, mfc = 'none', mec = 'dimgrey')
            elif mm[1] == [1,0,1,1]:
                sns.lineplot(x = 'period', y = 'f2_a2', data = dfbeta, label = 'F2, ' + option_label,
                             dashes=True, ax=ax1, ci = None, linewidth=0.4, color = 'dimgrey', marker = mark,
                             markers = True, ms = 5, markevery = marker_places, mfc = 'none', mec = 'dimgrey')

            ax1.grid(linestyle = '-', linewidth = 0.3)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['bottom'].set_visible(False)
            ax1.spines['left'].set_visible(False)
            inc += 1
        fig1.set(ylabel='idiosyncratic alpha', xlabel = 'period')
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
        fig1.set(title= sweep_type + ' sweep, idio alpha, ' + model)
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.savefig(os.path.join(graph_directory, sweep_type + '_plot1_overlap_idio_' + str(filebegin) + '.png'), bbox_inches='tight')
        plt.clf()


        ##############################
        ##Plot 1a - Idio beta Firm 1 #
        ##############################
        fig1a, ax1a = plt.subplots(1,dpi=200)
        inc = 0
        for option in option_list:
            df=master_df.loc[option,model]
            iw_df = initial_weights[option,model]
            dfbeta = df.loc[:, ['period', 'f1_int', 'f1_a1', 'f1_a2', 'f1_b', 'f2_int', 'f2_a1', 'f2_a2', 'f2_b']]
            dfbeta = pd.concat([dfbeta, iw_df], axis = 0, sort = True).sort_values(by=['period']).reset_index()
            graph_content = [df.f1_b]
            #run_fold = 'option_' + str(option)
            option_label = str(var_type) + ': ' + str(option)
            mark = marker_styles[inc%(len(marker_styles))]

            fig1a = sns.lineplot(x = 'period', y = 'f1_b', data = dfbeta, label = 'F1, ' + option_label,
                                ax=ax1a, ci = None, linewidth=0.4, color = 'black', marker = mark,
                                markers = True, ms = 5, markevery = marker_places)
            if compmod == 'so':
                #sophisticated
                if market_type == 'base_duop':
                    #Sophisticated base_duop
                    label_params = [0.02, 0.045, 0.02, 0]
                if market_type == 'monop_duop':
                    #Sophisticated monop_duop
                    label_params = [0.10, 0.015, 0.10, 0]
            elif compmod == 'sr':
                if market_type == 'base_duop':
                    #Self-reflective base_duop
                    label_params = [0.25, 0.04, 0, 0]
                if market_type == 'monop_duop':
                    #Self-reflective monop_duop
                    label_params = [0.25, 0.25, 0, 0]


            #graph_label(fig1, df, graph_content, graph_colors, label_params)
            ax1a.grid(linestyle = '-', linewidth = 0.3)
            ax1a.spines['top'].set_visible(False)
            ax1a.spines['right'].set_visible(False)
            ax1a.spines['bottom'].set_visible(False)
            ax1a.spines['left'].set_visible(False)
            inc += 1
        fig1a.set(ylabel='F1 idio. beta', xlabel = 'period')
        ax1a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
        fig1a.set(title= sweep_type + ' sweep, ind beta F1, ' + model)
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.savefig(os.path.join(graph_directory, sweep_type + '_plot1a_overlap_beta_F1_' + str(filebegin) + '.png'), bbox_inches='tight')
        plt.clf()


        ##############################
        ##Plot 1b - Idio beta Firm 2 #
        ##############################
        fig1b, ax1b = plt.subplots(1,dpi=200)
        inc = 0
        for option in option_list:
            df=master_df.loc[option,model]
            iw_df = initial_weights[option,model]
            dfbeta = df.loc[:, ['period', 'f1_int', 'f1_a1', 'f1_a2', 'f1_b', 'f2_int', 'f2_a1', 'f2_a2', 'f2_b']]
            dfbeta = pd.concat([dfbeta, iw_df], axis = 0, sort = True).sort_values(by=['period']).reset_index()

            #run_fold = 'option_' + str(option)
            option_label = str(var_type) + ': ' + str(option)
            mark = marker_styles[inc%(len(marker_styles))]


            fig1b = sns.lineplot(x = 'period', y = 'f2_b', data = dfbeta, label = 'F2, ' + option_label,
                             dashes=True, ax=ax1b, ci = None, linewidth=0.4, color = 'dimgrey', marker = mark,
                             markers = True, ms = 5, markevery = marker_places, mfc = 'none', mec = 'dimgrey')

            ax1b.grid(linestyle = '-', linewidth = 0.3)
            ax1b.spines['top'].set_visible(False)
            ax1b.spines['right'].set_visible(False)
            ax1b.spines['bottom'].set_visible(False)
            ax1b.spines['left'].set_visible(False)
            inc += 1
        plt.ylabel('F2 idio. beta')
        plt.xlabel('period')
        ax1b.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
        fig1b.set(title= sweep_type + ' sweep, ind beta F2, ' + model)
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.savefig(os.path.join(graph_directory, sweep_type + '_plot1a_overlap_beta_F2_' + str(filebegin) + '.png'), bbox_inches='tight')
        plt.clf()


        '''
        #############################
        ##Plot 2 - Ind beta by Firm #
        #############################
        fig2, ax2 = plt.subplots(1,dpi=200)
        inc = 0
        for option in option_list:
            df=master_df.loc[option, model]
            iw_df = initial_weights[option,model]
            dfbeta = df.loc[:, ['period', 'f1_int', 'f1_a1', 'f1_a2', 'f1_b', 'f2_int', 'f2_a1', 'f2_a2', 'f2_b']]
            dfbeta = pd.concat([dfbeta, iw_df], axis = 0, sort = True).sort_values(by=['period']).reset_index()

            run_fold = 'option_' + str(option)
            option_label = str(var_type) + ': ' + str(option)
            mark = marker_styles[inc%(len(marker_styles))]

            fig2 = sns.lineplot(x = 'period', y = 'f1_b3', data = dfbeta, label = 'F1, ' + option_label,
                               ax=ax2, ci = None, linewidth=0.5, color = 'black', marker = mark,
                                markers = True, ms = 5, markevery = marker_places)
            sns.lineplot(x = 'period', y = 'f2_b', data = dfbeta, label = 'F2, ' + option_label,
                         ax=ax2, ci = None, linewidth=0.5, color = 'dimgrey', marker = mark,
                             markers = True, ms = 5, markevery = marker_places, mfc = 'none', mec = 'dimgrey')
            ax2.grid(linestyle = '-', linewidth = 0.3)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['bottom'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            inc += 1
        fig2.set(ylabel='industry variable', xlabel = 'period')
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
        fig2.set(title= sweep_type + ' sweep, industry beta, ' + model)
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.savefig(os.path.join(graph_directory, sweep_type + '_plot2_overlap_ind_' + str(filebegin) + '.png'), bbox_inches='tight')
        plt.clf()

        #############################
        ##Plot 2a - Ind beta Firm 1 #
        #############################
        fig2a, ax2a = plt.subplots(1,dpi=200)
        inc = 0
        for option in option_list:
            df=master_df.loc[option, model]
            iw_df = initial_weights[option,model]
            dfbeta = df.loc[:, ['period', 'f1_int', 'f1_b1', 'f1_b2', 'f1_b3', 'f2_int', 'f2_b1', 'f2_b2', 'f2_b3']]
            dfbeta = pd.concat([dfbeta, iw_df], axis = 0, sort = True).sort_values(by=['period']).reset_index()

            #run_fold = 'option_' + str(option)
            option_label = str(var_type) + ': ' + str(option)
            mark = marker_styles[inc%(len(marker_styles))]

            fig2a = sns.lineplot(x = 'period', y = 'f1_b3', data = dfbeta, label = 'F1, ' + option_label,
                               ax=ax2a, ci = None, linewidth=0.5, color = 'black', marker = mark,
                                markers = True, ms = 5, markevery = marker_places)
            ax2a.grid(linestyle = '-', linewidth = 0.3)
            ax2a.spines['top'].set_visible(False)
            ax2a.spines['right'].set_visible(False)
            ax2a.spines['bottom'].set_visible(False)
            ax2a.spines['left'].set_visible(False)
            inc += 1
        fig2a.set(ylabel='F1 ind. beta', xlabel = 'period')
        ax2a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
        fig2a.set(title= sweep_type + ' sweep, industry beta F1, ' + model)
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.savefig(os.path.join(graph_directory, sweep_type + '_plot2a_overlap_ind_F1_' + str(filebegin) + '.png'), bbox_inches='tight')
        plt.clf()

        #############################
        ##Plot 2b - Ind beta Firm 1 #
        #############################
        fig2b, ax2b = plt.subplots(1,dpi=200)
        inc = 0
        for option in option_list:
            df=master_df.loc[option, model]
            iw_df = initial_weights[option,model]
            dfbeta = df.loc[:, ['period', 'f1_int', 'f1_b1', 'f1_b2', 'f1_b3', 'f2_int', 'f2_b1', 'f2_b2', 'f2_b3']]
            dfbeta = pd.concat([dfbeta, iw_df], axis = 0, sort = True).sort_values(by=['period']).reset_index()

            option_label = str(var_type) + ': ' + str(option)
            mark = marker_styles[inc%(len(marker_styles))]
            fig2b = sns.lineplot(x = 'period', y = 'f2_b3', data = dfbeta, label = 'F2, ' + option_label,
                         ax=ax2b, ci = None, linewidth=0.5, color = 'dimgrey', marker = mark,
                             markers = True, ms = 5, markevery = marker_places, mfc = 'none', mec = 'dimgrey')
            ax2b.grid(linestyle = '-', linewidth = 0.3)
            ax2b.spines['top'].set_visible(False)
            ax2b.spines['right'].set_visible(False)
            ax2b.spines['bottom'].set_visible(False)
            ax2b.spines['left'].set_visible(False)
            inc += 1
        fig2b.set(ylabel='F2 ind. beta', xlabel = 'period')
        ax2b.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
        fig2b.set(title= sweep_type + ' sweep, industry beta F2, ' + model)
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.savefig(os.path.join(graph_directory, sweep_type + '_plot2b_overlap_ind_F2_' + str(filebegin) + '.png'), bbox_inches='tight')
        plt.clf()
        '''

        ##############################
        ##Plot 3 - Intercept by Firm #
        ##############################
        fig3, ax3 = plt.subplots(1,dpi=200)
        inc = 0
        for option in option_list:
            df=master_df.loc[option,model]
            iw_df = initial_weights[option,model]
            dfbeta = df.loc[:, ['period', 'f1_int', 'f1_a1', 'f1_a2', 'f1_b', 'f2_int', 'f2_a1', 'f2_a2', 'f2_b']]
            dfbeta = pd.concat([dfbeta, iw_df], axis = 0, sort = True).sort_values(by=['period']).reset_index()

            run_fold = 'option_' + str(option)
            option_label = str(var_type) + ': ' + str(option)
            mark = marker_styles[inc%(len(marker_styles))]

            fig3 = sns.lineplot(x = 'period', y = 'f1_int', data = dfbeta, label = 'F1, ' + option_label,
                                ax = ax3, ci = None, linewidth = 0.5, color = 'black', marker = mark,
                                markers = True, ms = 5, markevery = marker_places)
            sns.lineplot(x = 'period', y = 'f2_int', data = dfbeta, label = 'F2'+ option_label,
                         ax = ax3, ci = None, linewidth = 0.5, color = 'dimgrey', marker = mark,
                         markers = True, ms = 5, markevery = marker_places, mfc = 'none', mec = 'dimgrey')
            ax3.grid(linestyle = '-', linewidth = 0.3)
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.spines['bottom'].set_visible(False)
            ax3.spines['left'].set_visible(False)
            inc += 1
        fig3.set(ylabel='intercept', xlabel = 'period')
        ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
        fig3.set(title= sweep_type + ' sweep, intercept, ' + model)
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.savefig(os.path.join(graph_directory, sweep_type + '_plot3_int_' + str(filebegin) + '.png'), bbox_inches='tight')
        plt.clf()


        ##############################
        ##Plot 3a - Intercept Firm 1 #
        ##############################
        fig3a, ax3a = plt.subplots(1,dpi=200)
        inc = 0
        for option in option_list:
            df=master_df.loc[option,model]
            iw_df = initial_weights[option,model]
            dfbeta = df.loc[:, ['period', 'f1_int', 'f1_a1', 'f1_a2', 'f1_b', 'f2_int', 'f2_a1', 'f2_a2', 'f2_b']]
            dfbeta = pd.concat([dfbeta, iw_df], axis = 0, sort = True).sort_values(by=['period']).reset_index()

            run_fold = 'option_' + str(option)
            option_label = str(var_type) + ': ' + str(option)
            mark = marker_styles[inc%(len(marker_styles))]

            fig3a = sns.lineplot(x = 'period', y = 'f1_int', data = dfbeta, label = 'F1, ' + option_label,
                                ax = ax3a, ci = None, linewidth = 0.5, color = 'black', marker = mark,
                                markers = True, ms = 5, markevery = marker_places)
            ax3a.grid(linestyle = '-', linewidth = 0.3)
            ax3a.spines['top'].set_visible(False)
            ax3a.spines['right'].set_visible(False)
            ax3a.spines['bottom'].set_visible(False)
            ax3a.spines['left'].set_visible(False)
            inc += 1
        fig3a.set(ylabel='intercept', xlabel = 'period')
        ax3a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
        fig3a.set(title= sweep_type + ' sweep, intercept, ' + model)
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.savefig(os.path.join(graph_directory, sweep_type + '_plot3a_int_F1_' + str(filebegin) + '.png'), bbox_inches='tight')
        plt.clf()


        ##############################
        ##Plot 3b - Intercept Firm 2 #
        ##############################
        fig3b, ax3b = plt.subplots(1,dpi=200)
        inc = 0
        for option in option_list:
            df=master_df.loc[option,model]
            iw_df = initial_weights[option,model]
            dfbeta = df.loc[:, ['period', 'f1_int', 'f1_a1', 'f1_a2', 'f1_b', 'f2_int', 'f2_a1', 'f2_a2', 'f2_b']]
            dfbeta = pd.concat([dfbeta, iw_df], axis = 0, sort = True).sort_values(by=['period']).reset_index()

            run_fold = 'option_' + str(option)
            option_label = str(var_type) + ': ' + str(option)
            mark = marker_styles[inc%(len(marker_styles))]

            fig3b = sns.lineplot(x = 'period', y = 'f2_int', data = dfbeta, label = 'F2, '+ option_label,
                         ax = ax3b, ci = None, linewidth = 0.5, color = 'dimgrey', marker = mark,
                         markers = True, ms = 5, markevery = marker_places, mfc = 'none', mec = 'dimgrey')
            ax3b.grid(linestyle = '-', linewidth = 0.3)
            ax3b.spines['top'].set_visible(False)
            ax3b.spines['right'].set_visible(False)
            ax3b.spines['bottom'].set_visible(False)
            ax3b.spines['left'].set_visible(False)
            inc += 1
        fig3b.set(ylabel='intercept', xlabel = 'period')
        ax3b.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
        fig3b.set(title= sweep_type + ' sweep, intercept, ' + model)
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.savefig(os.path.join(graph_directory, sweep_type + '_plot3b_int_F2_' + str(filebegin) + '.png'), bbox_inches='tight')
        plt.clf()

        marker_places = [0,10,20,30,40,50,60,70,80,90,99]

        ##############################
        ##Plot 4 - Quantity by Firm #
        ##############################
        fig4, ax4 = plt.subplots(1,dpi=200)
        inc = 0
        for option in option_list:

            df=master_df.loc[option,model]
            run_fold = 'option_' + str(option)
            option_label = str(var_type) + ': ' + str(option)
            mark = marker_styles[inc%(len(marker_styles))]

            fig4 = sns.lineplot(x = 'period', y = 'Q1', data = df, label = 'F1, ' + option_label,
                                 ax = ax4, ci = None, linewidth = 0.5, color = 'black', marker = mark,
                                 markers = True, ms = 5, markevery = marker_places)
            sns.lineplot(x = 'period', y = 'Q2', data = df, label = 'F2, '+ option_label,
                                 ax = ax4, ci = None, linewidth = 0.5, color = 'grey', marker = mark,
                                 markers = True, ms = 5, markevery = marker_places)
            ax4.grid(linestyle = '-', linewidth = 0.3)
            ax4.spines['top'].set_visible(False)
            ax4.spines['right'].set_visible(False)
            ax4.spines['bottom'].set_visible(False)
            ax4.spines['left'].set_visible(False)
            inc += 1
        fig4.set(ylabel='Quantity', xlabel = 'period')
        ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
        fig4.set(title= sweep_type + ' sweep, Quantity, ' + model)
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.savefig(os.path.join(graph_directory, sweep_type + '_plot4_Q_' + str(filebegin) + '.png'), bbox_inches='tight')
        plt.clf()


        ##############################
        ##Plot 4a - Quantity, Firm 1 #
        ##############################
        fig4a, ax4a = plt.subplots(1,dpi=200)
        inc = 0
        for option in option_list:

            df=master_df.loc[option,model]
            run_fold = 'option_' + str(option)
            option_label = str(var_type) + ': ' + str(option)
            mark = marker_styles[inc%(len(marker_styles))]

            fig4a = sns.lineplot(x = 'period', y = 'Q1', data = df, label = 'F1, ' + option_label,
                                 ax = ax4a, ci = None, linewidth = 0.5, color = 'black', marker = mark,
                                 markers = True, ms = 5, markevery = marker_places)
            ax4a.grid(linestyle = '-', linewidth = 0.3)
            ax4a.spines['top'].set_visible(False)
            ax4a.spines['right'].set_visible(False)
            ax4a.spines['bottom'].set_visible(False)
            ax4a.spines['left'].set_visible(False)
            inc += 1
        fig4a.set(ylabel='Quantity', xlabel = 'period')
        ax4a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
        fig4a.set(title= sweep_type + ' sweep, F1 quantity, ' + model)
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.savefig(os.path.join(graph_directory, sweep_type + '_plot4a_q1' + str(filebegin) + '.png'), bbox_inches='tight')
        plt.clf()

        ##############################
        ##Plot 4b - Quantity, Firm 2 #
        ##############################
        fig4b, ax4b = plt.subplots(1,dpi=200)
        inc = 0
        for option in option_list:
            df=master_df.loc[option,model]
            run_fold = 'option_' + str(option)
            option_label = str(var_type) + ': ' + str(option)
            mark = marker_styles[inc%(len(marker_styles))]

            fig4b = sns.lineplot(x = 'period', y = 'Q2', data = df, label = 'F2, '+ option_label,
                                 ax = ax4b, ci = None, linewidth = 0.5, color = 'black', marker = mark,
                                 markers = True, ms = 5, markevery = marker_places)
            ax4b.grid(linestyle = '-', linewidth = 0.3)
            ax4b.spines['top'].set_visible(False)
            ax4b.spines['right'].set_visible(False)
            ax4b.spines['bottom'].set_visible(False)
            ax4b.spines['left'].set_visible(False)
            inc += 1
        fig4b.set(ylabel='Quantity', xlabel = 'period')
        ax4b.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
        #ax4b.set_ylim([0.8,2.5])
        fig4b.set(title= sweep_type + ' sweep, F2 quantity, ' + model)
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.savefig(os.path.join(graph_directory, sweep_type + '_plot4b_q2' + str(filebegin) + '.png'), bbox_inches='tight')
        plt.clf()

        ##################
        ##Plot 5 - Price #
        ##################
        fig5, ax5 = plt.subplots(1, dpi=200)
        inc = 0
        for option in option_list:
            df=master_df.loc[option,model]
            run_fold = 'option_' + str(option)
            option_label = str(var_type) + ': ' + str(option)
            mark = marker_styles[inc%(len(marker_styles))]

            fig5 = sns.lineplot(x = 'period', y = 'price', data = df, label = option_label,
                                   ax = ax5, ci = None, linewidth = 0.5, color = 'black', marker = mark,
                                 markers = True, ms = 5, markevery = marker_places)
            ax5.grid(linestyle = '-', linewidth = 0.3)
            ax5.spines['top'].set_visible(False)
            ax5.spines['right'].set_visible(False)
            ax5.spines['bottom'].set_visible(False)
            ax5.spines['left'].set_visible(False)
            inc += 1
        fig5.set(ylabel='Price', xlabel = 'period')
        ax5.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
        #ax5.set_ylim([55,70])
        fig5.set(title= sweep_type + ' sweep, price, ' + model)
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.savefig(os.path.join(graph_directory, sweep_type + '_plot5_p_' + str(filebegin) + '.png'), bbox_inches='tight')
        plt.clf()


        ###################
        ##Plot 6 - Profit #
        ###################
        fig6, ax6 = plt.subplots(1,dpi=200)
        inc = 0
        for option in option_list:
            df=master_df.loc[option,model]
            run_fold = 'option_' + str(option)
            option_label = str(var_type) + ': ' + str(option)
            mark = marker_styles[inc%(len(marker_styles))]

            fig6 =  sns.lineplot(x = 'period', y = 'prof1', data = df, label = 'F1, '+ option_label,
                        ax = ax6, ci = None, linewidth = 0.5, color = 'black', marker = mark,
                        markers = True, ms = 5, markevery = marker_places)
            sns.lineplot(x = 'period', y = 'prof2', data = df, label = 'F2, '+ option_label,
                        ax = ax6, ci = None, linewidth = 0.5, color = 'dimgrey', marker = mark,
                         markers = True, ms = 5, markevery = marker_places, mfc = 'none', mec = 'dimgrey')
            ax6.grid(linestyle = '-', linewidth = 0.3)
            ax6.spines['top'].set_visible(False)
            ax6.spines['right'].set_visible(False)
            ax6.spines['bottom'].set_visible(False)
            ax6.spines['left'].set_visible(False)
            inc += 1
        fig6.set(ylabel='Absolute Profits', xlabel = 'period', title= sweep_type + ' sweep, abs prof both firms, ' + model)
        ax6.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
        #ax6.set_ylim([-15,20])
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.savefig(os.path.join(graph_directory, sweep_type + '_plot6_profit_' + str(filebegin) + '.png'), bbox_inches='tight')
        plt.clf()

        ############################
        ##Plot 6a - Profit, Firm 1 #
        ############################
        fig6a, ax6a = plt.subplots(1,dpi=200)
        inc = 0
        for option in option_list:
            df = master_df.loc[option,model]
            run_fold = 'option = ' + str(option)
            option_label = str(var_type) + ': ' + str(option)
            mark = marker_styles[inc%(len(marker_styles))]

            fig6a =  sns.lineplot(x = 'period', y = 'prof1', data = df, label = 'F1, '+ option_label,
                        ax = ax6a, ci = None, linewidth = 0.5, color = 'black', marker = mark,
                        markers = True, ms = 5, markevery = marker_places)
            ax6a.grid(linestyle = '-', linewidth = 0.3)
            ax6a.spines['top'].set_visible(False)
            ax6a.spines['right'].set_visible(False)
            ax6a.spines['bottom'].set_visible(False)
            ax6a.spines['left'].set_visible(False)
            inc += 1
        fig6a.set(ylabel='Absolute Profits', xlabel = 'period',
                  title= sweep_type + ' sweep, abs prof F1, ' + model)
        ax6a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.savefig(os.path.join(graph_directory, sweep_type + '_plot6a_profitF1_' + str(filebegin) + '.png'), bbox_inches='tight')
        plt.clf()

        ############################
        ##Plot 6b - Profit, Firm 2 #
        ############################
        fig6b, ax6b = plt.subplots(1,dpi=200)
        inc = 0
        for option in option_list:
            df = master_df.loc[option,model]
            run_fold = 'option = ' + str(option)
            option_label = str(var_type) + ': ' + str(option)
            mark = marker_styles[inc%(len(marker_styles))]

            fig6b =  sns.lineplot(x = 'period', y = 'prof2', data = df, label = 'F2, '+ option_label,
                        ax = ax6b, ci = None, linewidth = 0.5, color = 'black', marker = mark,
                        markers = True, ms = 5, markevery = marker_places)
            ax6b.grid(linestyle = '-', linewidth = 0.3)
            ax6b.spines['top'].set_visible(False)
            ax6b.spines['right'].set_visible(False)
            ax6b.spines['bottom'].set_visible(False)
            ax6b.spines['left'].set_visible(False)
            inc += 1
        ax6b.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
        fig6b.set(ylabel='Absolute Profits', xlabel = 'period',
                  title= sweep_type + ' sweep, abs prof F2, ' + model)
        #ax4b.set_ylim([0, 25])
        #plt.rcParams.update({'figure.max_open_warning': 0})
        plt.savefig(os.path.join(graph_directory, sweep_type + '_plot6b_profitF2_' + str(filebegin) + '.png'), bbox_inches='tight')
        plt.clf()

def graph_trick_overlap(df_dict, tamount_space, model_list, directory, dir_list, initial_weights):
    ''' WORK IN PROGRESS.
    [?] Rewrite this description better
    [?] Is it possible to incorporate the other graphs into this? If not, write why it's not possible.
        Graphing all of the time series plots from graph_sns so that all of the information for multiple levels of beta
        are visible on the same graph for comparison.

        Inputs: dataframe dictionary, tamount_space, model list, directory, all params from original runs.

        parameters for other graphing fcns: (dataframes, tamount_space, model_list, directory)

        '''
    #Bringing in dataframes and variables
    #Creates master df, which concatenates all of the dataframes in df_dict and converts its elements to float.
    dfs, tamount_space, model_list, directory, dir_list = df_dict, tamount_space, model_list, directory, dir_list
    master_df = pd.concat(dfs, sort = True).astype(float)

    master_df.sort_index()
    #print('master df:')
    #print(master_df)

    #Some variables for graphing: list of colors, positive increment (pos_inc), and setting overall graphing to seaborn dark style.
    #[?] Does pos_inc ever get used or is this from the old graph positioning?
    sns.set(style = 'white')

    f1_edgecolor = 'black'
    f2_edgecolor = 'grey'
    f1_boxcolor = '1.0'
    f2_boxcolor = '0.8'
    graph_colors = [f1_edgecolor, f2_edgecolor, f1_boxcolor, f2_boxcolor]

    ''''''

    #Loops over each model graphing separately.
    ###In the trick master, this only considers sophisticated models, the idea being that an unsophisticated F1 wouldn't be able to trick F2.
    for model in model_list:
        #saves the subdirectory and the model parameters from the directory list
        #[?] clarify what's in dir_list
        #assigns names to each of the params and trick_params
        subdirect, params = dir_list[model][0], dir_list[model][1]
        [firmnum, unknown, compmod, market_type, truebetas, init_betas, mm, prehistlen,
        dist_params, cost, periods, periods_examind, max_q, mult_cap, iterations,
        trick_params, monop_firm, so_firm,
        figure_style, directory, filebegin, param_num] = params
        [trick_type, ta, tamount_space, tperiod_space, trick_len, trick_freq] = trick_params

        print('tamount space: ' + str(tamount_space))

        marker_styles = ['^', 'o', 'D', '*', '', 'X']
        marker_places = [0,10,20,30,40,50,60,70,80,90,100]
        #colors = ['black', 'grey']

        ##############################
        ##Plot 1 - Idio beta by Firm #
        ##############################
        fig1, ax1 = plt.subplots(1,dpi=200) #Creates the plot and axis
        inc = 0

        for ta in tamount_space:
            df=master_df.loc[ta,model]
            iw_df = initial_weights[ta,model]
            dfbeta = df.loc[:, ['period', 'f1_int', 'f1_b1', 'f1_b2', 'f1_b3', 'f2_int', 'f2_b1', 'f2_b2', 'f2_b3']]
            dfbeta = pd.concat([dfbeta, iw_df], axis = 0, sort = True).sort_values(by=['period']).reset_index()
            graph_content = {}
            #If the trick amount is 1000, this indicates no trick.
            if ta == 1000:
                trick_label = 'No trick'
            else:
                trick_label = 'trick: ' + str(round(ta,2))
            mark = marker_styles[inc%(len(marker_styles))]

            fig1 = sns.lineplot(x = 'period', y = 'f1_b1', data = dfbeta, label = 'F1, ' + trick_label,
                                ax=ax1, ci = None, linewidth=0.5, color = 'black', marker = mark,
                                markers = True, ms = 5, markevery = marker_places)
            graph_content[0] = dfbeta.f1_b1
            #If F2 has the correct mental model, and doesn't consider b1, then compare F2's b1.
            if mm[1] == [1,1,0,1]:
                sns.lineplot(x = 'period', y = 'f2_b1', data = dfbeta, label = 'F2, ' + trick_label,
                             dashes=True, ax=ax1, ci = None, linewidth=0.5, color = 'dimgrey', marker = mark,
                             markers = True, ms = 5, markevery = marker_places, mfc = 'none', mec = 'dimgrey')
                f2_beta = dfbeta.f2_b1
            #If F2 has the incorrect mental model and does not consider b1, then compare F2's b2.
            elif mm[1] == [1,0,1,1]:
                sns.lineplot(x = 'period', y = 'f2_b2', data = dfbeta, label = 'F2, ' + trick_label,
                             dashes=True, ax=ax1, ci = None, linewidth=0.5, color = 'dimgrey', marker = mark,
                             markers = True, ms = 5, markevery = marker_places, mfc = 'none', mec = 'dimgrey')
                f2_beta = dfbeta.f2_b2

            graph_content[1] = f2_beta
            if market_type == 'base_duop':
                #Sophisticated base_duop
                label_params = [0.32, 0.25, 0.02, 0]
            if market_type == 'monop_duop':
                #Sophisticated monop_duop -- label_params not adjusted yet.
                label_params = [0.10, 0.015, 0.10, 0]
            graph_label(fig1, df, graph_content, graph_colors, label_params)
            #Note on label params: F1_inc, F2_inc, zero_inc, zero_horiz = label_params[0], label_params[1], label_params[2], label_params[3]

            ax1.grid(linestyle = '-', linewidth = 0.3)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['bottom'].set_visible(False)
            ax1.spines['left'].set_visible(False)
            inc += 1
        plt.ylabel('idiosyncratic variable')
        plt.xlabel('period')
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
        ax1.set_title('idio beta, ' + model)
        plt.savefig(os.path.join(directory, 'plot1_overlap_idio' + str(filebegin) + '.png'), bbox_inches='tight')
        plt.clf()

        ##############################
        ##Plot 1a - Idio beta Firm 1 #
        ##############################
        fig1a, ax1a = plt.subplots(1,dpi=200) #Creates the plot and axis
        inc = 0
        for ta in tamount_space:
            df=master_df.loc[ta,model]
            iw_df = initial_weights[ta,model]
            dfbeta = df.loc[:, ['period', 'f1_int', 'f1_b1', 'f1_b2', 'f1_b3', 'f2_int', 'f2_b1', 'f2_b2', 'f2_b3']]
            dfbeta = pd.concat([dfbeta, iw_df], axis = 0, sort = True).sort_values(by=['period']).reset_index()
            #If the trick amount is 1000, this indicates no trick.
            if ta == 1000:
                trick_label = 'No trick'
            else:
                trick_label = 'trick: ' + str(round(ta,2))
            mark = marker_styles[inc%(len(marker_styles))]

            fig1a = sns.lineplot(x = 'period', y = 'f1_b1', data = dfbeta, label = 'F1, ' + trick_label,
                                ax=ax1a, ci = None, linewidth=0.5, color = 'black', marker = mark,
                                markers = True, ms = 5, markevery = marker_places)
            ax1a.grid(linestyle = '-', linewidth = 0.3)
            ax1a.spines['top'].set_visible(False)
            ax1a.spines['right'].set_visible(False)
            ax1a.spines['bottom'].set_visible(False)
            ax1a.spines['left'].set_visible(False)
            inc += 1
        plt.ylabel('idiosyncratic variable')
        plt.xlabel('period')
        ax1a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
        ax1a.set_title('idio beta, ' + model)
        plt.savefig(os.path.join(directory, 'plot1a_overlap_idio' + str(filebegin) + '.png'), bbox_inches='tight')
        plt.clf()

        ##############################
        ##Plot 1b - Idio beta Firm 2 #
        ##############################
        fig1b, ax1b = plt.subplots(1,dpi=200) #Creates the plot and axis
        inc = 0
        for ta in tamount_space:
            df=master_df.loc[ta,model]
            iw_df = initial_weights[ta,model]
            dfbeta = df.loc[:, ['period', 'f1_int', 'f1_b1', 'f1_b2', 'f1_b3', 'f2_int', 'f2_b1', 'f2_b2', 'f2_b3']]
            dfbeta = pd.concat([dfbeta, iw_df], axis = 0, sort = True).sort_values(by=['period']).reset_index()

            #If the trick amount is 1000, this indicates no trick.
            if ta == 1000:
                trick_label = 'No trick'
            else:
                trick_label = 'trick: ' + str(round(ta,2))
            mark = marker_styles[inc%(len(marker_styles))]

            if mm[1] == [1,1,0,1]:
                sns.lineplot(x = 'period', y = 'f2_b1', data = dfbeta, label = 'F2, ' + trick_label,
                             dashes=True, ax=ax1b, ci = None, linewidth=0.5, color = 'dimgrey', marker = mark,
                             markers = True, ms = 5, markevery = marker_places, mfc = 'none', mec = 'dimgrey')
            #If F2 has the incorrect mental model and does not consider b1, then compare F2's b2.
            elif mm[1] == [1,0,1,1]:
                sns.lineplot(x = 'period', y = 'f2_b2', data = dfbeta, label = 'F2, ' + trick_label,
                             dashes=True, ax=ax1b, ci = None, linewidth=0.5, color = 'dimgrey', marker = mark,
                             markers = True, ms = 5, markevery = marker_places, mfc = 'none', mec = 'dimgrey')
            ax1b.grid(linestyle = '-', linewidth = 0.3)
            ax1b.spines['top'].set_visible(False)
            ax1b.spines['right'].set_visible(False)
            ax1b.spines['bottom'].set_visible(False)
            ax1b.spines['left'].set_visible(False)
            inc += 1
        plt.ylabel('idiosyncratic variable')
        plt.xlabel('period')
        ax1b.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
        ax1b.set(title='idio beta, ' + model)
        plt.savefig(os.path.join(directory, 'plot1b_overlap_idio' + str(filebegin) + '.png'), bbox_inches='tight')
        plt.clf()


        #############################
        ##Plot 2 - Ind beta by Firm #
        #############################
        fig2, ax2 = plt.subplots(1,dpi=200)
        inc, line_inc = 0, 0
        for ta in tamount_space:
            df=master_df.loc[ta,model]
            iw_df = initial_weights[ta,model]
            dfbeta = df.loc[:, ['period', 'f1_int', 'f1_b1', 'f1_b2', 'f1_b3', 'f2_int', 'f2_b1', 'f2_b2', 'f2_b3']]
            dfbeta = pd.concat([dfbeta, iw_df], axis = 0, sort = True).sort_values(by=['period']).reset_index()
            graph_content = {0: dfbeta.f1_b3, 1: dfbeta.f2_b3}
            if ta == 1000:
                trick_label = 'No trick'
            else:
                trick_label = 'trick: ' + str(round(ta,2))
            mark = marker_styles[inc%(len(marker_styles))]
            fig2 = sns.lineplot(x = 'period', y = 'f1_b3', data = dfbeta, label = 'F1, ' + trick_label,
                               ax=ax2, color = 'black', ci = None, linewidth=0.5, marker = mark,
                             markers = True, ms = 5, markevery = marker_places)
            sns.lineplot(x = 'period', y = 'f2_b3', data = dfbeta, label = 'F2, ' + trick_label,
                         ax=ax2, color = 'dimgrey', ci = None, linewidth=0.5, marker = mark,
                             markers = True, ms = 5, markevery = marker_places, mfc = 'none', mec = 'dimgrey')
            if market_type == 'base_duop':
                #Sophisticated base_duop
                label_params = [0.32, 0.25, 0.02, 0]
            if market_type == 'monop_duop':
                #Sophisticated monop_duop -- label params not adjusted yet
                label_params = [0.10, 0.015, 0.10, 0]
            graph_label(fig2, df, graph_content, graph_colors, label_params)

            ax2.grid(linestyle = '-', linewidth = 0.3)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['bottom'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            inc += 1
        plt.ylabel('industry variable')
        plt.xlabel('period')
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
        ax2.set(title='ind beta, ' + model)
        plt.savefig(os.path.join(directory, 'plot2_overlap_ind_' + str(filebegin) + '.png'), bbox_inches='tight')
        plt.clf()


        ##############################
        ##Plot 3 - Intercept by Firm #
        ##############################
        fig3, ax3 = plt.subplots(1,dpi=200)
        inc = 0
        for ta in tamount_space:
            df = master_df.loc[ta, model]
            iw_df = initial_weights[ta,model]
            dfbeta = df.loc[:, ['period', 'f1_int', 'f1_b1', 'f1_b2', 'f1_b3', 'f2_int', 'f2_b1', 'f2_b2', 'f2_b3']]
            dfbeta = pd.concat([dfbeta, iw_df], axis = 0, sort = True).sort_values(by=['period']).reset_index()
            graph_content = {}

            if ta == 1000:
                trick_label = 'No trick'
            else:
                trick_label = 'trick: ' + str(round(ta,2))
            mark = marker_styles[inc%(len(marker_styles))]
            fig3 = sns.lineplot(x = 'period', y = 'f1_int', data = dfbeta, label = 'F1, ' + trick_label,
                               ax = ax3, ci = None, linewidth = 0.5, color = 'black', marker = mark,
                                markers = True, ms = 5, markevery = marker_places)
            sns.lineplot(x = 'period', y = 'f2_int', data = dfbeta, label = 'F2'+ trick_label,
                        ax = ax3, ci = None, linewidth = 0.5, color = 'dimgrey', marker = mark,
                         markers = True, ms = 5, markevery = marker_places, mfc = 'none', mec = 'dimgrey')
            graph_content[0] = df.f1_int
            graph_content[1] = df.f2_int
            if market_type == 'base_duop':
                #Sophisticated base_duop
                label_params = [0.32, 0.25, 0.02, 0]
            if market_type == 'monop_duop':
                #Sophisticated monop_duop -- label params not adjusted yet
                label_params = [0.10, 0.015, 0.10, 0]
            graph_label(fig3, df, graph_content, graph_colors, label_params)

            ax3.grid(linestyle = '-', linewidth = 0.3)
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.spines['bottom'].set_visible(False)
            ax3.spines['left'].set_visible(False)
            inc += 1
        plt.ylabel('intercept')
        plt.xlabel('period')
        ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
        ax3.set_title('intercept, ' + model)
        plt.savefig(os.path.join(directory, 'plot3_int_' + str(filebegin) + '.png'), bbox_inches='tight')
        plt.clf()


        marker_places = [0,10,20,30,40,50,60,70,80,90,99]

        ##############################
        ##Plot 4a - Quantity, Firm 1 #
        ##############################
        fig4a, ax4a = plt.subplots(1,dpi=200)
        inc = 0
        for ta in tamount_space:
            graph_content = {}
            df = master_df.loc[ta, model]
            if ta == 1000:
                trick_label = 'No trick'
            else:
                trick_label = 'trick: ' + str(round(ta,2))
            mark = marker_styles[inc%(len(marker_styles))]
            fig4a = sns.lineplot(x = 'period', y = 'Q1', data = df, label = 'F1, ' + trick_label,
                               ax = ax4a, ci = None, linewidth = 0.5, color = 'black', marker = mark,
                                 markers = True, ms = 5, markevery = marker_places)


            ax4a.grid(linestyle = '-', linewidth = 0.3)
            ax4a.spines['top'].set_visible(False)
            ax4a.spines['right'].set_visible(False)
            ax4a.spines['bottom'].set_visible(False)
            ax4a.spines['left'].set_visible(False)
            inc += 1
        plt.ylabel('quantity')
        plt.xlabel('period')
        ax4a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
        ax4a.set_title('Trick: '+ trick_type + ', ' + model)
        plt.savefig(os.path.join(directory, 'plot4a_q1' + str(filebegin) + '.png'), bbox_inches='tight')
        plt.clf()

        fig4b, ax4b = plt.subplots(1,dpi=200)
        inc = 0
        for ta in tamount_space:
            df = master_df.loc[ta, model]
            if ta == 1000:
                trick_label = 'No trick'
            else:
                trick_label = 'trick: ' + str(round(ta,2))
            mark = marker_styles[inc%(len(marker_styles))]
            fig4b = sns.lineplot(x = 'period', y = 'Q2', data = df, label = 'F2 '+ trick_label,
                        ax = ax4b, ci = None, linewidth = 0.5, color = 'black', marker = mark,
                                 markers = True, ms = 5, markevery = marker_places)

            ax4b.grid(linestyle = '-', linewidth = 0.3)
            ax4b.spines['top'].set_visible(False)
            ax4b.spines['right'].set_visible(False)
            ax4b.spines['bottom'].set_visible(False)
            ax4b.spines['left'].set_visible(False)
            inc += 1
        plt.ylabel('quantity')
        plt.xlabel('period')
        ax4b.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
        ax4b.set_title('F2 Quantity - Trick: '+ trick_type + ', ' + model)
        plt.savefig(os.path.join(directory, 'plot4b_q2' + str(filebegin) + '.png'), bbox_inches='tight')


        ##Plot 5 - price
        fig5, ax5 = plt.subplots(1, dpi=200)
        inc = 0
        for ta in tamount_space:
            df = master_df.loc[ta, model]
            if ta == 1000:
                trick_label = 'No trick'
            else:
                trick_label = 'trick: ' + str(round(ta,2))
            mark = marker_styles[inc%(len(marker_styles))]
            fig5 = sns.lineplot(x = 'period', y = 'price', data = df, label = 'trick: ' + trick_label,
                                 ax = ax5, ci = None, linewidth = 0.5, color = 'black', marker = mark,
                                 markers = True, ms = 5, markevery = marker_places)
            ax5.grid(linestyle = '-', linewidth = 0.3)
            ax5.spines['top'].set_visible(False)
            ax5.spines['right'].set_visible(False)
            ax5.spines['bottom'].set_visible(False)
            ax5.spines['left'].set_visible(False)
            inc += 1
        plt.ylabel('price')
        plt.xlabel('period')
        ax5.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
        ax5.set_title('Price - Trick: '+ trick_type + ', ' + model)
        plt.savefig(os.path.join(directory, 'plot5_p_' + str(filebegin) + '.png'), bbox_inches='tight')
        plt.clf()



        ##The below graphs aren't annotated:

        ##Plot 6 - absolute profit by firm
        fig6, ax6 = plt.subplots(1,dpi=200)
        inc = 0
        for ta in tamount_space:
            df = master_df.loc[ta,model]
            if ta == 1000:
                trick_label = 'No trick'
            else:
                trick_label = 'trick: ' + str(round(ta,2))
            mark = marker_styles[inc%(len(marker_styles))]
            fig6 =  sns.lineplot(x = 'period', y = 'prof1', data = df, label = 'F1 '+ trick_label,
                        ax = ax6, ci = None, linewidth = 0.5, color = 'black', marker = mark,
                        markers = True, ms = 5, markevery = marker_places)
            sns.lineplot(x = 'period', y = 'prof2', data = df, label = 'F2 '+ trick_label,
                        ax = ax6, ci = None, linewidth = 0.5, color = 'dimgrey', marker = mark,
                         markers = True, ms = 5, markevery = marker_places, mfc = 'none', mec = 'dimgrey')
            ax6.grid(linestyle = '-', linewidth = 0.3)
            ax6.spines['top'].set_visible(False)
            ax6.spines['right'].set_visible(False)
            ax6.spines['bottom'].set_visible(False)
            ax6.spines['left'].set_visible(False)
            inc += 1
        plt.ylabel('Absolute Profits')
        plt.xlabel('period')
        ax6.set_title('Absolute Profits - Trick: '+ trick_type + ', ' + model)
        ax6.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
        plt.savefig(os.path.join(directory, 'plot6_profit_' + str(filebegin) + '.png'), bbox_inches='tight')
        plt.clf()


        #Plots 6a and 6b, profits by firm on separate plots
        fig6a, ax6a = plt.subplots(1,dpi=200)
        inc = 0
        for ta in tamount_space:
            df = master_df.loc[ta,model]
            if ta == 1000:
                trick_label = 'No trick'
            else:
                trick_label = 'trick: ' + str(round(ta,2))
            mark = marker_styles[inc%(len(marker_styles))]
            fig6a =  sns.lineplot(x = 'period', y = 'prof1', data = df, label = 'F1 '+ trick_label,
                        ax = ax6a, color = 'black', ci = None, linewidth = 0.5, marker = mark,
                        markers = True, ms = 5, markevery = marker_places)
            ax6a.grid(linestyle = '-', linewidth = 0.3)
            ax6a.spines['top'].set_visible(False)
            ax6a.spines['right'].set_visible(False)
            ax6a.spines['bottom'].set_visible(False)
            ax6a.spines['left'].set_visible(False)
            inc += 1
        plt.ylabel('Absolute Profits')
        plt.xlabel('period')
        ax6a.set_title('Absolute Profits F1 - Trick: '+ trick_type + ', ' + model)
        ax6a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
        plt.savefig(os.path.join(directory, 'plot6a_profitF1_' + str(filebegin) + '.png'), bbox_inches='tight')
        plt.clf()


        fig6b, ax6b = plt.subplots(1,dpi=200)
        inc = 0
        for ta in tamount_space:
            df = master_df.loc[ta,model]
            if ta == 1000:
                trick_label = 'No trick'
            else:
                trick_label = 'trick: ' + str(round(ta,2))
            mark = marker_styles[inc%(len(marker_styles))]
            fig6b =  sns.lineplot(x = 'period', y = 'prof2', data = df, label = 'F2 '+ trick_label,
                        ax = ax6b, ci = None, linewidth = 0.5, color = 'black', marker = mark,
                         markers = True, ms = 5, markevery = marker_places)
            ax6b.grid(linestyle = '-', linewidth = 0.3)
            ax6b.spines['top'].set_visible(False)
            ax6b.spines['right'].set_visible(False)
            ax6b.spines['bottom'].set_visible(False)
            ax6b.spines['left'].set_visible(False)
            inc += 1
        plt.ylabel('Absolute Profits')
        plt.xlabel('period')
        ax6b.set_title('Absolute Profits F2 - Trick: '+ trick_type + ', ' + model)
        ax6b.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
        plt.savefig(os.path.join(directory, 'plot6b_profitF2_' + str(filebegin) + '.png'), bbox_inches='tight')
        plt.clf()

        '''
        ##Plot 6C - absolute profit for non-trick firm only
        ##### This is for comparison to the so_base_duop from V2_4, to see if this model is making the right comparison.
        fig6c, ax6c = plt.subplots(1,dpi=200)
        inc, line_inc = 0,0
        df = master_df.loc[1000,model]
        col = 'red'
        trick_label = 'No trick'
        fig6c =  sns.lineplot(x = 'period', y = 'prof1', data = df, label = 'F1, no trick',
                    ax = ax6, color = col, ci = None, linewidth = 0.5)
        sns.lineplot(x = 'period', y = 'prof2', data = df, label = 'F2, no trick ',
                    ax = ax6, color = col, ci = None, linewidth = 0.5)
        fig6c.lines[line_inc].set_linestyle("--")
        inc += 1
        line_inc += 2
        prof1_round = df.prof1.round(2)

        fig6c.set(ylabel='Absolute Profits', xlabel = 'period',
                 title='Absolute Profits - Trick: '+ trick_type + ', ' + model)
        ax6c.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
        plt.savefig(os.path.join(directory, 'plot6c_profit_notrickonly_' + str(filebegin) + '.png'), bbox_inches='tight')
        plt.clf()
        '''


        ##Plot 7 - cumulative profit by firm
        fig7, ax7 = plt.subplots(1,dpi=200)
        inc, line_inc = 0,0
        for ta in tamount_space:
            df = master_df.loc[ta,model]
            df['cumprof1'] = df['prof1'].cumsum()
            df['cumprof2'] = df['prof2'].cumsum()
            if ta == 1000:
                trick_label = 'No trick'
            else:
                trick_label = 'trick: ' + str(round(ta,2))
            mark = marker_styles[inc%(len(marker_styles))]
            fig7 =  sns.lineplot(x = 'period', y = 'cumprof1', data = df, label = 'F1 '+ trick_label,
                        ax = ax7, ci = None, linewidth = 0.5, color = 'black', marker = mark,
                         markers = True, ms = 5, markevery = marker_places)
            sns.lineplot(x = 'period', y = 'cumprof2', data = df, label = 'F2 '+ trick_label,
                        ax = ax7, ci = None, linewidth = 0.5, color = 'dimgrey', marker = mark,
                         markers = True, ms = 5, markevery = marker_places, mfc = 'none', mec = 'dimgrey')
            ax7.grid(linestyle = '-', linewidth = 0.3)
            ax7.spines['top'].set_visible(False)
            ax7.spines['right'].set_visible(False)
            ax7.spines['bottom'].set_visible(False)
            ax7.spines['left'].set_visible(False)
            inc += 1
        plt.ylabel('Cumulative Profits')
        plt.xlabel('period')
        ax7.set_title('Cumulative Profits - Trick: '+ trick_type + ', ' + model)
        ax7.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
        plt.savefig(os.path.join(directory, 'plot7_cumprofit_' + str(filebegin) + '.png'), bbox_inches='tight')
        plt.clf()



        ##Plots 7a and 7b - cumulative profit by firm, on separate plots
        fig7a, ax7a = plt.subplots(1,dpi=200)
        inc = 0
        for ta in tamount_space:
            df = master_df.loc[ta,model]
            df['cumprof1'] = df['prof1'].cumsum()
            if ta == 1000:
                trick_label = 'No trick'
                col = 'red'
            else:
                trick_label = 'trick: ' + str(round(ta,2))
                col = 'black'
            mark = marker_styles[inc%(len(marker_styles))]
            fig7a =  sns.lineplot(x = 'period', y = 'cumprof1', data = df, label = 'F1 '+ trick_label,
                        ax = ax7a, ci = None, linewidth = 0.5, color = col, marker = mark,
                         markers = True, ms = 5, markevery = marker_places)
            ax7a.grid(linestyle = '-', linewidth = 0.3)
            ax7a.spines['top'].set_visible(False)
            ax7a.spines['right'].set_visible(False)
            ax7a.spines['bottom'].set_visible(False)
            ax7a.spines['left'].set_visible(False)
            inc += 1
        plt.ylabel('Cumulative Profits')
        plt.xlabel('period')
        ax7a.set_title('Cumulative Profits F1 - Trick: '+ trick_type + ', ' + model)
        ax7a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
        plt.savefig(os.path.join(directory, 'plot7a_cumprofitF1_' + str(filebegin) + '.png'), bbox_inches='tight')
        plt.clf()

        fig7b, ax7b = plt.subplots(1,dpi=200)
        inc = 0
        for ta in tamount_space:
            df = master_df.loc[ta,model]
            df['cumprof2'] = df['prof2'].cumsum()
            if ta == 1000:
                trick_label = 'No trick'
                col = 'red'
            else:
                trick_label = 'trick: ' + str(round(ta,2))
                col = 'black'
            mark = marker_styles[inc%(len(marker_styles))]
            fig7b =  sns.lineplot(x = 'period', y = 'cumprof2', data = df, label = 'F2 '+ trick_label,
                        ax = ax7b, ci = None, linewidth = 0.5, color = col, marker = mark,
                         markers = True, ms = 5, markevery = marker_places)
            ax7b.grid(linestyle = '-', linewidth = 0.3)
            ax7b.spines['top'].set_visible(False)
            ax7b.spines['right'].set_visible(False)
            ax7b.spines['bottom'].set_visible(False)
            ax7b.spines['left'].set_visible(False)
            inc += 1
        plt.ylabel('Cumulative Profits')
        plt.xlabel('period')
        ax7b.set_title('Cumulative Profits F2 - Trick: '+ trick_type + ', ' + model)
        ax7b.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
        plt.savefig(os.path.join(directory, 'plot7b_cumprofitF2_' + str(filebegin) + '.png'), bbox_inches='tight')
        plt.clf()



        ##Plot 8 - Non-cumulative advantage of Firm 1 (profit difference)
        fig8, ax8 = plt.subplots(1,dpi=200)
        inc = 0
        for ta in tamount_space:
            df = master_df.loc[ta,model]
            df['soph_adv'] = df['prof1'] - df['prof2']
            if ta == 1000:
                trick_label = 'No trick'
                col = 'red'
            else:
                trick_label = 'trick: ' + str(round(ta,2))
                col = 'black'
            mark = marker_styles[inc%(len(marker_styles))]
            fig8 =  sns.lineplot(x = 'period', y = 'soph_adv', data = df, label = trick_label,
                        ax = ax8, ci = None, linewidth = 0.5, color = col, marker = mark,
                         markers = True, ms = 5, markevery = marker_places)
            sns.lineplot(x = 'period', y = 1, data = df, ax=ax8, color = 'yellow', ci = None, linewidth = 0.4)
            ax8.grid(linestyle = '-', linewidth = 0.3)
            ax8.spines['top'].set_visible(False)
            ax8.spines['right'].set_visible(False)
            ax8.spines['bottom'].set_visible(False)
            ax8.spines['left'].set_visible(False)
            inc += 1
        plt.ylabel('Sophisticated Advantage, non-cumulative')
        plt.xlabel('period')
        ax8.set_title('Absolute Profit F1 Advantage - Trick: '+ trick_type + ', ' + model)
        ax8.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
        plt.savefig(os.path.join(directory, 'plot8_f1adv_' + str(filebegin) + '.png'), bbox_inches='tight')
        plt.clf()


        ##Plot 9 - Cumulative advantage of Firm 1 (profit difference)
        fig9, ax9 = plt.subplots(1,dpi=200)
        inc = 0
        for ta in tamount_space:
            df = master_df.loc[ta,model]
            df['cumprof1'] = df['prof1'].cumsum()
            df['cumprof2'] = df['prof2'].cumsum()
            df['cum_soph_adv'] = df['cumprof1'] - df['cumprof2']
            if ta == 1000:
                trick_label = 'No trick'
                col = 'red'
            else:
                trick_label = 'trick: ' + str(round(ta,2))
                col = 'black'
            mark = marker_styles[inc%(len(marker_styles))]
            fig9 =  sns.lineplot(x = 'period', y = 'cum_soph_adv', data = df, label = 'F1 '+ trick_label,
                        ax = ax9, ci = None, linewidth = 0.5, color = col, marker = mark,
                         markers = True, ms = 5, markevery = marker_places)
            ax9.grid(linestyle = '-', linewidth = 0.3)
            ax9.spines['top'].set_visible(False)
            ax9.spines['right'].set_visible(False)
            ax9.spines['bottom'].set_visible(False)
            ax9.spines['left'].set_visible(False)
            inc += 1
        plt.ylabel('Sophisticated Advantage, cumulative')
        plt.xlabel('period')
        ax9.set_title('Cumulative Profit F1 Advantage - Trick: '+ trick_type + ', ' + model)
        ax9.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
        plt.savefig(os.path.join(directory, 'plot9_cumf1adv_' + str(filebegin) + '.png'), bbox_inches='tight')
        plt.clf()


        ##Plot 10 - Ratio of profits (Firm 1/Firm 2), non-cumulative
        fig10, ax10 = plt.subplots(1,dpi=200)
        inc = 0
        for ta in tamount_space:
            df = master_df.loc[ta,model]
            df['prof_ratio'] = df['prof1'] / df['prof2']
            if ta == 1000:
                trick_label = 'No trick'
                col= 'red'
            else:
                trick_label = 'trick: ' + str(round(ta,2))
                col = 'black'
            mark = marker_styles[inc%(len(marker_styles))]
            fig10 =  sns.lineplot(x = 'period', y = 'prof_ratio', data = df, label = 'F1 '+ trick_label,
                        ax = ax10, ci = None, linewidth = 0.5,  color = col, marker = mark,
                        markers = True, ms = 5, markevery = marker_places)
            sns.lineplot(x = 'period', y = 0, data = df, ax=ax10, color = 'yellow', ci = None, linewidth = 0.4)
            ax10.grid(linestyle = '-', linewidth = 0.3)
            ax10.spines['top'].set_visible(False)
            ax10.spines['right'].set_visible(False)
            ax10.spines['bottom'].set_visible(False)
            ax10.spines['left'].set_visible(False)
            inc += 1
        plt.ylabel('Profit ratio, F1/F2')
        plt.xlabel('period')
        ax10.set_title('Absolute Profit Ratio F1/F2 - Trick: '+ trick_type + ', ' + model)
        ax10.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
        plt.savefig(os.path.join(directory, 'plot10_profratio_' + str(filebegin) + '.png'), bbox_inches='tight')
        plt.clf()


        ##Plot 10a - Ratio of profits (Firm 1/Firm 2), cumulative
        fig10a, ax10a = plt.subplots(1,dpi=200)
        inc, line_inc = 0,0
        for ta in tamount_space:
            df = master_df.loc[ta,model]
            df['cumprof1'] = df['prof1'].cumsum()
            df['cumprof2'] = df['prof2'].cumsum()
            df['prof_ratio'] = df['cumprof1'] / df['cumprof2']
            if ta == 1000:
                trick_label = 'No trick'
                color = 'red'
            else:
                trick_label = 'trick: ' + str(round(ta,2))
                color = 'black'
            mark = marker_styles[inc%(len(marker_styles))]
            fig10a =  sns.lineplot(x = 'period', y = 'prof_ratio', data = df, label = 'F1 '+ trick_label,
                        ax = ax10a, ci = None, linewidth = 0.5, marker = mark, color = col,
                        markers = True, ms = 5, markevery = marker_places)
            sns.lineplot(x = 'period', y = 0, data = df, ax=ax10a, color = 'yellow', ci = None, linewidth = 0.4)
            ax10a.grid(linestyle = '-', linewidth = 0.3)
            ax10a.spines['top'].set_visible(False)
            ax10a.spines['right'].set_visible(False)
            ax10a.spines['bottom'].set_visible(False)
            ax10a.spines['left'].set_visible(False)
            inc += 1
        plt.ylabel('Cumulative profit ratio, F1/F2')
        plt.xlabel('period')
        ax10a.set_title('Cumulative Profit Ratio F1/F2 - Trick: '+ trick_type + ', ' + model)
        ax10a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
        plt.savefig(os.path.join(directory, 'plot10a_cumprofratio_' + str(filebegin) + '.png'), bbox_inches='tight')
        plt.clf()

        ###Comparisons to no-trick version:###


        if 1000 in tamount_space:

            ##Plot 11 - Comparison to no trick, noncumulative

            fig11, ax11 = plt.subplots(1,dpi=200)
            inc = 0
            df_notrick = master_df.loc[1000, model]
            for ta in tamount_space:
                df = master_df.loc[ta, model]
                df['abs_improvement'] = df['prof1'] - df_notrick['prof1']
                if ta == 1000:
                    trick_label = 'No trick'
                else:
                    trick_label = 'trick: ' + str(round(ta,2))
                mark = marker_styles[inc%(len(marker_styles))]
                fig11 =  sns.lineplot(x = 'period', y = 'abs_improvement', data = df, label = 'F1 '+ trick_label,
                            ax = ax11, ci = None, linewidth = 0.5, color = 'black', marker = mark,
                        markers = True, ms = 5, markevery = marker_places)
                sns.lineplot(x = 'period', y = 0, data = df, ax=ax11, color = 'yellow', ci = None, linewidth = 0.4)
                ax11.grid(linestyle = '-', linewidth = 0.3)
                ax11.spines['top'].set_visible(False)
                ax11.spines['right'].set_visible(False)
                ax11.spines['bottom'].set_visible(False)
                ax11.spines['left'].set_visible(False)
                inc += 1
            plt.ylabel('Profit gained by trick, non-cumulative')
            plt.xlabel('period')
            ax11.set_title('F1 improvement relative to self w no trick: '+ trick_type + ', ' + model)
            ax11.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
            plt.savefig(os.path.join(directory, 'plot11_absimprovement_' + str(filebegin) + '.png'), bbox_inches='tight')
            plt.clf()


            ##Plot 12 - Comparison to no trick, cumulative

            fig12, ax12 = plt.subplots(1,dpi=200)
            inc = 0
            df_notrick = master_df.loc[1000, model]
            for ta in tamount_space:
                df = master_df.loc[ta, model]
                df['cumprof1'] = df['prof1'].cumsum()
                df_notrick['cumprof1'] = df_notrick['prof1'].cumsum()
                df['cum_abs_improvement'] = df['cumprof1'] - df_notrick['cumprof1']
                if ta == 1000:
                    trick_label = 'No trick'
                else:
                    trick_label = 'trick: ' + str(round(ta,2))
                mark = marker_styles[inc%(len(marker_styles))]
                fig12 =  sns.lineplot(x = 'period', y = 'cum_abs_improvement', data = df, label = 'F1 '+ trick_label,
                            ax = ax12, ci = None, linewidth = 0.5, color = 'black', marker = mark,
                        markers = True, ms = 5, markevery = marker_places)
                sns.lineplot(x = 'period', y = 0, data = df, ax=ax12, color = 'yellow', ci = None, linewidth = 0.4)
                ax12.grid(linestyle = '-', linewidth = 0.3)
                ax12.spines['top'].set_visible(False)
                ax12.spines['right'].set_visible(False)
                ax12.spines['bottom'].set_visible(False)
                ax12.spines['left'].set_visible(False)
                inc += 1
            plt.ylabel('Profit gained by trick, cumulative')
            plt.xlabel('period')
            ax12.set_title('F1 cum. improvement relative to self w no trick: '+ trick_type + ', ' + model)
            ax12.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
            plt.savefig(os.path.join(directory, 'plot12_cumabsimprovement_' + str(filebegin) + '.png'), bbox_inches='tight')
            plt.clf()


################################################################################
###                          End of functions                                ###
################################################################################



# %% Running the model.
############################## INTERCEPT MODEL #################################

