# -*- coding: utf-8 -*-
"""
Created on Thu May  6 08:02:40 2021

@author: rsalisbury
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['figure.max_open_warning'] = 0
import matplotlib.pyplot as plt


import multiprocessing as mp
#print ("Number of processors: ", mp.cpu_count()) # 8 processors

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
                   figure_style, directory, filebegin, param_num, file_id):

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
                   #"\n x_var:" + str(x_var) +
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
    csv_name = str(filebegin) + '_' +  file_id + '_sim_data.csv'
    av_df.to_csv(os.path.join(directory, csv_name), header = True)
    total_capvals_df.to_csv(os.path.join(directory, str(filebegin) + '_' +  file_id + '_captest_results.csv'), header = True)
    
    return csv_name, initial_weights