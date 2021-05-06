# -*- coding: utf-8 -*-
"""
Created on Thu May  6 08:07:06 2021

@author: rsalisbury
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['figure.max_open_warning'] = 0
import matplotlib.pyplot as plt
import seaborn as sns


import multiprocessing as mp
#print ("Number of processors: ", mp.cpu_count()) # 8 processors

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

def graph_sns_overlap(df_dict, sweep_values, model_list, directory, dir_list, initial_weights, sweep_variables):
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
    dfs, option_list, model_list, directory, dir_list, sweep_variables = df_dict, sweep_values, model_list, directory, dir_list, sweep_variables
    master_df = pd.concat(dfs, sort = True).astype(float)
    print(df_dict)

    var_type = sweep_variables
    sweep_type = ''
    for var in var_type:
        sweep_type += var + '_'

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
