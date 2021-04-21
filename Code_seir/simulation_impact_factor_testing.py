from seirsplus.models import *
from seirsplus.networks import *
from seirsplus.sim_loops import *
from seirsplus.utilities import *
import networkx
import time
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
from seirsplus.utilities import load_model, save_model
from General_information.graph_information import graph_info
from General_information.Health_information import health_information
import yaml
import os
import random


def add_message(message):
    str_message = ''
    str_message += str("#####################################")
    str_message += str('   ') + str(message) + str('   ')
    for i in range(50 - len(message)):
        str_message += str('#')

    print(str_message[:])


def evenets_param(random=False):
    # airport ,two_grades ,pre_school = 1,8,8
    # restaurants ,work ,starting_prec =8,5,20
    airport, two_grades, pre_school = 4, 4, 3
    restaurants, work, starting_prec = 3, 0, 1
    holiday, quarantine_per, minimum, max_value = 3, 91, 19, 66
    if random:
        max_value = np.random.randint(50, 80)
        airport = np.random.randint(0, 5)
        two_grades = np.random.randint(0, 5)
        pre_school = np.random.randint(0, 5)
        restaurants = np.random.randint(0, 5)
        work = np.random.randint(0, 5)
        starting_prec = np.random.randint(0, 5)
        holiday = np.random.randint(0, 5)
        quarantine_per = np.random.randint(85, 98)
        minimum = np.random.randint(5, 30)

    return airport, two_grades, pre_school, restaurants, work, starting_prec, holiday, \
           quarantine_per, minimum, max_value


def save_graphs(graphs, housholds, individual_ageGroups, names, directory):
    for name, graph in zip(names, graphs):
        networkx.write_gpickle(g, f"{directory}/G_distancingScale{name}.gpickle")

    pickle.dump(housholds, open(str(directory) + "/households.p", "wb"))  # save it into a file named save.p
    pickle.dump(individual_ageGroups, open(str(directory) + "/individual_ageGroups.p", "wb"))
    add_message('graph is saved')


def load_graphs(names, directory):
    for name in names:
        g = networkx.read_gpickle(f"{directory}/G_distancingScale{name}.gpickle")
        graphs_for_sim.append(g.copy())

    households = pickle.load(open(f"{directory}/households.p", "rb"))
    individual_ageGroups = pickle.load(open(f"{directory}/individual_ageGroups.p", "rb"))

    add_message('Loading graph')
    return graphs_for_sim, households, individual_ageGroups


if __name__ == "__main__":
    """
    Second wave:
    01.06.2020 --- 2006 ---------- everything is open (G_baseline) -0
    05.06.2020 -------------------start learning in zoom (G_0_prec_less)-5
    19.06.2020 -------------------hofesh gadol 
    01.07  ----------------------- closing <50 open <20 close (G_24_prec_les) -30
    01.08 -------------------------closing work (G_60_prec_les) -60
    01.09--------------------------schools are back with yashivot (G_24_prec_les)-90
    25.09 ------------------------ full seger (G_98_prec) - 115
    01.10 -------------------------maximum 72k  - 120
    17.10-------------------------kindergarden back (G_85_prec) -137
    1.11-------------------------grades 1-4 back (G_70_prec) - 150
    8.11------------------------- stores outside open - 157
    10.11 minimum--------------- 8000  - 160   
    24.11------------------------- grades 5-6 back - 173 (G_50_prec)
    29.11------------------------- grades 9-12 back - 178 (G_40_prec)
    06.12------------------------- grades 7-8 back - 185 (G_30_prec)
    Third wave:  
    20.12 ---------------------- started vaccinations for 60+ and doctors - 200 
    23.12---------------------- people returning from abroad in hotels - 203 
    23.12 ---------------------- british mutation in israel - 203 (R needs to be enlarged by 40%) assume 10% have the british variant
    27.12---------------------- beginning of 3rd seger, grades 7-10 closed - 207 (G_30_prec)
    8.1 ---------------------- more restrictions in 3rd seger, closed all schools - 218 (G_95_prec)
    9.1 ---------------------- south african mutation in israel - 219 (most cases are south african or british) update R
    12.1---------------------- teachers and over 50 can vaccinate - 222 
    19.1 ---------------------- over 40 can vaccinate - 229
    19.1 ---------------------- 25% of population got first vaccination - 229
    23.1 ---------------------- 16-18 can vaccinate - 233
    25.1 ---------------------- closed airports - 235 (G_98_prec)
    4.2 ---------------------- all population over 16 can vaccinate - 245
    7.2 ---------------------- 1-4 in orange+ cities back to school + opening work - 247 (G_85_prec)
    10.2 ---------------------- 25% of population gets the second vaccination - 250
    14.2 ---------------------- opening airports for up to 3000 people - 254 (G_80_prec)
    21.2 ---------------------- 5-6 and 11-12 back to school + stores - 261 (G_60_prec)
    25.2 ---------------------- purim, closed in the evening - 265  (G_50_prec)
    2.3 ---------------------- people post covid can vaccinate - 272
    7.3 ---------------------- restaurants open for vaccinated population, 7-10 back to school - 276 (G_30_prec)
    16.3 ---------------------- flights allowed back in israel - 285 (G_20_prec)
    21.3 ---------------------- everything normal - 290 (G_10_prec)
    30.3 ---------------------- no red cities - 299
    
    
    time = [160, 173, 178, 185, 203, 207, 218, 219, 235, 247, 254, 261, 265, 276, 285, 290]
    
    event= [160: 'Minimum active cases', 173: '5-6 back to school', 178: '9-12 back to school',
            185: '7-8 back to school', 203: 'British variant', 207: '3rd lockdown',
            218: 'schools close', 219: 'Most cases new mutations', 235: 'Closed airport',
            247: 'back:1-4,work', 25s: 'airports semi open', 261: 'back:5-6,11-12',
            265: 'purim', 276:'back:restaurants,7-10' , 285: 'airports fully opened', 'normal']
            
            [-pre_school - two_grades - work, -two_grades, -two_grades,
                               -two_grades, 0, +work + restaurants + two_grades * 3,
                               +two_grades, 0, +airport,
                               -pre_school - work, -airport, -two_grades * 2,
                               -holiday, -restaurants - two_grades, -airport, -normal]
    percentage_out = [60, 50, 40, 
                      30, 30, 80,
                      90, 90, 95,
                      85, 80, 70,
                      50, 30, 20, 5]
    """

    running_info = open("General_information/parameters_impact_factor.yaml")
    running_info = yaml.load(running_info, Loader=yaml.FullLoader)

    N = 10000
    start_day = 0
    INIT_EXP = int(0.05 * N)

    General_information = running_info['General_data']
    vaccination_data = pd.read_csv(General_information['vaccination_data'])
    graph_info = graph_info(N)

    ###############################################################################
    ############################### Load vaccination data #########################
    ###############################################################################

    vaccination_data['date'] = pd.to_datetime(vaccination_data['date'], format='%Y-%m-%d')
    vaccination_data['num_day'] = (vaccination_data['date'] -
                                   pd.to_datetime(General_information['simulation_start_date'],
                                                  format='%Y-%m-%d')).dt.days
    vaccination_data['num_day'] = vaccination_data['num_day'] + start_day

    vaccination_data.iloc[:, 2:-1] = np.ceil(vaccination_data.iloc[:, 2:-1] * N / 9e6).astype(int)

    ###############################################################################
    ############################### Creating events  ##############################
    ###############################################################################

    add_message('creating events')

    event = ['Minimum active cases', '5-6 back to school', '9-12 back to school',
             '7-8 back to school', 'British variant', '3rd lockdown',
             'schools close', 'Most cases new mutations', 'Closed airport',
             'back:1-4,work', 'airports semi open', 'back:5-6,11-12',
             'purim', 'back:restaurants,7-10', 'airports fully opened', 'normal']

    times_for_sim = np.array(
        [160, 173, 178, 185, 203, 207, 218, 219, 235, 247, 254, 261, 265, 276, 285, 290]) - 160 + start_day
    optimize = General_information['optimize']['status']
    # decision = [General_information['optimize']['Number'] if optimize else 1]
    decision = General_information['optimize']['Number']
    curr_date = time.strftime("%Y_%m_%d")
    directory = General_information['Save_model']["directory"] + ("/") + str(curr_date)
    if not os.path.exists(directory):
        os.makedirs(directory)
    files_name = os.listdir(directory)
    intial = int(len(files_name) / 2)

    for number_of_runs in range(intial, intial + decision):
        if General_information['seed']['status']:
            seed = running_info['General_data']['seed']['Number']
            np.random.seed(seed)
            random.seed(seed)
        else:
            seed = np.random.randint(0, 1000)
            np.random.seed(seed)
            random.seed(seed)
        add_message(f'seed:{seed}')
        add_message(str(number_of_runs))
        distancing_scales = [0.7]
        ###############################################################################
        ######################## Load or save graph  ##################################
        ###############################################################################

        graphs_for_sim = []
        if not General_information['Load_graph']['status']:

            demographic_graphs, individual_ageGroups, households = generate_demographic_contact_network(
                N=N, demographic_data=household_country_data('ISRAEL'),
                distancing_scales=distancing_scales, isolation_groups=[], verbose=False,
                layer_info=graph_info.layers_info, households_data=graph_info.housholds,
                number_of_people_each_household=graph_info.numb_each_household, seed=seed)

            G_baseline = demographic_graphs['baseline']

            names = []
            for s in distancing_scales:
                names.append(str(s).replace('.', '_'))
                g = demographic_graphs[f'distancingScale{s}']
                graphs_for_sim.append(g.copy())

            if General_information['Save_graph']['status']:
                directory = General_information['Save_graph']["directory"]
                curr_save_folder = f"{directory}/{curr_date}/{number_of_runs}"
                if not os.path.exists(curr_save_folder):
                    os.makedirs(curr_save_folder)
                networkx.write_gpickle(G_baseline, f"{curr_save_folder}/G_baseline.gpickle")
                save_graphs(graphs_for_sim, households, individual_ageGroups, names, curr_save_folder)

        else:
            load_directory = General_information['Load_graph']["directory"]
            G_baseline = networkx.read_gpickle(f"{load_directory}/G_baseline.gpickle")
            names = []
            for s in distancing_scales:
                names.append(str(s).replace('.', '_'))

            graphs_for_sim, households, individual_ageGroups = load_graphs(names, load_directory)

        ###############################################################################
        ######################## defining the model ###################################
        ###############################################################################

        health_info = health_information(N, individual_ageGroups,
                                         optimze=General_information['optimize']['health_parameters'])
        add_message(f"vaccination policy:{General_information['vacc_policy']}")
        if General_information['vacc_policy'] == "regular":
            vaccination_data_out = vaccination_data.drop(columns=["num_tot"])
        elif General_information['vacc_policy'] == "no_policy":
            # vaccination_data_out = None
            vaccination_data.iloc[:, 2:-1] = 0 * vaccination_data.iloc[:, 2:-1]
            vaccination_data_out = vaccination_data.drop(columns=["num_tot"])

        else:
            vaccination_data_out = health_info.create_vaccination_df(N, total_vacc_per_day_df=vaccination_data,
                                                                     individual_ages_list=individual_ageGroups,
                                                                     vacc_policy=General_information['vacc_policy'])
            vaccination_data_out = vaccination_data_out.drop(columns=["num_tot"])

        BETA_1p1 = [b * 1.1 for b in health_info.BETA]
        BETA_1p15 = [b * 1.15 for b in health_info.BETA]
        BETA_1p2 = [b * 1.2 for b in health_info.BETA]
        BETA_1p25 = [b * 1.25 for b in health_info.BETA]
        BETA_1p3 = [b * 1.4 for b in health_info.BETA]
        BETA_1p4 = [b * 1.4 for b in health_info.BETA]

        BETA_Q_1p1 = [b * 1.1 for b in health_info.BETA]
        BETA_Q_1p15 = [b * 1.15 for b in health_info.BETA]
        BETA_Q_1p2 = [b * 1.2 for b in health_info.BETA]
        BETA_Q_1p25 = [b * 1.25 for b in health_info.BETA]
        BETA_Q_1p3 = [b * 1.4 for b in health_info.BETA]
        BETA_Q_1p4 = [b * 1.4 for b in health_info.BETA]

        households_indices = [household['indices'] for household in households]

        # checkpoints = {'t': times_for_sim,
        #                'G': graphs_for_sim[:-1],
        #                'beta': [health_info.BETA, health_info.BETA, health_info.BETA,
        #                         BETA_1p1, BETA_1p1, BETA_1p1,
        #                         BETA_1p15, BETA_1p15, BETA_1p2,
        #                         BETA_1p2, BETA_1p25, BETA_1p25,
        #                         BETA_1p3, BETA_1p4, BETA_1p4, BETA_1p4],
        #                'beta_Q': [health_info.BETA_Q, health_info.BETA_Q, health_info.BETA_Q,
        #                           BETA_Q_1p1, BETA_Q_1p1, BETA_Q_1p1,
        #                           BETA_Q_1p15, BETA_Q_1p15, BETA_Q_1p2,
        #                           BETA_Q_1p2, BETA_Q_1p25, BETA_Q_1p25,
        #                           BETA_Q_1p3, BETA_Q_1p4, BETA_Q_1p4, BETA_Q_1p4]}
        checkpoints = None

        # -------------------------model over time --------------------------------
        if General_information['optimize']['health_parameters']:
            # interactions being with incidental or casual contacts outside their set of close contacts
            p_glob_kids = np.random.randint(5, 15) / 100
            p_glob_adults = np.random.randint(10, 30) / 100
            # interaction when a person is in quarantine
            q_glob_kids = [np.random.randint(5, 20) / 100, np.random.randint(5, 20) / 100]
            q_glob_adults = np.random.randint(1, 10) / 100
            percentage_edges_removed_vacc = np.random.randint(50, 100) / 100
        else:
            # interactions being with incidental or casual contacts outside their set of close contacts
            p_glob_kids = 0
            p_glob_adults = 0
            # interaction when a person is in quarantine
            q_glob_kids = [0, 0]
            q_glob_adults = 0
            percentage_edges_removed_vacc = 0.9
        Q_GLOBALINTXN = []
        for age in individual_ageGroups:
            if age == '10-19':
                Q_GLOBALINTXN.append(q_glob_kids[0])
            elif age == '20-29':
                Q_GLOBALINTXN.append(q_glob_kids[1])
            else:
                Q_GLOBALINTXN.append(q_glob_adults)
        P_GLOBALINTXN = [p_glob_kids if age in ['0-9', '10-19'] else p_glob_adults for age in individual_ageGroups]
        # Q_GLOBALINTXN = [q_glob_kids if age in ['10-19', '20-29'] else q_glob_adults for age in
        #                  individual_ageGroups]

        PCT_ASYMPTOMATIC = 0  # percent of asymptomatic

        node_groups = {k: np.where(np.array(individual_ageGroups) == k)[0] for k in np.unique(individual_ageGroups)}
        model = ExtSEIRSNetworkModel(G=G_baseline, p=P_GLOBALINTXN,
                                     beta=health_info.BETA, sigma=health_info.SIGMA, lamda=health_info.LAMDA,
                                     gamma=health_info.GAMMA,
                                     gamma_asym=health_info.GAMMA, eta=health_info.ETA, gamma_H=health_info.GAMMA_H,
                                     mu_H=health_info.MU_H,
                                     a=PCT_ASYMPTOMATIC, h=health_info.PCT_HOSPITALIZED, f=health_info.PCT_FATALITY,
                                     alpha=health_info.ALPHA, beta_pairwise_mode=health_info.BETA_PAIRWISE_MODE,
                                     delta_pairwise_mode=health_info.DELTA_PAIRWISE_MODE,
                                     G_Q=graphs_for_sim[-1], q=0, beta_Q=health_info.BETA_Q, isolation_time=10,
                                     initE=int(0.8 * INIT_EXP),
                                     initQ_E=INIT_EXP - int(0.8 * INIT_EXP), initI_sym=int(0.8 * 0),
                                     initI_asym=0,
                                     initH=0 - int(0.8 * 0), seed=seed, node_groups=node_groups,
                                     per_remove_vacc_edges=percentage_edges_removed_vacc, initR=0,
                                     store_Xseries=True)

        ###############################################################################
        ######################## defining the runing parameters #######################
        ###############################################################################

        general = running_info['model_param']['General']
        testing = running_info['model_param']['Testing']
        isolation = running_info['model_param']['ISOLATION']

        TESTING_COMPLIANCE_RANDOM = (numpy.random.rand(N) < testing['TESTING_COMPLIANCE_RATE_RANDOM'])
        TESTING_COMPLIANCE_TRACED = (numpy.random.rand(N) < testing['TESTING_COMPLIANCE_RATE_TRACED'])
        TESTING_COMPLIANCE_SYMPTOMATIC = (numpy.random.rand(N) < testing['TESTING_COMPLIANCE_RATE_SYMPTOMATIC'])
        TRACING_COMPLIANCE = (numpy.random.rand(N) < general['TRACING_COMPLIANCE_RATE'])
        ISOLATION_COMPLIANCE_SYMPTOMATIC_INDIVIDUAL = (
                numpy.random.rand(N) < isolation['ISOLATION_COMPLIANCE_RATE_SYMPTOMATIC_INDIVIDUAL'])
        ISOLATION_COMPLIANCE_SYMPTOMATIC_GROUPMATE = (
                numpy.random.rand(N) < isolation['ISOLATION_COMPLIANCE_RATE_SYMPTOMATIC_GROUPMATE'])
        ISOLATION_COMPLIANCE_POSITIVE_INDIVIDUAL = (
                numpy.random.rand(N) < isolation['ISOLATION_COMPLIANCE_RATE_POSITIVE_INDIVIDUAL'])
        ISOLATION_COMPLIANCE_POSITIVE_GROUPMATE = (
                numpy.random.rand(N) < isolation['ISOLATION_COMPLIANCE_RATE_POSITIVE_GROUPMATE'])
        ISOLATION_COMPLIANCE_POSITIVE_CONTACT = (
                numpy.random.rand(N) < isolation['ISOLATION_COMPLIANCE_RATE_POSITIVE_CONTACT'])
        ISOLATION_COMPLIANCE_POSITIVE_CONTACTGROUPMATE = (
                numpy.random.rand(N) < isolation['ISOLATION_COMPLIANCE_RATE_POSITIVE_CONTACTGROUPMATE'])

        T = 200
        input_json = {'N': N,
                      'p_global_kids': p_glob_kids,
                      'p_global_adults': p_glob_adults,
                      'q_global_kids': q_glob_kids,
                      'q_global_adults': q_glob_adults,
                      'seed': seed,
                      'percentage_edges_removed_vacc': percentage_edges_removed_vacc,
                      'vacc_policy': General_information['vacc_policy'],
                      'R0_factor': health_info.factors['R0']}
        add_message(f"inputs are:{input_json}")
        run_tti_sim(model, T,
                    intervention_start_pct_infected=general['INTERVENTION_START_PCT_INFECTED'],
                    average_introductions_per_day=general['AVERAGE_INTRODUCTIONS_PER_DAY'],
                    testing_cadence=testing['TESTING_CADENCE'], pct_tested_per_day=testing['PCT_TESTED_PER_DAY'],
                    test_falseneg_rate=testing['TEST_FALSENEG_RATE'],
                    testing_compliance_symptomatic=TESTING_COMPLIANCE_SYMPTOMATIC,
                    max_pct_tests_for_symptomatics=testing['MAX_PCT_TESTS_FOR_SYMPTOMATICS'],
                    testing_compliance_traced=TESTING_COMPLIANCE_TRACED,
                    max_pct_tests_for_traces=testing['MAX_PCT_TESTS_FOR_TRACES'],
                    testing_compliance_random=TESTING_COMPLIANCE_RANDOM,
                    random_testing_degree_bias=testing['RANDOM_TESTING_DEGREE_BIAS'],
                    tracing_compliance=TRACING_COMPLIANCE, pct_contacts_to_trace=general['PCT_CONTACTS_TO_TRACE'],
                    tracing_lag=general['TRACING_LAG'],
                    isolation_compliance_symptomatic_individual=ISOLATION_COMPLIANCE_SYMPTOMATIC_INDIVIDUAL,
                    isolation_compliance_symptomatic_groupmate=ISOLATION_COMPLIANCE_SYMPTOMATIC_GROUPMATE,
                    isolation_compliance_positive_individual=ISOLATION_COMPLIANCE_POSITIVE_INDIVIDUAL,
                    isolation_compliance_positive_groupmate=ISOLATION_COMPLIANCE_POSITIVE_GROUPMATE,
                    isolation_compliance_positive_contact=ISOLATION_COMPLIANCE_POSITIVE_CONTACT,
                    isolation_compliance_positive_contactgroupmate=ISOLATION_COMPLIANCE_POSITIVE_CONTACTGROUPMATE,
                    isolation_lag_symptomatic=isolation['ISOLATION_LAG_SYMPTOMATIC'],
                    isolation_lag_positive=isolation['ISOLATION_LAG_POSITIVE'],
                    isolation_groups=households_indices, checkpoints=checkpoints,
                    vaccinations_df=vaccination_data_out)  # vaccination_data

        if General_information['Save_model']['status']:
            curr_date = time.strftime("%Y_%m_%d")
            directory = General_information['Save_model']["directory"] + ("/") + str(curr_date)
            name = "model" + str(number_of_runs)
            path_name = f"{directory}/{name}"
            with open(f"{path_name}.pickle", 'wb') as handle:
                pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

            name = "input_config" + str(number_of_runs)
            path_name = f"{directory}/{name}"
            with open(f"{path_name}.pickle", 'wb') as handle:
                pickle.dump(input_json, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(f"{path_name}.pickle", 'wb') as handle:
                pickle.dump(input_json, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if General_information['plot']['status']:
            results_summary(model)

            plt.plot(model.tseries, (model.numPositive * 9e6) / N, label="simulation data")
            for t, event in zip(times_for_sim, event):
                plt.axvline(int(t) + start_day, 0, 1, c='k')
                plt.text(int(t) + 0.1 + start_day, 0, str(event), rotation=90)
            accumulated_data = pd.read_csv("accumulated_data_for_sim.csv")
            time_acc = accumulated_data.weeks_from * 7
            num_acc = accumulated_data.accumulated_cases_country
            plt.plot(time_acc + start_day, num_acc, '-r', label="real data")
            plt.legend()
            plt.figure()
            plt.plot(model.tseries, (model.numR * 9e6) / N, label="num recovered", marker='*', color='r')
            plt.plot(model.tseries, (model.numPositive * 9e6) / N, label="num positive", marker='.', color='g')
            plt.legend()
            plt.figure()
            age_dist = np.array([0.197, 0.164, 0.14, 0.13, 0.118, 0.091, 0.081, 0.049, 0.03])
            for num, key in enumerate(model.nodeGroupData.keys()):
                mean_age = np.mean(model.nodeGroupData[key]['numPositive'] / (model.numPositive + 1e-5))

                plt.bar(key, (mean_age / age_dist[num]).astype(float))
            fig, ax = model.figure_infections(combine_Q_infected=False)
            plt.show()
