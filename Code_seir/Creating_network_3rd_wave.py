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

def add_message(message):
    str_message=''
    str_message += str("#####################################")
    str_message += str('   ')+ str(message) + str('   ')
    for i in range(50-len(message)):
        str_message+=str('#')

    print(str_message[:])

def evenets_param(random=False):
    airport ,two_grades ,pre_school = 0,6,6
    restaurants ,work ,starting_prec =10,11, 69
    holiday ,quarantine_per,normal,max_value=7,90,19,80
    if random:
        max_value =np.random.randint(80, 95)
        airport = np.random.randint(0, 6)
        two_grades = np.random.randint(5, 15)
        pre_school = np.random.randint(5, 15)
        restaurants = np.random.randint(5, 15)
        work = np.random.randint(5, 15)
        starting_prec = np.random.randint(50, 80)
        holiday = np.random.randint(0, 10)
        quarantine_per = np.random.randint(85, 95)
        normal = np.random.randint(0, 20)

    return airport,two_grades,pre_school,restaurants,work,starting_prec,holiday,\
           quarantine_per,normal,max_value

def save_graphs(graphs, housholds,individual_ageGroups,names,directory):

    for name,graph in zip(names,graphs):
        networkx.write_gpickle(g, f"{directory}/G_distancingScale{name}.gpickle")

    pickle.dump(housholds, open(str(directory)+"/households.p", "wb"))  # save it into a file named save.p
    pickle.dump(individual_ageGroups, open(str(directory)+"/individual_ageGroups.p", "wb"))
    add_message('graph is saved')

def load_graphs(names,directory):
    for name in names:
        g = networkx.read_gpickle(f"{directory}/G_distancingScale{name}.gpickle")
        graphs_for_sim.append(g.copy())

    households = pickle.load(open(f"{directory}/households.p", "rb"))
    individual_ageGroups = pickle.load(open(f"{directory}/individual_ageGroups.p", "rb"))

    add_message('Loading graph')
    return graphs_for_sim,households,individual_ageGroups


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
    event= ['Minimum active cases', '5-6 back to school', '9-12 back to school',
            '7-8 back to school', 'British variant',  '3rd lockdown',
            'schools close', 'Most cases new mutations', 'Closed airport',
            'back:1-4,work', 'airports semi open', 'back:5-6,11-12',
            'purim', 'back:restaurants,7-10', 'airports fully opened', 'normal']
    percentage_out = [60, 50, 40, 
                      30, 30, 80,
                      90, 90, 95,
                      85, 80, 70,
                      50, 30, 20, 5]
    """

    running_info = open("General_information/parameters.yaml")
    running_info = yaml.load(running_info, Loader=yaml.FullLoader)

    np.random.seed(0)
    N = 10000
    INIT_ISYM = int((13045 / 9e6) * N)
    INIT_EXP = 0.1 * INIT_ISYM
    INIT_ASYM = (1.25*INIT_ISYM)*0.25

    General_information= running_info['General_data']
    vaccination_data = pd.read_csv(General_information['vaccination_data'])
    graph_info = graph_info(N)

    ###############################################################################
    ############################### Load vaccination data #########################
    ###############################################################################

    vaccination_data['date'] = pd.to_datetime(vaccination_data['date'], format='%Y-%m-%d')
    vaccination_data['num_day'] = (vaccination_data['date'] -
                                              pd.to_datetime(General_information['simulation_start_date'], format='%Y-%m-%d')).dt.days

    vaccination_data.iloc[:, 2:-1] = np.ceil(vaccination_data.iloc[:, 2:-1] * N / 9e6).astype(int)
    vaccination_data.iloc[1:, 2:-1] = vaccination_data.iloc[1:, 2:-1] - vaccination_data.iloc[:-1, 2:-1].values
    #vaccination_data['num_days'] = vaccination_data['num_day'] + 10

    ###############################################################################
    ############################### Creating events  ##############################
    ###############################################################################

    add_message('creating events')

    event = ['Minimum active cases', '5-6 back to school', '9-12 back to school',
             '7-8 back to school', 'British variant', '3rd lockdown',
             'schools close', 'Most cases new mutations', 'Closed airport',
             'back:1-4,work', 'airports semi open', 'back:5-6,11-12',
             'purim', 'back:restaurants,7-10', 'airports fully opened', 'normal']

    times_for_sim = np.array([160, 173, 178, 185, 203, 207, 218, 219, 235, 247, 254, 261, 265, 276, 285, 290]) - 160
    optimize =General_information['optimize']['status']
    decision = [General_information['optimize']['Number'] if optimize else 1 ]

    for number_of_runs in range(decision[0]):
        add_message(str(number_of_runs))
        airport, two_grades, pre_school, restaurants, work, starting_prec, holiday, \
        quarantine_per, normal,max_value =evenets_param(optimize)


        delta_prec = np.array([-pre_school - two_grades - work, -two_grades, -two_grades,
                               -two_grades, 0, +work + restaurants + two_grades * 3,
                               +two_grades, 0, +airport,
                               -pre_school - work, -airport, -two_grades * 2,
                               -holiday, -restaurants - two_grades, -airport, -normal])

        percentage_out = starting_prec + numpy.cumsum(delta_prec)
        percentage_out = ((percentage_out - (np.min(percentage_out))) /
                          (np.max(percentage_out) - (np.min(percentage_out) - 5)) * max_value + 5)
        percentage_out = numpy.append(percentage_out, quarantine_per)


        ###############################################################################
        ######################## change precentage to scale  ##########################
        ###############################################################################

        per_to_scale_data = numpy.load("percentage_edges_removed_by_scale.npy")
        scale_out = convert_percentage_to_scale(percentage=percentage_out/100, per_to_scale_data=per_to_scale_data)

        ###############################################################################
        ######################## Load or save graph  ##################################
        ###############################################################################

        graphs_for_sim = []
        if not General_information['Load_graph']['status']:

            demographic_graphs, individual_ageGroups, households = generate_demographic_contact_network(
                N=N, demographic_data=household_country_data('ISRAEL'),
                distancing_scales=list(scale_out), isolation_groups=[], verbose=False,
                layer_info=graph_info.layers_info, households_data=graph_info.housholds, number_of_people_each_household=graph_info.numb_each_household)

            G_baseline = demographic_graphs['baseline']

            names=[]
            for s in scale_out:
                names.append(str(s).replace('.', '_'))
                g = demographic_graphs[f'distancingScale{s}']
                graphs_for_sim.append(g.copy())


            if General_information['Save_graph']['status']:
                directory=General_information['Save_graph']["directory"]
                if not os.path.exists(directory):
                    os.makedirs(directory)
                networkx.write_gpickle(G_baseline,directory+"/G_baseline.gpickle")
                save_graphs(graphs_for_sim, households,individual_ageGroups,names,directory)

        else:
            load_directory = General_information['Load_graph']["directory"]
            G_baseline = networkx.read_gpickle(f"{load_directory}/G_baseline.gpickle")
            names=[]
            for s in scale_out:
                names.append(str(s).replace('.', '_'))


            graphs_for_sim,households,individual_ageGroups = load_graphs(names,load_directory)



        ###############################################################################
        ######################## defining the model ###################################
        ###############################################################################
        health_info = health_information(N, individual_ageGroups,optimze=General_information['optimize']['health_parameters'])


        BETA_1p1 = [b / 1.1 for b in health_info.BETA]
        BETA_1p15 = [b / 1.15 for b in health_info.BETA]
        BETA_1p2 = [b / 1.2 for b in health_info.BETA]
        BETA_1p25 = [b / 1.25 for b in health_info.BETA]
        BETA_1p3 = [b / 1.3 for b in health_info.BETA]
        BETA_1p4 = [b / 1.4 for b in health_info.BETA]

        BETA_Q_1p1 = [b / 1.1 for b in health_info.BETA]
        BETA_Q_1p15 = [b / 1.15 for b in health_info.BETA]
        BETA_Q_1p2 = [b / 1.2 for b in health_info.BETA]
        BETA_Q_1p25 = [b / 1.25 for b in health_info.BETA]
        BETA_Q_1p3 = [b / 1.3 for b in health_info.BETA]
        BETA_Q_1p4 = [b / 1.4 for b in health_info.BETA]


        households_indices = [household['indices'] for household in households]

        checkpoints = {'t': times_for_sim,
                       'G': graphs_for_sim[:-1],
                       'beta': [health_info.BETA, health_info.BETA, health_info.BETA,
                                BETA_1p1, BETA_1p1, BETA_1p1,
                                BETA_1p15, BETA_1p15, BETA_1p2,
                                BETA_1p2, BETA_1p25, BETA_1p25,
                                BETA_1p3, BETA_1p4, BETA_1p4, BETA_1p4],
                       'beta_Q': [health_info.BETA_Q, health_info.BETA_Q, health_info.BETA_Q,
                                  BETA_Q_1p1, BETA_Q_1p1, BETA_Q_1p1,
                                  BETA_Q_1p15, BETA_Q_1p15, BETA_Q_1p2,
                                  BETA_Q_1p2, BETA_Q_1p25, BETA_Q_1p25,
                                  BETA_Q_1p3, BETA_Q_1p4, BETA_Q_1p4, BETA_Q_1p4]}


        #-------------------------model over time --------------------------------
        # P_GLOBALINTXN = 0.1

        P_GLOBALINTXN = [0.09 if age in ['0-9', '10-19'] else 0.13 for age in individual_ageGroups]
        # interactions being with incidental or casual contacts outside their set of close contacts
        # Q_GLOBALINTXN = 0.05
        Q_GLOBALINTXN = [0.13 if age in ['10-19', '20-29'] else 0.06 for age in individual_ageGroups]
        # interaction when a person is in quarantine
        PCT_ASYMPTOMATIC = 0.25  # percent of asymptomatic


        node_groups = {k: np.where(np.array(individual_ageGroups) == k)[0] for k in np.unique(individual_ageGroups)}
        model = ExtSEIRSNetworkModel(G=G_baseline, p=P_GLOBALINTXN,
                                     beta=health_info.BETA, sigma=health_info.SIGMA, lamda=health_info.LAMDA, gamma=health_info.GAMMA,
                                     gamma_asym=health_info.GAMMA, eta=health_info.ETA, gamma_H=health_info.GAMMA_H, mu_H=health_info.MU_H,
                                     a=PCT_ASYMPTOMATIC, h=health_info.PCT_HOSPITALIZED, f=health_info.PCT_FATALITY,
                                     alpha=health_info.ALPHA, beta_pairwise_mode=health_info.BETA_PAIRWISE_MODE,
                                     delta_pairwise_mode=health_info.DELTA_PAIRWISE_MODE,
                                     G_Q=graphs_for_sim[-1], q=0, beta_Q=health_info.BETA_Q, isolation_time=10, initE=int(0.8 * INIT_EXP),
                                     initQ_E=INIT_EXP - int(0.8 * INIT_EXP), initI_sym=int(0.8 * INIT_ISYM),initI_asym=INIT_ASYM,
                                     initH=INIT_ISYM - int(0.8 * INIT_ISYM), seed=1, node_groups=node_groups,
                                     per_remove_vacc_edges=0.5)


        ###############################################################################
        ######################## defining the runing parameters #######################
        ###############################################################################

        general = running_info['model_param']['General']
        testing=running_info['model_param']['Testing']
        isolation = running_info['model_param']['ISOLATION']

        TESTING_COMPLIANCE_RANDOM = (numpy.random.rand(N) < testing['TESTING_COMPLIANCE_RATE_RANDOM'])
        TESTING_COMPLIANCE_TRACED = (numpy.random.rand(N) < testing['TESTING_COMPLIANCE_RATE_TRACED'])
        TESTING_COMPLIANCE_SYMPTOMATIC = (numpy.random.rand(N) < testing['TESTING_COMPLIANCE_RATE_SYMPTOMATIC'])
        TRACING_COMPLIANCE = (numpy.random.rand(N) < general['TRACING_COMPLIANCE_RATE'])
        ISOLATION_COMPLIANCE_SYMPTOMATIC_INDIVIDUAL = (numpy.random.rand(N) < isolation['ISOLATION_COMPLIANCE_RATE_SYMPTOMATIC_INDIVIDUAL'])
        ISOLATION_COMPLIANCE_SYMPTOMATIC_GROUPMATE = (numpy.random.rand(N) < isolation['ISOLATION_COMPLIANCE_RATE_SYMPTOMATIC_GROUPMATE'])
        ISOLATION_COMPLIANCE_POSITIVE_INDIVIDUAL = (numpy.random.rand(N) < isolation['ISOLATION_COMPLIANCE_RATE_POSITIVE_INDIVIDUAL'])
        ISOLATION_COMPLIANCE_POSITIVE_GROUPMATE = (numpy.random.rand(N) < isolation['ISOLATION_COMPLIANCE_RATE_POSITIVE_GROUPMATE'])
        ISOLATION_COMPLIANCE_POSITIVE_CONTACT = (numpy.random.rand(N) < isolation['ISOLATION_COMPLIANCE_RATE_POSITIVE_CONTACT'])
        ISOLATION_COMPLIANCE_POSITIVE_CONTACTGROUPMATE = (numpy.random.rand(N) < isolation['ISOLATION_COMPLIANCE_RATE_POSITIVE_CONTACTGROUPMATE'])

        T = 140
        run_tti_sim(model, T,
                    intervention_start_pct_infected=general['INTERVENTION_START_PCT_INFECTED'],
                    average_introductions_per_day=general['AVERAGE_INTRODUCTIONS_PER_DAY'],
                    testing_cadence=testing['TESTING_CADENCE'], pct_tested_per_day=testing['PCT_TESTED_PER_DAY'],
                    test_falseneg_rate=testing['TEST_FALSENEG_RATE'],
                    testing_compliance_symptomatic=TESTING_COMPLIANCE_SYMPTOMATIC,
                    max_pct_tests_for_symptomatics=testing['MAX_PCT_TESTS_FOR_SYMPTOMATICS'],
                    testing_compliance_traced=TESTING_COMPLIANCE_TRACED, max_pct_tests_for_traces=testing['MAX_PCT_TESTS_FOR_TRACES'],
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
                    isolation_lag_symptomatic=isolation['ISOLATION_LAG_SYMPTOMATIC'], isolation_lag_positive=isolation['ISOLATION_LAG_POSITIVE'],
                    isolation_groups=households_indices, checkpoints=checkpoints, vaccinations_df=vaccination_data)  #vaccination_data


        curr_date = time.strftime("%Y_%m_%d")

        directory =General_information['Save_graph']["directory"] +("/")+str(curr_date)
        if not os.path.exists(directory):
            os.makedirs(directory)
        name = "model"+str(number_of_runs)
        path_name = f"{directory}/{name}"
        with open(f"{path_name}.pickle", 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


        input_json = {'N': N,
                      'airport': airport,
                      'two_grades': two_grades,
                      'pre_school': pre_school,
                      'restaurants': restaurants,
                      'work': work,
                      'starting_prec': starting_prec,
                      'holiday': holiday,
                      'quarantine_per': quarantine_per,
                      'normal': normal,
                      'max_value': max_value,
                      'percentage_out': percentage_out,
                      'R0_factor' : health_info.factors['R0']}

        name = "input_config"+str(number_of_runs)
        path_name = f"{directory}/{name}"
        with open(f"{path_name}.pickle", 'wb') as handle:
            pickle.dump(input_json, handle, protocol=pickle.HIGHEST_PROTOCOL)


        if General_information['plot']['status']:
            results_summary(model)

            plt.plot(model.tseries, ((model.numI_sym+model.numH)*9e6)/N)
            for time, event in zip(times_for_sim, event):
                plt.axvline(int(time), 0, 1, c='k')
                plt.text(int(time)+0.1, 0, str(event), rotation=90)

            time = numpy.load('g_third_lockdown/time_3rd_wave.npy')
            active_casese = numpy.load('g_third_lockdown/active_cases_3rd_wave.npy')
            plt.plot(time, active_casese, '-r')

            plt.show()

            age_dist = np.array([0.197, 0.164, 0.14, 0.13, 0.118, 0.091, 0.081, 0.049, 0.03])
            for num, key in enumerate(model.nodeGroupData.keys()):
                mean_age = np.mean(model.nodeGroupData[key]['numPositive']/(model.numPositive+1e-5))


                plt.bar(key,(mean_age/age_dist[num]).astype(float))
            plt.show()
            fig, ax = model.figure_infections(combine_Q_infected=False)
            plt.show()
