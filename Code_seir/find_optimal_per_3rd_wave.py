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
from seirsplus.utilities import save_model
import pathlib

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
    id_run = 5
    N = 10000
    # simulation params -
    ISOLATION_LAG_SYMPTOMATIC = 1  # number of days between onset of symptoms and self-isolation of symptomatics
    ISOLATION_LAG_POSITIVE = 2  # test turn-around time (TAT): number of days between administration of test and isolation of positive cases
    ISOLATION_LAG_CONTACT = 0  # number of days between a contact being traced and that contact self-isolating
    INTERVENTION_START_PCT_INFECTED = 0.1 / 100
    TESTING_COMPLIANCE_RATE_SYMPTOMATIC = 0.5
    TESTING_COMPLIANCE_RATE_TRACED = 1.0
    TESTING_COMPLIANCE_RATE_RANDOM = 0.8

    TRACING_COMPLIANCE_RATE = 0.8
    AVERAGE_INTRODUCTIONS_PER_DAY = 0

    TESTING_CADENCE = 'everyday'  # how often to do tracing testing and random testing
    PCT_TESTED_PER_DAY = 1 / 100  # max daily test allotment defined as a percent of population size
    TEST_FALSENEG_RATE = 'temporal'  # test false negative rate, will use FN rate that varies with disease time
    MAX_PCT_TESTS_FOR_SYMPTOMATICS = 1.0  # max percent of daily test allotment to use on self-reporting symptomatics
    MAX_PCT_TESTS_FOR_TRACES = 1.0  # max percent of daily test allotment to use on contact traces
    RANDOM_TESTING_DEGREE_BIAS = 0  # magnitude of degree bias in random selections for testing, none here

    PCT_CONTACTS_TO_TRACE = 0.5  # percentage of primary cases' contacts that are traced
    TRACING_LAG = 2  # number of cadence testing days between primary tests and tracing tests

    ISOLATION_COMPLIANCE_RATE_SYMPTOMATIC_INDIVIDUAL = 0.3
    ISOLATION_COMPLIANCE_RATE_SYMPTOMATIC_GROUPMATE = 0.0
    ISOLATION_COMPLIANCE_RATE_POSITIVE_INDIVIDUAL = 0.8
    ISOLATION_COMPLIANCE_RATE_POSITIVE_GROUPMATE = 0.8
    ISOLATION_COMPLIANCE_RATE_POSITIVE_CONTACT = 0.8
    ISOLATION_COMPLIANCE_RATE_POSITIVE_CONTACTGROUPMATE = 0.0
    TESTING_COMPLIANCE_RANDOM = (numpy.random.rand(N) < TESTING_COMPLIANCE_RATE_RANDOM)
    TESTING_COMPLIANCE_TRACED = (numpy.random.rand(N) < TESTING_COMPLIANCE_RATE_TRACED)
    TESTING_COMPLIANCE_SYMPTOMATIC = (numpy.random.rand(N) < TESTING_COMPLIANCE_RATE_SYMPTOMATIC)
    TRACING_COMPLIANCE = (numpy.random.rand(N) < TRACING_COMPLIANCE_RATE)
    ISOLATION_COMPLIANCE_SYMPTOMATIC_INDIVIDUAL = (
            numpy.random.rand(N) < ISOLATION_COMPLIANCE_RATE_SYMPTOMATIC_INDIVIDUAL)
    ISOLATION_COMPLIANCE_SYMPTOMATIC_GROUPMATE = (
            numpy.random.rand(N) < ISOLATION_COMPLIANCE_RATE_SYMPTOMATIC_GROUPMATE)
    ISOLATION_COMPLIANCE_POSITIVE_INDIVIDUAL = (numpy.random.rand(N) < ISOLATION_COMPLIANCE_RATE_POSITIVE_INDIVIDUAL)
    ISOLATION_COMPLIANCE_POSITIVE_GROUPMATE = (numpy.random.rand(N) < ISOLATION_COMPLIANCE_RATE_POSITIVE_GROUPMATE)
    ISOLATION_COMPLIANCE_POSITIVE_CONTACT = (numpy.random.rand(N) < ISOLATION_COMPLIANCE_RATE_POSITIVE_CONTACT)
    ISOLATION_COMPLIANCE_POSITIVE_CONTACTGROUPMATE = (
            numpy.random.rand(N) < ISOLATION_COMPLIANCE_RATE_POSITIVE_CONTACTGROUPMATE)

    T = 200

    vaccination_data = pd.read_csv("../israel_data/vaccination_data_for_simulation.csv")
    simulation_start_date = '2020-11-10'
    vaccination_data['date'] = pd.to_datetime(vaccination_data['date'], format='%Y-%m-%d')
    vaccination_data['num_day'] = (vaccination_data['date'] -
                                   pd.to_datetime(simulation_start_date, format='%Y-%m-%d')).dt.days

    vaccination_data.iloc[:, 2:-1] = np.ceil(vaccination_data.iloc[:, 2:-1] * N / 9e6).astype(int)
    vaccination_data.iloc[1:, 2:-1] = vaccination_data.iloc[1:, 2:-1] - vaccination_data.iloc[:-1, 2:-1].values
    # create graph information for different policies and interventions
    times_for_sim = np.array([160, 173, 178, 185, 203, 207, 218, 219, 235, 247, 254, 261, 265, 276, 285, 290]) - 160

    INIT_ISYM = int((13045 / 9e6) * N)
    per_to_scale_data = numpy.load("percentage_edges_removed_by_scale.npy")
    event = ['Minimum active cases', '5-6 back to school', '9-12 back to school',
             '7-8 back to school', 'British variant', '3rd lockdown',
             'schools close', 'Most cases new mutations', 'Closed airport',
             'back:1-4,work', 'airports semi open', 'back:5-6,11-12',
             'purim', 'back:restaurants,7-10', 'airports fully opened', 'normal']

    for i in range(100):
        INIT_EXP = np.random.random() * INIT_ISYM
        airport = np.random.randint(0, 6)
        two_grades = np.random.randint(5, 15)
        pre_school = np.random.randint(5, 15)
        restaurants = np.random.randint(5, 15)
        work = np.random.randint(5, 15)
        starting_prec = np.random.randint(50, 80)
        holiday = np.random.randint(0, 10)
        quarantine_per = np.random.randint(85, 95)
        normal = np.random.randint(0, 20)
        # airport = 5
        # two_grades = 10
        # pre_school = 5
        # restaurants = 15
        # work = 10
        # starting_prec = 75
        # holiday = 5
        # quarantine_per = 98
        # normal = 0
        max_value = np.random.randint(80, 95)

        delta_prec = np.array([-pre_school - two_grades - work, -two_grades, -two_grades,
                               -two_grades, 0, +work + restaurants + two_grades * 3,
                               +two_grades, 0, +airport,
                               -pre_school - work, -airport, -two_grades * 2,
                               -holiday, -restaurants - two_grades, -airport, -normal])
        percentage_out = starting_prec + numpy.cumsum(delta_prec)
        percentage_out = ((percentage_out - (np.min(percentage_out))) /
                          (np.max(percentage_out) - (np.min(percentage_out) - 5)) * max_value + 5)
        percentage_out = numpy.append(percentage_out, quarantine_per)
        scale_out = convert_percentage_to_scale(percentage=percentage_out / 100, per_to_scale_data=per_to_scale_data)

        graphs_for_sim = []
        demographic_graphs, individual_ageGroups, households = generate_demographic_contact_network(
            N=N, demographic_data=household_country_data('ISRAEL'),
            distancing_scales=list(scale_out), isolation_groups=[], verbose=False)
        G_baseline = demographic_graphs['baseline']
        # -------------------------model over time --------------------------------
        # interactions being with incidental or casual contacts outside their set of close contacts
        p_glob_kids = np.random.randint(5, 15)/100
        p_glob_adults = np.random.randint(10, 30)/100
        P_GLOBALINTXN = [p_glob_kids if age in ['0-9', '10-19'] else p_glob_adults for age in individual_ageGroups]
        # interaction when a person is in quarantine
        q_glob_kids = np.random.randint(5, 20) / 100
        q_glob_adults = np.random.randint(1, 10) / 100
        Q_GLOBALINTXN = [q_glob_kids if age in ['10-19', '20-29'] else q_glob_adults for age in individual_ageGroups]
        PCT_ASYMPTOMATIC = 0.25  # percent of asymptomatic

        # ------------------------------------------------------------------------------#
        # -------------------Hospitalization data----------------------------------------#
        # -------------------------------------------------------------------------------#

        ageGroup_pctHospitalized = {'0-9': 0.0000,  # We need to use the minister of health data
                                    '10-19': 0.0004,
                                    '20-29': 0.0104,
                                    '30-39': 0.0343,
                                    '40-49': 0.0425,
                                    '50-59': 0.0816,
                                    '60-69': 0.118,
                                    '70-79': 0.166,
                                    '80+': 0.184}

        PCT_HOSPITALIZED = [ageGroup_pctHospitalized[ageGroup] for ageGroup in individual_ageGroups]

        # ------------------------------------------------------------------------------#
        # -------------------Fatality Rate data------------------------------------------#
        # -------------------------------------------------------------------------------#

        ageGroup_hospitalFatalityRate = {'0-9': 0.0000,
                                         '10-19': 0.3627,
                                         '20-29': 0.0577,
                                         '30-39': 0.0426,
                                         '40-49': 0.0694,
                                         '50-59': 0.1532,
                                         '60-69': 0.3381,
                                         '70-79': 0.5187,
                                         '80+': 0.7283}
        PCT_FATALITY = [ageGroup_hospitalFatalityRate[ageGroup] for ageGroup in individual_ageGroups]
        for s in scale_out:
            s_str = str(s).replace('.', '_')
            g = demographic_graphs[f'distancingScale{s}']
            graphs_for_sim.append(g.copy())
        G_quarantine = graphs_for_sim.pop()
        pickle.dump(households, open("g_third_lockdown/households.p", "wb"))  # save it into a file named save.p
        pickle.dump(individual_ageGroups, open("g_third_lockdown/individual_ageGroups.p", "wb"))
        households_indices = [household['indices'] for household in households]

        # -----------------------pandemic parameters ----------------------------

        latentPeriod_mean, latentPeriod_coeffvar = 3.0, 0.6
        presymptomaticPeriod_mean, presymptomaticPeriod_coeffvar = 2.2, 0.5
        symptomaticPeriod_mean, symptomaticPeriod_coeffvar = 4.0, 0.4
        onsetToHospitalizationPeriod_mean, onsetToHospitalizationPeriod_coeffvar = 11.0, 0.45
        hospitalizationToDischargePeriod_mean, hospitalizationToDischargePeriod_coeffvar = 11.0, 0.45
        hospitalizationToDeathPeriod_mean, hospitalizationToDeathPeriod_coeffvar = 7.0, 0.45
        """
        1/Lamda  = latent period
        1/SIGMA = pre-symptomatic period
        BETA = Individual Transmissibility Value = R0/ infectius period
        GAMMA =  recoviring rate = inverse of the expected symptomatic periods assigned to each individual [1/min]
        ETA =  1/ expected onset-to-hospitalization periods
        gamma_H  = 1/ hospitalization-to-discharge
        MU_H  =  1/  hospitalization-to-death
        ALPHA = are all individual susceptibilities to corona the same? 1.0 defult.... The paper used 0.5 for childrenss i changed it.
        BETA_PAIRWISE_MODE= see for options, https://github.com/ryansmcgee/seirsplus/wiki/ExtSEIRSNetworkModel-Class#pairwise-transmissibility-values
        DELTA_PAIRWISE_MODE  = when two individuals whose average degree is an order of magnitude greater than the average degree of the population overall, then the propensity of exposure in their interaction is weighted to be twice that of two averagely-connected individuals. 
        BETA_Q = transmissibility of quarantined individuals
        """

        SIGMA = 1 / gamma_dist(latentPeriod_mean, latentPeriod_coeffvar, N)
        LAMDA = 1 / gamma_dist(presymptomaticPeriod_mean, presymptomaticPeriod_coeffvar, N)
        GAMMA = 1 / gamma_dist(symptomaticPeriod_mean, symptomaticPeriod_coeffvar, N)
        infectiousPeriod = 1 / LAMDA + 1 / GAMMA
        R0_mean = 2.5
        R0_coeffvar = 0.2
        R0 = gamma_dist(R0_mean, R0_coeffvar, N)
        factor_r_kids = np.random.random() + 1
        BETA = [(1 / infectiousPeriod[i] * (factor_r_kids * R0[i])) if age in ['10-19', '20-29']
                else (1 / infectiousPeriod[i] * R0[i]) for i, age in enumerate(individual_ageGroups)]
        ETA = 1 / gamma_dist(onsetToHospitalizationPeriod_mean, onsetToHospitalizationPeriod_coeffvar, N)
        GAMMA_H = 1 / gamma_dist(hospitalizationToDischargePeriod_mean, hospitalizationToDischargePeriod_coeffvar, N)
        MU_H = 1 / gamma_dist(hospitalizationToDeathPeriod_mean, hospitalizationToDeathPeriod_coeffvar, N)
        alpha_kids = np.random.randint(50, 100) / 100
        ALPHA = [alpha_kids if age in ['0-9', '10-19', '20-29'] else 1 for age in individual_ageGroups]
        BETA_PAIRWISE_MODE = 'infected'
        DELTA_PAIRWISE_MODE = 'mean'
        BETA_Q = [b * (0.3 / R0_mean) for b in BETA]

        # ------------------------- add checkpoints for interventions --------------------------------
        # ['Minimum active cases', '5-6 back to school', '9-12 back to school',
        #              '7-8 back to school', 'British variant', '3rd lockdown',
        #              'schools close', 'Most cases new mutations', 'Closed airport',
        #              'back:1-4,work', 'airports semi open', 'back:5-6,11-12',
        #              'purim', 'back:restaurants,7-10', 'airports fully opened', 'normal']

        BETA_1p1 = [b / 1.1 for b in BETA]
        BETA_1p15 = [b / 1.15 for b in BETA]
        BETA_1p2 = [b / 1.2 for b in BETA]
        BETA_1p25 = [b / 1.25 for b in BETA]
        BETA_1p3 = [b / 1.3 for b in BETA]
        BETA_1p4 = [b / 1.4 for b in BETA]

        BETA_Q_1p1 = [b / 1.1 for b in BETA]
        BETA_Q_1p15 = [b / 1.15 for b in BETA]
        BETA_Q_1p2 = [b / 1.2 for b in BETA]
        BETA_Q_1p25 = [b / 1.25 for b in BETA]
        BETA_Q_1p3 = [b / 1.3 for b in BETA]
        BETA_Q_1p4 = [b / 1.4 for b in BETA]

        checkpoints = {'t': times_for_sim,
                       'G': graphs_for_sim,
                       'beta': [BETA, BETA, BETA,
                                BETA_1p1, BETA_1p1, BETA_1p1,
                                BETA_1p15, BETA_1p15, BETA_1p2,
                                BETA_1p2, BETA_1p25, BETA_1p25,
                                BETA_1p3, BETA_1p4, BETA_1p4, BETA_1p4],
                       'beta_Q': [BETA_Q, BETA_Q, BETA_Q,
                                  BETA_Q_1p1, BETA_Q_1p1, BETA_Q_1p1,
                                  BETA_Q_1p15, BETA_Q_1p15, BETA_Q_1p2,
                                  BETA_Q_1p2, BETA_Q_1p25, BETA_Q_1p25,
                                  BETA_Q_1p3, BETA_Q_1p4, BETA_Q_1p4, BETA_Q_1p4]}

        node_groups = {k: np.where(np.array(individual_ageGroups) == k)[0] for k in np.unique(individual_ageGroups)}
        model = ExtSEIRSNetworkModel(G=G_baseline, p=P_GLOBALINTXN,
                                     beta=BETA, sigma=SIGMA, lamda=LAMDA, gamma=GAMMA,
                                     gamma_asym=GAMMA, eta=ETA, gamma_H=GAMMA_H, mu_H=MU_H,
                                     a=PCT_ASYMPTOMATIC, h=PCT_HOSPITALIZED, f=PCT_FATALITY,
                                     alpha=ALPHA, beta_pairwise_mode=BETA_PAIRWISE_MODE,
                                     delta_pairwise_mode=DELTA_PAIRWISE_MODE,
                                     G_Q=G_quarantine, q=0, beta_Q=BETA_Q, isolation_time=10, initE=int(0.8 * INIT_EXP),
                                     initQ_E=INIT_EXP - int(0.8 * INIT_EXP), initI_sym=int(0.8 * INIT_ISYM),
                                     initH=INIT_ISYM - int(0.8 * INIT_ISYM), seed=1, node_groups=node_groups)
        run_tti_sim(model, T,
                    intervention_start_pct_infected=INTERVENTION_START_PCT_INFECTED,
                    average_introductions_per_day=AVERAGE_INTRODUCTIONS_PER_DAY,
                    testing_cadence=TESTING_CADENCE, pct_tested_per_day=PCT_TESTED_PER_DAY,
                    test_falseneg_rate=TEST_FALSENEG_RATE,
                    testing_compliance_symptomatic=TESTING_COMPLIANCE_SYMPTOMATIC,
                    max_pct_tests_for_symptomatics=MAX_PCT_TESTS_FOR_SYMPTOMATICS,
                    testing_compliance_traced=TESTING_COMPLIANCE_TRACED,
                    max_pct_tests_for_traces=MAX_PCT_TESTS_FOR_TRACES,
                    testing_compliance_random=TESTING_COMPLIANCE_RANDOM,
                    random_testing_degree_bias=RANDOM_TESTING_DEGREE_BIAS,
                    tracing_compliance=TRACING_COMPLIANCE, pct_contacts_to_trace=PCT_CONTACTS_TO_TRACE,
                    tracing_lag=TRACING_LAG,
                    isolation_compliance_symptomatic_individual=ISOLATION_COMPLIANCE_SYMPTOMATIC_INDIVIDUAL,
                    isolation_compliance_symptomatic_groupmate=ISOLATION_COMPLIANCE_SYMPTOMATIC_GROUPMATE,
                    isolation_compliance_positive_individual=ISOLATION_COMPLIANCE_POSITIVE_INDIVIDUAL,
                    isolation_compliance_positive_groupmate=ISOLATION_COMPLIANCE_POSITIVE_GROUPMATE,
                    isolation_compliance_positive_contact=ISOLATION_COMPLIANCE_POSITIVE_CONTACT,
                    isolation_compliance_positive_contactgroupmate=ISOLATION_COMPLIANCE_POSITIVE_CONTACTGROUPMATE,
                    isolation_lag_symptomatic=ISOLATION_LAG_SYMPTOMATIC, isolation_lag_positive=ISOLATION_LAG_POSITIVE,
                    isolation_groups=households_indices, checkpoints=checkpoints,
                    vaccinations_df=vaccination_data)  # vaccination_data

        curr_date = time.strftime("%Y_%m_%d")
        dir_loc = f"g_third_lockdown/test_coeffs/{id_run}"
        pathlib.Path(dir_loc).mkdir(parents=True, exist_ok=True)
        save_model(model, f"{dir_loc}/model_out")
        # save input params
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
                      'percentage_out': percentage_out,
                      'beta_values': checkpoints['beta'],
                      'beta_q_values': checkpoints['beta_Q'],
                      'alpha_kids': alpha_kids,
                      'factor_r_kids': factor_r_kids,
                      'P_GLOBALINTXN': P_GLOBALINTXN,
                      'p_global_kids': p_glob_kids,
                      'p_global_adults': p_glob_adults,
                      'Q_GLOBALINTXN': Q_GLOBALINTXN,
                      'q_global_kids': q_glob_kids,
                      'q_global_adults': q_glob_adults,
                      'max_value': max_value,
                      'checkpoint_times': checkpoints['t']}
        with open(f"{dir_loc}/input_config.pickle", 'wb') as handle:
            pickle.dump(input_json, handle, protocol=pickle.HIGHEST_PROTOCOL)
        results_summary(model)
        id_run += 1
