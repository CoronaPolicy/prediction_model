from seirsplus.models import *
from seirsplus.networks import *
from seirsplus.sim_loops import *
from seirsplus.utilities import *
import networkx
import matplotlib.pyplot as pyplot
import matplotlib.pyplot as plt
import pickle

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
    10.11 minimum---------------8000  - 160
    
    time = [5,30,60,90,115,120,137,150,160]
    event= ['hofesh gadol', 'close <50,20', 'close work', 'schools back', 'full seger', 'maximum-72k','kindergarden back','grades 1-4 back','minimum-8000']
    G = [100,10,4,10,0.5,1,1.5]
    """
    N = 20000
    INIT_EXPOSED = int((2006/9e6)*N)
    Load_graph=True

    if not Load_graph:
        demographic_graphs, individual_ageGroups, households = generate_demographic_contact_network(
            N=N, demographic_data=household_country_data('ISRAEL'),
            distancing_scales=[300,9,3,0.5,5,10,11], isolation_groups=[],verbose=False)

        G_baseline = demographic_graphs['baseline']
        G_less_connections_1=demographic_graphs['distancingScale300']
        G_less_connections_2 = demographic_graphs['distancingScale9']
        G_less_connections_3 = demographic_graphs['distancingScale3']
        G_less_connections_4=demographic_graphs['distancingScale0.5']
        G_less_connections_5 = demographic_graphs['distancingScale5']
        G_less_connections_6 = demographic_graphs['distancingScale10']
        G_less_connections_7 = demographic_graphs['distancingScale11']

        G_quarantine = demographic_graphs['distancingScale0.5']

        networkx.write_gpickle(G_baseline, "G_baseline.gpickle")
        networkx.write_gpickle(G_less_connections_1, "G_less_connections_1.gpickle")
        networkx.write_gpickle(G_less_connections_2, "G_less_connections_2.gpickle")
        networkx.write_gpickle(G_less_connections_3, "G_less_connections_3.gpickle")
        networkx.write_gpickle(G_less_connections_4, "G_less_connections_4.gpickle")
        networkx.write_gpickle(G_less_connections_5, "G_less_connections_5.gpickle")
        networkx.write_gpickle(G_less_connections_6, "G_less_connections_6.gpickle")
        networkx.write_gpickle(G_less_connections_7, "G_less_connections_7.gpickle")

        networkx.write_gpickle(G_quarantine, "G_quarantine.gpickle")

        pickle.dump(households, open("households.p", "wb"))  # save it into a file named save.p
        pickle.dump(individual_ageGroups, open("individual_ageGroups.p", "wb"))
    else:
        G_baseline =networkx.read_gpickle("G_baseline.gpickle")
        G_less_connections_1 =networkx.read_gpickle("G_less_connections_1.gpickle")
        G_less_connections_2 = networkx.read_gpickle("G_less_connections_2.gpickle")
        G_less_connections_3 = networkx.read_gpickle("G_less_connections_3.gpickle")
        G_less_connections_4 =networkx.read_gpickle("G_less_connections_4.gpickle")
        G_less_connections_5 = networkx.read_gpickle("G_less_connections_5.gpickle")
        G_less_connections_6 = networkx.read_gpickle("G_less_connections_6.gpickle")

        G_quarantine =networkx.read_gpickle("G_quarantine.gpickle")
        households = pickle.load(open("households.p", "rb"))
        individual_ageGroups=pickle.load( open("individual_ageGroups.p", "rb"))
    households_indices = [household['indices'] for household in households]
    #plot_degree_distn(G_baseline, max_degree=100)
    #plot_degree_distn(G_quarantine, max_degree=100)
   #-----------------------pandemic parameters ----------------------------

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
    BETA = 1 / infectiousPeriod * R0
    ETA = 1 / gamma_dist(onsetToHospitalizationPeriod_mean, onsetToHospitalizationPeriod_coeffvar, N)
    GAMMA_H = 1 / gamma_dist(hospitalizationToDischargePeriod_mean, hospitalizationToDischargePeriod_coeffvar, N)
    MU_H = 1 / gamma_dist(hospitalizationToDeathPeriod_mean, hospitalizationToDeathPeriod_coeffvar, N)
    ALPHA = [1 if age in ['0-9', '10-19','20-29'] else 1 for age in individual_ageGroups]
    BETA_PAIRWISE_MODE = 'infected'
    DELTA_PAIRWISE_MODE = 'mean'
    BETA_Q = BETA * (0.3 / R0_mean)
    #-------------------------model over time --------------------------------


    P_GLOBALINTXN = 0.1  # interactions being with incidental or casual contacts outside their set of close contacts
    Q_GLOBALINTXN = 0.03 # interaction when a person is in quarntine
    PCT_ASYMPTOMATIC = 0.25 # precent of asymptomatic

    # ------------------------------------------------------------------------------#
    #-------------------Hospitalization data----------------------------------------#
    #-------------------------------------------------------------------------------#

    ageGroup_pctHospitalized = {'0-9': 0.0000,               # We need to use the minister of health data
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
    #-------------------Fatality Rate data------------------------------------------#
    #-------------------------------------------------------------------------------#

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




    model = ExtSEIRSNetworkModel(G=G_baseline, p=P_GLOBALINTXN,
                                 beta=BETA, sigma=SIGMA, lamda=LAMDA, gamma=GAMMA,
                                 gamma_asym=GAMMA, eta=ETA, gamma_H=GAMMA_H, mu_H=MU_H,
                                 a=PCT_ASYMPTOMATIC, h=PCT_HOSPITALIZED, f=PCT_FATALITY,
                                 alpha=ALPHA, beta_pairwise_mode=BETA_PAIRWISE_MODE,
                                 delta_pairwise_mode=DELTA_PAIRWISE_MODE,
                                 G_Q=G_quarantine, q=0, beta_Q=BETA_Q, isolation_time=10,
                                 initE=INIT_EXPOSED,seed=1)


    ISOLATION_LAG_SYMPTOMATIC = 1  # number of days between onset of symptoms and self-isolation of symptomatics
    ISOLATION_LAG_POSITIVE = 2  # test turn-around time (TAT): number of days between administration of test and isolation of positive cases
    ISOLATION_LAG_CONTACT = 0  # number of days between a contact being traced and that contact self-isolating
    INTERVENTION_START_PCT_INFECTED= 0.1/100
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
    ISOLATION_COMPLIANCE_SYMPTOMATIC_INDIVIDUAL = (numpy.random.rand(N) < ISOLATION_COMPLIANCE_RATE_SYMPTOMATIC_INDIVIDUAL)
    ISOLATION_COMPLIANCE_SYMPTOMATIC_GROUPMATE = (numpy.random.rand(N) < ISOLATION_COMPLIANCE_RATE_SYMPTOMATIC_GROUPMATE)
    ISOLATION_COMPLIANCE_POSITIVE_INDIVIDUAL = (numpy.random.rand(N) < ISOLATION_COMPLIANCE_RATE_POSITIVE_INDIVIDUAL)
    ISOLATION_COMPLIANCE_POSITIVE_GROUPMATE = (numpy.random.rand(N) < ISOLATION_COMPLIANCE_RATE_POSITIVE_GROUPMATE)
    ISOLATION_COMPLIANCE_POSITIVE_CONTACT = (numpy.random.rand(N) < ISOLATION_COMPLIANCE_RATE_POSITIVE_CONTACT)
    ISOLATION_COMPLIANCE_POSITIVE_CONTACTGROUPMATE = (numpy.random.rand(N) < ISOLATION_COMPLIANCE_RATE_POSITIVE_CONTACTGROUPMATE)

    #time = [5,30,60,90,115,137,150]
    #G = [100,10,4,10,0.5,1,1.5]

    checkpoints = {'t': [5,30,60,90,115,137,150],
                   'G': [G_less_connections_1,G_less_connections_2,
                         G_less_connections_3,G_less_connections_7,
                         G_less_connections_4,G_less_connections_5,G_less_connections_6]}

    #model.run(T=200, checkpoints=checkpoints)
    T = 200
    run_tti_sim(model, T,
                intervention_start_pct_infected=INTERVENTION_START_PCT_INFECTED,
                average_introductions_per_day=AVERAGE_INTRODUCTIONS_PER_DAY,
                testing_cadence=TESTING_CADENCE, pct_tested_per_day=PCT_TESTED_PER_DAY,
                test_falseneg_rate=TEST_FALSENEG_RATE,
                testing_compliance_symptomatic=TESTING_COMPLIANCE_SYMPTOMATIC,
                max_pct_tests_for_symptomatics=MAX_PCT_TESTS_FOR_SYMPTOMATICS,
                testing_compliance_traced=TESTING_COMPLIANCE_TRACED, max_pct_tests_for_traces=MAX_PCT_TESTS_FOR_TRACES,
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
                isolation_groups=households_indices,checkpoints=checkpoints)



    run_tti_sim(model, T,intervention_start_pct_infected=0.0,isolation_lag_symptomatic=ISOLATION_LAG_SYMPTOMATIC, isolation_lag_positive=ISOLATION_LAG_POSITIVE, isolation_groups=households_indices)


    results_summary(model)

    time_line = [5, 30, 60, 90, 115, 120, 137, 150, 160]
    events = ['hofesh gadol', 'close <50,20', 'close work', 'schools back', 'full seger', 'maximum-72k',
             'kindergarden back', 'grades 1-4 back', 'minimum-8000']

    plt.plot(model.tseries, ((model.numI_sym+model.numH)*9e6)/N)
    for time,event in zip(time_line,events):
        plt.axvline(int(time), 0, 1, c='k')
        plt.text(int(time)+0.1, 0, str(event), rotation=90)

    time = numpy.load('time.npy')
    active_casese = numpy.load('active_cases.npy')
    plt.plot(time-2,active_casese,'-r')

    plt.show()

    fig, ax = model.figure_infections(combine_Q_infected=False)
    plt.show()
    """
    intervention_start_pct_infected = 0, 
    average_introductions_per_day = 0,
    testing_cadence = 'everyday', 
    pct_tested_per_day = 1.0, 
    test_falseneg_rate = 'temporal',
    testing_compliance_symptomatic = [None],
     max_pct_tests_for_symptomatics = 1.0,
    testing_compliance_traced = [None],
     max_pct_tests_for_traces = 1.0,
    testing_compliance_random = [None], 
    random_testing_degree_bias = 0,
    tracing_compliance = [None], 
    num_contacts_to_trace = None, 
    pct_contacts_to_trace = 1.0,
     tracing_lag = 1,
    isolation_compliance_symptomatic_individual = [None],
     isolation_compliance_symptomatic_groupmate = [None],
    isolation_compliance_positive_individual = [None],
     isolation_compliance_positive_groupmate = [None],
    isolation_compliance_positive_contact = [None], 
    isolation_compliance_positive_contactgroupmate = [None],
    isolation_lag_symptomatic = 1, 
    isolation_lag_positive = 1, 
    isolation_lag_contact = 0, 
    isolation_groups = None,
    cadence_testing_days = None, 
    cadence_cycle_length = 28, 
    temporal_falseneg_rates = None
    """