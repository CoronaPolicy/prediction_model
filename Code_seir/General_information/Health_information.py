from seirsplus.utilities import *
import numpy as np


class health_information(object):
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

    def __init__(self, N=10000, individual_ageGroups=None, latentPeriod_mean=3.0, latentPeriod_coeffvar=0.6,
                 presymptomaticPeriod_mean=2.2, presymptomaticPeriod_coeffvar=0.5,
                 symptomaticPeriod_mean=4.0, symptomaticPeriod_coeffvar=0.4,
                 onsetToHospitalizationPeriod_mean=11.0, onsetToHospitalizationPeriod_coeffvar=0.45,
                 hospitalizationToDischargePeriod_mean=11.0, hospitalizationToDischargePeriod_coeffvar=0.45,
                 hospitalizationToDeathPeriod_mean=7.0, hospitalizationToDeathPeriod_coeffvar=0.45,
                 R0_mean=2.5, R0_coeffvar=0.2, alpha=None, optimze=False):

        if alpha is None:
            alpha = {'0-9': 0.5,
                     '10-19': 1.5,
                     '20-29': 1.5,
                     '30-39': 1.2,
                     '40-49': 1.0,
                     '50-59': 1.0,
                     '60-69': 0.8,
                     '70-79': 0.8,
                     '80+': 0.8}
        print(f"alpha is:{alpha}")

        self.ALPHA = [alpha[age] for age in individual_ageGroups]
        self.SIGMA = 1 / gamma_dist(latentPeriod_mean, latentPeriod_coeffvar, N)
        self.LAMDA = 1 / gamma_dist(presymptomaticPeriod_mean, presymptomaticPeriod_coeffvar, N)
        self.GAMMA = 1 / gamma_dist(symptomaticPeriod_mean, symptomaticPeriod_coeffvar, N)
        self.infectiousPeriod = 1 / self.LAMDA + 1 / self.GAMMA

        self.R0 = gamma_dist(R0_mean, R0_coeffvar, N)
        # BETA = 1 / infectiousPeriod * R0
        # self.BETA = [(1 / self.infectiousPeriod[i] * (1.0 * self.R0[i])) if age in ['10-19', '20-29']
        #         else (1 / self.infectiousPeriod[i] * self.R0[i]) for i, age in enumerate(self.individual_ageGroups)]
        self.factors = {}
        if optimze:
            self.factors['R0'] = numpy.random.random() + 1
            self.BETA = [(1 / self.infectiousPeriod[i] * (self.factors['R0'] * self.R0[i])) if age in ['10-19', '20-29']
                         else (1 / self.infectiousPeriod[i] * self.R0[i]) for i, age in enumerate(individual_ageGroups)]
        else:
            self.factors['R0'] = 1.6
        self.BETA = [(1 / self.infectiousPeriod[i] * (self.factors['R0'] * self.R0[i])) if age in ['10-19', '20-29']
                     else (1 / self.infectiousPeriod[i] * self.R0[i]) for i, age in enumerate(individual_ageGroups)]
        self.BETA_Q = [b * (0.3 / R0_mean) for b in self.BETA]
        self.ETA = 1 / gamma_dist(onsetToHospitalizationPeriod_mean, onsetToHospitalizationPeriod_coeffvar, N)
        self.GAMMA_H = 1 / gamma_dist(hospitalizationToDischargePeriod_mean, hospitalizationToDischargePeriod_coeffvar,
                                      N)
        self.MU_H = 1 / gamma_dist(hospitalizationToDeathPeriod_mean, hospitalizationToDeathPeriod_coeffvar, N)

        self.BETA_PAIRWISE_MODE = 'infected'
        self.DELTA_PAIRWISE_MODE = 'mean'
        hospitalization = self.infected_to_hospitalization()
        fatality = self.fatality()
        self.PCT_HOSPITALIZED = [hospitalization[ageGroup] for ageGroup in individual_ageGroups]
        self.PCT_FATALITY = [fatality[ageGroup] for ageGroup in individual_ageGroups]

    def infected_to_hospitalization(self):
        H = {'0-9': 0.0000,  # We need to use the minister of health data
             '10-19': 0.0004,
             '20-29': 0.0104,
             '30-39': 0.0343,
             '40-49': 0.0425,
             '50-59': 0.0816,
             '60-69': 0.118,
             '70-79': 0.166,
             '80+': 0.184}
        return H

    def fatality(self):
        f = {'0-9': 0.0000,
             '10-19': 0.3627,
             '20-29': 0.0577,
             '30-39': 0.0426,
             '40-49': 0.0694,
             '50-59': 0.1532,
             '60-69': 0.3381,
             '70-79': 0.5187,
             '80+': 0.7283}

        return f

    def create_vaccination_df(self, N, total_vacc_per_day_df, individual_ages_list, vacc_policy=None):
        total_vacc_per_day = total_vacc_per_day_df.num_tot * N / (9 * 10 ** 6)
        unique, counts = np.unique(individual_ages_list, return_counts=True)
        num_per_age = dict(zip(unique, counts))
        age_index = {k: i for i, k in enumerate(np.unique(individual_ages_list))}
        age_order = []
        if vacc_policy == "old_to_young":
            # start with oldest people and go down
            age_order = ['80+', '70-79', '60-69', '50-59', '40-49', '30-39', '20-29', '10-19'][::-1]
        elif vacc_policy == "young_to_old":
            # start with youngest people and go up
            age_order = ['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+'][::-1]
        elif vacc_policy == "triangle":
            age_order = ['80+', '70-79', '60-69', '10-19', '20-29', '30-39', '40-49'][::-1]
        num_vacc_per_day = np.zeros(shape=(total_vacc_per_day.size, len(unique)))
        current_age = age_order.pop()
        for i_d, t in enumerate(total_vacc_per_day_df.num_tot):
            while (t > 0) and len(age_order) > 0:
                if num_per_age[current_age] > 0:
                    num_vacc = np.min([t, num_per_age[current_age]])
                    #                     print(f"t:{t}, num_vacc:{num_vacc}, len age order:{len(age_order)}")
                    num_per_age[current_age] = num_per_age[current_age] - num_vacc
                    num_vacc_per_day[i_d, age_index[current_age]] = int(num_vacc)
                    t = t - num_vacc
                    if t > 0 and len(age_order) > 0:
                        print(f"finished age:{current_age}")
                        current_age = age_order.pop()
        total_vacc_per_day_df.iloc()[:, 2:2 + 8] = num_vacc_per_day[:, :-1].astype(np.int64)
        return total_vacc_per_day_df
