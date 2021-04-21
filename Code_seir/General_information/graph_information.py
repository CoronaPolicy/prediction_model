import numpy as np


class graph_info(object):
    def __init__(self, N=10000, alone_parma=0.2, couples_without_kids_param=0.5,
                 kids_left_house_p=0.34, old_kids=0.6, two_kids_young=0.28,
                 two_kids_old=0.032, three_kids_young=0.35, three_kids_old=0.15,
                 one_kid_young_per=0.7, four_kids_young=0.1, four_kids_old=0.2, dict_ages=None):

        self.housholds = self.houshold(alone_parma, couples_without_kids_param, kids_left_house_p
                                       , old_kids, two_kids_young, two_kids_old, three_kids_young,
                                       three_kids_old, one_kid_young_per, four_kids_young, four_kids_old)

        self.house_size = {'alone': 1,
                           'students_app': 5,
                           'soldier': 4,
                           'couples_without_kids': 2,
                           'couples_kids_left_the_house': 2,
                           'couples_with_one_young_kid': 3,
                           'couples_with_one_old_kid': 3,
                           'couples_with_two_kid': 4,
                           'couples_with_three_kid': 5,
                           'couples_with_four_kid_pluse': 7,
                           }

        if dict_ages is None:
            dict_ages = {
                '0-9': 1,
                '10-19': 1.2,
                '20-49': 1.2,
                '50-69': 1,
                '70+': 1,
            }
        self.demographic_data = self.demographic_data_israel()
        self.layers_info = self.layers_info(dict_ages)

        self.individual_ageGroups = self.demographic_data['age_distn'].copy()
        for key in self.individual_ageGroups.keys():
            self.individual_ageGroups[key] = self.individual_ageGroups[key] * N

    def houshold(self, alone_parma, couples_without_kids_param, kids_left_house_p
                 , old_kids, two_kids_young, two_kids_old, three_kids_young,
                 three_kids_old, one_kid_young_per, four_kids_young, four_kids_old):

        household_data = {
            'alone': (0.05 * 0.9, [],
                      [0, 0, 1 * alone_parma / 4, 2 * alone_parma / 4, 1 * alone_parma / 4, 0 * (1 - alone_parma) / 8,
                       4 * (1 - alone_parma) / 8, 2 * (1 - alone_parma) / 8, 2 * (1 - alone_parma) / 8]),
            'students_app': (0.05 * 0.1, [], [0, 0.2, 0.8, 0, 0, 0, 0, 0, 0]),
            'soldier': (0.015, [], [0.0, 0.7, 0.3, 0, 0, 0, 0, 0, 0]),
            'couples_without_kids': (
                0.935 * 0.28 * 0.25, [],
                [0, 0, couples_without_kids_param, 1 - couples_without_kids_param, 0, 0, 0, 0, 0]),
            'couples_kids_left_the_house': (0.935 * 0.28 * 0.75, [],
                                            [0, 0, 0, 0, 0, kids_left_house_p, 16 * (1 - kids_left_house_p) / 30,
                                             10 * (1 - kids_left_house_p) / 30, 4 * (1 - kids_left_house_p) / 30]),
            'couples_with_one_young_kid': (
                0.935 * 0.18 * 0.75, [0.7, 0.3], [0, 0, one_kid_young_per, 1 - one_kid_young_per, 0, 0, 0, 0, 0]),
            'couples_with_one_old_kid': (
                0.935 * 0.18 * 0.25, [0, 0.8, 0.2], [0, 0, 0, old_kids, 1 - old_kids, 0, 0, 0, 0]),
            'couples_with_two_kid': (0.935 * 0.16, [0.51, 0.46, 0.03],
                                     [0, 0, 0.05, two_kids_young, 1 - two_kids_young - two_kids_old - 0.05,
                                      two_kids_old, 0, 0, 0]),
            'couples_with_three_kid': (0.935 * 0.18, [0.51, 0.44, 0.05],
                                       [0, 0, 0.05, three_kids_young, 1 - three_kids_old - three_kids_young - 0.05,
                                        three_kids_old, 0, 0, 0]),
            'couples_with_four_kid_pluse': (0.935 * 0.2, [0.5, 0.45, 0.05],
                                            [0, 0, 0.07, four_kids_young, 1 - four_kids_old - four_kids_young - 0.07,
                                             four_kids_old, 0, 0, 0]),
        }

        return household_data

    def layers_info(self, dict_ages):
        age_distn = self.demographic_data['age_distn']
        # ageBrackets_20to40 = ['20-29', '30-39']
        # ageBrackets_50to80 = ['50-59', '60-69', '70-79', '80+']
        # totalPct20to60 = np.sum([age_distn[bracket] for bracket in ageBrackets_20to40])
        # totalPct50to80 = np.sum([age_distn[bracket] for bracket in ageBrackets_50to80])
        # age_distn_given20to60 = {bracket: pct / totalPct20to60 for bracket, pct in age_distn.items() if
        #                          bracket in ageBrackets_20to40}
        # age_distn_given50to80 = {bracket: pct / totalPct50to80 for bracket, pct in age_distn.items() if
        #                          bracket in ageBrackets_50to80}

        layer_info = {'0-9': {'ageBrackets': ['0-9'],
                              'meanDegree': dict_ages['0-9'] * 8,
                              'meanDegree_CI': (dict_ages['0-9'] * 0,
                                                dict_ages['0-9'] * 16)},
                      '10-19': {'ageBrackets': ['10-19'],
                                'meanDegree': dict_ages['10-19'] * 15,
                                'meanDegree_CI': (dict_ages['10-19'] * 10,
                                                  dict_ages['10-19'] * 20)},
                      '20-49': {'ageBrackets': ['20-29', '30-39', '40-49'],
                                'meanDegree': dict_ages['20-49'] * 20,
                                'meanDegree_CI': (dict_ages['20-49'] * 15,
                                                  dict_ages['20-49'] * 25)},
                      '50-69': {'ageBrackets': ['50-59', '60-69'],
                                'meanDegree': dict_ages['50-69'] * 10,
                                'meanDegree_CI': (dict_ages['50-69'] * 5,
                                                  dict_ages['50-69'] * 15)},
                      '70+': {'ageBrackets': ['70-79', '80+'],
                              'meanDegree': dict_ages['70+'] * 10,
                              'meanDegree_CI': (dict_ages['70+'] * 5,
                                                dict_ages['70+'] * 15)},
                      }
        return layer_info

    def demographic_data_israel(self):
        household_data = {
            'household_size_distn': {1: 0.1,
                                     2: 0.15,
                                     3: 0.3,
                                     4: 0.3,
                                     5: 0.1,
                                     6: 0.025,
                                     7: 0.025},

            'age_distn': {'0-9': 0.197,
                          '10-19': 0.181,
                          '20-29': 0.135,
                          '30-39': 0.123,
                          '40-49': 0.121,
                          '50-59': 0.084,
                          '60-69': 0.081,
                          '70-79': 0.050,
                          '80+': 0.028},
        }
        return household_data


if __name__ == "__main__":
    Count = {
        'alone': [],
        'students_app': [],
        'soldier': [],
        'couples_without_kids': [],
        'couples_kids_left_the_house': [],
        'couples_with_one_young_kid': [],
        'couples_with_one_old_kid': [],
        'couples_with_two_kid': [],
        'couples_with_three_kid': [],
        'couples_with_four_kid_pluse': [],
    }
    N = 1000000
    number_of_people_array = [1, 4, 4, 2, 2, 3, 3, 4, 5, 8]
    housholds_object = graph_info()
    for num, (key, values) in enumerate(housholds_object.housholds.items()):

        precent_group = values[0]
        p_ages_kids = values[1]
        p_ages_parents = values[2]

        keys_ages = housholds_object.demographic_data['age_distn'].keys()

        number_of_people = N * precent_group
        num_of_people_in_house = number_of_people_array[num]
        num_parents = 2
        num_kids = num_of_people_in_house - num_parents

        if num < 5:
            choise = np.random.choice(list(keys_ages), p=p_ages_parents, size=int(number_of_people))
            Count[key].append(choise)

        elif (num >= 5 and num < 7):
            kids_amount = int((num_kids / num_of_people_in_house) * number_of_people)
            kids_choise = np.random.choice(list(keys_ages)[:len(p_ages_kids)], p=p_ages_kids, size=kids_amount)
            adults_choise = np.random.choice(list(keys_ages), p=p_ages_parents,
                                             size=int(number_of_people - kids_amount))
            Count[key].append(np.array(list(kids_choise) + list(adults_choise)))
        else:
            kids_amount = int((num_kids / num_of_people_in_house) * number_of_people)
            kids_choise = np.random.choice(list(keys_ages)[:len(p_ages_kids)], p=p_ages_kids, size=kids_amount)
            adults_choise = np.random.choice(list(keys_ages), p=p_ages_parents,
                                             size=int(number_of_people - kids_amount))
            Count[key].append(np.array(list(kids_choise) + list(adults_choise)))

    Count_2 = {'0-9': 0,
               '10-19': 0,
               '20-29': 0,
               '30-39': 0,
               '40-49': 0,
               '50-59': 0,
               '60-69': 0,
               '70-79': 0,
               '80+': 0
               }

    for num, (keys, values) in enumerate(Count.items()):
        for i in np.squeeze(values):
            Count_2[i] += 1

    values_all = np.array(list(Count_2.items()))[:, 1]
    ages_all = np.array(list(housholds_object.demographic_data['age_distn'].items()))
    number_all = ages_all[:, 1].astype(float) * 1000000
    print((((values_all.astype(int) - number_all) / number_all) * 100).astype(int))
