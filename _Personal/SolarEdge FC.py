import os
import math
import random
import statistics
from itertools import combinations
if os.getlogin() == "eddy.a":
    from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
    import Plot_Graphs_with_Sliders as _G
    import my_tools


# list_of_players = {"Eddy A.": 9, "Yan O.": 7, "Omer Sh.": 7, "Shlomi A.": 6, "Max K.": 6, "Elroee Sh.": 5, "Elad T.": 4, "Alon M.": 4,
#                    "Noam M.": 4, "Tal O.": 4, "Kobi S.": 4, "Amit N.": 4, "Ido G.": 4, "Sagie Q.": 4, "Tomer H.": 3, "Yogev F.": 3,
#                    "Alon G.": 3, "Tomer Y.": 2, "Yaniv A.": 2, "Bar M.": 2, "Elad M.": 2, "Doron A.": 2, "Oshri A.": 1, "Tomer B.": 1,
#                    "Ido K.": 1, "Itai Y.": 1, "Daniel Z.": 1, "Lior S.": 1, "Ori A.": 1, "Dudu O.": 1, "Doron E.": 1, "Gabriel G.": 1, "Roy Sh.": 1}
list_of_players = {"Eddy A.": 9, "Yan O.": 7, "Omer Sh.": 7, "Shlomi A.": 6, "Max K.": 6, "Elroee Sh.": 5, "Elad T.": 4, "Alon M.": 4, "Noam M.": 4, "Tal O.": 4, "Kobi S.": 4, "Amit N.": 4, "Ido G.": 4, "Sagie Q.": 4, "Tomer H.": 3}
print_for_debug = True

class Player:
    min_rating = 0
    max_rating = 10

    def __init__(self, name, rating):
        self.name = name
        self.rating = rating
        self.played_with = {}
        self.played_against = {}
        self.performance = 0

    def __get__(self, instance, owner):
        return self.name

    def new_game(self, played_with, played_against, performance):
        self.played_with.update(played_with)
        self.played_against.update(played_against)
        self.performance += performance

    def calculate_new_rating(self, performance_weight):
        temp_rating = self.rating + (self.performance * performance_weight)
        if temp_rating > self.max_rating:
            self.rating = self.max_rating
        elif temp_rating < self.min_rating:
            self.rating = self.min_rating
        else:
            self.rating = temp_rating

    def get_player(self):
        # return self.name, self.rating
        return {"Name": self.name, "Rating": self.rating}


def group_distribution(number_of_groups, total_players):
    if total_players % number_of_groups == 0:
        return [total_players // number_of_groups] * number_of_groups
    elif total_players % number_of_groups == 1:
        return [total_players // number_of_groups + 1] + [total_players // 3] * 2
    else:
        return [total_players // number_of_groups + 1] * 2 + [total_players // number_of_groups]


def find_possible_teams(players, group_sizes):
    sum_of_possible_combinations = int(math.comb(sum(group_sizes), group_sizes[0]) * math.comb(sum(group_sizes) - group_sizes[0], group_sizes[1]) / 6)
    print(f'\nfind_possible_teams({len(players) = }, {group_sizes = }, possible combinations = {sum_of_possible_combinations}')
    for group_1 in combinations(players, group_sizes[0]):
        remaining_players_1 = [player for player in players if player not in group_1]
        for group_2 in combinations(remaining_players_1, group_sizes[1]):
            remaining_players_2 = [player for player in remaining_players_1 if player not in group_2]
            for group_3 in combinations(remaining_players_2, group_sizes[2]):
                group_1 = tuple(sorted(group_1))
                group_2 = tuple(sorted(group_2))
                group_3 = tuple(sorted(group_3))
                yield tuple(sorted((group_1, group_2, group_3)))


def find_close_sum_groups(teams, sum_pass, sd_pass):
    min_sum_ratio = (2**63-1, 0)
    min_sd_ratio = (0, 2**63-1)
    min_ratio = (2**63-1, 2**63-1)
    for group_1, group_2, group_3 in teams:
        group_1 = dict(group_1)
        group_2 = dict(group_2)
        group_3 = dict(group_3)
        if len(group_3) < len(group_2):
            group_3.update({"MANA": max(*group_1.values(), *group_2.values())})
        elif len(group_1) > len(group_2):
            group_2.update({"MANA": max(*group_1.values(), *group_3.values())})
            group_3.update({"MANA": max(*group_1.values(), *group_2.values())})

        sum_1 = sum(group_1.values())
        sum_2 = sum(group_2.values())
        sum_3 = sum(group_3.values())
        avg_1 = statistics.mean(group_1.values())
        avg_2 = statistics.mean(group_2.values())
        avg_3 = statistics.mean(group_3.values())
        sd_1 = statistics.pstdev(group_1.values())
        sd_2 = statistics.pstdev(group_2.values())
        sd_3 = statistics.pstdev(group_3.values())

        sum_ratio = statistics.pstdev([sum_1, sum_2, sum_3])
        sd_ratio = statistics.pstdev([sd_1, sd_2, sd_3])
        ratio = (sum_ratio + sd_ratio) / 2
        if sum_ratio < min_sum_ratio[0]:
            min_sum_ratio = (sum_ratio, sd_ratio)
        if sd_ratio < min_sd_ratio[1]:
            min_sd_ratio = (sum_ratio, sd_ratio)
        if ratio < statistics.mean(min_ratio):
            min_ratio = (sum_ratio, sd_ratio)

        if print_for_debug:
            print(f'\n{sum_ratio = }')
            print(f'{sd_ratio = }')
            print(f'{ratio = }')

        if sum_ratio <= sum_pass and sd_ratio <= sd_pass:
            group_1 = dict(sorted(group_1.items()))
            group_2 = dict(sorted(group_2.items()))
            group_3 = dict(sorted(group_3.items()))
            yield (group_1, sum_1, avg_1, sd_1), (group_2, sum_2, avg_2, sd_2), (group_3, sum_3, avg_3, sd_3)
    return None     # If no close sum groups are found, return None


number_of_groups = 3    # this script works only for 3 groups (because of the loops)
# sum_pass = 0.2
sum_pass = 0
# sd_pass = 0.5
sd_pass = 0

available_players = list(list_of_players.items())
group_sizes = group_distribution(number_of_groups, len(available_players))
possible_teams = list(set(find_possible_teams(available_players, group_sizes)))
random.shuffle(possible_teams)

for g_index, groups in enumerate(find_close_sum_groups(possible_teams, sum_pass, sd_pass)):
    if groups:
        for index, sub_group in enumerate(groups):
            print(f'Match {g_index + 1:06} Group {index + 1:02}: Players = {", ".join(sub_group[0].keys())} (Sum = {sub_group[1]}, Avg = {sub_group[2]:.1f}, SD = {sub_group[3]:.1f})')
        print()
    else:
        print("Close sum groups not found.")
