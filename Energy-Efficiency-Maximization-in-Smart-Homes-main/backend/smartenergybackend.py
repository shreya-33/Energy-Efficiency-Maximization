import numpy as np
import pandas as pd
import random
import math
import time
import matplotlib.pyplot as plt
from copy import deepcopy
from IPython.display import display

def generate_energy_recommendations(input_data):

    num_slots = 8

    appliance_data = [['Ref', 'NS', 100, 0.5, [1, num_slots]], ['AS', 'NS', 100, 0.1, [1, num_slots]], 
                      ['AC1', 'PS', 50, [0.7, 1.4, 0.1], [1, num_slots]], 
                      ['AC2', 'PS', 50, [0.7, 1.4, 0.1], [1, num_slots]], 
                      ['H', 'PS', 50, [0.5, 1.5, 0.1], [1, num_slots]], 
                      ['L1', 'TS', 0.02, 0.7, [5, 7]],
                      ['L2', 'TS', 0.02, 0.7, [4, 7]],
                      ['WM', 'TS', 0.02, 0.7, [6, 8]],
                      ['DW', 'TS', 0.06, 0.3, [6, 8]]]

    appliance_df = pd.DataFrame(appliance_data, columns=['Name', 'Type', 'Diss. Coeff.', 'Power Rating (kWh)', 'Time Slot'])

    appliance_df = appliance_df.sort_values('Diss. Coeff.', ascending=False)
    
    num_points = 1000

    def generate_electricity_prices(num_slots, ep_mx_q1, ep_mx_q2, ep_mx_q3, ep_mx_q4, ep_k, min_price):
        electricity_prices = []
        distribution = []

        for ep_mx, num_bins in zip([ep_mx_q1, ep_mx_q2, ep_mx_q3, ep_mx_q4], [num_slots, num_slots, num_slots, num_slots]):
            mu, sigma = 0, 1.0 / (np.sqrt(2 * np.pi) * ep_mx)
            s = np.random.normal(mu, sigma, num_points)
            count, bins, ignored = plt.hist(s, num_bins, density=True)
            plt.clf()
            electricity_prices += [max(0, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (i - mu) ** 2 / (2 * sigma ** 2)) + random.uniform(-ep_k, ep_k)) for i in bins[:-1]]
            distribution += [1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (i - mu) ** 2 / (2 * sigma ** 2)) for i in bins[:-1]]

        electricity_prices = [i + min_price for i in electricity_prices]
        distribution = [i + min_price for i in distribution]

        return electricity_prices, distribution

    def generate_solar_power(num_slots, sp_mx_q1, sp_mx_q2, sp_mx_q3, sp_mx_q4, sp_k):
        solar_power = []
        distribution = []

        for sp_mx, num_bins in zip([sp_mx_q1, sp_mx_q2, sp_mx_q3, sp_mx_q4], [num_slots // 2, num_slots // 2, num_slots // 2, num_slots // 2]):
            mu, sigma = 0, 1.0 / (np.sqrt(2 * np.pi) * sp_mx)
            s = np.random.normal(mu, sigma, num_points)
            count, bins, ignored = plt.hist(s, num_bins, density=True)
            plt.clf()
            solar_power += [max(0, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (i - mu) ** 2 / (2 * sigma ** 2)) + random.uniform(-sp_k, sp_k)) for i in bins[:-1]]
            distribution += [1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (i - mu) ** 2 / (2 * sigma ** 2)) for i in bins[:-1]]

        return solar_power, distribution

    electricity_prices, _ = generate_electricity_prices(num_slots, 35, 35, 30, 20, 5, 7)
    solar_power, _ = generate_solar_power(num_slots, 200, 200, 250, 100, 20)

    states = [(electricity_prices[i], solar_power[i]) for i in range(num_slots)]

    agents = appliance_df['Name'].tolist()
    agent_actions = {}
    for agent in agents:
        agent_data = appliance_df.loc[appliance_df['Name'] == agent]
        if (agent_data['Type'] == 'NS').bool():
            agent_actions[agent] = ['on']
        elif (agent_data['Type'] == 'PS').bool():
            pr = agent_data['Power Rating (kWh)'].tolist()[0]
            agent_actions[agent] = [round(i, 1) for i in np.arange(pr[0], pr[1] + pr[2], pr[2])]
        elif (agent_data['Type'] == 'TS').bool():
            agent_actions[agent] = ['on', 'off']

    def in_slot(t, ts):
        return t + 1 >= ts[0] and t + 1 <= ts[1]

    without_DR_actions = {}
    for t in range(num_slots):
        without_DR_actions[t] = {}
        for agent in agents:
            without_DR_actions[t][agent] = None
            act = None
            agent_data = appliance_df.loc[appliance_df['Name'] == agent]
            if (agent_data['Type'] == 'NS').bool():
                ts = agent_data['Time Slot'].tolist()[0]
                if in_slot(t, ts):
                    pr = agent_data['Power Rating (kWh)'].tolist()[0]
                    act = 'on'
            elif (agent_data['Type'] == 'PS').bool():
                ts = agent_data['Time Slot'].tolist()[0]
                if in_slot(t, ts):
                    pr = agent_data['Power Rating (kWh)'].tolist()[0]
                    act = pr[1]
            elif (agent_data['Type'] == 'TS').bool():
                ts = agent_data['Time Slot'].tolist()[0]
                if in_slot(t, ts):
                    pr = agent_data['Power Rating (kWh)'].tolist()[0]
                    if t + 1 == ts[0]:
                        act = 'on'
                    else:
                        act = 'off'
            without_DR_actions[t][agent] = act

    def get_reward_DR(st, t, agent, act, started):
        reward = 0.0
        cost = 0.0
        agent_data = appliance_df.loc[appliance_df['Name'] == agent]
        if (agent_data['Type'] == 'NS').bool():
            ts = agent_data['Time Slot'].tolist()[0]
            if in_slot(t, ts):
                pr = agent_data['Power Rating (kWh)'].tolist()[0]
                reward = (-st[0] * max(0, (pr)))
                cost = (-st[0] * max(0, (pr)))
        elif (agent_data['Type'] == 'PS').bool():
            ts = agent_data['Time Slot'].tolist()[0]
            if in_slot(t, ts):
                pr = agent_data['Power Rating (kWh)'].tolist()[0]
                dc = agent_data['Diss. Coeff.'].tolist()[0]
                reward = ((-st[0] * max(0, (act))) - (dc * math.pow(pr[1] - act, 2)))
                cost = (-st[0] * max(0, (act)))
        elif (agent_data['Type'] == 'TS').bool():
            ts = agent_data['Time Slot'].tolist()[0]
            if in_slot(t, ts):
                pr = agent_data['Power Rating (kWh)'].tolist()[0]
                dc = agent_data['Diss. Coeff.'].tolist()[0]
                if (act == 'on' or (t + 1 == ts[1])) and started[agent] != 1:
                    reward = - (st[0] * max(0, pr) - (dc * math.pow(ts[0] - t - 1, 2)))
                    cost = (-st[0] * max(0, (pr)))
                    started[agent] = 1
        return round(reward, 6), cost, started

    def get_reward_without_DR(st, t, agent):
        reward = 0.0
        cost = 0.0
        agent_data = appliance_df.loc[appliance_df['Name'] == agent]
        if (agent_data['Type'] == 'NS').bool():
            ts = agent_data['Time Slot'].tolist()[0]
            if in_slot(t, ts):
                pr = agent_data['Power Rating (kWh)'].tolist()[0]
                reward = (-st[0] * max(0, (pr)))
                cost = (-st[0] * max(0, (pr)))
        elif (agent_data['Type'] == 'PS').bool():
            ts = agent_data['Time Slot'].tolist()[0]
            if in_slot(t, ts):
                pr = agent_data['Power Rating (kWh)'].tolist()[0]
                reward = (-st[0] * max(0, (pr[1])))
                cost = (-st[0] * max(0, (pr[1])))

        elif (agent_data['Type'] == 'TS').bool():
            ts = agent_data['Time Slot'].tolist()[0]
            if in_slot(t, ts):
                pr = agent_data['Power Rating (kWh)'].tolist()[0]
                if t + 1 == ts[0]:
                    reward = - (st[0] * max(0, pr))
                    cost = - (st[0] * max(0, pr))
        return round(reward, 6), cost

    num_timeslots = num_slots
    num_episodes = 1000
    epsilon = 0.3
    gamma = 0.9
    theta = 0.3
    converged = 1
    threshold = 0.001

    recommendations = {}
    actually_started = {}

    for t in range(num_timeslots):
        actually_started[t] = {}
        for agent in agents:
            actually_started[t][agent] = 0

    started = {}
    for agent in agents:
        started[agent] = 0

    Without_DR_total_reward = 0.0
    With_DR_total_reward = 0.0
    Without_DR_total_electricity_cost = 0.0
    With_DR_total_electricity_cost = 0.0

    for t in range(num_timeslots):
        recommendations[t] = {}
        Q = {}
        Q_prev = {}
        R = {}
        C = {}
        for st in states:
            Q[st] = {}
            Q_prev[st] = {}
            R[st] = {}
            C[st] = {}
            for agent in agents:
                Q[st][agent] = {}
                Q_prev[st][agent] = {}
                R[st][agent] = {}
                C[st][agent] = {}
                recommendations[t][agent] = None
        converged = 1
        for agent in agents:
            for st in states:
                for act in agent_actions[agent]:
                    Q[st][agent][act] = 0.0
                    Q_prev[st][agent][act] = 0.0
            if t > 0:
                started[agent] = actually_started[t - 1][agent]
            else:
                started[agent] = 0
            for eps in range(num_episodes):
                if t > 0:
                    if actually_started[t - 1][agent] != 1:
                        started[agent] = 0
                else:
                    started[agent] = 0
                Q_prev = deepcopy(Q)
                st = states[t]
                itr = 0
                curr_t = t
                while curr_t < num_slots - 1:
                    p = np.random.random()
                    if p < epsilon:
                        j = np.random.choice(len(agent_actions[agent]))
                    else:
                        j = np.argmax([Q[st][agent][a] for a in agent_actions[agent]])
                    at = agent_actions[agent][j]
                    sdash = states[curr_t + 1]
                    R[st][agent][at], C[st][agent][at], started = get_reward_DR(st, curr_t, agent, at, started)
                    Q[st][agent][at] = Q[st][agent][at] + theta * (R[st][agent][at] + gamma * max([Q[sdash][agent][act] for act in agent_actions[agent]]) - Q[st][agent][at])
                    st = sdash
                    curr_t += 1
                converged = 1
                for s in states:
                    for a in agent_actions[agent]:
                        if abs(Q[s][agent][a] - Q_prev[s][agent][a]) > threshold:
                            converged = 0
                if converged == 1:
                    break
            recommendations[t][agent] = agent_actions[agent][np.argmax([Q[states[t]][agent][a] for a in agent_actions[agent]])]
            agent_data = appliance_df.loc[appliance_df['Name'] == agent]
            slot = agent_data['Time Slot'].tolist()[0]
            if not in_slot(t, slot):
                recommendations[t][agent] = None
            if (appliance_df.loc[appliance_df['Name'] == agent]['Type'] == 'TS').bool() and actually_started[t][agent] == 1:
                recommendations[t][agent] = 'off'
            if recommendations[t][agent] == 'on' and (appliance_df.loc[appliance_df['Name'] == agent]['Type'] == 'TS').bool():
                if t + 1 < num_timeslots:
                    actually_started[t + 1][agent] = 1
            if recommendations[t][agent] == 'off' and (appliance_df.loc[appliance_df['Name'] == agent]['Type'] == 'TS').bool() and actually_started[t][agent] == 0 and t + 1 == slot[1]:
                if t + 1 < num_timeslots:
                    actually_started[t + 1][agent] = 1
                recommendations[t][agent] = 'on'

    Appliance_Without_DR_Cost = {}
    Appliance_With_DR_Cost = {}

    Timeslot_Without_DR_Cost = {}
    Timeslot_With_DR_Cost = {}

    Appliance_Without_DR_Reward = {}
    Appliance_With_DR_Reward = {}

    Without_DR_total_reward = 0
    With_DR_total_reward = 0
    Without_DR_total_electricity_cost = 0
    With_DR_total_electricity_cost = 0

    for agent in agents:
        Appliance_Without_DR_Cost[agent] = 0
        Appliance_With_DR_Cost[agent] = 0
        Appliance_Without_DR_Reward[agent] = 0
        Appliance_With_DR_Reward[agent] = 0

    for t in range(num_timeslots):
        Timeslot_Without_DR_Cost[t] = 0
        Timeslot_With_DR_Cost[t] = 0
        for agent in agents:
            temp = deepcopy(actually_started)

            rew, c, actually_started[t] = get_reward_DR(states[t], t, agent, recommendations[t][agent], actually_started[t])
            actually_started = temp
            With_DR_total_reward += rew
            Appliance_With_DR_Reward[agent] -= rew
            With_DR_total_electricity_cost += c
            Appliance_With_DR_Cost[agent] -= c
            Timeslot_With_DR_Cost[t] -= c

            rew, c = get_reward_without_DR(states[t], t, agent)
            Without_DR_total_reward += rew
            Appliance_Without_DR_Reward[agent] -= rew
            Without_DR_total_electricity_cost += c
            Appliance_Without_DR_Cost[agent] -= c
            Timeslot_Without_DR_Cost[t] -= c

    recom_data = {}
    Recom_df = {}
    for agent in agents:
        recom_data[agent] = [[t+1, without_DR_actions[t][agent], recommendations[t][agent], states[t][0]] for t in range(num_timeslots)]
        Recom_df[agent] = pd.DataFrame(recom_data[agent], columns=['Slot', 'Action without DR', 'Action with DR', 'Elec. Price ($/MWk)'])

    return {
        "recommendations": recommendations,
        "without_DR_actions": without_DR_actions,
        "ECWithoutDR": -Without_DR_total_electricity_cost,
        "ECWithDR": -With_DR_total_electricity_cost
    }

