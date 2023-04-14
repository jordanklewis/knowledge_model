# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 21:01:14 2022

@author: Lewis
"""
import time
import os
import datetime
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from knowledge_model import KnowledgeModel

# time duration of model should be 30% of max knowledge
# 30% of max knowledge should equate to 30% of a career
# career is assumed to be 50 years
# model should run for 15 years

# 4 time resolutions for 15 years- 30000 hours, 3750 Days, 750 Weeks, 180 Months
# assume 5 days in a week, 50 weeks in a year
# employees gain 1 unit of know per 1 unit of time
# corresponding max_know - 100,000, 12,500, 2,500, 600

num_emp=40
steps = 180
innov_rate=0.5
seed=0
grid_res = 10

total_time = time.time()
data_path = "C:\\Users\\Lewis\\Documents\\data"
data_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
file_info = "NumEmp%s_Steps%s_Grid%s_Seed%s" % (num_emp, steps, grid_res, seed)
data_file = os.path.join(data_path, data_time+'_'+file_info+'.csv')

config_num = -1
model_data = []
max_know = int(steps/0.3)
know_cat_ct = int(max_know/10)

it_count = 0
for i in range(20, 105, grid_res):
    for w in range(0, 105, grid_res):
        if (i + w) > 100:
            continue
        if (i + w) <= 50:
            continue
        it_count += 1
total_steps = it_count*steps

t_time = time.time()
for i in range(20, 105, grid_res):
    for w in range(0, 105, grid_res):
        if (i + w) > 100:
            continue
        if (i + w) <= 50:
            continue
        config_num += 1
        avail = i/100
        busy = w/100
        docs = abs(round(1 - avail - busy, 2))
        model = KnowledgeModel(num_emp=num_emp,
                               avail=avail,
                               busy=busy,
                               know_cat_ct=know_cat_ct,
                               max_know=max_know,
                               innov_rate=innov_rate,
                               config_num = config_num,
                               seed=seed)

        for s in range(steps):
            model.step()

            if (s % 25 == 0) & (s > 0):
                pct_comp = (config_num*steps+s+1)/total_steps
                elap_time = time.time() - t_time
                remain_time = (1-pct_comp)*elap_time/pct_comp
                comp_time = str(datetime.datetime.now() +
                                datetime.timedelta(seconds=remain_time))
                remain_time = str(datetime.timedelta(seconds=remain_time))
                print(("\rRunning Model: A:%s, B:%s, D:%s,"+
                      "Config: %s, Step:%s, Remain Time: %s, End Time: %s    ")
                      % (avail, busy, docs, config_num, s,
                         remain_time, comp_time), end='\r')

            if (s % 1000 == 0) | (s+1 == steps):
                df = pd.DataFrame(model.step_data)
                if os.path.isfile(data_file):
                    df.to_csv(data_file, mode='a', index=False, header=False)
                else:
                    df.to_csv(data_file, index=False)
                model.step_data = []
                df = pd.DataFrame()

print("%.2f Sec Total Time" % (time.time() - total_time))

# insert 'complete' to the beginning to mark the csv data as a complete simulation
complete_data_file = os.path.join(data_path, 'complete_'+data_time+'_'+file_info+'.csv')
os.rename(data_file, complete_data_file)

model.plot_company_know()
model.plot_company_know_subplots()

# company population pyramid
plt.figure(figsize=(10,6))
plt.grid()
labels = [str(int(l)) for l in np.linspace(max_know/10,max_know,10)]
bottom = np.zeros(len(labels)).astype(int)

e = Counter(model.roster.Exp[model.roster.Dept=='SE'])
values = [e[int(l)] for l in np.linspace(max_know/10,max_know,10)]
plt.barh(labels,values,
          color='red', alpha=1, label='SE', left=bottom)
bottom += values

e = Counter(model.roster.Exp[model.roster.Dept=='SW'])
values = [e[int(l)] for l in np.linspace(max_know/10,max_know,10)]
plt.barh(labels,values,
          color='green', alpha=0.5, label='SW', left=bottom)
bottom+=values

e = Counter(model.roster.Exp[model.roster.Dept=='EE'])
values = [e[int(l)] for l in np.linspace(max_know/10,max_know,10)]
plt.barh(labels,values,
          color='yellow', alpha=0.75, label='EE', left=bottom)
bottom+= values

e = Counter(model.roster.Exp[model.roster.Dept=='ME'])
values = [e[int(l)] for l in np.linspace(max_know/10,max_know,10)]
plt.barh(labels,values,
          color='blue', alpha=0.25, label='ME', left=bottom)
plt.legend()
plt.ylabel('Years of Experience')
plt.xlabel('Employee Count')
plt.title('Employee Populaiton Pyramid\n'
          + 'Num Emp: ' + str(num_emp)
          + '    Know Cat Ct: ' + str(know_cat_ct)
          + '    Max_Know: ' + str(max_know)
          + '    Innov Rate: ' + str(innov_rate)
          + '    Seed: ' + str(seed))
plt.show()