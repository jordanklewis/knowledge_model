# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 11:13:50 2023

@author: Lewis
"""
# import time
import os
# import datetime
# from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data_path = "C:\\Users\\Lewis\\Documents\\data"
data_file = "complete_20230327_232139_NumEmp40_Steps3750_Grid5_Seed0_random_library.csv"
data_file = os.path.join(data_path, data_file)
model_data = pd.read_csv(data_file)

steps = max(model_data.step)+1
num_emp = max(model_data.num_emp)

comp_task_value = {}
for c in np.unique(model_data.config_num):
    keys = (int(list(model_data.avail[model_data.config_num==c]*100)[-1]),
            int(list(model_data.busy[model_data.config_num==c]*100)[-1]))
    comp_task_value[keys] = model_data.task_complexity[(model_data.task_completed)
                                & (model_data.config_num==c)].sum()
ser = pd.Series(list(comp_task_value.values()),
                  index=pd.MultiIndex.from_tuples(comp_task_value.keys()))
df = ser.unstack()

plt.figure(figsize=(10,6))
ax = sns.heatmap(df, annot=False, fmt='.0f', cmap='coolwarm', robust=True)
# ax = sns.heatmap(df, annot=False, fmt='.0f', cmap='coolwarm', robust=True, center=1.3e6)
ax.invert_yaxis()
plt.xlabel('Busy')
plt.ylabel('Available')
plt.title('Company Task Value\n'
          + 'Num Emp: ' + str(model_data.num_emp.iloc[-1])
          + '    Know Cat Ct: ' + str(model_data.know_cat_ct.iloc[-1])
          + '    Max_Know: ' + str(model_data.max_know.iloc[-1])
          + '    Innov Rate: ' + str(model_data.innov_rate.iloc[-1])
          + '    Seed: ' + str(model_data.seed.iloc[-1]))
plt.show()

comp_know = {}
for c in np.unique(model_data.config_num):
    keys = (int(list(model_data.avail[model_data.config_num==c]*100)[-1]),
            int(list(model_data.busy[model_data.config_num==c]*100)[-1]))
    comp_know[keys] = (model_data.total_knowledge[(model_data.config_num==c) &
                                                 (model_data.step==(steps-1))].sum() - 
                      model_data.total_knowledge[(model_data.config_num==c) &
                                                 (model_data.step==0)].sum())
ser = pd.Series(list(comp_know.values()),
                  index=pd.MultiIndex.from_tuples(comp_know.keys()))
df = ser.unstack()

plt.figure(figsize=(10,6))
ax = sns.heatmap(df, annot=False, fmt=".0f", cmap='coolwarm', robust=True)
ax.invert_yaxis()
plt.xlabel('Busy')
plt.ylabel('Available')
plt.title('Company Knowledge Growth\n'
          + 'Num Emp: ' + str(model_data.num_emp.iloc[-1])
          + '    Know Cat Ct: ' + str(model_data.know_cat_ct.iloc[-1])
          + '    Max_Know: ' + str(model_data.max_know.iloc[-1])
          + '    Innov Rate: ' + str(model_data.innov_rate.iloc[-1])
          + '    Seed: ' + str(model_data.seed.iloc[-1]))
plt.show()



doc_know_pct = {}
read_know_pct = {}
learn_know_pct = {}
res_know_pct = {}

for c in np.unique(model_data.config_num):
    keys = (int(list(model_data.avail[model_data.config_num==c]*100)[-1]),
            int(list(model_data.busy[model_data.config_num==c]*100)[-1]))
    doc_know = model_data[['document_know']][model_data.config_num==c].sum().iloc[0]
    read_know = model_data[['read_know']][model_data.config_num==c].sum().iloc[0]
    learn_know = model_data[['learn_know']][model_data.config_num==c].sum().iloc[0]
    res_know = model_data[['research_know']][model_data.config_num==c].sum().iloc[0]
    total = model_data[['research_know',
                        'learn_know',
                        'teach_know',
                        'document_know',
                        'read_know']][model_data.config_num==c].sum().sum()
    
    doc_know_pct[keys] = round(doc_know/total*100)
    read_know_pct[keys] = round(read_know/total*100)
    learn_know_pct[keys] = round(learn_know/total*100)
    res_know_pct[keys] = round(res_know/total*100)


doc_know_pct = pd.Series(list(doc_know_pct.values()),
                  index=pd.MultiIndex.from_tuples(doc_know_pct.keys()))
doc_know_pct = doc_know_pct.unstack()

read_know_pct = pd.Series(list(read_know_pct.values()),
                  index=pd.MultiIndex.from_tuples(read_know_pct.keys()))
read_know_pct = read_know_pct.unstack()

learn_know_pct = pd.Series(list(learn_know_pct.values()),
                  index=pd.MultiIndex.from_tuples(learn_know_pct.keys()))
learn_know_pct = learn_know_pct.unstack()

res_know_pct = pd.Series(list(res_know_pct.values()),
                  index=pd.MultiIndex.from_tuples(res_know_pct.keys()))
res_know_pct = res_know_pct.unstack()

fig,axis = plt.subplots(2, 2, figsize=(15,9))
sns.heatmap(res_know_pct, annot=True, fmt=".0f", cmap='coolwarm', robust=True, ax=axis[0][0])
axis[0][0].invert_yaxis()
axis[0][0].set_xlabel('Busy')
axis[0][0].set_ylabel('Available')
axis[0][0].set_title('Researching')

sns.heatmap(learn_know_pct, annot=True, fmt=".0f", cmap='coolwarm', robust=True, ax=axis[0][1])
axis[0][1].invert_yaxis()
axis[0][1].set_xlabel('Busy')
axis[0][1].set_ylabel('Available')
axis[0][1].set_title('Teaching/Learning')

sns.heatmap(doc_know_pct, annot=True, fmt=".0f", cmap='coolwarm', robust=True, ax=axis[1][0])
axis[1][0].invert_yaxis()
axis[1][0].set_xlabel('Busy')
axis[1][0].set_ylabel('Available')
axis[1][0].set_title('Documenting')

sns.heatmap(read_know_pct, annot=True, fmt=".0f", cmap='coolwarm', robust=True, ax=axis[1][1])
axis[1][1].invert_yaxis()
axis[1][1].set_xlabel('Busy')
axis[1][1].set_ylabel('Available')
axis[1][1].set_title('Reading')
fig.suptitle('How Employees Spend Their Time\n'
          + 'Num Emp: ' + str(model_data.num_emp.iloc[-1])
          + '    Know Cat Ct: ' + str(model_data.know_cat_ct.iloc[-1])
          + '    Max_Know: ' + str(model_data.max_know.iloc[-1])
          + '    Innov Rate: ' + str(model_data.innov_rate.iloc[-1])
          + '    Seed: ' + str(model_data.seed.iloc[-1]))
plt.tight_layout()
plt.show()

# model.plot_company_know()
# model.plot_company_know_subplots()

# # company population pyramid
# plt.figure(figsize=(10,6))
# plt.grid()
# labels = [str(int(l)) for l in np.linspace(max_know/10,max_know,10)]
# bottom = np.zeros(len(labels)).astype(int)

# e = Counter(model.roster.Exp[model.roster.Dept=='SE'])
# values = [e[int(l)] for l in np.linspace(max_know/10,max_know,10)]
# plt.barh(labels,values,
#           color='red', alpha=1, label='SE', left=bottom)
# bottom += values

# e = Counter(model.roster.Exp[model.roster.Dept=='SW'])
# values = [e[int(l)] for l in np.linspace(max_know/10,max_know,10)]
# plt.barh(labels,values,
#           color='green', alpha=0.5, label='SW', left=bottom)
# bottom+=values

# e = Counter(model.roster.Exp[model.roster.Dept=='EE'])
# values = [e[int(l)] for l in np.linspace(max_know/10,max_know,10)]
# plt.barh(labels,values,
#           color='yellow', alpha=0.75, label='EE', left=bottom)
# bottom+= values

# e = Counter(model.roster.Exp[model.roster.Dept=='ME'])
# values = [e[int(l)] for l in np.linspace(max_know/10,max_know,10)]
# plt.barh(labels,values,
#           color='blue', alpha=0.25, label='ME', left=bottom)
# plt.legend()
# plt.ylabel('Experience Level')
# plt.xlabel('Employee Count')
# plt.title('Employee Populaiton Pyramid\n'
#           + 'Num Emp: ' + str(model_data.num_emp.iloc[-1])
#           + '    Know Cat Ct: ' + str(model_data.know_cat_ct.iloc[-1])
#           + '    Max_Know: ' + str(model_data.max_know.iloc[-1])
#           + '    Innov Rate: ' + str(model_data.innov_rate.iloc[-1])
#           + '    Seed: ' + str(model_data.seed.iloc[-1]))
# plt.show()

# create an employee aggregate data frame
emp_ids = sorted(pd.unique(model_data.employee_id))
emp_agg_df = pd.DataFrame([])
for c in np.unique(model_data.config_num):
    model_agg = []
    for i in emp_ids:
        model_data_emp = model_data[(model_data.employee_id==i) &
                                    (model_data.config_num==c)]
        model_agg.append({'config_num': c,
                          'num_emp': model_data_emp.num_emp.iloc[-1],
                          'avail': int(model_data_emp.avail.iloc[-1]*100),
                          'busy': int(model_data_emp.busy.iloc[-1]*100),
                          'docs': int(model_data_emp.docs.iloc[-1]*100),
                          'know_cat_ct': model_data_emp.know_cat_ct.iloc[-1],
                          'max_know': model_data_emp.max_know.iloc[-1],
                          'innov_rate': model_data_emp.innov_rate.iloc[-1],
                          'seed': model_data_emp.seed.iloc[-1],
                          'emp_id': i,
                          'dept': model_data_emp.employee_dept.iloc[-1],
                          'start_know': model_data_emp.total_knowledge.iloc[0],
                          'current_know': model_data_emp.total_knowledge.iloc[-1],
                          'know_growth': (model_data_emp.total_knowledge.iloc[-1]
                                          - model_data_emp.total_knowledge.iloc[0]),
                          'task_count': model_data_emp.task_completed.sum(),
                          'task_value': model_data_emp.task_complexity[model_data_emp.task_completed].sum(),
                          'research_know': model_data_emp.research_know.iloc[-1],
                          'learn_know': model_data_emp.learn_know.iloc[-1],
                          'teach_know': model_data_emp.teach_know.iloc[-1],
                          'doc_know': model_data_emp.document_know.iloc[-1],
                          'read_know': model_data_emp.read_know.iloc[-1]
                          })
    model_agg = pd.DataFrame(model_agg).sort_values(['start_know'])
    emp_agg_df = pd.concat([emp_agg_df, model_agg], ignore_index=True)

avail_arr = np.unique(emp_agg_df.avail)
avail_arr = avail_arr[avail_arr%10==0]
busy_arr = np.unique(emp_agg_df.busy)
busy_arr = busy_arr[busy_arr%10==0]
fig,axis = plt.subplots(len(avail_arr),
                        len(busy_arr),
                        figsize=(15,9), sharex=True,
                        sharey=True,constrained_layout=True)
for row in range(len(avail_arr)):
    for col in range(len(busy_arr)):
        axis[row][col].axis('off')
        if row == len(avail_arr)-1:
            axis[row][col].set_xlabel(busy_arr[col])
        if col == 0:
            axis[row][col].set_ylabel(avail_arr[len(busy_arr) - row - 1])
for c in np.unique(model_data.config_num):
    config_filt = emp_agg_df[emp_agg_df.config_num==c]
    if not (np.any(avail_arr == config_filt.avail.iloc[-1]) &
            np.any(busy_arr == config_filt.busy.iloc[-1])):
        continue
    row = len(avail_arr) - np.where(avail_arr == config_filt.avail.iloc[-1])[0][0] -1
    col = np.where(busy_arr == config_filt.busy.iloc[-1])[0][0]
    axis[row][col].stackplot(config_filt.start_know, config_filt.research_know,
                  config_filt.learn_know, config_filt.read_know,
                  labels = ['Research', 'Learn', 'Read'])
    axis[row][col].axis('on')
    if row == len(avail_arr)-1:
        axis[row][col].set_xlabel(str(config_filt.busy.iloc[-1]))
    if col == 0:
        axis[row][col].set_ylabel(str(config_filt.avail.iloc[-1]))
        
    for r in np.where(avail_arr < config_filt.avail.iloc[-1])[0]:
        for c in np.where(busy_arr < config_filt.busy.iloc[-1])[0]:
            row = len(avail_arr) - r - 1
            col = c
            axis[row][col].axis('on')
            
fig.suptitle('Employee Knowledge Source\n'
          + 'Num Emp: ' + str(emp_agg_df.num_emp.iloc[-1])
          + '    Know Cat Ct: ' + str(emp_agg_df.know_cat_ct.iloc[-1])
          + '    Max_Know: ' + str(emp_agg_df.max_know.iloc[-1])
          + '    Innov Rate: ' + str(emp_agg_df.innov_rate.iloc[-1])
          + '    Seed: ' + str(emp_agg_df.seed.iloc[-1]))
fig.supxlabel('Start Know Quantity\nBusy')
fig.supylabel('Available\nKnow Growth')
h, l = axis[row][col].get_legend_handles_labels()
fig.legend(h, l)
fig.tight_layout()
plt.show()

avail_arr = np.unique(emp_agg_df.avail)
avail_arr = avail_arr[avail_arr%10==0]
busy_arr = np.unique(emp_agg_df.busy)
busy_arr = busy_arr[busy_arr%10==0]
fig,axis = plt.subplots(len(avail_arr),
                        len(busy_arr),
                        figsize=(15,9), sharex=True,
                        sharey=True,constrained_layout=True)

for row in range(len(avail_arr)):
    for col in range(len(busy_arr)):
        axis[row][col].axis('off')
        if row == len(avail_arr)-1:
            axis[row][col].set_xlabel(busy_arr[col])
        if col == 0:
            axis[row][col].set_ylabel(avail_arr[len(busy_arr) - row - 1])
        
for c in np.unique(model_data.config_num):
    config_filt = emp_agg_df[emp_agg_df.config_num==c]
    if not (np.any(avail_arr == config_filt.avail.iloc[-1]) &
            np.any(busy_arr == config_filt.busy.iloc[-1])):
        continue
    row = len(avail_arr) - np.where(avail_arr == config_filt.avail.iloc[-1])[0][0] -1
    col = np.where(busy_arr == config_filt.busy.iloc[-1])[0][0]
    axis[row][col].stackplot(config_filt.start_know, config_filt.teach_know,
                  config_filt.doc_know,
                  labels = ['Teach', 'Document'])
    axis[row][col].axis('on')
    if row == len(avail_arr)-1:
        axis[row][col].set_xlabel(str(config_filt.busy.iloc[-1]))
    if col == 0:
        axis[row][col].set_ylabel(str(config_filt.avail.iloc[-1]))

    for r in np.where(avail_arr < config_filt.avail.iloc[-1])[0]:
        for c in np.where(busy_arr < config_filt.busy.iloc[-1])[0]:
            row = len(avail_arr) - r - 1
            col = c
            axis[row][col].axis('on')

fig.suptitle('Employee Knowledge Transfer Method\n'
          + 'Num Emp: ' + str(emp_agg_df.num_emp.iloc[-1])
          + '    Know Cat Ct: ' + str(emp_agg_df.know_cat_ct.iloc[-1])
          + '    Max_Know: ' + str(emp_agg_df.max_know.iloc[-1])
          + '    Innov Rate: ' + str(emp_agg_df.innov_rate.iloc[-1])
          + '    Seed: ' + str(emp_agg_df.seed.iloc[-1]))
fig.supxlabel('Start Know Quantity\nBusy')
fig.supylabel('Available\nKnow Share')
h, l = axis[row][col].get_legend_handles_labels()
fig.legend(h, l)
fig.tight_layout()
plt.show()

avail_arr = np.unique(emp_agg_df.avail)
avail_arr = avail_arr[avail_arr%10==0]
busy_arr = np.unique(emp_agg_df.busy)
busy_arr = busy_arr[busy_arr%10==0]
fig,axis = plt.subplots(len(avail_arr),
                        len(busy_arr),
                        figsize=(15,9), sharex=True,
                        sharey=True,constrained_layout=True)
for row in range(len(avail_arr)):
    for col in range(len(busy_arr)):
        axis[row][col].axis('off')
        if row == len(avail_arr)-1:
            axis[row][col].set_xlabel(busy_arr[col])
        if col == 0:
            axis[row][col].set_ylabel(avail_arr[len(busy_arr) - row - 1])
for c in np.unique(model_data.config_num):
    config_filt = emp_agg_df[emp_agg_df.config_num==c]
    if not (np.any(avail_arr == config_filt.avail.iloc[-1]) &
            np.any(busy_arr == config_filt.busy.iloc[-1])):
        continue
    row = len(avail_arr) - np.where(avail_arr == config_filt.avail.iloc[-1])[0][0] -1
    col = np.where(busy_arr == config_filt.busy.iloc[-1])[0][0]
    axis[row][col].stackplot(config_filt.start_know, config_filt.task_value)
    axis[row][col].axis('on')
    if row == len(avail_arr)-1:
        axis[row][col].set_xlabel(str(config_filt.busy.iloc[-1]))
    if col == 0:
        axis[row][col].set_ylabel(str(config_filt.avail.iloc[-1]))

    for r in np.where(avail_arr < config_filt.avail.iloc[-1])[0]:
        for c in np.where(busy_arr < config_filt.busy.iloc[-1])[0]:
            row = len(avail_arr) - r - 1
            col = c
            axis[row][col].axis('on')
                
fig.suptitle('Employee Task Value\n'
          + 'Num Emp: ' + str(emp_agg_df.num_emp.iloc[-1])
          + '    Know Cat Ct: ' + str(emp_agg_df.know_cat_ct.iloc[-1])
          + '    Max_Know: ' + str(emp_agg_df.max_know.iloc[-1])
          + '    Innov Rate: ' + str(emp_agg_df.innov_rate.iloc[-1])
          + '    Seed: ' + str(emp_agg_df.seed.iloc[-1]))
fig.supxlabel('Start Know Quantity\nBusy')
fig.supylabel('Available\nTask Value')
h, l = axis[row][col].get_legend_handles_labels()
fig.legend(h, l)
fig.tight_layout()
plt.show()

avail_arr = np.unique(emp_agg_df.avail)
avail_arr = avail_arr[avail_arr%10==0]
busy_arr = np.unique(emp_agg_df.busy)
busy_arr = busy_arr[busy_arr%10==0]
fig,axis = plt.subplots(len(avail_arr),
                        len(busy_arr),
                        figsize=(15,9), sharex=True,
                        sharey=True,constrained_layout=True)
for row in range(len(avail_arr)):
    for col in range(len(busy_arr)):
        axis[row][col].axis('off')
        if row == len(avail_arr)-1:
            axis[row][col].set_xlabel(busy_arr[col])
        if col == 0:
            axis[row][col].set_ylabel(avail_arr[len(busy_arr) - row - 1])
for c in np.unique(model_data.config_num):
    config_filt = emp_agg_df[emp_agg_df.config_num==c]
    if not (np.any(avail_arr == config_filt.avail.iloc[-1]) &
            np.any(busy_arr == config_filt.busy.iloc[-1])):
        continue
    row = len(avail_arr) - np.where(avail_arr == config_filt.avail.iloc[-1])[0][0] -1
    col = np.where(busy_arr == config_filt.busy.iloc[-1])[0][0]
    axis[row][col].stackplot(config_filt.start_know, config_filt.task_count)
    axis[row][col].axis('on')
    if row == len(avail_arr)-1:
        axis[row][col].set_xlabel(str(config_filt.busy.iloc[-1]))
    if col == 0:
        axis[row][col].set_ylabel(str(config_filt.avail.iloc[-1]))

    for r in np.where(avail_arr < config_filt.avail.iloc[-1])[0]:
        for c in np.where(busy_arr < config_filt.busy.iloc[-1])[0]:
            row = len(avail_arr) - r - 1
            col = c
            axis[row][col].axis('on')
                
fig.suptitle('Employee Task Count\n'
          + 'Num Emp: ' + str(emp_agg_df.num_emp.iloc[-1])
          + '    Know Cat Ct: ' + str(emp_agg_df.know_cat_ct.iloc[-1])
          + '    Max_Know: ' + str(emp_agg_df.max_know.iloc[-1])
          + '    Innov Rate: ' + str(emp_agg_df.innov_rate.iloc[-1])
          + '    Seed: ' + str(emp_agg_df.seed.iloc[-1]))
fig.supxlabel('Start Know Quantity\nBusy')
fig.supylabel('Available\nTask Count')
h, l = axis[row][col].get_legend_handles_labels()
fig.legend(h, l)
fig.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
plt.grid()
plt.plot(model_data.total_knowledge[model_data.task_completed==True],
          model_data.needed_know[model_data.task_completed==True], '.',alpha=0.25)
plt.ylabel('Needed Task Knowledge')
plt.xlabel('Employee Total Knowledge')
plt.title('Employee Task Difficulty\n'
          + 'Num Emp: ' + str(model_data.num_emp.iloc[-1])
          + '    Know Cat Ct: ' + str(model_data.know_cat_ct.iloc[-1])
          + '    Max_Know: ' + str(model_data.max_know.iloc[-1])
          + '    Innov Rate: ' + str(model_data.innov_rate.iloc[-1])
          + '    Seed: ' + str(model_data.seed.iloc[-1]))
plt.show()

plt.figure(figsize=(10,6))
plt.grid()
for c in np.unique(model_data.config_num):
    config_filt = model_data[model_data.config_num==c]
    if config_filt.busy.iloc[-1] == 0.2:
        plt.plot(config_filt.step, config_filt.comp_lib_know,
                 label="Doc Pct: %s" % (int(config_filt.docs.iloc[-1]*100)))
plt.title('Company Library Knowledge\n'
          + 'Num Emp: ' + str(model_data.num_emp.iloc[-1])
          + '    Know Cat Ct: ' + str(model_data.know_cat_ct.iloc[-1])
          + '    Max_Know: ' + str(model_data.max_know.iloc[-1])
          + '    Innov Rate: ' + str(model_data.innov_rate.iloc[-1])
          + '    Seed: ' + str(model_data.seed.iloc[-1]))
# plt.legend(loc='outside right')
# if c < 20:
plt.legend(bbox_to_anchor =(1, 1.15))
plt.xlabel('Step')
plt.ylabel('Knowledge Quantity')
plt.show()

growth = []
for c in np.unique(model_data.config_num):
    config_filt = model_data[model_data.config_num==c]
    growth.append({"config_num": config_filt.config_num.iloc[-1],
                    "avail": config_filt.avail.iloc[-1],
                    "busy": config_filt.busy.iloc[-1],
                    "docs": config_filt.docs.iloc[-1],
                    "SE_dept_know": max(config_filt.SE_dept_know) - min(config_filt.SE_dept_know),
                    "SW_dept_know": max(config_filt.SW_dept_know) - min(config_filt.SW_dept_know),
                    "EE_dept_know": max(config_filt.EE_dept_know) - min(config_filt.EE_dept_know),
                    "ME_dept_know": max(config_filt.ME_dept_know) - min(config_filt.ME_dept_know),
                    "comp_lib_know": max(config_filt.comp_lib_know) - min(config_filt.comp_lib_know),
                    "comp_know": max(config_filt.comp_know) - min(config_filt.comp_know)})
growth = pd.DataFrame(growth)

SE_dept_know = {}
SW_dept_know = {}
EE_dept_know = {}
ME_dept_know = {}
comp_lib_know = {}
comp_know = {}

for c in np.unique(growth.config_num):
    keys = (int(list(growth.avail[growth.config_num==c]*100)[-1]),
            int(list(growth.busy[growth.config_num==c]*100)[-1]))
    
    SE_dept_know[keys] = growth[['SE_dept_know']][growth.config_num==c].iloc[0][0]
    SW_dept_know[keys] = growth[['SW_dept_know']][growth.config_num==c].iloc[0][0]
    EE_dept_know[keys] = growth[['EE_dept_know']][growth.config_num==c].iloc[0][0]
    ME_dept_know[keys] = growth[['ME_dept_know']][growth.config_num==c].iloc[0][0]
    comp_lib_know[keys] = growth[['comp_lib_know']][growth.config_num==c].iloc[0][0]
    comp_know[keys] = growth[['comp_know']][growth.config_num==c].iloc[0][0]


SE_dept_know = pd.Series(list(SE_dept_know.values()),
                  index=pd.MultiIndex.from_tuples(SE_dept_know.keys()))
SE_dept_know = SE_dept_know.unstack()

SW_dept_know = pd.Series(list(SW_dept_know.values()),
                  index=pd.MultiIndex.from_tuples(SW_dept_know.keys()))
SW_dept_know = SW_dept_know.unstack()

EE_dept_know = pd.Series(list(EE_dept_know.values()),
                  index=pd.MultiIndex.from_tuples(EE_dept_know.keys()))
EE_dept_know = EE_dept_know.unstack()

ME_dept_know = pd.Series(list(ME_dept_know.values()),
                  index=pd.MultiIndex.from_tuples(ME_dept_know.keys()))
ME_dept_know = ME_dept_know.unstack()

comp_lib_know = pd.Series(list(comp_lib_know.values()),
                  index=pd.MultiIndex.from_tuples(comp_lib_know.keys()))
comp_lib_know = comp_lib_know.unstack()

comp_know = pd.Series(list(comp_know.values()),
                  index=pd.MultiIndex.from_tuples(comp_know.keys()))
comp_know = comp_know.unstack()

fig,axis = plt.subplots(2, 2, figsize=(15,9))
sns.heatmap(SE_dept_know, annot=False, fmt=".0f", cmap='coolwarm', robust=True, ax=axis[0][0])
axis[0][0].invert_yaxis()
axis[0][0].set_xlabel('Busy')
axis[0][0].set_ylabel('Available')
axis[0][0].set_title('SE Knowledge')

sns.heatmap(SW_dept_know, annot=False, fmt=".0f", cmap='coolwarm', robust=True, ax=axis[0][1])
axis[0][1].invert_yaxis()
axis[0][1].set_xlabel('Busy')
axis[0][1].set_ylabel('Available')
axis[0][1].set_title('SW Knowledge')

sns.heatmap(EE_dept_know, annot=False, fmt=".0f", cmap='coolwarm', robust=True, ax=axis[1][0])
axis[1][0].invert_yaxis()
axis[1][0].set_xlabel('Busy')
axis[1][0].set_ylabel('Available')
axis[1][0].set_title('EE Knowledge')

sns.heatmap(ME_dept_know, annot=False, fmt=".0f", cmap='coolwarm', robust=True, ax=axis[1][1])
axis[1][1].invert_yaxis()
axis[1][1].set_xlabel('Busy')
axis[1][1].set_ylabel('Available')
axis[1][1].set_title('ME Knowledge')
fig.suptitle('Knowledge Growth by Department\n'
          + 'Num Emp: ' + str(model_data.num_emp.iloc[-1])
          + '    Know Cat Ct: ' + str(model_data.know_cat_ct.iloc[-1])
          + '    Max_Know: ' + str(model_data.max_know.iloc[-1])
          + '    Innov Rate: ' + str(model_data.innov_rate.iloc[-1])
          + '    Seed: ' + str(model_data.seed.iloc[-1]))
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
ax = sns.heatmap(comp_lib_know, annot=False, fmt='.0f', cmap='coolwarm', robust=True)
ax.invert_yaxis()
plt.xlabel('Busy')
plt.ylabel('Available')
plt.title('Company Library Knowledge\n'
          + 'Num Emp: ' + str(model_data.num_emp.iloc[-1])
          + '    Know Cat Ct: ' + str(model_data.know_cat_ct.iloc[-1])
          + '    Max_Know: ' + str(model_data.max_know.iloc[-1])
          + '    Innov Rate: ' + str(model_data.innov_rate.iloc[-1])
          + '    Seed: ' + str(model_data.seed.iloc[-1]))
plt.show()

plt.figure(figsize=(10,6))
ax = sns.heatmap(comp_know, annot=False, fmt='.0f', cmap='coolwarm', robust=True)
ax.invert_yaxis()
plt.xlabel('Busy')
plt.ylabel('Available')
plt.title('Orgainzational Knowledge Growth\n'
          + 'Num Emp: ' + str(model_data.num_emp.iloc[-1])
          + '    Know Cat Ct: ' + str(model_data.know_cat_ct.iloc[-1])
          + '    Max_Know: ' + str(model_data.max_know.iloc[-1])
          + '    Innov Rate: ' + str(model_data.innov_rate.iloc[-1])
          + '    Seed: ' + str(model_data.seed.iloc[-1]))
plt.show()

'''
# plt.figure(figsize=(10,6))
# plt.grid()
# for c in np.unique(emp_agg_df.config_num):
#     config_filt = emp_agg_df[emp_agg_df.config_num==c]
#     plt.plot(config_filt.start_know, config_filt.know_growth,
#              label = "A:%s W:%s D: %s" % (config_filt.avail.iloc[-1],
#                                           config_filt.busy.iloc[-1],
#                                           config_filt.docs.iloc[-1]))
# plt.title('Employee Knowledge Growth\n'
#           + 'Num Emp: ' + str(emp_agg_df.num_emp.iloc[-1])
#           + '    Know Cat Ct: ' + str(emp_agg_df.know_cat_ct.iloc[-1])
#           + '    Max_Know: ' + str(emp_agg_df.max_know.iloc[-1])
#           + '    Innov Rate: ' + str(emp_agg_df.innov_rate.iloc[-1])
#           + '    Seed: ' + str(emp_agg_df.seed.iloc[-1]))
# plt.legend(bbox_to_anchor =(1, 1.15))
# plt.xlabel('Start Know Quantity')
# plt.ylabel('Know Growth Quantity')
# plt.show()

# plt.figure(figsize=(10,6))
# plt.grid()
# for c in np.unique(emp_agg_df.config_num):
#     config_filt = emp_agg_df[emp_agg_df.config_num==c]
#     plt.plot(config_filt.start_know, config_filt.task_value,
#              label = "A:%s W:%s D: %s" % (config_filt.avail.iloc[-1],
#                                           config_filt.busy.iloc[-1],
#                                           config_filt.docs.iloc[-1]))
# plt.title('Employee Task Value\n'
#           + 'Num Emp: ' + str(emp_agg_df.num_emp.iloc[-1])
#           + '    Know Cat Ct: ' + str(emp_agg_df.know_cat_ct.iloc[-1])
#           + '    Max_Know: ' + str(emp_agg_df.max_know.iloc[-1])
#           + '    Innov Rate: ' + str(emp_agg_df.innov_rate.iloc[-1])
#           + '    Seed: ' + str(emp_agg_df.seed.iloc[-1]))
# plt.legend(bbox_to_anchor =(1, 1.15))
# plt.xlabel('Start Know Quantity')
# plt.ylabel('Task Value')
# plt.show()

# plt.figure(figsize=(10,6))
# plt.grid()
# for c in np.unique(emp_agg_df.config_num):
#     config_filt = emp_agg_df[emp_agg_df.config_num==c]
#     plt.plot(config_filt.start_know, config_filt.task_count,
#              label = "A:%s W:%s D: %s" % (config_filt.avail.iloc[-1],
#                                           config_filt.busy.iloc[-1],
#                                           config_filt.docs.iloc[-1]))
# plt.title('Employee Task Count\n'
#           + 'Num Emp: ' + str(emp_agg_df.num_emp.iloc[-1])
#           + '    Know Cat Ct: ' + str(emp_agg_df.know_cat_ct.iloc[-1])
#           + '    Max_Know: ' + str(emp_agg_df.max_know.iloc[-1])
#           + '    Innov Rate: ' + str(emp_agg_df.innov_rate.iloc[-1])
#           + '    Seed: ' + str(emp_agg_df.seed.iloc[-1]))
# plt.legend(bbox_to_anchor =(1, 1.15))
# plt.xlabel('Start Know Quantity')
# plt.ylabel('Task Count')
# plt.show()




model_data = pd.DataFrame(model.step_data)


# Data Analysis Objectives
# 1. Employee knowledge growth
# 2. Employee personal task performance (total completed tasks, task complexity)
# 3. Company knowledge growth (max knowledge in company, average knowledge in company)
# 4. Company task performance (total completed tasks, task complexity)

# Heat Map for avail, busy, docs rates with colors for
    # company knowledge growth
    # company task count
    # company task value
    # department growth
    # research know
    # learn know
    # Teach know
    # Read know
    # Doc know

# Employee Aggregate Table
# Generate plot with X axis employee start knowledge, then all other variables
# can be plotted againsts it showing how varying experience levels are responding
# to the model configurations


# Iterate through each configuration
# Generate employee aggragete plots









# Current problem:
    # Employees who learn through perosnal research grow knowledge at the same
    # rate as employees who receive help (1 knowledge per step). This means
    # employees are promoted at the same rate whether the receive help or not.
    # In theory, employees who receive help will accomplish tasks quicker.
    # Accopmlishing tasks quicker is better for the company. The user of this
    # model is corporate management, who's top prioriety is maximizing company
    # performance.

# There are 2 types of plots:
    # 1. Plots that verify the moddel is working corretly and demonstrate how the model works
    # 2. Plots that measure the model performance and answer the research question

# this analysis and plot demonstrates the promotion/performance cycle
# you are a top performer as you near a promotion, once promoted you have
# new responsibilities causing you to take longer to complete tasks. The more
# expereienced/promoted an employee is, the quicker they can complete tasks
emp_ids = pd.unique(model_data.employee_id)
plt.figure(figsize=(10,6))
for i in emp_ids:
    comp_ndx = model_data.index[(model_data.employee_id==i) &
                                (model_data.task_completed==True)]
    s = model_data.step[comp_ndx]
    x = model_data.step[comp_ndx].diff()
    pro = model_data.employee_exp[comp_ndx].diff()==model.max_know/10
    plt.plot(s, x, label=i)
    plt.plot(s[pro], x[pro], '*')
# plt.legend()
plt.title('Promotion vs Task Performance Cycle')
plt.xlabel('Step')
plt.ylabel('Steps to Complete the Task')
plt.show()

# this analysis and plot demonstrates how employee value to the company grows
# over time. The more tasks of higher company value that are completed, the
# more valueable the employee is to the company.
emp_ids = sorted(pd.unique(model_data.employee_id))
plt.figure(figsize=(10,6))
for i in emp_ids:
    comp_ndx = model_data.index[(model_data.employee_id==i) &
                                (model_data.task_completed==True)]
    s = model_data.step[comp_ndx]
    # e = model_data.employee_exp[model_data.employee_id==i]
    # d = model_data.employee_dept[model_data.employee_id==i]

    # t = model_data.teach_know[model_data.employee_id==i]
    # l = model_data.learn_know[model_data.employee_id==i]
    # t = model_data.teach_know[model_data.employee_id==i]
    # r = model_data.research_know[model_data.employee_id==i]
    # doc = model_data.document_know[model_data.employee_id==i]
    # read = model_data.read_know[model_data.employee_id==i]

    x = np.cumsum(model_data.task_complexity[comp_ndx])
    pro = model_data.employee_exp[comp_ndx].diff()==model.max_know/10
    plt.plot(s, x, label=i)
    plt.plot(s[pro], x[pro], '*')
    # print('Emp ID: ' + str(i) + '    Dept: ' + str(list(d)[-1])
    #       + '    Exp: ' + str(list(e)[-1])
    #       + '    Task Value: ' + str(list(x)[-1])
    #       +'    Research Know:' + str(list(r)[-1])
    #       +'    Learn Know:' + str(list(l)[-1])
    #       +'    Teach Know:' + str(list(t)[-1])
    #       +'    Doc Know:' + str(list(doc)[-1])
    #       +'    Read Know:' + str(list(read)[-1]))
plt.legend()
plt.ylabel('Cumulative Completed Task Value')
plt.xlabel('Step')
plt.title('Employee Company Value')
plt.show()
plt.show()

# This plot demonstrates how each department is growing in knowledge over time
unique_steps, ndx = np.unique(model_data.step, return_index=True)
plt.figure(figsize=(10,6))
plt.plot(unique_steps, model_data.SE_dept_know.iloc[ndx],
         color='red', alpha=1, label='SE')
plt.plot(unique_steps, model_data.SW_dept_know.iloc[ndx],
         color='green', alpha=0.5, label='SW')
plt.plot(unique_steps, model_data.EE_dept_know.iloc[ndx],
         color='yellow', alpha=1, label='EE')
plt.plot(unique_steps, model_data.ME_dept_know.iloc[ndx],
         color='blue', alpha=0.25, label='ME')
plt.legend()
plt.ylabel('Department Knowledge')
plt.xlabel('Step')
plt.title('Department Knowledge Growth')
plt.grid()
plt.show()

# This plot demonstrates how the company is growing in knowledge over time
unique_steps, ndx = np.unique(model_data.step, return_index=True)
plt.figure(figsize=(10,6))
plt.plot(unique_steps, model_data.comp_know.iloc[ndx] - model_data.comp_know.iloc[0],
         color='red', alpha=1)
plt.ylabel('Company Knowledge')
plt.xlabel('Step')
plt.title('Company Knowledge Growth')
plt.grid()
plt.show()
print("Company Knowledge Growth: "
      + str(model_data.comp_know.iloc[-1] - model_data.comp_know.iloc[0]))

# This plot demonstrates how the value of the tasks the company is completing
# over time.
comp_ndx = model_data.index[model_data.task_completed==True]
x = model_data.iloc[comp_ndx]
x = np.cumsum(x.groupby('step')['task_complexity'].sum())
plt.figure(figsize=(10,6))
plt.plot(x.index, x,
         color='blue', alpha=1)
plt.ylabel('Cumulative Completed Task Value')
plt.xlabel('Step')
plt.title('Company Task Value')
plt.grid()
plt.show()
print("Cum. Completed Task Value: " + str(list(x)[-1]))

# create an aggregate data frame
emp_ids = sorted(pd.unique(model_data.employee_id))
model_agg = []
for i in emp_ids:
    model_data_emp = model_data[model_data.employee_id==i]

    model_agg.append({'emp_id': i,
                      'dept': model_data_emp.employee_dept.iloc[-1],
                      'start_know': model_data_emp.total_knowledge.iloc[0],
                      'current_know': model_data_emp.total_knowledge.iloc[-1],
                      'know_growth': (model_data_emp.total_knowledge.iloc[-1]
                                      - model_data_emp.total_knowledge.iloc[0]),
                      'task_count': model_data_emp.task_completed.sum(),
                      'task_value': model_data_emp.task_complexity[model_data_emp.task_completed==True].sum(),
                      'research_know': model_data_emp.research_know.iloc[-1],
                      'learn_know': model_data_emp.learn_know.iloc[-1],
                      'teach_know': model_data_emp.teach_know.iloc[-1],
                      'doc_know': model_data_emp.document_know.iloc[-1],
                      'read_know': model_data_emp.read_know.iloc[-1]
                      })
model_agg = pd.DataFrame(model_agg).sort_values(['start_know'])
'''
















'''
##############################################################################
# generate a model for a company's departments and knowledge scope
##############################################################################

# the distribution indicates a department bredth of knowledge
# systems
mu1, sigma1, N1 = 0, 30, 1000
X1 = np.random.normal(mu1, sigma1, N1).astype(int)
X1[X1>=50] -= 100
X1[X1<-50] += 100

# software
mu2, sigma2, N2 = 25, 20, 1000
X2 = np.random.normal(mu2, sigma2, N2).astype(int)
X2[X2>=50] -= 100
X2[X2<-50] += 100
# electrical
mu3, sigma3, N3 = 50, 20, 1000
X3 = np.random.normal(mu3, sigma3, N3).astype(int)
X3[X3>=50] -= 100
X3[X3<-50] += 100
# mechanical
mu4, sigma4, N4 = -25, 20, 1000
X4 = np.random.normal(mu4, sigma4, N4).astype(int)
X4[X4>=50] -= 100
X4[X4<-50] += 100

plt.figure(figsize=(10,6))
plt.hist(X1, bins=np.arange(X1.min()-0.5, X1.max()+0.5),
          color='red', alpha = 1, label='SE')
plt.hist(X2, bins=np.arange(X2.min()-0.5, X2.max()+0.5),
          color='green', alpha = 0.75, label='SW')
# plt.hist(X3, bins=np.arange(X3.min()-0.5, X3.max()+0.5),
#           color='yellow', alpha = 0.5, label='EE')
# plt.hist(X4, bins=np.arange(X4.min()-0.5, X4.max()+0.5),
#           color='blue', alpha = 0.3, label='ME')
plt.legend(loc='upper right')
plt.xlabel('Knowledge Category')
plt.ylabel('Knowledge Quantity')
plt.show()

##############################################################################
# generate database of all company knowledge from all departments
##############################################################################

know_key_ct = 100
know_key = list(np.arange(know_key_ct).astype(int)-int(know_key_ct/2))


se_ct = Counter(X1)
sw_ct = Counter(X2)
ee_ct = Counter(X3)
me_ct = Counter(X4)

for n in know_key:
    print(str(n) + ': ' + str(me_ct[n]))


diff_ct = se_ct - sw_ct

diff_ct = se_ct.subtract(sw_ct)


plt.figure(figsize=(10,6))
plt.hist(X1, bins=np.arange(X1.min()-0.5, X1.max()+0.5),
          color='red', alpha = 1, label='SE')
plt.hist(X2, bins=np.arange(X2.min()-0.5, X2.max()+0.5),
          color='green', alpha = 0.75, label='SW')
plt.hist(list((sw_ct+diff_ct).elements()), bins=np.arange(X2.min()-0.5, X2.max()+0.5),
          color='black', alpha = 0.5, label='SW+SE')
plt.legend(loc='upper right')
plt.xlabel('Knowledge Category')
plt.ylabel('Knowledge Quantity')
plt.show()


to_learn = list((se_ct - sw_ct).elements())



to_learn = list((c_t - c_k).elements())

##############################################################################
# creating a company of employees with varying skill levels
##############################################################################
total_employees = 20
employee_exp = []
for n in range(total_employees):
    employee_exp.append(np.random.choice(9, p=[.1, .1, .14, .2, .2, .1, .1, .05, .01])+1)
Counter(np.sort(employee_exp))

name_list = pd.read_csv("employee_names.csv")
name_list = name_list["Names"].tolist()
name_list = list(np.random.choice(name_list,
                                  total_employees, replace=False))
dept_list = ['SE', 'SW', 'EE', 'ME']
Counter(np.random.choice(dept_list, 20))



##############################################################################
# creating a mechanical engineer and choosing a task for them
##############################################################################

# electrical engineering department scope (for reference)
# plt.hist(X3, bins=np.arange(X3.min()-0.5, X3.max()+0.5),
#           color='yellow', alpha = 0.5)
fig = plt.figure(figsize=(15,8))
# distribution of department scope
plt.hist(X4, bins=np.arange(X4.min()-0.5, X4.max()+0.5),
          color='blue', histtype='step')

# create an Mech employee's knowledge distribution
exp_level = 900
mech_know = np.random.normal(mu4, sigma4, exp_level).astype(int)
mech_know[mech_know>50] -= 100
mech_know[mech_know<-50] += 100
plt.hist(mech_know, bins=np.arange(X4.min()-0.5, X4.max()+0.5),
          color='red', alpha = 1)

task = np.random.choice(X4, 600).astype(int)
plt.hist(task, bins=np.arange(X4.min()-0.5, X4.max()+0.5),
          color='black', alpha = .5)
plt.show()

##############################################################################
# creating a systems engineer and choosing a task for them
##############################################################################

# electrical engineering department scope (for reference)
# plt.hist(X3, bins=np.arange(X3.min()-0.5, X3.max()+0.5),
#           color='yellow', alpha = 0.5)

fig = plt.figure(figsize=(15,8))
# distribution of department scope
plt.hist(X1, bins=np.arange(X1.min()-0.5, X1.max()+0.5),
          color='blue', histtype='step')

# create an Mech employee's knowledge distribution
exp_level = 900
sys_know = np.random.normal(mu1, sigma1, exp_level).astype(int)
sys_know[sys_know>50] -= 100
sys_know[sys_know<-50] += 100
plt.hist(sys_know, bins=np.arange(X1.min()-0.5, X1.max()+0.5),
          color='red', alpha = .5)

task = np.random.choice(X1, 600).astype(int)
plt.hist(task, bins=np.arange(X1.min()-0.5, X1.max()+0.5),
          color='black', alpha = .5)
plt.grid()
plt.show()

##############################################################################
# how to tune mu, sigma, and N for each company department
##############################################################################
# only adjusting N
mu, sigma, N = 0, 5, 1000
X = np.random.normal(mu, sigma, N).astype(int)
plt.hist(X, bins=np.arange(X.min()-0.5, X.max()+0.5),
          color='blue', alpha = 0.25)

mu, sigma, N = 0, 5, 500
X = np.random.normal(mu, sigma, N).astype(int)
plt.hist(X, bins=np.arange(X.min()-0.5, X.max()+0.5),
          color='red', alpha = 0.25)

mu, sigma, N = 0, 5, 250
X = np.random.normal(mu, sigma, N).astype(int)
plt.hist(X, bins=np.arange(X.min()-0.5, X.max()+0.5),
          color='green', alpha = 0.25)

mu, sigma, N = 0, 5, 100
X = np.random.normal(mu, sigma, N).astype(int)
plt.hist(X, bins=np.arange(X.min()-0.5, X.max()+0.5),
          color='black', alpha = 0.25)

plt.show()

# only adjusting sigma
mu, sigma, N = 0, 1, 500
X = np.random.normal(mu, sigma, N).astype(int)
plt.hist(X, bins=np.arange(X.min()-0.5, X.max()+0.5),
          color='blue', alpha = 0.25)

mu, sigma, N = 0, 2, 1000
X = np.random.normal(mu, sigma, N).astype(int)
plt.hist(X, bins=np.arange(X.min()-0.5, X.max()+0.5),
          color='red', alpha = 0.25)

mu, sigma, N = 0, 4, 1000
X = np.random.normal(mu, sigma, N).astype(int)
plt.hist(X, bins=np.arange(X.min()-0.5, X.max()+0.5),
          color='green', alpha = 0.25)

mu, sigma, N = 0, 8, 1000
X = np.random.normal(mu, sigma, N).astype(int)
plt.hist(X, bins=np.arange(X.min()-0.5, X.max()+0.5),
          color='black', alpha = 0.25)

plt.show()

##############################################################################
# generate a model for an employee researching to learn new knowledge
##############################################################################

# generate a distribution of an employee's current knowledge
mu, sigma = 55, 6
X_curr = np.round(np.random.normal(mu, sigma, N)).astype(int)

# generate a normal distribution for learning to the target level
# learning to a deisred level is not a linear path. This is simulated as a
# normal distribution of knowledge centered around the target knowledge
steps = np.array([]).astype(int)
for i in range(10):
    X_tot = np.array([]).astype(int)
    mu4, sigma4 = 50, 1 # mu is target knowledge, sigma is randomness in learning
    knowledge_level = 10
    step_count = 0
    while True:
        X4 = np.round(np.random.normal(mu4, sigma4, 1)).astype(int)
        X_tot = np.concatenate([X_tot, X4])
        knowledge = np.count_nonzero(X_tot==mu4)
        if knowledge == knowledge_level:
            # plt.hist(X_tot, bins=np.arange(X_tot.min()-0.5, X_tot.max()+0.5), edgecolor='black')
            # plt.hist(X_tot, edgecolor='black')
            # plt.grid()
            # plt.show()

            plt.hist(X_curr, bins=np.arange(X_curr.min()-0.5, X_curr.max()+0.5),
                      color='red', alpha = 0.5, edgecolor='black')
            plt.grid()
            plt.hist(X_tot, bins=np.arange(X_tot.min()-0.5, X_tot.max()+0.5),
                      color='blue', alpha = 0.5, edgecolor='black')
            plt.show()

            # subtract the existing knowledge from the learning distribution
            X_learn = X_tot
            X_exist = X_curr

            ca = Counter(X_learn)
            cb = Counter(X_exist)

            to_learn = list((ca - cb).elements())
            # print(to_learn)

            step_count = len(to_learn)

            print("Knowledge: " + str(knowledge) + " Steps: " + str(step_count))
            steps = np.concatenate([steps, [step_count]])

            break

# on each step of the model, append one number from the to_learn distribution
# to the employee's knowledge distribution

plt.hist(steps, bins=range(steps.min(), steps.max() + 1), edgecolor='black')
plt.show()

'''
'''
# The model could be evaluated at the model level for
#    - The number of tasks completed company wide
#    - The complexity of tasks completed
#    - Growth in companey knowledge capability

# The model could be evaluated at the agent level for
#    - knowledge growth (number of values added to X)
#    - Knowledge diversity (range of X)

# Employee types

# - Company/Industry Lifer
#    - 30+ years of experience
#    - Knowledge follows a normal distribution centered around the company's knowledge

# - Employees with varying amounts of previous jobs (multimodal knowledge distribution)

# - New college hire

# Day in the life of employee
#    A. try to solve the currently assigned task
#    B. document a previously solved task
#    C. help another employee solve their task
#    D. if an expert is helping another employee, then the expert is not accomplishing a task
#    E. Experts can only help one employee at a time with their task
#    F. Employees can search for experts in the company and ask them for help if they are available


# Task definitions
#    - tasks are a list of numbers. The assigned employee must have all numbers
#      in their knowledge base to complete the task.

# Learning mechanisms
#    A. When an employee completes a task, a list of the unique numbers in the
#       task are added to the employee's knowledge base. Every knowledge category
#       required to complete the task increases by 1 for the employee' (I Disagree now)
#    B. If an employee is assigned a task that which they do not have the knowledge
#       to complete, then normal distribution sampling method.
#    C. If an employee asks an employee for help that has the knowledge, the needed
#       category can grow by one on each cycle. By asking for help, the employee
#       can get the needed information to solve their task but will misout on any
#       foundational knowledge. There is no knowledge pyramid supporting their knowledge

# When researching something new the learning process is not linear.
# In the process of acquiring the desired knowledge to solve the task
# you will also explore other areas that are not directly related to the solution.

# When given guidence from a how-to document or person, the learning process becomes
# linear


# knobs to turn
# 1. Distribution of possible company departments
# 2. Aggressivness of companies to assign tasks requiring new knowledge
# 2. Employee starting knowledge distribution
# 3. Employee rate of learning (the sigma in the knowledge acquistion distribution)

# model limitations
# 1. Employees in real life do not know exactly what other employees are experts in to ask for help


'''
