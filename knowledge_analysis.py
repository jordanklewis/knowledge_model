# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 11:13:50 2023

@author: Lewis
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data_path = "C:\\Users\\Lewis\\Documents\\data"
data_file = "complete_20230414_115442_NumEmp40_Steps180_Grid10_Seed0.csv"
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
for row, _ in enumerate(avail_arr):
    for col, _ in enumerate(busy_arr):
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
for row, _ in enumerate(avail_arr):
    for col, _ in enumerate(busy_arr):
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
for row, _ in enumerate(avail_arr):
    for col, _ in enumerate(busy_arr):
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
for row, _ in enumerate(avail_arr):
    for col, _ in enumerate(busy_arr):
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
plt.plot(model_data.total_knowledge[model_data.task_completed],
          model_data.needed_know[model_data.task_completed], '.',alpha=0.25)
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
