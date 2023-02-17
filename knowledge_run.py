# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 21:01:14 2022

@author: Lewis
"""
import time
import datetime
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from knowledge_model import KnowledgeModel

total_time = time.time()

config_num = -1
model_data = []

# time duration of model should be 30% of max knowledge
# 30% of max knowledge should equate to 30% of a career
# career is assumed to be 50 years
# model should run for 15 years

# 4 time resolutions for 15 years- 30000 hours, 3750 Days, 750 Weeks, 180 Months
# assume 5 days in a week, 50 weeks in a year
# employees gain 1 unit of know per 1 unit of time
# corresponding max_know - 100,000, 12,500, 2,500, 600


steps = 180
max_know = int(steps/0.3)
know_cat_ct = int(max_know/10)

num_emp=10
# know_cat_ct=200
# max_know=2000
innov_rate=0.5
seed=0

for i in range(20, 100, 10):
    for w in range(20, 100, 10):
        if (i + w) > 100:
            continue
        config_num += 1
        avail = i/100
        busy = w/100
        docs = abs(round(1 - avail - busy, 2))
        print('Running Model: avail=' + str(avail)
              + ', busy=' + str(busy)
              + ', docs=' + str(docs))
        iter_time = time.time()
        model = KnowledgeModel(num_emp=num_emp,
                               avail=avail,
                               busy=busy,
                               know_cat_ct=know_cat_ct,
                               max_know=max_know,
                               innov_rate=innov_rate,
                               config_num = config_num,
                               seed=seed)

        for s in range(steps):
            # start_time = time.time()
            model.step()
            # print("%s Model Step" % int((time.time() - start_time)*1000))
        print("%.2f Sec Iter Time" % (time.time() - iter_time))
        print(datetime.datetime.now())
        model_data += model.step_data

model_data = pd.DataFrame(model_data)
print("%.2f Sec Total Time" % (time.time() - total_time))

comp_task_value = {}
for c in np.unique(model_data.config_num):
    keys = (list(model_data.avail[model_data.config_num==c])[-1],
            list(model_data.busy[model_data.config_num==c])[-1])
    comp_task_value[keys] = model_data.task_complexity[(model_data.task_completed is True)
                                & (model_data.config_num==c)].sum()

ser = pd.Series(list(comp_task_value.values()),
                  index=pd.MultiIndex.from_tuples(comp_task_value.keys()))
df = ser.unstack()

plt.figure(figsize=(10,6))
ax = sns.heatmap(df, annot=True, cmap='coolwarm', robust=True)
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

model.plot_company_know()
model.plot_company_know_subplots()

# company population pyramid
e = Counter(model.roster.Exp)
labels = [str(int(l)) for l in np.linspace(max_know/10,max_know,10)]
values = [e[int(l)] for l in np.linspace(max_know/10,max_know,10)]
plt.figure(figsize=(10,6))
plt.grid()
plt.barh(labels, values)
plt.ylabel('Experience Level')
plt.xlabel('Employee Count')
plt.title('Employee Populaiton Pyramid')
plt.show()

# create a step aggregate data frame
step_agg_df = pd.DataFrame([])
for c in np.unique(model_data.config_num):
    model_agg = []
    for s in np.unique(model_data.step):
        model_data_step = model_data[(model_data.step==s) &
                                    (model_data.config_num==c)]
        model_agg.append({'config_num': c,
                          'step': s,
                          'num_emp': model_data_step.num_emp.iloc[-1],
                          'avail': model_data_step.avail.iloc[-1],
                          'busy': model_data_step.busy.iloc[-1],
                          'docs': model_data_step.docs.iloc[-1],
                          'know_cat_ct': model_data_step.know_cat_ct.iloc[-1],
                          'max_know': model_data_step.max_know.iloc[-1],
                          'innov_rate': model_data_step.innov_rate.iloc[-1],
                          'seed': model_data_step.seed.iloc[-1],
                          'task_count': model_data_step.task_completed.sum(),
                          'task_value': model_data_step.task_complexity[model_data_step.task_completed is True].sum(),
                          'research_know': model_data_step.research_know.iloc[-1],
                          'learn_know': model_data_step.learn_know.iloc[-1],
                          'teach_know': model_data_step.teach_know.iloc[-1],
                          'doc_know': model_data_step.document_know.iloc[-1],
                          'read_know': model_data_step.read_know.iloc[-1],
                          'SE_dept_know': model_data_step.SE_dept_know.iloc[-1],
                          'SW_dept_know': model_data_step.SW_dept_know.iloc[-1],
                          'EE_dept_know': model_data_step.EE_dept_know.iloc[-1],
                          'ME_dept_know': model_data_step.ME_dept_know.iloc[-1],
                          'comp_lib_know': model_data_step.comp_lib_know.iloc[-1],
                          'comp_know': model_data_step.comp_know.iloc[-1]
                          })
    model_agg = pd.DataFrame(model_agg)
    step_agg_df = pd.concat([step_agg_df, model_agg], ignore_index=True)

plt.figure(figsize=(10,6))
plt.grid()
for c in np.unique(step_agg_df.config_num):
    config_filt = step_agg_df[step_agg_df.config_num==c]
    plt.plot(config_filt.step, config_filt.comp_lib_know,
             label = "A:%s W:%s D: %s" % (config_filt.avail.iloc[-1],
                                          config_filt.busy.iloc[-1],
                                          config_filt.docs.iloc[-1]))
plt.title('Company Library Knowledge\n'
          + 'Num Emp: ' + str(step_agg_df.num_emp.iloc[-1])
          + '    Know Cat Ct: ' + str(step_agg_df.know_cat_ct.iloc[-1])
          + '    Max_Know: ' + str(step_agg_df.max_know.iloc[-1])
          + '    Innov Rate: ' + str(step_agg_df.innov_rate.iloc[-1])
          + '    Seed: ' + str(step_agg_df.seed.iloc[-1]))
plt.legend()
plt.xlabel('Step')
plt.ylabel('Knowledge Quantity')
plt.show()

plt.figure(figsize=(10,6))
plt.grid()
for c in np.unique(step_agg_df.config_num):
    config_filt = step_agg_df[step_agg_df.config_num==c]
    plt.plot(config_filt.step, config_filt.comp_know,
             label = "A:%s W:%s D: %s" % (config_filt.avail.iloc[-1],
                                          config_filt.busy.iloc[-1],
                                          config_filt.docs.iloc[-1]))
plt.title('Company Knowledge\n'
          + 'Num Emp: ' + str(step_agg_df.num_emp.iloc[-1])
          + '    Know Cat Ct: ' + str(step_agg_df.know_cat_ct.iloc[-1])
          + '    Max_Know: ' + str(step_agg_df.max_know.iloc[-1])
          + '    Innov Rate: ' + str(step_agg_df.innov_rate.iloc[-1])
          + '    Seed: ' + str(step_agg_df.seed.iloc[-1]))
plt.legend()
plt.xlabel('Step')
plt.ylabel('Knowledge Quantity')
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
                          'avail': model_data_emp.avail.iloc[-1],
                          'busy': model_data_emp.busy.iloc[-1],
                          'docs': model_data_emp.docs.iloc[-1],
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
                          'task_value': model_data_emp.task_complexity[model_data_emp.task_completed is True].sum(),
                          'research_know': model_data_emp.research_know.iloc[-1],
                          'learn_know': model_data_emp.learn_know.iloc[-1],
                          'teach_know': model_data_emp.teach_know.iloc[-1],
                          'doc_know': model_data_emp.document_know.iloc[-1],
                          'read_know': model_data_emp.read_know.iloc[-1]
                          })
    model_agg = pd.DataFrame(model_agg).sort_values(['start_know'])
    emp_agg_df = pd.concat([emp_agg_df, model_agg], ignore_index=True)

plt.figure(figsize=(10,6))
plt.grid()
for c in np.unique(emp_agg_df.config_num):
    config_filt = emp_agg_df[emp_agg_df.config_num==c]
    plt.plot(config_filt.start_know, config_filt.know_growth,
             label = "A:%s W:%s D: %s" % (config_filt.avail.iloc[-1],
                                          config_filt.busy.iloc[-1],
                                          config_filt.docs.iloc[-1]))
plt.title('Employee Knowledge Growth\n'
          + 'Num Emp: ' + str(emp_agg_df.num_emp.iloc[-1])
          + '    Know Cat Ct: ' + str(emp_agg_df.know_cat_ct.iloc[-1])
          + '    Max_Know: ' + str(emp_agg_df.max_know.iloc[-1])
          + '    Innov Rate: ' + str(emp_agg_df.innov_rate.iloc[-1])
          + '    Seed: ' + str(emp_agg_df.seed.iloc[-1]))
plt.legend()
plt.xlabel('Start Know Quantity')
plt.ylabel('Know Growth Quantity')
plt.show()

plt.figure(figsize=(10,6))
plt.grid()
for c in np.unique(emp_agg_df.config_num):
    config_filt = emp_agg_df[emp_agg_df.config_num==c]
    plt.plot(config_filt.start_know, config_filt.task_value,
             label = "A:%s W:%s D: %s" % (config_filt.avail.iloc[-1],
                                          config_filt.busy.iloc[-1],
                                          config_filt.docs.iloc[-1]))
plt.title('Employee Task Value\n'
          + 'Num Emp: ' + str(emp_agg_df.num_emp.iloc[-1])
          + '    Know Cat Ct: ' + str(emp_agg_df.know_cat_ct.iloc[-1])
          + '    Max_Know: ' + str(emp_agg_df.max_know.iloc[-1])
          + '    Innov Rate: ' + str(emp_agg_df.innov_rate.iloc[-1])
          + '    Seed: ' + str(emp_agg_df.seed.iloc[-1]))
plt.legend()
plt.xlabel('Start Know Quantity')
plt.ylabel('Task Value')
plt.show()

plt.figure(figsize=(10,6))
plt.grid()
for c in np.unique(emp_agg_df.config_num):
    config_filt = emp_agg_df[emp_agg_df.config_num==c]
    plt.plot(config_filt.start_know, config_filt.task_count,
             label = "A:%s W:%s D: %s" % (config_filt.avail.iloc[-1],
                                          config_filt.busy.iloc[-1],
                                          config_filt.docs.iloc[-1]))
plt.title('Employee Task Count\n'
          + 'Num Emp: ' + str(emp_agg_df.num_emp.iloc[-1])
          + '    Know Cat Ct: ' + str(emp_agg_df.know_cat_ct.iloc[-1])
          + '    Max_Know: ' + str(emp_agg_df.max_know.iloc[-1])
          + '    Innov Rate: ' + str(emp_agg_df.innov_rate.iloc[-1])
          + '    Seed: ' + str(emp_agg_df.seed.iloc[-1]))
plt.legend()
plt.xlabel('Start Know Quantity')
plt.ylabel('Task Count')
plt.show()


plt.figure(figsize=(10,6))
plt.grid()
for c in np.unique(emp_agg_df.config_num):
    config_filt = emp_agg_df[emp_agg_df.config_num==c]
    plt.plot(config_filt.start_know, config_filt.task_count,
             label = "A:%s W:%s D: %s" % (config_filt.avail.iloc[-1],
                                          config_filt.busy.iloc[-1],
                                          config_filt.docs.iloc[-1]))
plt.title('Employee Task Count\n'
          + 'Num Emp: ' + str(emp_agg_df.num_emp.iloc[-1])
          + '    Know Cat Ct: ' + str(emp_agg_df.know_cat_ct.iloc[-1])
          + '    Max_Know: ' + str(emp_agg_df.max_know.iloc[-1])
          + '    Innov Rate: ' + str(emp_agg_df.innov_rate.iloc[-1])
          + '    Seed: ' + str(emp_agg_df.seed.iloc[-1]))
plt.legend()
plt.xlabel('Start Know Quantity')
plt.ylabel('Task Count')
plt.show()


'''

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
