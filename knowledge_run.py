# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 21:01:14 2022

@author: Lewis
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from knowledge_model import KnowledgeModel

model = KnowledgeModel(20)

for i in range(3000):
    model.step()

model_data = pd.DataFrame(model.step_data)


# Data Analysis Objectives
# 1. Employee personal knowledge growth (hours until promotion)
# 2. Employee personal task performance (total completed tasks, task complexity)
# 3. Company knowledge growth (max knowledge in company, average knowledge in company)
# 4. Company task performance (total completed tasks, task complexity)

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
emp_ids = pd.unique(model_data.employee_id)
plt.figure(figsize=(10,6))
for i in emp_ids:
    comp_ndx = model_data.index[(model_data.employee_id==i) &
                                (model_data.task_completed==True)]
    s = model_data.step[comp_ndx]
    x = np.cumsum(model_data.task_complexity[comp_ndx])
    pro = model_data.employee_exp[comp_ndx].diff()==model.max_know/10
    plt.plot(s, x, label=i)
    plt.plot(s[pro], x[pro], '*')
# plt.legend()
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

# This plot demonstrates how the company knowledge is growing in knowledge over time
unique_steps, ndx = np.unique(model_data.step, return_index=True)
plt.figure(figsize=(10,6))
plt.plot(unique_steps, model_data.comp_know.iloc[ndx],
         color='red', alpha=1)
plt.ylabel('Company Knowledge')
plt.xlabel('Step')
plt.title('Company Knowledge Growth')
plt.grid()
plt.show()


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