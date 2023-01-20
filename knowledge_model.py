# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 20:55:49 2022

@author: Lewis
"""
from collections import Counter
import mesa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class EmployeeAgent(mesa.Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, unique_id, exp, name, dept, know, model):
        super().__init__(unique_id, model)
        self.exp = exp
        self.name = name
        self.dept = dept
        self.emp_know = know
        self.task_num = 0
        self.model = model
        self.busy = False
        self.task = Counter()
        self.emp_know_to_learn = Counter()
        self.task_learn_path = Counter()

        self.get_new_task()

    def get_new_task(self):
        task_level = int(self.exp * self.model.innovation_rate)
        task = np.random.normal(self.model.dept_know[self.dept]['mu'],
                                   self.model.dept_know[self.dept]['sigma'],
                                   task_level).astype(int)
        task[task>=self.model.know_cat_ct/2] -= self.model.know_cat_ct
        task[task<-self.model.know_cat_ct/2] += self.model.know_cat_ct
        self.task = Counter(task)
        # the difference betwen the employee's task and knowledge is what they need to learn
        self.emp_know_to_learn = self.task - self.emp_know
        self.generate_learning_path(plot=True)
        self.plot_emp_know_post_task()

    def plot_employee_task(self):
        # plot employee task
        plt.figure(figsize=(10,6))
        plt.bar(self.emp_know.keys(),
                 self.emp_know.values(),
                  color='red', alpha =0.5, label='Employee')
        plt.bar(self.task.keys(),
                 self.task.values(),
                  color='black', alpha =0.5, label='Task')
        plt.step(self.model.know_cats,
                 [self.model.comp_know[n] for n in self.model.know_cats],
                  color='black', alpha =1, label='Company', where='mid')
        plt.step(self.model.know_cats,
                 [self.model.dept_know[self.dept]['dist'][n] for n in self.model.know_cats],
                  color='blue', alpha =1, label='Dept', where='mid')
        plt.legend()
        plt.title(self.name + ' [' + self.dept
                  + ' Exp: ' + str(self.exp) + ']\n'
                  + 'Needed Knowledge: ' + str(len(list(self.emp_know_to_learn()))))
        plt.xlim([min(self.model.know_cats)-0.5, max(self.model.know_cats)+0.5])
        plt.minorticks_on()
        plt.xticks(np.arange(min(self.model.know_cats), max(self.model.know_cats)+1, 5))
        plt.tick_params(axis='x', which='minor', length=5, width=1)
        plt.tick_params(axis='x', which='major', length=7, width=2)
        plt.xlabel('Knowledge Categories')
        plt.ylabel('Knowledge Quantity')
        plt.show()

    def generate_learning_path(self, plot=False):
        # generate the learning path for the employee to find all needed knowledge without help

        # generate a normal distribution for learning to the target level
        # learning to a deisred level is not a linear path. This is simulated as a
        # normal distribution of knowledge centered around the target knowledge
        # Later, existing knowledge can be subtracted from the normal distribution
        # to speed up the learning process because the employee already has existing
        # backgrond knowledge.
        self.task_learn_path = Counter()
        for cat in self.emp_know_to_learn.keys(): # iterate all needed know cats
            new_learn = Counter()
            mu_learn = cat # mu is target knowledge
            sigma_learn = 1 # sigma is randomness in learning
            needed_know = self.task[cat] # amount of knowledge needed for the task
            while True: # while desired knowledge has not been attained...
                rand_learn = np.random.normal(mu_learn, sigma_learn, 1).astype(int)
                new_learn[rand_learn.item()] += 1
                if new_learn[cat] == needed_know: # then knowledge has been attained
                    # subtract the existing knowledge from the learning distribution
                    # new knowledge is knowledge that you didn't already know
                    # and knowledge that you did't just find out on the learning path
                    c_n_k = new_learn - self.emp_know - self.task_learn_path # counter new knowledge

                    # add new knowledge to the learning path
                    self.task_learn_path = c_n_k + self.task_learn_path

                    if plot:
                        # plot employee task
                        plt.figure(figsize=(10,6))
                        plt.step(self.model.know_cats,
                                [self.model.dept_know[self.dept]['dist'][n]
                                 for n in self.model.know_cats],
                                  color='red', alpha=1, label='SE Dept', where='mid')
                        plt.bar(self.emp_know.keys(),
                                 self.emp_know.values(),
                                 color='blue', alpha =0.25, label='Employee', width=1)
                        plt.step(self.model.know_cats,
                                [self.task[n] for n in self.model.know_cats],
                                  color='black', alpha=1, label='Task', where='mid')
                        plt.bar(c_n_k.keys(),
                                c_n_k.values(),
                                 color='green', alpha =0.75, label='Know Post Task',
                                 bottom=[self.emp_know[n] for n in c_n_k.keys()])
                        plt.legend()
                        plt.title(self.name + ' [' + self.dept
                                  + ' Exp: ' + str(self.exp) + ']\n'
                                  + 'Target Category: ' + str(cat)
                                  + '    Target Level: ' + str(needed_know)
                                  + '    Current Level: ' + str(self.emp_know[cat]))
                        plt.xlim([min(self.model.know_cats)-0.5, max(self.model.know_cats)+0.5])
                        plt.minorticks_on()
                        plt.xticks(np.arange(min(self.model.know_cats),
                                             max(self.model.know_cats)+1, 5))
                        plt.tick_params(axis='x', which='minor', length=5, width=1)
                        plt.tick_params(axis='x', which='major', length=7, width=2)
                        plt.xlabel('Knowledge Categories')
                        plt.ylabel('Knowledge Quantity')
                        plt.grid()
                        plt.show()

                    break

    def plot_emp_know_post_task(self):
        plt.figure(figsize=(10,6))
        # stacked bar chart?
        plt.bar(self.emp_know.keys(),
                self.emp_know.values(),
                color='blue', alpha =0.25, label='Know Pre Task')
        plt.bar(self.task_learn_path.keys(),
                self.task_learn_path.values(),
                 color='green', alpha =0.75, label='Know Post Task',
                 bottom=[self.emp_know[n] for n in self.task_learn_path.keys()])
        plt.step(self.model.know_cats,
                [self.task[n] for n in self.model.know_cats],
                  color='black', alpha=1, label='Task', where='mid')
        plt.step(self.model.know_cats,
                 [self.model.dept_know[self.dept]['dist'][n] for n in self.model.know_cats],
                  color='red', alpha =1, label=str(self.dept)+' Dept', where='mid')
        plt.step(self.model.know_cats,
                 [self.model.comp_know[n] for n in self.model.know_cats],
                  'k--', alpha =1, label='Company', where='mid')
        plt.legend()
        plt.xlim([min(self.model.know_cats)-0.5, max(self.model.know_cats)+0.5])
        plt.minorticks_on()
        plt.xticks(np.arange(min(self.model.know_cats), max(self.model.know_cats)+1, 5))
        plt.tick_params(axis='x', which='minor', length=5, width=1)
        plt.tick_params(axis='x', which='major', length=7, width=2)
        plt.title(self.name + ' [' + self.dept
                  + ' Exp: ' + str(self.exp) + ']\n'
                  + 'Needed Knowledge: ' + str(len(list(self.emp_know_to_learn.elements())))
                  + '    Gained Knowledge: ' + str(len(list(self.task_learn_path.elements()))))
        plt.xlabel('Knowledge Categories')
        plt.ylabel('Knowledge Quantity')
        plt.grid()
        plt.show()

    def step(self):
        # The agent's step will go here.
        print("Agent: " + str(self.unique_id)
              + ", Knowledge: " + str(self.emp_know)
              + ", ActiveTask: " + str(self.task_num))


class KnowledgeModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, N):
        # company parameters
        self.innovation_rate = 0.5 # value must be betwen 0 and 1
        self.know_cat_ct = 100
        self.know_cats = list(np.arange(self.know_cat_ct).astype(int)
                              - int(self.know_cat_ct/2))
        self.max_know = 1000
        self.num_employees = N
        self.task_num = 0
        self.task_dict = {}
        self.roster = []

        # department descriptive statistics
        self.dept_know = {'SE': {'mu': 0, 'sigma': 30},
                     'SW': {'mu': 25, 'sigma': 20},
                     'EE': {'mu': 50, 'sigma': 20},
                     'ME': {'mu': -25, 'sigma': 20}}

        # model parameters
        self.schedule = mesa.time.RandomActivation(self)
        self.running = True

        # create company knowledge
        self.create_company_know_dist()
        self.plot_company_know()

        # generate company employees
        name_list = self.create_employee_name_list()
        for i in range(self.num_employees):

            # assign employee a name
            name = name_list.pop()

            # assign employee experience
            exp = self.get_employee_experience()

            # assign employee a department
            dept = np.random.choice(list(self.dept_know.keys()))

            # generate employee's knowledge distribution
            know = self.get_employee_knowledge_dist(dept, exp)

            # add employee to the company roster for later reference
            self.roster.append({'Name': name, 'Dept': dept, 'Exp': exp})

            # create employee agent
            agent = EmployeeAgent(i, exp, name, dept, know, self)
            self.schedule.add(agent)

        self.roster = pd.DataFrame(self.roster)

        self.datacollector = mesa.DataCollector(
            agent_reporters={"Knowledge": "knowledge"})

    # create knowledge distribution for each company department
    def create_dept_know_dist(self):
        # systems engineering department distribution
        se_know = np.random.normal(self.dept_know['SE']['mu'],
                                   self.dept_know['SE']['sigma'],
                                   self.max_know).astype(int)
        # wrap to +/- self.know_cat_ct/2 to create department overlap
        se_know[se_know>=self.know_cat_ct/2] -= self.know_cat_ct
        se_know[se_know<-self.know_cat_ct/2] += self.know_cat_ct
        self.dept_know['SE']['dist'] = Counter(se_know)


        # software engineering department distribution
        sw_know = np.random.normal(self.dept_know['SW']['mu'],
                                   self.dept_know['SW']['sigma'],
                                   self.max_know).astype(int)
        # wrap to +/- self.know_cat_ct/2 for department overlap
        sw_know[sw_know>=self.know_cat_ct/2] -= self.know_cat_ct
        sw_know[sw_know<-self.know_cat_ct/2] += self.know_cat_ct
        self.dept_know['SW']['dist'] = Counter(sw_know)

        # electrical engineering department distribution
        ee_know = np.random.normal(self.dept_know['EE']['mu'],
                                   self.dept_know['EE']['sigma'],
                                   self.max_know).astype(int)
        # wrap to +/- self.know_cat_ct/2 for department overlap
        ee_know[ee_know>=self.know_cat_ct/2] -= self.know_cat_ct
        ee_know[ee_know<-self.know_cat_ct/2] += self.know_cat_ct
        self.dept_know['EE']['dist'] = Counter(ee_know)

        # mechanical engineering department distribution
        me_know = np.random.normal(self.dept_know['ME']['mu'],
                                   self.dept_know['ME']['sigma'],
                                   self.max_know).astype(int)
        # wrap to +/- self.know_cat_ct/2 for department overlap
        me_know[me_know>=self.know_cat_ct/2] -= self.know_cat_ct
        me_know[me_know<-self.know_cat_ct/2] += self.know_cat_ct
        self.dept_know['ME']['dist'] = Counter(me_know)

    def create_company_know_dist(self):
        # create department knowledge
        self.create_dept_know_dist()

        # the company knowledge is defined as the maximum knowledge for a knowledge
        # category in any department
        self.comp_know = {}
        for cat in self.know_cats:
            self.comp_know[cat] = max([self.dept_know['SE']['dist'][cat],
                                       self.dept_know['SW']['dist'][cat],
                                       self.dept_know['EE']['dist'][cat],
                                       self.dept_know['ME']['dist'][cat]])
        self.comp_know = Counter(self.comp_know)

    def plot_company_know(self):
        # plot company knowledge for each department
        plt.figure(figsize=(10,6))
        plt.bar(self.dept_know['SE']['dist'].keys(),
                self.dept_know['SE']['dist'].values(),
                  color='red', alpha=1, label='SE', width=1)
        plt.bar(self.dept_know['SW']['dist'].keys(),
                self.dept_know['SW']['dist'].values(),
                  color='green', alpha=0.5, label='SW', width=1)
        plt.bar(self.dept_know['EE']['dist'].keys(),
                self.dept_know['EE']['dist'].values(),
                  color='yellow', alpha=0.5, label='EE', width=1)
        plt.bar(self.dept_know['ME']['dist'].keys(),
                self.dept_know['ME']['dist'].values(),
                  color='blue', alpha=0.25, label='ME', width=1)
        plt.xlim([min(self.know_cats)-0.5, max(self.know_cats)+0.5])
        plt.minorticks_on()
        plt.xticks(np.arange(min(self.know_cats), max(self.know_cats)+1, 5))
        plt.tick_params(axis='x', which='minor', length=5, width=1)
        plt.tick_params(axis='x', which='major', length=7, width=2)
        plt.legend(loc='upper right')
        plt.title('Company Knowledge by Department')
        plt.xlabel('Knowledge Categories')
        plt.ylabel('Knowledge Quantity')
        plt.show()

    def plot_company_know_subplots(self):
        # plot distributions on subplots
        _,ax = plt.subplots(2, 2, figsize=(10,6))
        ax[0][0].bar(self.dept_know['SE']['dist'].keys(),
                self.dept_know['SE']['dist'].values(),
                  color='red', alpha=1, label='SE', width=1)
        ax[0][0].set_title('Systems Enginereing')
        ax[0][0].set_xlabel('Knowledge Category')
        ax[0][0].set_ylabel('Knowledge Quantity')
        ax[0][0].set_xlim([min(self.know_cats)-0.5, max(self.know_cats)+0.5])
        ax[0][0].set_xticks(np.arange(min(self.know_cats), max(self.know_cats)+1, 10))
        ax[0][0].minorticks_on()
        ax[0][0].tick_params(axis='x', which='minor', length=5, width=1)
        ax[0][0].tick_params(axis='x', which='major', length=7, width=2)

        ax[0][1].bar(self.dept_know['SW']['dist'].keys(),
                self.dept_know['SW']['dist'].values(),
                  color='green', alpha=0.5, label='SE', width=1)
        ax[0][1].set_title('Software Enginereing')
        ax[0][1].set_xlabel('Knowledge Category')
        ax[0][1].set_ylabel('Knowledge Quantity')
        ax[0][1].set_xlim([min(self.know_cats)-0.5, max(self.know_cats)+0.5])
        ax[0][1].set_xticks(np.arange(min(self.know_cats), max(self.know_cats)+1, 10))
        ax[0][1].minorticks_on()
        ax[0][1].tick_params(axis='x', which='minor', length=5, width=1)
        ax[0][1].tick_params(axis='x', which='major', length=7, width=2)

        ax[1][0].bar(self.dept_know['EE']['dist'].keys(),
                self.dept_know['EE']['dist'].values(),
                  color='yellow', alpha=0.5, label='SE', width=1)
        ax[1][0].set_title('Electrical Enginereing')
        ax[1][0].set_xlabel('Knowledge Category')
        ax[1][0].set_ylabel('Knowledge Quantity')
        ax[1][0].set_xlim([min(self.know_cats)-0.5, max(self.know_cats)+0.5])
        ax[1][0].set_xticks(np.arange(min(self.know_cats), max(self.know_cats)+1, 10))
        ax[1][0].minorticks_on()
        ax[1][0].tick_params(axis='x', which='minor', length=5, width=1)
        ax[1][0].tick_params(axis='x', which='major', length=7, width=2)

        ax[1][1].bar(self.dept_know['ME']['dist'].keys(),
                self.dept_know['ME']['dist'].values(),
                  color='blue', alpha=0.25, label='SE', width=1)
        ax[1][1].set_title('Mechanical Enginereing')
        ax[1][1].set_xlabel('Knowledge Category')
        ax[1][1].set_ylabel('Knowledge Quantity')
        ax[1][1].set_xlim([min(self.know_cats)-0.5, max(self.know_cats)+0.5])
        ax[1][1].set_xticks(np.arange(min(self.know_cats), max(self.know_cats)+1, 10))
        ax[1][1].minorticks_on()
        ax[1][1].tick_params(axis='x', which='minor', length=5, width=1)
        ax[1][1].tick_params(axis='x', which='major', length=7, width=2)

        plt.tight_layout()
        plt.show()

    def create_employee_name_list(self):
        # Create employee names
        name_list = pd.read_csv("employee_names.csv")
        name_list = name_list["Names"].tolist()
        return list(np.random.choice(name_list,
                                     self.num_employees, replace=False))

    def get_employee_experience(self):
        return (np.random.choice(9, p=[.1, .1, .14, .2,
                                      .2, .1, .1, .05, .01])+1)*100

    def get_employee_knowledge_dist(self, dept, exp):
        know = np.random.normal(self.dept_know[dept]['mu'],
                                self.dept_know[dept]['sigma'],
                                exp).astype(int)
        know[know>=self.know_cat_ct/2] -= self.know_cat_ct
        know[know<-self.know_cat_ct/2] += self.know_cat_ct
        return Counter(know)

    def step(self):
        """Advance the model by one step."""
        self.task_dict[self.task_num] = [2,3,7,8]
        self.task_num += 1
        self.datacollector.collect(self)
        self.schedule.step()
