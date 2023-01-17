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
        self.knowledge = know
        self.task_num = 0
        self.busy = False

        task_level = int(self.exp * model.innovation_rate)
        self.task = np.random.normal(model.dept_know[self.dept]['mu'],
                                   model.dept_know[self.dept]['sigma'],
                                   task_level).astype(int)
        self.task[self.task>=model.know_cat_ct/2] -= model.know_cat_ct
        self.task[self.task<-model.know_cat_ct/2] += model.know_cat_ct

        c_t = Counter(self.task)            # counter task
        c_k = Counter(self.knowledge)       # counter knowledge
        c_t_l = c_t - c_k                   # counter to learn

        # plot employee task
        plt.figure(figsize=(10,6))
        plt.hist(self.knowledge,
                 bins=np.arange(model.comp_know.min()-0.5,
                                model.comp_know.max()+0.5),
                  color='red', alpha =0.5, label='Employee')
        plt.hist(self.task,
                 bins=np.arange(model.comp_know.min()-0.5,
                                model.comp_know.max()+0.5),
                  color='black', alpha =0.5, label='Task')
        plt.hist(model.comp_know,
                 bins=np.arange(model.comp_know.min()-0.5,
                                model.comp_know.max()+0.5),
                  color='black', alpha =1, label='Company', histtype='step')
        plt.hist(model.dept_know[self.dept]['dist'],
                 bins=np.arange(model.comp_know.min()-0.5,
                                model.comp_know.max()+0.5),
                  color='blue', alpha =1, label='Dept', histtype='step')
        plt.legend()
        plt.title(self.name + ' [' + self.dept
                  + ' Exp: ' + str(self.exp) + ']\n'
                  + 'Needed Knowledge: ' + str(len(list(c_t_l.elements()))))
        plt.xlabel('Knowledge Categories')
        plt.ylabel('Knowledge Quantity')
        plt.show()
        
        # generate the learning path for the employee to find all needed knowledge without help

        # generate a normal distribution for learning to the target level
        # learning to a deisred level is not a linear path. This is simulated as a 
        # normal distribution of knowledge centered around the target knowledge
        c_learn_path = Counter()
        for cat in c_t_l.keys():
            new_learn = np.array([]).astype(int)
            mu_learn = cat # mu is target knowledge
            sigma_learn = 1 # sigma is randomness in learning
            needed_know = c_t[cat]
            while True: # while desired knowledge has not been attained...
                rand_learn = np.random.normal(mu_learn, sigma_learn, 1).astype(int)
                new_learn = np.concatenate([new_learn, rand_learn])
                learned_know = np.count_nonzero(new_learn==mu_learn)
                if learned_know == needed_know: # then knowledge has been attained
                    # plot employee task
                    # plt.figure(figsize=(10,6))
                    # plt.hist(self.knowledge,
                    #          bins=np.arange(model.comp_know.min()-0.5,
                    #                         model.comp_know.max()+0.5),
                    #          color='red', alpha =0.5, label='Employee')
                    # plt.hist(self.task,
                    #           bins=np.arange(model.comp_know.min()-0.5,
                    #                         model.comp_know.max()+0.5),
                    #           color='black', alpha=1, label='Task', histtype='step')
                    # plt.hist(new_learn,
                    #          bins=np.arange(model.comp_know.min()-0.5,
                    #                         model.comp_know.max()+0.5), 
                    #          color='blue', alpha = 0.5, edgecolor='black',
                    #          label='Research')
                    # plt.legend()
                    # plt.title(self.name + ' [' + self.dept
                    #           + ' Exp: ' + str(self.exp) + ']\n'
                    #           + 'Target Research: ' + str(cat)
                    #           + '    Target Level: ' + str(needed_know))
                    # plt.xlabel('Knowledge Categories')
                    # plt.ylabel('Knowledge Quantity')
                    # plt.grid()
                    # plt.show()
            
                    # subtract the existing knowledge from the learning distribution
                    c_l = Counter(new_learn) # counter learn

                    # new knowledge is knowledge that you didn't already know
                    # and knowledge that you did't just find out on the learning path
                    c_n_k = c_l - c_k - c_learn_path # counter new knowledge

                    # add new knowledge to the learning path
                    c_learn_path = c_n_k + c_learn_path
                    break
        
        plt.figure(figsize=(10,6))
        plt.hist(list((c_k + c_learn_path).elements()),
                 bins=np.arange(model.comp_know.min()-0.5,
                                model.comp_know.max()+0.5),
                 color='red', alpha =0.5, label='Know Post Task')
        plt.hist(self.knowledge,
                 bins=np.arange(model.comp_know.min()-0.5,
                                model.comp_know.max()+0.5),
                 color='blue', alpha =0.5, label='Know Pre Task')
        plt.hist(self.task,
                  bins=np.arange(model.comp_know.min()-0.5,
                                model.comp_know.max()+0.5),
                  color='black', alpha=1, label='Task', histtype='step')
        plt.legend()
        plt.title(self.name + ' [' + self.dept
                  + ' Exp: ' + str(self.exp) + ']\n'
                  + 'Needed Knowledge: ' + str(len(list(c_t_l.elements())))
                  + '    Gained Knowledge: ' + str(len(list(c_learn_path.elements()))))
        plt.xlabel('Knowledge Categories')
        plt.ylabel('Knowledge Quantity')
        plt.grid()
        plt.show()
        x = 5

    def step(self):
        # The agent's step will go here.
        print("Agent: " + str(self.unique_id)
              + ", Knowledge: " + str(self.knowledge)
              + ", ActiveTask: " + str(self.task_num))


class KnowledgeModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, N):
        # company parameters
        self.innovation_rate = 0.5 # value must be betwen 0 and 1
        self.know_cat_ct = 100
        self.know_cats = list(np.arange(self.know_cat_ct).astype(int)
                              - int(self.know_cat_ct/2))
        self.num_employees = N
        self.task_num = 0
        self.task_dict = {}
        self.roster = []

        # model parameters
        self.schedule = mesa.time.RandomActivation(self)
        self.running = True

        # create knowledge distribution for each company department
        max_know = 1000

        # department descriptive statistics
        self.dept_know = {'SE': {'mu': 0, 'sigma': 30},
                     'SW': {'mu': 25, 'sigma': 20},
                     'EE': {'mu': 50, 'sigma': 20},
                     'ME': {'mu': -25, 'sigma': 20}}

        # systems engineering department distribution
        self.dept_know['SE']['dist'] = np.random.normal(self.dept_know['SE']['mu'],
                                   self.dept_know['SE']['sigma'],
                                   max_know).astype(int)
        # wrap to +/- self.know_cat_ct/2 to create department overlap
        self.dept_know['SE']['dist'][self.dept_know['SE']['dist']
                                     >=self.know_cat_ct/2] -= self.know_cat_ct
        self.dept_know['SE']['dist'][self.dept_know['SE']['dist']
                                     <-self.know_cat_ct/2] += self.know_cat_ct

        # software engineering department distribution
        self.dept_know['SW']['dist'] = np.random.normal(self.dept_know['SW']['mu'],
                                   self.dept_know['SW']['sigma'],
                                   max_know).astype(int)
        # wrap to +/- self.know_cat_ct/2 for department overlap
        self.dept_know['SW']['dist'][self.dept_know['SW']['dist']
                                     >=self.know_cat_ct/2] -= self.know_cat_ct
        self.dept_know['SW']['dist'][self.dept_know['SW']['dist']
                                     <-self.know_cat_ct/2] += self.know_cat_ct

        # electrical engineering department distribution
        self.dept_know['EE']['dist'] = np.random.normal(self.dept_know['EE']['mu'],
                                   self.dept_know['EE']['sigma'],
                                   max_know).astype(int)
        # wrap to +/- self.know_cat_ct/2 for department overlap
        self.dept_know['EE']['dist'][self.dept_know['EE']['dist']
                                     >=self.know_cat_ct/2] -= self.know_cat_ct
        self.dept_know['EE']['dist'][self.dept_know['EE']['dist']
                                     <-self.know_cat_ct/2] += self.know_cat_ct

        # mechanical engineering department distribution
        self.dept_know['ME']['dist'] = np.random.normal(self.dept_know['ME']['mu'],
                                   self.dept_know['ME']['sigma'],
                                   max_know).astype(int)
        # wrap to +/- self.know_cat_ct/2 for department overlap
        self.dept_know['ME']['dist'][self.dept_know['ME']['dist']
                                     >=self.know_cat_ct/2] -= self.know_cat_ct
        self.dept_know['ME']['dist'][self.dept_know['ME']['dist']
                                     <-self.know_cat_ct/2] += self.know_cat_ct

        # database of all company knowledge
        se_ct = Counter(self.dept_know['SE']['dist'])
        sw_ct = Counter(self.dept_know['SW']['dist'])
        ee_ct = Counter(self.dept_know['EE']['dist'])
        me_ct = Counter(self.dept_know['ME']['dist'])

        # the company knowledge is defined as the maximum knowledge for a knowledge
        # category in any department
        self.comp_know = {}
        for n in self.know_cats:
            print(str(n) + ': ' + str(se_ct[n]) + ', ' 
                  + str(sw_ct[n]) + ', ' 
                  + str(ee_ct[n])  + ', ' 
                  + str(me_ct[n]))
            self.comp_know[n] = max([se_ct[n], sw_ct[n], ee_ct[n], me_ct[n]])

        self.comp_know = Counter(self.comp_know)
        self.comp_know = np.array(list(self.comp_know.elements()))

        # plot company knowledge for each department
        plt.figure(figsize=(10,6))
        plt.hist(self.dept_know['SE']['dist'],
                 bins=np.arange(self.comp_know.min()-0.5,
                                self.comp_know.max()+0.5),
                  color='red', alpha=1, label='SE')
        plt.hist(self.dept_know['SW']['dist'],
                 bins=np.arange(self.comp_know.min()-0.5,
                                self.comp_know.max()+0.5),
                  color='green', alpha=0.5, label='SW')
        plt.hist(self.dept_know['EE']['dist'],
                 bins=np.arange(self.comp_know.min()-0.5,
                                self.comp_know.max()+0.5),
                  color='yellow', alpha=0.5, label='EE')
        plt.hist(self.dept_know['ME']['dist'],
                 bins=np.arange(self.comp_know.min()-0.5,
                                self.comp_know.max()+0.5),
                  color='blue', alpha=0.25, label='ME')
        plt.legend(loc='upper right')
        plt.title('Company Knowledge by Department')
        plt.xlabel('Knowledge Categories')
        plt.ylabel('Knowledge Quantity')
        plt.show()
        
        # plot company knowledge for all departments
        # plt.hist(self.comp_know,
        #          bins=np.arange(self.comp_know.min()-0.5,
        #                         self.comp_know.max()+0.5),
        #           color='red', alpha=1, label='ALL')
        # plt.legend(loc='upper right')
        # plt.show()

        # Create employee names
        name_list = pd.read_csv("employee_names.csv")
        name_list = name_list["Names"].tolist()
        name_list = list(np.random.choice(name_list,
                                          self.num_employees, replace=False))

        for i in range(self.num_employees):
            # assign employee experience
            exp = (np.random.choice(9, p=[.1, .1, .14, .2,
                                          .2, .1, .1, .05, .01])+1)*100

            # assign employee a name
            name = name_list.pop()

            # assign employee a department
            dept = np.random.choice(list(self.dept_know.keys()))

            # generate employee's knowledge distribution
            know = np.random.normal(self.dept_know[dept]['mu'],
                                    self.dept_know[dept]['sigma'],
                                    exp).astype(int)
            know[know>=self.know_cat_ct/2] -= self.know_cat_ct
            know[know<-self.know_cat_ct/2] += self.know_cat_ct

            # add employee to the company roster for later reference
            self.roster.append({'Name': name, 'Dept': dept, 'Exp': exp})

            # plot employee's knowledge distribution
            # plt.hist(know, bins=np.arange(know.min()-0.5, know.max()+0.5),
            #           color='red', alpha=1)
            # plt.title(name + ' ' + dept + ' ' + str(exp))
            # plt.xlabel('Knowledge Categories')
            # plt.ylabel('Knowledge Quantity')
            # plt.show()

            # create employee agent
            agent = EmployeeAgent(i, exp, name, dept, know, self)
            self.schedule.add(agent)

        self.roster = pd.DataFrame(self.roster)

        self.datacollector = mesa.DataCollector(
            agent_reporters={"Knowledge": "knowledge"}
        )

    def step(self):
        """Advance the model by one step."""
        self.task_dict[self.task_num] = [2,3,7,8]
        self.task_num += 1
        self.datacollector.collect(self)
        self.schedule.step()