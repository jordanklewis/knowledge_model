# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 20:55:49 2022

@author: Lewis
"""
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
        dept_know = {'SE': {'mu': 0, 'sigma': 30},
                     'SW': {'mu': 25, 'sigma': 20},
                     'EE': {'mu': 50, 'sigma': 20},
                     'ME': {'mu': -25, 'sigma': 20}}
        
        # systems engineering department distribution
        se_know = np.random.normal(dept_know['SE']['mu'],
                                   dept_know['SE']['sigma'],
                                   max_know).astype(int)
        se_know[se_know>50] -= 100 # wrap to +/- 50 for department overlap
        se_know[se_know<-50] += 100
        
        # software engineering department distribution
        sw_know = np.random.normal(dept_know['SW']['mu'],
                                   dept_know['SW']['sigma'],
                                   max_know).astype(int)
        sw_know[sw_know>50] -= 100 # wrap to +/- 50 for department overlap
        sw_know[sw_know<-50] += 100
        
        # electrical engineering department distribution
        ee_know = np.random.normal(dept_know['EE']['mu'],
                                   dept_know['EE']['sigma'],
                                   max_know).astype(int)
        ee_know[ee_know>50] -= 100 # wrap to +/- 50 for department overlap
        ee_know[ee_know<-50] += 100
        
        # mechanical engineering department distribution
        me_know = np.random.normal(dept_know['ME']['mu'],
                                   dept_know['ME']['sigma'],
                                   max_know).astype(int)
        me_know[me_know>50] -= 100 # wrap to +/- 50 for department overlap
        me_know[me_know<-50] += 100
        
        plt.hist(se_know, bins=np.arange(se_know.min()-0.5, se_know.max()+0.5), 
                  color='red', alpha=1, label='SE')
        plt.hist(sw_know, bins=np.arange(sw_know.min()-0.5, sw_know.max()+0.5), 
                  color='green', alpha=0.75, label='SW')
        plt.hist(ee_know, bins=np.arange(ee_know.min()-0.5, ee_know.max()+0.5), 
                  color='yellow', alpha=0.5, label='EE')
        plt.hist(me_know, bins=np.arange(me_know.min()-0.5, me_know.max()+0.5), 
                  color='blue', alpha=0.25, label='ME')
        plt.legend(loc='upper right')
        plt.show()
        
        # Create employee names
        name_list = pd.read_csv("employee_names.csv")
        name_list = name_list["Names"].tolist()
        name_list = list(np.random.choice(name_list, 
                                          self.num_employees, replace=False))
        
        for i in range(self.num_employees):
            exp = (np.random.choice(9, p=[.1, .1, .14, .2,
                                          .2, .1, .1, .05, .01])+1)*100
            name = name_list.pop()
            dept = np.random.choice(list(dept_know.keys()))
            know = np.random.normal(dept_know[dept]['mu'],
                                    dept_know[dept]['sigma'],
                                    exp).astype(int)
            know[know>50] -= 100
            know[know<-50] += 100
            
            self.roster.append({'Name': name, 'Dept': dept, 'Exp': exp})
            
            # plt.hist(know, bins=np.arange(know.min()-0.5, know.max()+0.5), 
            #           color='red', alpha=1)
            # plt.title(name + ' ' + dept + ' ' + str(exp))
            # plt.xlabel('Knowledge Categories')
            # plt.ylabel('Knowledge Quantity')
            # plt.show()
            
            a = EmployeeAgent(i, exp, name, dept, know, self)
            self.schedule.add(a)
            
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