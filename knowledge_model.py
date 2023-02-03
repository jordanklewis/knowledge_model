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
        self.emp_id = unique_id
        self.exp = exp
        self.name = name
        self.dept = dept
        self.emp_know_pre_task = know
        self.emp_know = know
        self.emp_task_num = -1
        self.research_know_ct = 0
        self.learn_know_ct = 0
        self.read_know_ct = 0
        self.teach_know_ct = 0
        self.document_know_ct = 0
        self.model = model
        self.task_completed = True
        # Status: Idle, Teaching, Learning, Documenting, Reading, Researching
        # Coworker: Name, None
        self.status = 'Idle'
        self.coworker = 'None'
        self.task = Counter()
        self.emp_know_to_learn = Counter()
        self.emp_remain_know_to_learn = Counter()
        self.task_learn_path = Counter()
        self.know_cat_research = Counter()
        self.emp_performance = []

    def step(self):
        # the end of a step is initaited by
        # 1. Employee gains new knowledge
        # 2. Employee completes a task

        # agent must be available to continue step
        if self.status != 'Idle':
            return

        # before starting step, see if employee is eligable for promotion
        self.check_for_promotion() # maybe only promote when a task is completed ???????

        # if a current task is not assigned, then assign one
        if self.task_completed:
            self.get_new_task()
            self.emp_know_pre_task = self.emp_know
            if self.check_for_task_completion():
                self.task_completed = True
                print('Task completed with no new knowledge.' +
                      ' This should never happen')
                return

        # Always search for help on tasks so they can be finished quicker
        got_help  = self.find_help_for_task()
        # got_help = False

        # if no help is currently availalbe, they should research the low hanging fruit
        if not(got_help): # if help is unavailable for a know cat, then research
            self.work_on_task_without_help()

        # At the end of every step, check for task completion and log data
        self.task_completed = self.check_for_task_completion()
        if self.task_completed:
            self.update_company_and_dept_know()
        self.log_step_data()

    def find_help_for_task(self):
        # employees always want to get help from the lowest possible senority

        # first loop through all remaining know to learn ordering from most
        # challenging to least
        # employees should ask for help on the most needed knowledge category
        for i, k in enumerate(self.emp_remain_know_to_learn.most_common()):
            know_cat = self.emp_remain_know_to_learn.most_common()[i][0]

            # find a helpful and available employee for the know_cat starting
            # with the lowest ranking employee
            for agent in reversed(self.model.schedule.agents):
                if ((agent.emp_know[know_cat] - self.emp_know[know_cat]) > 1
                    and agent.status == 'Idle'):

                    # self.plot_employee_help_know_cat(know_cat, agent)

                    # agents can only help one person at a time
                    agent.status = 'Teaching'
                    agent.coworker = self.name
                    agent.teach_know_ct += 1

                    # add new know to the emp_know
                    self.status = 'Learning'
                    self.coworker = agent.name
                    self.emp_know[know_cat] += 1
                    self.learn_know_ct += 1

                    # determine needed know still remaining
                    self.emp_remain_know_to_learn = self.task - self.emp_know
                    return True
        # if no help is available for all needed categories, return false
        return False
    
    # for a in reversed(self.model.schedule.agents):
    #     print(a.name + ': ' + str(a.emp_know[know_cat]))

    def plot_employee_help_know_cat(self, know_cat, agent):
        # verification plot to demonstrate that the chosen employee has
        # the know needed to teach the requesting employee
        _,axis = plt.subplots(2, 1, figsize=(10,12))
        axis[0].bar(self.emp_know.keys(),
                self.emp_know.values(),
                color='blue', alpha =0.25, label='Know')
        axis[0].bar(know_cat,
                self.emp_remain_know_to_learn[know_cat],
                 color='red', alpha =0.75, label='Target Needed Know',
                 bottom=self.emp_know[know_cat])
        axis[0].step(self.model.know_cats,
                [self.task[n] for n in self.model.know_cats],
                  color='black', alpha=1, label='Task', where='mid')
        axis[0].step(self.model.know_cats,
                 [self.model.comp_know[n] for n in self.model.know_cats],
                  'k--', alpha =1, label='Company', where='mid')
        axis[0].set_title(self.name + ' [' + self.dept
                  + ' Exp: ' + str(self.exp) + ']'
                  + '    Total Knowledge: ' + str(len(list(self.emp_know.elements()))) + '\n'
                  + 'Step: ' + str(self.model.step_num)
                  + '    Task Number: ' + str(self.emp_task_num)
                  + '    Target Know Category: ' + str(know_cat)
                  + '    Target Know Quantity: ' + str(self.task[know_cat])
                  + '    Current Know Quantity: ' + str(self.emp_know[know_cat]))
        axis[0].set_xlabel('Knowledge Category')
        axis[0].set_ylabel('Knowledge Quantity')
        axis[0].set_xlim([min(self.model.know_cats)-0.5, max(self.model.know_cats)+0.5])
        axis[0].set_xticks(np.arange(min(self.model.know_cats),
                                      max(self.model.know_cats)+1,
                                      self.model.know_cat_ct/20))
        axis[0].minorticks_on()
        axis[0].tick_params(axis='x', which='minor', length=5, width=1)
        axis[0].tick_params(axis='x', which='major', length=7, width=2)
        axis[0].legend()

        axis[1].bar(agent.emp_know.keys(),
                agent.emp_know.values(),
                color='green', alpha =0.25, label='Know')
        axis[1].bar(know_cat,
                agent.emp_know[know_cat],
                 color='green', alpha=1, label='Target Avail Know')
        axis[1].step(self.model.know_cats,
                [self.task[n] for n in self.model.know_cats],
                  color='black', alpha=1, label='Task', where='mid')
        axis[1].step(self.model.know_cats,
                 [self.model.comp_know[n] for n in self.model.know_cats],
                  'k--', alpha =1, label='Company', where='mid')
        axis[1].set_title(('Ask For Help From:    ' + agent.name + ' [' + agent.dept
                  + ' Exp: ' + str(agent.exp) + ']'
                  + '    Total Knowledge: ' + str(len(list(agent.emp_know.elements()))) + '\n'
                  + 'Status: ' + agent.status
                  + '    Target Knowledge Category: ' + str(know_cat)
                  + '    Target Knowledge Quantity: ' + str(agent.emp_know[know_cat])))
        axis[1].set_xlabel('Knowledge Category')
        axis[1].set_ylabel('Knowledge Quantity')
        axis[1].set_xlim([min(self.model.know_cats)-0.5, max(self.model.know_cats)+0.5])
        axis[1].set_xticks(np.arange(min(self.model.know_cats),
                                      max(self.model.know_cats)+1,
                                      self.model.know_cat_ct/20))
        axis[1].minorticks_on()
        axis[1].tick_params(axis='x', which='minor', length=5, width=1)
        axis[1].tick_params(axis='x', which='major', length=7, width=2)
        axis[1].legend()

        plt.tight_layout()
        plt.show()

    def update_company_and_dept_know(self):
        new_dept_know = self.task - self.model.dept_know[self.dept]['dist']
        if new_dept_know:
            # self.plot_new_dept_know()
            self.model.dept_know[self.dept]['dist'] += new_dept_know

        new_comp_know = self.task - self.model.comp_know
        if new_comp_know:
            # self.plot_new_comp_know()
            self.model.comp_know += new_comp_know

    def plot_new_dept_know(self):
        new_dept_know = self.task - self.model.dept_know[self.dept]['dist']
        plt.figure(figsize=(10,6))
        plt.bar(self.model.dept_know[self.dept]['dist'].keys(),
                self.model.dept_know[self.dept]['dist'].values(),
                color='orange', alpha =0.25, label='Prev Dept Know')
        plt.bar(new_dept_know.keys(),
                new_dept_know.values(),
                 color='green', alpha =0.75, label='New Dept Know',
                 bottom=[self.model.dept_know[self.dept]['dist'][n]
                         for n in new_dept_know.keys()])
        plt.step(self.model.know_cats,
                 [self.model.comp_know[n] for n in self.model.know_cats],
                  'k--', alpha =1, label='Company', where='mid')
        plt.step(self.model.know_cats,
                [self.task[n] for n in self.model.know_cats],
                  color='black', alpha=1, label='Task', where='mid')
        plt.legend()
        plt.xlim([min(self.model.know_cats)-0.5, max(self.model.know_cats)+0.5])
        plt.minorticks_on()
        plt.xticks(np.arange(min(self.model.know_cats),
                             max(self.model.know_cats)+1,
                             self.model.know_cat_ct/20))
        plt.tick_params(axis='x', which='minor', length=5, width=1)
        plt.tick_params(axis='x', which='major', length=7, width=2)
        plt.title('Total ' + self.dept + ' Dept Knowledge: ' +
                  str(len(list((self.model.dept_know[self.dept]['dist']
                                + new_dept_know).elements())))
                  + '\nContributor: ' + self.name + ' [' + self.dept
                            + ' Exp: ' + str(self.exp) + ']'
                            + '    Total Knowledge: '
                            + str(len(list(self.emp_know.elements()))))
        plt.xlabel('Knowledge Categories')
        plt.ylabel('Knowledge Quantity')
        plt.show()

    def plot_new_comp_know(self):
        new_comp_know = self.task - self.model.comp_know
        plt.figure(figsize=(10,6))
        plt.bar(self.model.comp_know.keys(),
                self.model.comp_know.values(),
                color='purple', alpha =0.25, label='Prev Comp Know')
        plt.bar(new_comp_know.keys(),
                new_comp_know.values(),
                 color='green', alpha =0.75, label='New Comp Know',
                 bottom=[self.model.comp_know[n]
                         for n in new_comp_know.keys()])
        plt.step(self.model.know_cats,
                [self.task[n] for n in self.model.know_cats],
                  color='black', alpha=1, label='Task', where='mid')
        plt.legend()
        plt.xlim([min(self.model.know_cats)-0.5, max(self.model.know_cats)+0.5])
        plt.minorticks_on()
        plt.xticks(np.arange(min(self.model.know_cats),
                             max(self.model.know_cats)+1,
                             self.model.know_cat_ct/20))
        plt.tick_params(axis='x', which='minor', length=5, width=1)
        plt.tick_params(axis='x', which='major', length=7, width=2)
        plt.title('Total Company Knowledge: ' +
                  str(len(list((self.model.comp_know
                                + new_comp_know).elements())))
                  + '\nContributor: ' + self.name + ' [' + self.dept
                            + ' Exp: ' + str(self.exp) + ']'
                            + '    Total Knowledge: '
                            + str(len(list(self.emp_know.elements()))))
        plt.xlabel('Knowledge Categories')
        plt.ylabel('Knowledge Quantity')
        plt.show()

    def work_on_task_without_help(self):
        
        self.status = 'Researching'
        self.coworker = 'None'
        
        # pick the know_cat that will take the least research to finish the task
        know_cat = self.emp_remain_know_to_learn.most_common()[-1][0]

        # Continue researching until new knowledge is found in a category.
        # When researching, if you are discovering knowledge that is already
        # known, then not much time is spent because the information is already
        # known. Having foundational knowledge around the area that is needed
        # to be discovered will allow a person to make the discovery faster than
        # someone without foundational knowledge.
        while not self.know_cat_research - self.emp_know:
            self.research_know_cat(know_cat)

        # use this plot to demonstrate research through normal distributions
        # self.plot_employee_research()

        # add new know to the emp_know
        self.emp_know = self.emp_know + (self.know_cat_research - self.emp_know)

        # determine needed know still remaining
        self.emp_remain_know_to_learn = self.task - self.emp_know

        # clear the know_cat_research
        self.know_cat_research = Counter()

    def check_for_task_completion(self):
        # check to see if the employee has the needed knowledge to complete the task
        if self.task - self.emp_know: # if not empty, task is not completed
            return False
        # otherwise, the task has been completed
        # self.plot_emp_know_post_task()
        return True

    def log_step_data(self):
        self.model.step_data.append({'step': self.model.step_num,
        'employee_id': self.emp_id,
        'employee_name': self.name,
        'employee_dept': self.dept,
        'employee_exp': self.exp,
        'task_num': self.emp_task_num,
        'needed_know': len(list(self.emp_know_to_learn.elements())),
        'total_knowledge': len(list(self.emp_know.elements())),
        'task_complexity': len(list(self.task.elements())),
        'task_completed': self.task_completed,
        'research_know': 0,
        'helped_know': 0,
        'read_know': 0,
        'teach_know': 0,
        'SE_dept_know': len(list(self.model.dept_know['SE']['dist'].elements())),
        'SW_dept_know': len(list(self.model.dept_know['SW']['dist'].elements())),
        'EE_dept_know': len(list(self.model.dept_know['EE']['dist'].elements())),
        'ME_dept_know': len(list(self.model.dept_know['ME']['dist'].elements())),
        'comp_know': len(list(self.model.comp_know.elements()))})


    def check_for_promotion(self):
        # check if employee needs a promotion to be assigned more challenging tasks
        if len(list(self.emp_know.elements())) >= self.exp + self.model.max_know/10:
            self.exp = int(self.exp + self.model.max_know/10)

    def research_know_cat(self, cat):
        # generate a normal distribution for learning to the target level
        # learning to a deisred level is not a linear path. This is simulated as a
        # normal distribution of knowledge centered around the target knowledge
        mu_research = cat
        sigma_research = 1
        rand_research = np.random.normal(mu_research, sigma_research, 1).astype(int)
        rand_research[rand_research>=self.model.know_cat_ct/2] -= self.model.know_cat_ct
        rand_research[rand_research<-self.model.know_cat_ct/2] += self.model.know_cat_ct
        self.know_cat_research[rand_research.item()] += 1

    def get_new_task(self):
        task_level = int(self.exp * self.model.innovation_rate)
        self.emp_know_to_learn = Counter()

        # There is no such thing as a mindless job in 2023
        # Employees should not be assigned a task where ZERO new knowledge
        # is required to complete the task. If an employee's job scope is so
        # repetitive and well defined that that zero new knowledge is required,
        # it can be assumed that this employee's role is best substituted
        # for an autonomous process. As an employee approaches a promotion,
        # assigned tasks will require less and less new knowledge to complete
        # allowing the soon to be promoted employee to accomplish tasks faster.
        # As a result of the employee's star performance in their expected role,
        # the employee will soon be promoted to facing more challenging tasks.

        # this while loop generates a new task. If the generated task does not
        # require any new knowledge, then stay in the while loop
        while not self.emp_know_to_learn:
            task = np.random.normal(self.model.dept_know[self.dept]['mu'],
                                       self.model.dept_know[self.dept]['sigma'],
                                       task_level).astype(int)
            task[task>=self.model.know_cat_ct/2] -= self.model.know_cat_ct
            task[task<-self.model.know_cat_ct/2] += self.model.know_cat_ct
            self.task = Counter(task)

            # the difference betwen the employee's task and knowledge is what
            # they need to learn
            self.emp_know_to_learn = self.task - self.emp_know
            self.emp_remain_know_to_learn = self.emp_know_to_learn

        # assign a task number from the company
        self.emp_task_num = self.model.comp_task_num
        self.task_completed = False
        # increment comp task number so next assigned task has unique number
        self.model.comp_task_num += 1
        
    def plot_employee_task(self):
        emp_know_gain = self.emp_know - self.emp_know_pre_task

        plt.figure(figsize=(10,6))
        plt.bar(self.emp_know_pre_task.keys(),
                self.emp_know_pre_task.values(),
                color='blue', alpha =0.25, label='Know Pre Task')
        plt.bar(emp_know_gain.keys(),
                emp_know_gain.values(),
                 color='blue', alpha =0.75, label='Know Gain',
                 bottom=[self.emp_know_pre_task[n] for n in emp_know_gain.keys()])
        plt.bar(self.emp_remain_know_to_learn.keys(),
                self.emp_remain_know_to_learn.values(),
                 color='red', alpha =1, label='Needed Know',
                 bottom=[self.emp_know[n] for n in self.emp_remain_know_to_learn.keys()])
        plt.step(self.model.know_cats,
                [self.task[n] for n in self.model.know_cats],
                  color='black', alpha=1, label='Task', where='mid')
        plt.step(self.model.know_cats,
                 [self.model.dept_know[self.dept]['dist'][n] for n in self.model.know_cats],
                  'g:', alpha =1, label=str(self.dept)+' Dept', where='mid')
        plt.step(self.model.know_cats,
                 [self.model.comp_know[n] for n in self.model.know_cats],
                  'k--', alpha =1, label='Company', where='mid')
        plt.legend()
        plt.xlim([min(self.model.know_cats)-0.5, max(self.model.know_cats)+0.5])
        plt.minorticks_on()
        plt.xticks(np.arange(min(self.model.know_cats),
                             max(self.model.know_cats)+1,
                             self.model.know_cat_ct/20))
        plt.tick_params(axis='x', which='minor', length=5, width=1)
        plt.tick_params(axis='x', which='major', length=7, width=2)
        plt.title(self.name + ' [' + self.dept
                  + ' Exp: ' + str(self.exp) + ']'
                  + '    Total Knowledge: ' + str(len(list(self.emp_know.elements()))) + '\n'
                  + 'Step: ' + str(self.model.step_num)
                  + '    Task Number: ' + str(self.emp_task_num)
                  + '    Needed Knowledge: ' + str(len(list(self.emp_know_to_learn.elements())))
                  + '    Knowledge Gain: ' + str(len(list(emp_know_gain.elements()))))
        plt.xlabel('Knowledge Categories')
        plt.ylabel('Knowledge Quantity')
        plt.show()

    def plot_employee_research(self):
        emp_know_gain = self.emp_know - self.emp_know_pre_task
        breakthrough = self.know_cat_research - self.emp_know
        research = self.know_cat_research - breakthrough

        plt.figure(figsize=(10,6))
        plt.bar(self.emp_know_pre_task.keys(),
                self.emp_know_pre_task.values(),
                color='blue', alpha =0.25, label='Know Pre Task')
        plt.bar(emp_know_gain.keys(),
                emp_know_gain.values(),
                 color='blue', alpha =0.75, label='Know Gain',
                 bottom=[self.emp_know_pre_task[n] for n in emp_know_gain.keys()])
        plt.bar(self.emp_remain_know_to_learn.keys(),
                self.emp_remain_know_to_learn.values(),
                 color='red', alpha =1, label='Needed Know',
                 bottom=[self.emp_know[n] for n in self.emp_remain_know_to_learn.keys()])
        plt.bar(research.keys(),
                research.values(),
                color='black', alpha =0.5, label='Research')
        plt.bar(breakthrough.keys(),
                breakthrough.values(),
                 color='green', alpha =1, label='Breakthrough',
                 bottom=[research[n] for n in breakthrough.keys()])
        plt.step(self.model.know_cats,
                [self.task[n] for n in self.model.know_cats],
                  color='black', alpha=1, label='Task', where='mid')
        plt.step(self.model.know_cats,
                 [self.model.dept_know[self.dept]['dist'][n] for n in self.model.know_cats],
                  'g:', alpha =1, label=str(self.dept)+' Dept', where='mid')
        plt.step(self.model.know_cats,
                 [self.model.comp_know[n] for n in self.model.know_cats],
                  'k--', alpha =1, label='Company', where='mid')
        plt.legend()
        plt.xlim([min(self.model.know_cats)-0.5, max(self.model.know_cats)+0.5])
        plt.minorticks_on()
        plt.xticks(np.arange(min(self.model.know_cats),
                             max(self.model.know_cats)+1,
                             self.model.know_cat_ct/20))
        plt.tick_params(axis='x', which='minor', length=5, width=1)
        plt.tick_params(axis='x', which='major', length=7, width=2)
        plt.title(self.name + ' [' + self.dept
                  + ' Exp: ' + str(self.exp) + ']'
                  + '    Total Knowledge: ' + str(len(list(self.emp_know.elements()))) + '\n'
                  + 'Step: ' + str(self.model.step_num)
                  + '    Task Number: ' + str(self.emp_task_num)
                  + '    Needed Knowledge: ' + str(len(list(self.emp_know_to_learn.elements())))
                  + '    Knowledge Gain: ' + str(len(list(emp_know_gain.elements()))))
        plt.xlabel('Knowledge Categories')
        plt.ylabel('Knowledge Quantity')
        plt.show()

    def plot_emp_know_post_task(self):
        emp_know_gain = self.emp_know - self.emp_know_pre_task

        plt.figure(figsize=(10,6))
        plt.bar(self.emp_know_pre_task.keys(),
                self.emp_know_pre_task.values(),
                color='blue', alpha =0.25, label='Know Pre Task')
        plt.bar(emp_know_gain.keys(),
                emp_know_gain.values(),
                 color='green', alpha =0.75, label='Know Gain',
                 bottom=[self.emp_know_pre_task[n] for n in emp_know_gain.keys()])
        plt.step(self.model.know_cats,
                [self.task[n] for n in self.model.know_cats],
                  color='black', alpha=1, label='Task', where='mid')
        plt.step(self.model.know_cats,
                 [self.model.dept_know[self.dept]['dist'][n] for n in self.model.know_cats],
                  'r--', alpha =1, label=str(self.dept)+' Dept', where='mid')
        plt.step(self.model.know_cats,
                 [self.model.comp_know[n] for n in self.model.know_cats],
                  'k--', alpha =1, label='Company', where='mid')
        plt.legend()
        plt.xlim([min(self.model.know_cats)-0.5, max(self.model.know_cats)+0.5])
        plt.minorticks_on()
        plt.xticks(np.arange(min(self.model.know_cats),
                             max(self.model.know_cats)+1,
                             self.model.know_cat_ct/20))
        plt.tick_params(axis='x', which='minor', length=5, width=1)
        plt.tick_params(axis='x', which='major', length=7, width=2)
        plt.title(self.name + ' [' + self.dept
                  + ' Exp: ' + str(self.exp) + ']'
                  + '    Total Knowledge: ' + str(len(list(self.emp_know.elements()))) + '\n'
                  + 'Step: ' + str(self.model.step_num)
                  + '    Completed Task Number: ' + str(self.emp_task_num)
                  + '    Needed Knowledge: ' + str(len(list(self.emp_know_to_learn.elements())))
                  + '    Gained Knowledge: ' + str(len(list(emp_know_gain.elements()))))
        plt.xlabel('Knowledge Categories')
        plt.ylabel('Knowledge Quantity')
        plt.show()

class KnowledgeModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, N):
        super().__init__()
        # company parameters
        self.innovation_rate = 0.5 # value must be betwen 0 and 1
        self.know_cat_ct = 1000
        self.know_cats = list(np.arange(self.know_cat_ct).astype(int)
                              - int(self.know_cat_ct/2))
        self.max_know = 10000
        self.num_employees = N
        self.comp_task_num = 0
        self.task_dict = {}
        self.roster = []
        self.step_num = -1

        # department descriptive statistics
        self.dept_know = {'SE': {'mu': 0, 'sigma': self.know_cat_ct/3},
                     'SW': {'mu': self.know_cat_ct/4, 'sigma': self.know_cat_ct/5},
                     'EE': {'mu': self.know_cat_ct/2, 'sigma': self.know_cat_ct/5},
                     'ME': {'mu': self.know_cat_ct/-4, 'sigma': self.know_cat_ct/5}}

        # model parameters
        self.schedule = mesa.time.RandomActivation(self)
        # self.schedule = mesa.time.BaseScheduler(self)
        self.running = True
        self.step_data = []

        # create company knowledge
        self.create_company_know_dist()
        self.plot_company_know()
        #self.plot_company_know_subplots()

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
            self.roster.append({'Name': name, 'Dept': dept, 'Exp': exp,
                                'Start_Know': len(list(know.elements()))})

            # create employee agent
            agent = EmployeeAgent(i, exp, name, dept, know, self)
            self.schedule.add(agent)

        self.roster = pd.DataFrame(self.roster)
        self.roster.sort_values('Start_Know', ascending=False)

        self.datacollector = mesa.DataCollector(
            agent_reporters={"Knowledge": "knowledge"})
    
    def get_roster_status(self):
        self.roster = []
        for a in self.schedule.agents:
            self.roster.append({'Name': a.name, 'Dept': a.dept, 'Exp': a.exp,
                                'Current_Know': len(list(a.emp_know.elements())),
                               'Status': a.status, 'Coworker': a.coworker})
        self.roster = pd.DataFrame(self.roster)
        print(self.roster)

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
        plt.xticks(np.arange(min(self.know_cats),
                             max(self.know_cats)+1,
                             self.know_cat_ct/20))
        plt.tick_params(axis='x', which='minor', length=5, width=1)
        plt.tick_params(axis='x', which='major', length=7, width=2)
        plt.legend(loc='upper right')
        plt.title('Company Knowledge by Department')
        plt.xlabel('Knowledge Categories')
        plt.ylabel('Knowledge Quantity')
        plt.show()

    def plot_company_know_subplots(self):
        # plot distributions on subplots
        _,axis = plt.subplots(2, 2, figsize=(10,6))
        axis[0][0].bar(self.dept_know['SE']['dist'].keys(),
                self.dept_know['SE']['dist'].values(),
                  color='red', alpha=1, label='SE', width=1)
        axis[0][0].set_title('Systems Enginereing')
        axis[0][0].set_xlabel('Knowledge Category')
        axis[0][0].set_ylabel('Knowledge Quantity')
        axis[0][0].set_xlim([min(self.know_cats)-0.5, max(self.know_cats)+0.5])
        axis[0][0].set_xticks(np.arange(min(self.know_cats),
                                      max(self.know_cats)+1,
                                      self.know_cat_ct/10))
        axis[0][0].minorticks_on()
        axis[0][0].tick_params(axis='x', which='minor', length=5, width=1)
        axis[0][0].tick_params(axis='x', which='major', length=7, width=2)

        axis[0][1].bar(self.dept_know['SW']['dist'].keys(),
                self.dept_know['SW']['dist'].values(),
                  color='green', alpha=0.5, label='SE', width=1)
        axis[0][1].set_title('Software Enginereing')
        axis[0][1].set_xlabel('Knowledge Category')
        axis[0][1].set_ylabel('Knowledge Quantity')
        axis[0][1].set_xlim([min(self.know_cats)-0.5, max(self.know_cats)+0.5])
        axis[0][1].set_xticks(np.arange(min(self.know_cats),
                                      max(self.know_cats)+1,
                                      self.know_cat_ct/10))
        axis[0][1].minorticks_on()
        axis[0][1].tick_params(axis='x', which='minor', length=5, width=1)
        axis[0][1].tick_params(axis='x', which='major', length=7, width=2)

        axis[1][0].bar(self.dept_know['EE']['dist'].keys(),
                self.dept_know['EE']['dist'].values(),
                  color='yellow', alpha=0.5, label='SE', width=1)
        axis[1][0].set_title('Electrical Enginereing')
        axis[1][0].set_xlabel('Knowledge Category')
        axis[1][0].set_ylabel('Knowledge Quantity')
        axis[1][0].set_xlim([min(self.know_cats)-0.5, max(self.know_cats)+0.5])
        axis[1][0].set_xticks(np.arange(min(self.know_cats),
                                      max(self.know_cats)+1,
                                      self.know_cat_ct/10))
        axis[1][0].minorticks_on()
        axis[1][0].tick_params(axis='x', which='minor', length=5, width=1)
        axis[1][0].tick_params(axis='x', which='major', length=7, width=2)

        axis[1][1].bar(self.dept_know['ME']['dist'].keys(),
                self.dept_know['ME']['dist'].values(),
                  color='blue', alpha=0.25, label='SE', width=1)
        axis[1][1].set_title('Mechanical Enginereing')
        axis[1][1].set_xlabel('Knowledge Category')
        axis[1][1].set_ylabel('Knowledge Quantity')
        axis[1][1].set_xlim([min(self.know_cats)-0.5, max(self.know_cats)+0.5])
        axis[1][1].set_xticks(np.arange(min(self.know_cats),
                                      max(self.know_cats)+1,
                                      self.know_cat_ct/10))
        axis[1][1].minorticks_on()
        axis[1][1].tick_params(axis='x', which='minor', length=5, width=1)
        axis[1][1].tick_params(axis='x', which='major', length=7, width=2)

        plt.tight_layout()
        plt.show()

    def create_employee_name_list(self):
        # Create employee names
        name_list = pd.read_csv("employee_names.csv")
        name_list = name_list["Names"].tolist()
        return list(np.random.choice(name_list,
                                     self.num_employees, replace=False))

    def get_employee_experience(self):
        return int((np.random.choice(9, p=[.1, .1, .14, .2,
                                      .2, .1, .1, .05, .01])+1)*self.max_know/10)

    def get_employee_knowledge_dist(self, dept, exp):
        start_know = exp + np.random.randint(1,self.max_know/10)
        know = np.random.normal(self.dept_know[dept]['mu'],
                                self.dept_know[dept]['sigma'],
                                start_know).astype(int)
        know[know>=self.know_cat_ct/2] -= self.know_cat_ct
        know[know<-self.know_cat_ct/2] += self.know_cat_ct
        return Counter(know)

    def order_agents_by_know(self):
        # sort agents in the step scheduler by their knowledge levels so that
        # the most experienced agents can make the first moves in the model
        know_dict = {i: len(list(a.emp_know.elements()))
                    for i, a in enumerate(self.schedule.agents)}
        agent_keys = sorted(know_dict, key=know_dict.get, reverse=True)
        self.schedule._agents = {i : self.schedule._agents[k]
                        for i, k in enumerate(agent_keys)}

    def step(self):
        """Advance the model by one step."""
        #self.datacollector.collect(self)
        self.step_num += 1
        self.order_agents_by_know()
        self.schedule.step()
        
        # at the end of a step, reset all agent statuses
        # in the future, a probablility distribution could be used to assign
        # a status such as writing or busy to an agent for the next step. 
        for a in self.schedule.agents:
            a.status = 'Idle'
            a.coworker = 'None'
