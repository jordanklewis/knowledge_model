# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 12:14:54 2023

@author: Lewis
"""

import matplotlib.pyplot as plt
import numpy as np

class KnowledgePlots():

    def plot_emp_help_know_cat(self, know_cat, agent):
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

    def plot_emp_task(self):
        emp_know_gain = self.emp_know - self.emp_know_pre_task

        plt.figure(figsize=(10,6))
        plt.bar(self.emp_know_pre_task.keys(),
                self.emp_know_pre_task.values(),
                color='blue', alpha =0.25, label='Know Pre Task')
        if emp_know_gain:
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
                  + 'Task Number: ' + str(self.emp_task_num)
                  + '    Task Difficulty: ' + str(len(list(self.task.elements())))
                  + '    Needed Knowledge: ' + str(len(list(self.emp_know_to_learn.elements()))))
        plt.xlabel('Knowledge Categories')
        plt.ylabel('Knowledge Quantity')
        plt.show()
        
    def plot_emp_know(self):
        plt.figure(figsize=(10,6))
        plt.bar(self.emp_know.keys(),
                self.emp_know.values(),
                color='blue', alpha =0.25, label='Emp Know')
        plt.step(self.model.know_cats,
         [self.model.comp_know[n] for n in self.model.know_cats],
          'k--', alpha =1, label='Company', where='mid')
        plt.step(self.model.know_cats,
                 [self.model.dept_know[self.dept]['dist'][n] for n in self.model.know_cats],
                  'g', alpha =1, label=str(self.dept)+' Dept', where='mid')
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
                  + '    Total Knowledge: ' + str(len(list(self.emp_know.elements()))))
        plt.xlabel('Knowledge Categories')
        plt.ylabel('Knowledge Quantity')
        plt.show()

    def plot_comp_library(self):
        plt.figure(figsize=(10,6))
        plt.bar(self.comp_library.keys(),
                self.comp_library.values(),
                color='blue', alpha =0.25, label='Library Know', width=1)
        plt.step(self.know_cats,
                 [self.comp_know[n] for n in self.know_cats],
                  'k--', alpha =1, label='Company Know', where='mid')
        plt.legend()
        plt.xlim([min(self.know_cats)-0.5, max(self.know_cats)+0.5])
        plt.minorticks_on()
        plt.xticks(np.arange(min(self.know_cats),
                             max(self.know_cats)+1,
                             self.know_cat_ct/20))
        plt.tick_params(axis='x', which='minor', length=5, width=1)
        plt.tick_params(axis='x', which='major', length=7, width=2)
        plt.title('Company Knowledge Library')
        plt.xlabel('Knowledge Categories')
        plt.ylabel('Knowledge Quantity')
        plt.show()

    def plot_library_task(self):
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
                 [self.model.comp_library[n] for n in self.model.know_cats],
                  'g:', alpha =1, label='Library', where='mid')
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
                  + '    Task Number: ' + str(self.emp_task_num))
        plt.xlabel('Knowledge Categories')
        plt.ylabel('Knowledge Quantity')
        plt.show()

    def plot_emp_research(self):
        emp_know_gain = self.emp_know - self.emp_know_pre_task
        breakthrough = self.know_cat_research - self.emp_know
        research = self.know_cat_research - breakthrough

        plt.figure(figsize=(10,6))
        plt.bar(self.emp_know_pre_task.keys(),
                self.emp_know_pre_task.values(),
                color='blue', alpha =0.25, label='Know Pre Task')
        if emp_know_gain:
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
        if breakthrough:
            plt.bar(breakthrough.keys(),
                    breakthrough.values(),
                     color='green', alpha =1, label='Breakthrough',
                     bottom=[research[n] for n in breakthrough.keys()])
        # plt.step(self.model.know_cats,
        #         [self.task[n] for n in self.model.know_cats],
        #           color='black', alpha=1, label='Task', where='mid')
        # plt.step(self.model.know_cats,
        #          [self.model.dept_know[self.dept]['dist'][n] for n in self.model.know_cats],
        #           'g:', alpha =1, label=str(self.dept)+' Dept', where='mid')
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
                             self.know_cat_ct/10))
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
