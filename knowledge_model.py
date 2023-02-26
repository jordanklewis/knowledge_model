# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 20:55:49 2022

@author: Lewis
"""
from collections import Counter
import time
import mesa
import numpy as np
import pandas as pd
from knowledge_plots import KnowledgePlots

class EmployeeAgent(mesa.Agent, KnowledgePlots):
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
        # Status: avail, Teaching, Learning, docs, Reading, Researching
        # Coworker: Name, None
        self.status = 'avail'
        self.coworker = 'None'
        self.task = Counter()
        self.emp_know_to_learn = Counter()
        self.emp_remain_know_to_learn = Counter()
        self.know_cat_research = Counter()

    def step(self):
        # the end of a step is initaited by
        # 1. Employee gains new knowledge
        # 2. Employee completes a task

        # start_time = time.time()

        # before starting step, see if employee is eligable for promotion
        self.check_for_promotion() # maybe only promote when a task is completed ???????

        # if a current task is not assigned, then assign one
        if self.task_completed:
            self.get_new_task()
            self.emp_know_pre_task = self.emp_know
            # self.plot_emp_task()
            if self.check_for_task_completion():
                self.task_completed = True
                print('Task completed with no new knowledge.' +
                      ' This should never happen')
                return

        # agent has been assigned to produce documentation
        if self.status == 'docs':
            # Check if agent is successfully able to document anything
            if self.document_know():
                self.document_know_ct += 1
                self.log_step_data() # if success, log data and exit step
                # print("%s docs" % int((time.time() - start_time)*1000))
                return
            # if the agent has nothing to document, they should work on their task
            # Set status back to avail
            self.status = 'avail'

        # agent must be available to continue step
        if (self.status != 'avail') & (self.status != 'busy'):
            # print("%s Helping" % int((time.time() - start_time)*1000))
            self.log_step_data()
            return

        # before searching for help, agents should use the company library
        # as much as possible to complete their task
        if self.check_comp_library():
            if self.check_for_task_completion():
                self.update_company_and_dept_know()
            self.log_step_data()
            # print("%s Reading" % int((time.time() - start_time)*1000))
            return

        # Search for help on tasks so they can be finished quicker
        if self.find_help_for_task():
            if self.check_for_task_completion():
                self.update_company_and_dept_know()
            self.log_step_data()
            # print("%s Learning" % int((time.time() - start_time)*1000))
            return

        # if no help is currently availalbe, they should research the low hanging fruit
        self.work_on_task_without_help()
        # At the end of every step, check for task completion and log data
        if self.check_for_task_completion():
            self.update_company_and_dept_know()
        self.log_step_data()
        # print("%s Researching" % int((time.time() - start_time)*1000))

    def check_comp_library(self):
        # first loop through all remaining know to learn ordering from most
        # challenging to least
        # employees should first look for library knowledge on the most needed
        # knowledge category
        for i in self.emp_remain_know_to_learn.most_common():
            know_cat = i[0]

            # find helpful library know for the know_cat
            if (self.model.comp_library[know_cat] - self.emp_know[know_cat]) > 1:

                # self.plot_library_task()

                # add new know to the emp_know
                self.status = 'Reading'
                self.emp_know[know_cat] += 1
                self.read_know_ct += 1

                # determine needed know still remaining
                self.emp_remain_know_to_learn = self.task - self.emp_know
                return True

        # if no help is available for all needed categories, return false
        return False

    def find_help_for_task(self):
        # employees always want to get help from the lowest possible senority

        # first loop through all remaining know to learn ordering from most
        # challenging to least
        # employees should ask for help on the most needed knowledge category
        for i in self.emp_remain_know_to_learn.most_common():
            know_cat = i[0]

            # find a helpful and available employee for the know_cat starting
            # with the lowest ranking employee
            for agent in reversed(self.model.schedule.agents):
                if ((agent.emp_know[know_cat] - self.emp_know[know_cat]) > 1
                    and agent.status == 'avail'):

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

    def document_know(self):
        # employees should look at the library and contribute to the categories
        # where the knowledge quantity is the least

        # this is the know that the employee has that is not in the library
        # know_diff = self.emp_know - self.model.comp_library

        # employees can only document knowledge from their current task
        # know_diff = self.task - self.emp_remain_know_to_learn - self.model.comp_library

        # employees can only document new knowledge from their current task
        know_diff = self.emp_know - self.emp_know_pre_task - self.model.comp_library

        # if the employee has no know to contribute, do some other activity
        if not know_diff:
            return False

        # Iter through know_diff know_cats and add to the library for any know_cats
        # that are empty
        for know_cat in know_diff.keys():
            if self.model.comp_library[know_cat] == 0:
                self.model.comp_library[know_cat] += 1
                self.model.comp_library_log.append((self.model.step_num, know_cat))
                return True

        # if there is at least some knowledge in every know_cat, then add know
        # to the know_cat that has the least know
        for i in reversed(self.model.comp_library.most_common()):
            know_cat = i[0]
            if know_diff[know_cat] > 0:
                self.model.comp_library[know_cat] += 1
                self.model.comp_library_log.append((self.model.step_num, know_cat))
                return True

        # it shouldn't be possible to reach this point
        print('Something went wrong in document_know() logic')
        return False

    def update_company_and_dept_know(self):
        new_dept_know = self.task - self.model.dept_know[self.dept]['dist']
        if new_dept_know:
            # self.plot_new_dept_know()
            self.model.dept_know[self.dept]['dist'] += new_dept_know

        new_comp_know = self.task - self.model.comp_know
        if new_comp_know:
            # self.plot_new_comp_know()
            self.model.comp_know += new_comp_know

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
        # self.plot_emp_research()

        # add new know to the emp_know
        self.emp_know = self.emp_know + (self.know_cat_research - self.emp_know)

        # determine needed know still remaining
        self.emp_remain_know_to_learn = self.task - self.emp_know

        # clear the know_cat_research
        self.know_cat_research = Counter()
        self.research_know_ct += 1

    def check_for_task_completion(self):
        # check to see if the employee has the needed knowledge to complete the task
        if self.task - self.emp_know: # if not empty, task is not completed
            self.task_completed = False
            return False
        # otherwise, the task has been completed
        # self.plot_emp_know_post_task()
        self.task_completed = True
        return True

    def log_step_data(self):
        self.model.step_data.append({
        'config_num': self.model.config_num,
        'num_emp': self.model.num_employees,
        'avail': self.model.avail,
        'busy': self.model.busy,
        'docs': self.model.docs,
        'know_cat_ct': self.model.know_cat_ct,
        'max_know': self.model.max_know,
        'innov_rate': self.model.innovation_rate,
        'seed': self.model.seed,
        'step': self.model.step_num,
        'employee_id': self.emp_id,
        'employee_name': self.name,
        'employee_dept': self.dept,
        'employee_exp': self.exp,
        'task_num': self.emp_task_num,
        'needed_know': len(list(self.emp_know_to_learn.elements())),
        'total_knowledge': len(list(self.emp_know.elements())),
        'task_complexity': len(list(self.task.elements())),
        'task_completed': self.task_completed,
        'research_know': self.research_know_ct,
        'learn_know': self.learn_know_ct,
        'teach_know': self.teach_know_ct,
        'read_know': self.read_know_ct,
        'document_know': self.document_know_ct,
        'SE_dept_know': len(list(self.model.dept_know['SE']['dist'].elements())),
        'SW_dept_know': len(list(self.model.dept_know['SW']['dist'].elements())),
        'EE_dept_know': len(list(self.model.dept_know['EE']['dist'].elements())),
        'ME_dept_know': len(list(self.model.dept_know['ME']['dist'].elements())),
        'comp_lib_know': len(list(self.model.comp_library.elements())),
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
        # sigma 5 = 1:3 ratio
        #     with sigma 5, helping others is more company productive
        # sigma 3 = 1:2.25 ratio
        # sigma 2 = 1:2 ratio
        #     with sigma 2, researching on your own is more productive
        # sigma 1 = 1:0.7 ratio
        sigma_research = 5
        rand_research = np.random.normal(mu_research, sigma_research, 1).astype(int)
        rand_research %= self.model.know_cat_ct
        rand_research[rand_research>=self.model.know_cat_ct/2] -= self.model.know_cat_ct
        rand_research[rand_research<-self.model.know_cat_ct/2] += self.model.know_cat_ct
        self.know_cat_research[rand_research.item()] += 1

    def get_new_task(self):
        task_level = int(self.exp * self.model.innovation_rate)
        self.emp_know_to_learn = Counter()
        count = 0
        task_time = time.time()
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
            count += 1
            task = np.random.normal(self.model.dept_know[self.dept]['mu'],
                                       self.model.dept_know[self.dept]['sigma'],
                                       task_level).astype(int)
            task %= self.model.know_cat_ct
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
        if count > 20:
            print("New Task Attempts: %s" % count)
            print("New Task Time Time %.6f" % (time.time() - task_time))
class KnowledgeModel(mesa.Model, KnowledgePlots):
    """A model with some number of agents."""

    def __init__(self, num_emp, avail, busy, know_cat_ct,
                 max_know, innov_rate, config_num, seed=None):
        super().__init__(seed=seed)
        # company parameters
        self.config_num = config_num
        self.innovation_rate = innov_rate # value must be betwen 0 and 1
        self.know_cat_ct = know_cat_ct
        self.know_cats = list(np.arange(self.know_cat_ct).astype(int)
                              - int(self.know_cat_ct/2))
        self.max_know = max_know
        self.num_employees = num_emp
        self.comp_task_num = 0
        self.task_dict = {}
        self.roster = []
        self.step_num = -1
        self.seed = seed
        np.random.seed(seed)

        self.avail = avail
        self.busy = busy
        self.docs = abs(round(1 - avail - busy, 2))
        self.assigned_states = ['avail', 'busy', 'docs']
        self.state_prob = [self.avail, self.busy, self.docs]

        # department descriptive statistics
        self.dept_know = {'SE': {'mu': 0, 'sigma': self.know_cat_ct/3},
                     'SW': {'mu': self.know_cat_ct/4, 'sigma': self.know_cat_ct/5},
                     'EE': {'mu': self.know_cat_ct/2, 'sigma': self.know_cat_ct/5},
                     'ME': {'mu': self.know_cat_ct/-4, 'sigma': self.know_cat_ct/5}}
        self.comp_library = Counter()
        self.comp_library_log = []

        # model parameters
        self.schedule = mesa.time.RandomActivation(self)
        # self.schedule = mesa.time.BaseScheduler(self)
        self.running = True
        self.step_data = []

        # create company knowledge
        self.create_company_know_dist()
        # self.plot_company_know()
        # self.plot_company_know_subplots()

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
        for agt in self.schedule.agents:
            self.roster.append({'Name': agt.name, 'Dept': agt.dept, 'Exp': agt.exp,
                                'Current_Know': len(list(agt.emp_know.elements())),
                               'Status': agt.status, 'Coworker': agt.coworker})
        self.roster = pd.DataFrame(self.roster)
        print(self.roster.to_string())

    # create knowledge distribution for each company department
    def create_dept_know_dist(self):
        # systems engineering department distribution
        se_know = np.random.normal(self.dept_know['SE']['mu'],
                                   self.dept_know['SE']['sigma'],
                                   self.max_know).astype(int)
        # wrap to +/- self.know_cat_ct/2 to create department overlap
        se_know %= self.know_cat_ct
        se_know[se_know>=self.know_cat_ct/2] -= self.know_cat_ct
        se_know[se_know<-self.know_cat_ct/2] += self.know_cat_ct
        self.dept_know['SE']['dist'] = Counter(se_know)


        # software engineering department distribution
        sw_know = np.random.normal(self.dept_know['SW']['mu'],
                                   self.dept_know['SW']['sigma'],
                                   self.max_know).astype(int)
        # wrap to +/- self.know_cat_ct/2 for department overlap
        sw_know %= self.know_cat_ct
        sw_know[sw_know>=self.know_cat_ct/2] -= self.know_cat_ct
        sw_know[sw_know<-self.know_cat_ct/2] += self.know_cat_ct
        self.dept_know['SW']['dist'] = Counter(sw_know)

        # electrical engineering department distribution
        ee_know = np.random.normal(self.dept_know['EE']['mu'],
                                   self.dept_know['EE']['sigma'],
                                   self.max_know).astype(int)
        # wrap to +/- self.know_cat_ct/2 for department overlap
        ee_know %= self.know_cat_ct
        ee_know[ee_know>=self.know_cat_ct/2] -= self.know_cat_ct
        ee_know[ee_know<-self.know_cat_ct/2] += self.know_cat_ct
        self.dept_know['EE']['dist'] = Counter(ee_know)

        # mechanical engineering department distribution
        me_know = np.random.normal(self.dept_know['ME']['mu'],
                                   self.dept_know['ME']['sigma'],
                                   self.max_know).astype(int)
        # wrap to +/- self.know_cat_ct/2 for department overlap
        me_know %= self.know_cat_ct
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
        know %= self.know_cat_ct
        know[know>=self.know_cat_ct/2] -= self.know_cat_ct
        know[know<-self.know_cat_ct/2] += self.know_cat_ct
        return Counter(know)

    def order_agents_by_know(self):
        # sort agents in the step scheduler by their knowledge levels so that
        # the most experienced agents can make the first moves in the model
        know_dict = {i: len(list(agt.emp_know.elements()))
                    for i, agt in enumerate(self.schedule.agents)}
        agent_keys = sorted(know_dict, key=know_dict.get, reverse=True)
        self.schedule._agents = {i : self.schedule._agents[k]
                        for i, k in enumerate(agent_keys)}

    def remove_obsolete_library_data(self):
        # documentation is obsolete after a few years and is less valuable
        # There is an inflection point around the library and the number of
        # employees in the company. More employees allow more knowledge to be
        # added to the library before it becomes obsolete. As more knowledege
        # is documented, employees helping eachother becomes less effective
        # because employees can rely on the library for more efficient knowledge
        # transfer. The efficiency of library knowledge transfer outweighs the
        # time investment for employees spending time to write down knowledge.

        ob_ratio = 0.02 # corresponds to 1 year obsolecence
        # ob_ratio = 0.08 # corresponds to 4 year obsolecence
        for log in self.comp_library_log:
            if log[0] < (self.step_num - self.max_know*ob_ratio):
                self.comp_library[log[1]] -= 1

        self.comp_library_log = [l for l in self.comp_library_log
                                 if l[0] >= (self.step_num - self.max_know*ob_ratio)]

    def step(self):
        """Advance the model by one step."""
        #self.datacollector.collect(self)
        self.step_num += 1
        self.order_agents_by_know()
        self.remove_obsolete_library_data()

        # assign all agent statuses for the step
        for agt in self.schedule.agents:
            agt.status = np.random.choice(self.assigned_states, p=self.state_prob)
            agt.coworker = 'None'

        self.schedule.step()
