import numpy as np
from mesa.agent import Agent

"""
This file is used to define the agent class in the model.
"""
class Node(Agent):
    def __init__(self, unique_id, model, node_type, state, cf_num, demand, demand_rec, production, local_supply, local_demand, potential_firm,
                sensitivity, local_pollution, pollution):
        """
        Create a new node.
        Args:
            unique_id: Unique identifier for the node.
            model: Model in which the node lives.
            node_type: "customer" or "firm"
            state: 0 or 1, 0 means shortage, 1 means no shortage
            cf_num: number of customer-firm links
            demand: the amount of products the customer wants to buy
            demand_rec: the amount of products the customer actually buys
            production: the amount of products the firm produces
            local_supply: the amount of products the firm can offer to its customers
            local_demand: the amount of products the customer can gain from its firms
            potential_firm: the list of firms that the customer can make new links to
            sensitivity: the environmental sensitivity (consciousness) of the customer or firm
            local_pollution: the amount of pollution the customer expects to receive from its firms
            pollution: the amount of pollution the firm expects to send to its customers
        """
        # initialize the parameters of the agent as the inputs above
        # the meanings of the inputs are described above
        # the parameters have the same names and the meanings as the corresponding inputs
        super().__init__(unique_id = unique_id, model = model)
        self.node_type = node_type
        self.state = state
        self.cf_num = cf_num
        self.demand = demand
        self.demand_rec = demand_rec
        self.production = production
        self.local_supply = local_supply
        self.local_demand = local_demand
        self.potential_firm = potential_firm
        self.sensitivity = sensitivity
        self.local_pollution = local_pollution
        self.pollution = pollution
        
        # initialize "next" values of parameters and assign them to the current values
        self._next_state = state
        self._next_cf_num = cf_num
        self._next_demand_rec = demand_rec
        self._next_local_supply = local_supply
        self._next_local_demand = local_demand
        self._next_sensitivity = sensitivity
        self._next_local_pollution = local_pollution
        self._next_pollution = pollution

    
    # in the network, agents are represented by nodes
    # before every simulation, agent updates its parameters as the network changes
    def parameter_update(self):
        self.state = self.model.network.vs[self.unique_id]["state"]
        self.cf_num = self.model.network.vs[self.unique_id]["cf_num"]
        self.demand_rec = self.model.network.vs[self.unique_id]["demand_rec"]
        self.local_supply = self.model.network.vs[self.unique_id]["local_supply"]
        self.local_demand = self.model.network.vs[self.unique_id]["local_demand"]
        self.potential_firm = self.model.network.vs[self.unique_id]["potential_firm"]
        self.sensitivity = self.model.network.vs[self.unique_id]["sensitivity"]
        self.local_pollution = self.model.network.vs[self.unique_id]["local_pollution"]
        self.pollution = self.model.network.vs[self.unique_id]["pollution"]
    
    #######################################################################################################################
    #######################################################################################################################
    
    # after every simulation, agent updates its "next" parameters again, according to the network
    def next_parameter_update(self):
        self._next_state = self.model.network.vs[self.unique_id]["state"]
        self._next_cf_num = self.model.network.vs[self.unique_id]["cf_num"]
        self._next_demand_rec = self.model.network.vs[self.unique_id]["demand_rec"]
        self._next_local_supply = self.model.network.vs[self.unique_id]["local_supply"]
        self._next_local_demand = self.model.network.vs[self.unique_id]["local_demand"]
        self._next_potential_firm = self.model.network.vs[self.unique_id]["potential_firm"]
        self._next_sensitivity = self.model.network.vs[self.unique_id]["sensitivity"]
        self._next_local_pollution = self.model.network.vs[self.unique_id]["local_pollution"]
        self._next_pollution = self.model.network.vs[self.unique_id]["pollution"]

    #######################################################################################################################
    #######################################################################################################################
    
    # calculate the utility between a customer and one of its neighboring firm
    # determining the sign of the link between this customer and this firm
    def sign_assign_utility(self):
        for i in range(len(self.model.network.vs)):
            # calculate the number of pollution the customer expects to receive from its neighboring firms
            if self.model.network.vs[i]["type"] == "customer":
            
                f_neighbor = 0
                pollution_i = 0
            
                for j in self.model.network.neighbors(i):
                
                    if self.model.network.vs[j]["type"] == "firm":
                    
                        f_neighbor += 1
                
                pollution_i = self.model.network.vs[i]["demand"] / f_neighbor
                pollution_i /= self.model.network.vs[i]["sensitivity"]
            
                # the amount of pollution customer expects to receive from a neighboring firm
                self.model.network.vs[i]["pollution_exp"] = pollution_i       
                
            # calculate the number of pollution the firm expects to send to its neighboring customers
            if self.model.network.vs[i]["type"] == "firm":
            
                c_neighbor = 0
                pollution_k = 0
            
                for j in self.model.network.neighbors(i):
                
                    if self.model.network.vs[j]["type"] == "customer":
                    
                        c_neighbor += 1
                
                if c_neighbor > 0:
                    
                    pollution_k = self.model.network.vs[i]["production"] / c_neighbor
                else:
                    pollution_k = self.model.network.vs[i]["production"]     # very large number when there's no custoemrs link to the firm
                pollution_k /= self.model.network.vs[i]["sensitivity"]
            
                # the amount of pollution firm K expects to give to a neighboring customer 
                self.model.network.vs[i]["pollution_exp"] = pollution_k
    
        # assign the sign of links between customers and firms
        # utility = customer i's expected pollution - firm K's expected pollution
        # if utility > 0, this link is positive, otherwise negative
        
        # "alpha" is constant here
        alpha = self.model.alpha

        # calculate the sign of all the links between customers and firms that are connected
        for i in range(len(self.model.network.es)):
            if (self.model.network.vs[self.model.network.es[i].tuple[0]]["type"] == "customer")\
               & (self.model.network.vs[self.model.network.es[i].tuple[1]]["type"] == "firm"):
                
                if (alpha * self.model.network.vs[self.model.network.es[i].tuple[0]]["pollution_exp"] - \
                  self.model.network.vs[self.model.network.es[i].tuple[1]]["pollution_exp"])> self.model.uthr:
                    
                    self.model.network.es[i]["sign"] = 1
                else:
                    self.model.network.es[i]["sign"] = -1
                
            if (self.model.network.vs[self.model.network.es[i].tuple[0]]["type"] == "firm") & \
               (self.model.network.vs[self.model.network.es[i].tuple[1]]["type"] == "customer"):
                if (alpha * self.model.network.vs[self.model.network.es[i].tuple[1]]["pollution_exp"] -\
                  self.model.network.vs[self.model.network.es[i].tuple[0]]["pollution_exp"]) > self.model.uthr:
                    self.model.network.es[i]["sign"] = 1
                else:
                    self.model.network.es[i]["sign"] = -1
                    
    #######################################################################################################################
    #######################################################################################################################
    
    # calculate the states of agents after each time step
    
    def shortage_signed(self):
    
        sc_count = 0   # count the number of customers who suffer from demand shortage
        sf_count = 0   # count the number of firms who suffer from overproduction
        
    
        for i in range(len(self.model.network.vs)):
        
            sd = 0.0   # count the actual supply of firm or actual demand supplied of customer
        
        
            # figure out in which condition firm will suffer from overproduction
        
            if self.model.network.vs[i]["type"] == "firm":
                
                for j in self.model.network.neighbors(i):
                
                    if self.model.network.vs[j]["type"] == "customer":
                    
                        sk_t = 0  # calculate the denominator part of distribution factor
                        for k in self.model.network.neighbors(j):
                            if self.model.network.vs[k]["type"] == "firm":
                            
                                sk_t += np.exp(- self.model.network.vs[j]["sensitivity"] * \
                                           (self.model.network.vs[k]["pollution"]/\
                                            self.model.network.vs[j]["local_pollution"]))
                    
                        # calculate the numerator part of distribution factor
                        sk = np.exp(- self.model.network.vs[j]["sensitivity"] *\
                                    (self.model.network.vs[i]["pollution"]/self.model.network.vs[j]["local_pollution"]))
                    
                        # to get the sign of the link between node i and node j
                        link_ij = self.model.network.get_eid(i,j)
                        sign_ij = self.model.network.es[link_ij]["sign"]
                    
                        # calculate the demand of customer j from firm i, add it to total demand requirement of firm i
                        sd += self.model.network.vs[j]["demand"] *  sk / sk_t * ((1 + sign_ij) / 2)
                    
                self.model.network.vs[i]["local_demand"] = sd   # local demand of the firm
                self.model.network.vs[i]["local_supply"] = 0    # firms do not have local supply
                self.model.network.vs[i]["demand_rec"] = 0      # firms do not have demand received
                self.model.network.vs[i]["pollution_rec"] = 0   # firms do not have pollution received
                # the amount of products the firm has sold
                self.model.network.vs[i]["production_sold"] = min(self.model.network.vs[i]["local_demand"], \
                                                                  self.model.network.vs[i]["production"])
            
                # firm state is determined by its local_demand and its production
                if self.model.network.vs[i]["local_demand"] < 0.999 * self.model.network.vs[i]["production"]:
                    self.model.network.vs[i]["state"] = 1     # firms with overproduction have a state of 1
                
                    # the amount of products the firm has not sold
                    self.model.network.vs[i]["production_unsold"] = self.model.network.vs[i]["production"] -\
                                                     self.model.network.vs[i]["production_sold"]
                
                    sf_count += 1 # count the number of firms who suffer from overproduction
                else:
                    self.model.network.vs[i]["state"] = 0    # firms without overproduction have a state of 0
                    self.model.network.vs[i]["production_unsold"] = 0   # all products have been sold
            
           
            # figure out in which condition customer will suffer from demand shortage
            if self.model.network.vs[i]["type"] == "customer":
                ls = 0    # calculate local supply
                sk_t = 0  # calculate the denominator part of distribution factor
                P_rec = 0    # calculate the customer's pollution received
                for k in self.model.network.neighbors(i):
                    # calculate the denominator part for a customer's preference
                    if self.model.network.vs[k]["type"] == "firm":
                            
                        sk_t += np.exp(- self.model.network.vs[i]["sensitivity"] *\
                                       (self.model.network.vs[k]["pollution"]/self.model.network.vs[i]["local_pollution"]))
                        
                for j in self.model.network.neighbors(i):
                
                    if self.model.network.vs[j]["type"] == "firm":
                    
                        # to get the sign of the link between node i and node j
                        link_ij = self.model.network.get_eid(i,j)
                        sign_ij = self.model.network.es[link_ij]["sign"]
                    
                        # the max. amount production firm j can offer customer i
                        max_supply = self.model.network.vs[j]["production"] / self.model.network.vs[j]["cf_num"]* ((1 + sign_ij) / 2)
                    
                        # the max. amount production customer i want to gain from firm j
                        sk = np.exp(- self.model.network.vs[i]["sensitivity"] *\
                                    (self.model.network.vs[j]["pollution"]/self.model.network.vs[i]["local_pollution"]))

                        max_want = self.model.network.vs[i]["demand"] *  sk / sk_t * ((1 + sign_ij) / 2)
                    
                        # production customer i can really gain from firm j is the min. value of max_supply and max_want
                        sd += min(max_supply, max_want)
                        ls += max_supply
                        P_rec += min(max_supply, max_want) * self.model.network.vs[j]["unit_pollution"]
                    
                self.model.network.vs[i]["local_demand"] = 0         # customers do not have local demand 
                self.model.network.vs[i]["local_supply"] = ls        # customer's local supply
                self.model.network.vs[i]["demand_rec"] = min(sd, self.model.network.vs[i]["demand"])         # customer's demand received
                self.model.network.vs[i]["pollution_rec"] = P_rec    # customer's pollution received
                self.model.network.vs[i]["production_sold"] = 0      # customers do not have this attribute
                self.model.network.vs[i]["production_unsold"] = 0    # customers do not have this attribute
            
                # customer state is determined by demand_rec and the threshold of satisfaction
                if self.model.network.vs[i]["demand_rec"] < 0.999 * self.model.network.vs[i]["demand"]:
                    self.model.network.vs[i]["state"] = 0   # customer suffer from shortage has a state of 0
                    sc_count += 1
                else:
                    self.model.network.vs[i]["state"] = 1   # custoemr do not suffer from shortage has a state of 1
    
        # assign new attributes to network 
        self.model.network["sc_count"] = sc_count   # number of customers who suffer from shortage
        self.model.network["sf_count"] = sf_count   # number of firms who suffer from shortage
        self.model.network["scs"] = sc_count / self.model.network["customer_count"]   # share of customers who suffer from shortage
        self.model.network["sfs"] = sf_count / self.model.network["firm_count"]   # share of firms who suffer from shortage 
    
    #######################################################################################################################
    #######################################################################################################################
    
    def sign_check(self, c, f):
        """
        # check the utility of new link 
        # only when the new link is positive, the new link can be built
        # c is a customer node, f is a firm node
        """
        if self.model.network.vs[c]["sensitivity"]:
            # first, calculate the pollution the customer expects to receive
            f_neighbor = 0        # number of firms who are neighbors of customer c
            pollution_i = 0       # pollution customer c expects to receive

            for j in self.model.network.neighbors(c):
                if self.model.network.vs[j]["type"] == "firm":
                    f_neighbor += 1

            pollution_i = self.model.network.vs[c]["demand"] / (f_neighbor + 1) # new link should be included
            pollution_i /= self.model.network.vs[c]["sensitivity"]

            # second, calculate the pollution the firm expects to send
            c_neighbor = 0
            pollution_k = 0

            for j in self.model.network.neighbors(f):
                if self.model.network.vs[j]["type"] == "customer":
                    c_neighbor += 1

            pollution_k = self.model.network.vs[f]["production"] / (c_neighbor + 1)  # new link should be included
            pollution_k /= self.model.network.vs[f]["sensitivity"]

            # third, calculate the sign of the netowrk
            sign = 0
            if self.model.alpha * pollution_i - pollution_k > self.model.uthr:
                sign = 1
            else:
                sign = -1
        else:
            sign = 1
            
        return sign
      
    #######################################################################################################################
    #######################################################################################################################
    
    # in this part, customer will choose whether to 1) decrease sensitivity (environmental consciousness) or rewiring (build a new link)
    # results of rewiring depend on both customer's action and firms strategy (unilateral, bilateral or smart)
       
    def customer_action(self):
        
        # generate a random number, if it is smaller than "c_adapt", the customer will decrease its sensitivity by "c_rate"
        # else, the customer will try to build new links to firms in its "potential firm" list
        
        c_random = np.random.random()
        
        # customers only take action when it suffers from shortage
        if self.state == 0:
            # case 1: customer chooses to decrease its sensitivity
            if c_random <= self.model.c_adapt:   
            
                self.model.network.vs[self.unique_id]["sensitivity"] *= (1 - self.model.c_rate)
                
                # # recalcualte the signs of links
                # self.sign_assign_utility()                                    
                # # recalculate the states of all the agents
                # self.shortage_signed()
            
            # case 2: customer chooses to add new customer-firm links (rewiring)    
            else:
                # case 2.1: unilateral, customers can feel free to make new links to firms
                if self.model.unilateral == True:
                    
                    # no firms can make links to,
                    # the customer has no choice but to decide whether to decrease its sensitivity or not
                    if len(self.model.network.vs[self.unique_id]["potential_firm"]) == 0:
                        
                        c_r1 = np.random.random()
                        if c_r1 <= self.model.c_adapt:   
                            self.model.network.vs[self.unique_id]["sensitivity"] *= (1 - self.model.c_rate)
                
 
                    else:
                        new_firm = np.random.choice(self.model.network.vs[self.unique_id]["potential_firm"])
                        
                        # in unilateral case, customers can feel free to add new customer-firm links
                        self.model.network.add_edge(self.unique_id, new_firm)

                        self.model.network.vs[self.unique_id]["potential_firm"].remove(new_firm)
                    
                        # cf_num (number of customer-firm links) plus 1
                        self.model.network.vs[self.unique_id]["cf_num"] += 1
                        self.model.network.vs[new_firm]["cf_num"] += 1
                        
                      
                
                # case 2.2: bilateral, firms will accept new customers only when this firm is over-producing
                elif self.model.bilateral == True:
                    
                    # no firms can make links to,
                    # the customer has no choice but decide whether to decrease its sensitivity or not
                    if len(self.model.network.vs[self.unique_id]["potential_firm"]) == 0:
                        
                        c_r2 = np.random.random()
                        if c_r2 <= self.model.c_adapt:   
                            self.model.network.vs[self.unique_id]["sensitivity"] *= (1 - self.model.c_rate)
                
                    
                    else:
                        new_firm = np.random.choice(self.model.network.vs[self.unique_id]["potential_firm"])
                        
                        # in bilateral case,firms only accept new customers when this firm is over-producing
                        if self.model.network.vs[new_firm]["state"] == 1:
                        
                            self.model.network.add_edge(self.unique_id, new_firm)

                            self.model.network.vs[self.unique_id]["potential_firm"].remove(new_firm)
                    
                            # cf_num (number of customer-firm links) plus 1
                            self.model.network.vs[self.unique_id]["cf_num"] += 1
                            self.model.network.vs[new_firm]["cf_num"] += 1
                        

                        # the chosen firm can already sell out products and do not want to accept new customers
                        # the customer has no choice but decide whether to decrease its sensitivity or not
                        else:
                            c_r3 = np.random.random()
                            if c_r3 <= self.model.c_adapt:   
                                self.model.network.vs[self.unique_id]["sensitivity"] *= (1 - self.model.c_rate)
                
                # case 2.3: smart. Customer only make new link to a firm only if the new link is positive
                elif self.model.smart == True:
                    
                    # no firms can make links to,
                    # the customer has no choice but decide whether to decrease its sensitivity or not
                    if len(self.model.network.vs[self.unique_id]["potential_firm"]) == 0:
                        
                        c_r2 = np.random.random()
                        if c_r2 <= self.model.c_adapt:   
                            self.model.network.vs[self.unique_id]["sensitivity"] *= (1 - self.model.c_rate)
                
                    
                    else:
                        new_firm = np.random.choice(self.model.network.vs[self.unique_id]["potential_firm"])
                        new_sign = self.sign_check(self.unique_id, new_firm)
                        
                        # new link is positive and the new link will be built
                        if new_sign == 1:
                            self.model.network.add_edge(self.unique_id, new_firm)

                            self.model.network.vs[self.unique_id]["potential_firm"].remove(new_firm)
                    
                            # cf_num (number of customer-firm links) plus 1
                            self.model.network.vs[self.unique_id]["cf_num"] += 1
                            self.model.network.vs[new_firm]["cf_num"] += 1
                        
                        
                        # new link is negative and cannot be built
                        # customer has no choice but to decide whether to decrease its sensitivity
                        else:
                            c_r3 = np.random.random()
                            if c_r3 <= self.model.c_adapt:   
                                self.model.network.vs[self.unique_id]["sensitivity"] *= (1 - self.model.c_rate)
                
                    
    
    #######################################################################################################################
    #######################################################################################################################
    
    # in this part, a firm with over-production can choose to increase sensitivity (environmental consciousness) or not
    # unilateral or bilateral are coded in "customer-action" part
       
    def firm_action(self):
        
        # generate a random number, if it is smaller than "f_adapt", the firm will increase its sensitivity by "f_rate"
        # otherwise the firm will keep unchanged
        
        f_random = np.random.random()
        
        # adaptation only happens when the firm is over-producing
        if self.state == 1:
            if f_random <= self.model.f_adapt:   
            
                self.model.network.vs[self.unique_id]["sensitivity"] *= (1 + self.model.f_rate)                                                   
                # don't forget to update the pollution of the firm
                self.model.network.vs[self.unique_id]["pollution"] = self.model.network.vs[self.unique_id]["production"] / self.model.network.vs[self.unique_id]["sensitivity"] 
                
     
    #######################################################################################################################
    #######################################################################################################################    
    
    def step(self):
        '''
        Warning: in functions "parameter_update" and "next_parameter_update", only the currently activated agent updates its parameters
              Other agents' attributes may also change, but does not be updated by these two functions.
              So, in "model", we need to directly use the parameters from "network" rather than using agent.parameter
        '''
        # first, upgrade the parameters according to the network before taking the step
        self.parameter_update()
        
        # second, take actions
        if self.node_type == "customer":
            self.customer_action()
        elif self.node_type == "firm":
            self.firm_action()
        
        # third, update the _next parameters according to the network after taking the step
        self.next_parameter_update()            
    
    # update the parameters of agents according to the network after taking the step
    def advance(self):
        self.state = self._next_state
        self.cf_num = self._next_cf_num
        self.demand_rec = self._next_demand_rec
        self.local_supply = self._next_local_supply
        self.local_demand = self._next_local_demand
        self.potential_firm = self._next_potential_firm
        self.sensitivity = self._next_sensitivity
        self.local_pollution = self._next_local_pollution
        self.pollution = self._next_pollution

    