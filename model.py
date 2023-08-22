import numpy as np
from mesa.model import Model
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector
from agents import Node
import igraph as ig

# This file contains the model class of the model
# In this model firms are heterogeneous in production and customers are heterogeneous in demand 
# this model focuses on different strategies applied by firms and customers
# customers can choose to 1) rewiring (build new links) or 2) decrease their environmental sensitivities (consciousness)
# firms can choose to 1) let a new customer connect even if the firm can sell out its products, 2) only accept new customers when it is over-producing  

class SignedNetwork(Model):
    def __init__(self, network, alpha = 1, uthr = 0, c_adapt = 0.2, f_adapt = 0.2, c_rate = 0.31, f_rate = 0.6, 
                 unilateral = True, bilateral = False, smart = False, max_steps = 5, verbose = True):
        '''
        The model that simulates the signed network dynamics of firms and customers.
        At every time step, every customer is activated to make a decision and it can either form a link or adapt its environmental sensitivity.
        network: the network in which firms and customers are embedded
        alpha: parameter of utility function when determining the signs of links
        uthr: threshold of utility function when determining the signs of links
        c_adapt: the possibility that a customer chooses to decrease its sensitivity rather than rewiring
        f_adapt: the possibility that a firm chooses to increase its sensitivity
        c_rate: customers' rate to decrease environmental sensitivity
        f_rate: firms' rate to increase environmental sensitivity        
        unilateral: firms accept new customers without any conditions
        bilateral: firms only accept new customers when it is over-producing
        smart: customer will only make a new link with a firm only if the new link is positive
        max_steps: the maximum number of steps to simulate
        verbose: whether to print the information of the model
        '''
        
        # initialize the parameters of the model as the inputs above
        # the meanings of the inputs are described above
        # the parameters have the same names and the meanings as the corresponding inputs
        super().__init__()
        self.network = network
        self.max_steps = max_steps 
        self.schedule = SimultaneousActivation(self)
        self.node_num = network.vcount() # number of nodes in the network
        self.customer_num = network["customer_count"] # the number of customers
        self.alpha = alpha
        self.uthr = uthr
        self.c_adapt = c_adapt
        self.f_adapt = f_adapt
        self.c_rate = c_rate
        self.f_rate = f_rate
        self.unilateral = unilateral
        self.bilateral = bilateral
        self.smart = smart
        self.verbose = verbose 
                  
        # initialize the parameters of each node(agent)
        # the meaning of the parameters below are illustrated in "agents.py"
        for i in range(self.node_num):
            node_type = self.network.vs[i]["type"]
            state = self.network.vs[i]["state"]
            cf_num = self.network.vs[i]["cf_num"]
            demand = self.network.vs[i]["demand"]
            demand_rec = self.network.vs[i]["demand_rec"]      
            production = self.network.vs[i]["production"]
            local_supply = self.network.vs[i]["local_supply"]
            local_demand = self.network.vs[i]["local_demand"]           
            potential_firm = self.network.vs[i]["potential_firm"]
            sensitivity = self.network.vs[i]["sensitivity"]
            local_pollution = self.network.vs[i]["local_pollution"]
            pollution = self.network.vs[i]["pollution"]
            
            node = Node(i, self, node_type, state, cf_num, demand, demand_rec, production, local_supply, local_demand, potential_firm,
                              sensitivity, local_pollution, pollution)
                
            self.schedule.add(node)
        
        self.datacollector = DataCollector(
            model_reporters={"SCS": lambda m: m.scs,
                            "Demand satisfied": lambda m: m.demand_satisfied,
                            "Network": lambda m: m.network_storage,
                            "Firm eta": lambda m: m.firm_eta,
                            "Customer eta": lambda m: m.customer_eta,
                            "Gini": lambda m: m.gini_calculation}
                            
                           )
        """
        model_reporters: a dictionary of functions for collecting data from the model.
        "SCS": share of customers suffering from shortage
        "Demand satisfied": share of demand be satisfied
        "Network": store the network in every step
        "Firm eta": store firms' environmental sensitivity (consciousness) in every step
        "Customer eta": store customers' environmental sensitivity (consciousness) in every step
        "Gini": Gini coefficient of customers' satisfied demand in the network
        """
        
        self.datacollector.collect(self)
        
    @property
    # share of customers suffering from shortage
    def scs(self):
        scs = self.network["scs"]
        # if scs is too small, set it to 0
        if scs < 0.001:
            scs = 0
        return scs
        
    @property
    # share of demand be satisfied
    def demand_satisfied(self):
        customer_drec = 0   # products received by customers
        customer_d = 0     # customers' demands
        
        # calculate the total demand and the total products received by customers
        for i in self.network["customer_list"]:
            customer_drec += self.network.vs[i]["demand_rec"]
            customer_d += self.network.vs[i]["demand"]
        # calculate the share of demand be satisfied on the network level
        ds = customer_drec / customer_d

        # if ds is too large, set it to 1
        if ds > 0.999:
            ds =1
        return ds
               
    @property
    # store the network in every step
    def network_storage(self):
        return self.network.copy()
    
    @property
    # store customers' environmental sensitivity in every step
    def customer_eta(self):
        ce = []
        for i in self.network["customer_list"]:
           
            ce.append(self.network.vs[i]["sensitivity"])
        return ce
    
    @property
    # store firms' environmental sensitivity in every step
    def firm_eta(self):
        fe = []
        for i in self.network["firm_list"]:
            
            fe.append(self.network.vs[i]["sensitivity"])
        return fe
    
    @property
    # calculate the gini coefficient of of customers
    def gini_calculation(self):
        wealth = []
        for i in self.network["customer_list"]:
            
            wealth.append(self.network.vs[i]["demand_rec"])
        
        cum_wealths = np.cumsum(sorted(np.append(wealth,0)))
        sum_wealths = cum_wealths[-1]
        
        xarray = np.array(range(0, len(cum_wealths))) / np.float(len(cum_wealths) - 1)
        yarray = cum_wealths / sum_wealths

        B = np.trapz(yarray, x = xarray)
        A = 0.5 - B
        
        gini = A / (A + B)
        
        if gini <= 0.001:
            gini = 0
        elif gini >=0.999:
            gini = 1
        
        return gini
    
    ##############################################
    ##############################################
    def local_pollution_calculation(self):
        """
        calculate the local pollution of customers in the network
        """
        for i in range(len(self.network.vs)):
        
            if self.network.vs[i]["type"] == "customer":
                # calculate the total pollution from customer i's neighboring firms
                self.network.vs[i]["local_pollution"] = 0
                for j in self.network.neighbors(i):
                    if self.network.vs[j]["type"] == "firm":
                        self.network.vs[i]["local_pollution"] += self.network.vs[j]["pollution"]
                
            # firms do not have local pollution, set as 0
            elif self.network.vs[i]["type"] == "firm":
                self.network.vs[i]["local_pollution"] = 0  # firms do not have local pollution

    #######################################################################################################################
    #######################################################################################################################
    def sign_assign_utility(self):
        """
        calculate the utility between a customer and one of its neighboring firm
        determining the sign of the link between this customer and this firm
        """
        for i in range(len(self.network.vs)):
        
            if self.network.vs[i]["type"] == "customer":
            
                f_neighbor = 0            # the number of neighboring firms
                pollution_i = 0           # the amount of pollution customer expects to receive from a neighboring firm
            
                for j in self.network.neighbors(i):
                
                    if self.network.vs[j]["type"] == "firm":
                    
                        f_neighbor += 1
                
                pollution_i = self.network.vs[i]["demand"] / f_neighbor
                if self.network.vs[i]["sensitivity"]!=0:
                    pollution_i /= self.network.vs[i]["sensitivity"]
                # if the customer is not sensitive to pollution, set the pollution_i to a very large number
                else:
                    pollution_i = 1000000
                    if self.verbose == True:
                        print(self.c_adapt, self.c_rate, self.f_adapt, self.f_rate)
                        self.verbose = False
                # the amount of pollution customer expects to receive from a neighboring firm
                self.network.vs[i]["pollution_exp"] = pollution_i       
                
            if self.network.vs[i]["type"] == "firm":
            
                c_neighbor = 0      # the number of neighboring customers
                pollution_k = 0     # the amount of pollution firm K expects to give to a neighboring customer
            
                for j in self.network.neighbors(i):
                
                    if self.network.vs[j]["type"] == "customer":
                    
                        c_neighbor += 1
                
                if c_neighbor > 0:
                    
                    pollution_k = self.network.vs[i]["production"] / c_neighbor
                
                # very large number when there's no custoemrs link to the firm
                else:
                    pollution_k = self.network.vs[i]["production"]     
                pollution_k /= self.network.vs[i]["sensitivity"]
            
                # the amount of pollution firm K expects to give to a neighboring customer 
                self.network.vs[i]["pollution_exp"] = pollution_k
    
        # assign the sign of links between customers and firms
        # utility = customer i's expected pollution - firm K's expected pollution
        # if utility > 0, this link is positive, otherwise negative
        
        # "alpha" is constant here
        alpha = self.alpha
        uthr = self.uthr
        for i in range(len(self.network.es)):
            if (self.network.vs[self.network.es[i].tuple[0]]["type"] == "customer")\
               & (self.network.vs[self.network.es[i].tuple[1]]["type"] == "firm"):
                
                if (alpha * self.network.vs[self.network.es[i].tuple[0]]["pollution_exp"] - \
                  self.network.vs[self.network.es[i].tuple[1]]["pollution_exp"])> uthr:
                    
                    self.network.es[i]["sign"] = 1
                else:
                    self.network.es[i]["sign"] = -1
                
            if (self.network.vs[self.network.es[i].tuple[0]]["type"] == "firm") & \
               (self.network.vs[self.network.es[i].tuple[1]]["type"] == "customer"):
                if (alpha * self.network.vs[self.network.es[i].tuple[1]]["pollution_exp"] -\
                  self.network.vs[self.network.es[i].tuple[0]]["pollution_exp"]) > uthr:
                    self.network.es[i]["sign"] = 1
                else:
                    self.network.es[i]["sign"] = -1
                    
    ##############################################
    ##############################################    
    
    def shortage_signed(self):
        """
        Update the state of costumers and firms based on the demand and supply mismatch.
        """
        sc_count = 0   # count the number of customers who suffer from shortage
        sf_count = 0   # count the number of firms wo suffer from shortage
        
        for i in range(len(self.network.vs)):
            sd = 0.0   # count the actual supply of firm or actual demand supplied of customer
        
            # figure out in which condition firm will suffer from shortage
            if self.network.vs[i]["type"] == "firm":
    
                for j in self.network.neighbors(i):
                
                    if self.network.vs[j]["type"] == "customer":
                    
                        sk_t = 0  # calculate the denominator part of distribution factor
                        for k in self.network.neighbors(j):
                            if self.network.vs[k]["type"] == "firm":
                            
                                sk_t += np.exp(- self.network.vs[j]["sensitivity"] * \
                                           (self.network.vs[k]["pollution"]/\
                                            self.network.vs[j]["local_pollution"]))
                    
                        # calculate the numerator part of distribution factor
                        sk = np.exp(- self.network.vs[j]["sensitivity"] *\
                                    (self.network.vs[i]["pollution"]/self.network.vs[j]["local_pollution"]))
                    
                        # to get the sign of the link between node i and node j
                        link_ij = self.network.get_eid(i,j)
                        sign_ij = self.network.es[link_ij]["sign"]
                    
                        # calculate the demand of customer j from firm i, add it to total demand requirement of firm i
                        sd += self.network.vs[j]["demand"] *  sk / sk_t * ((1 + sign_ij) / 2)
                    
                self.network.vs[i]["local_demand"] = sd   # local demand of the firm
                self.network.vs[i]["local_supply"] = 0    # firms do not have local supply
                self.network.vs[i]["demand_rec"] = 0      # firms do not have demand received
                self.network.vs[i]["pollution_rec"] = 0   # firms do not have pollution received
                # the amount of products the firm has sold
                self.network.vs[i]["production_sold"] = min(self.network.vs[i]["local_demand"], \
                                                                  self.network.vs[i]["production"])
            
                # firm state is determined by local_demand and average production
                if self.network.vs[i]["local_demand"] < 0.999 * self.network.vs[i]["production"]:
                    self.network.vs[i]["state"] = 1     # firms with overproduction have a state of 1
                
                    # the amount of products the firm has not sold
                    self.network.vs[i]["production_unsold"] = self.network.vs[i]["production"] -\
                                                     self.network.vs[i]["production_sold"]
                
                    sf_count += 1
                else:
                    self.network.vs[i]["state"] = 0    # firms without overproduction have a state of 0
                    self.network.vs[i]["production_unsold"] = 0   # all products have been sold
            
           
            # figure out in which condition customer will suffer from shortage
            if self.network.vs[i]["type"] == "customer":
                ls = 0    # calculate local supply
                sk_t = 0  # calculate the denominator part of distribution factor
                P_rec = 0    # calculate the customer's pollution received
                for k in self.network.neighbors(i):
                    if self.network.vs[k]["type"] == "firm":
                            
                        sk_t += np.exp(- self.network.vs[i]["sensitivity"] *\
                                       (self.network.vs[k]["pollution"]/self.network.vs[i]["local_pollution"]))
                        
                for j in self.network.neighbors(i):
                
                    if self.network.vs[j]["type"] == "firm":
                    
                        # to get the sign of the link between node i and node j
                        link_ij = self.network.get_eid(i,j)
                        sign_ij = self.network.es[link_ij]["sign"]
                    
                        # the max. amount production firm j can offer customer i
                        max_supply = self.network.vs[j]["production"] / self.network.vs[j]["cf_num"]* ((1 + sign_ij) / 2)
                    
                        # the max. amount production customer i want to gain from firm j
                        sk = np.exp(- self.network.vs[i]["sensitivity"] *\
                                    (self.network.vs[j]["pollution"]/self.network.vs[i]["local_pollution"]))

                        max_want = self.network.vs[i]["demand"] *  sk / sk_t * ((1 + sign_ij) / 2)
                    
                        # production customer i can really gain from firm j is the min. value of max_supply and max_want
                        sd += min(max_supply, max_want)
                        ls += max_supply
                        P_rec += min(max_supply, max_want) * self.network.vs[j]["unit_pollution"]
                    
                self.network.vs[i]["local_demand"] = 0         # customers do not have local demand 
                self.network.vs[i]["local_supply"] = ls        # customer's local supply
                self.network.vs[i]["demand_rec"] = min(sd, self.network.vs[i]["demand"])         # customer's demand received
                self.network.vs[i]["pollution_rec"] = P_rec    # customer's pollution received
                self.network.vs[i]["production_sold"] = 0      # customers do not have this attribute
                self.network.vs[i]["production_unsold"] = 0    # customers do not have this attribute
            
                # customer state is determined by demand_rec and the threshold of satisfaction
                if self.network.vs[i]["demand_rec"] < 0.999 * self.network.vs[i]["demand"]:
                    self.network.vs[i]["state"] = 0   # customer suffer from shortage has a state of 0
                    sc_count += 1
                else:
                    self.network.vs[i]["state"] = 1   # custoemr do not suffer from shortage has a state of 1
    
        # assign new attributes to network 
        self.network["sc_count"] = sc_count   # number of customers who suffer from shortage
        self.network["sf_count"] = sf_count   # number of firms who suffer from shortage
        self.network["scs"] = sc_count / self.network["customer_count"]   # share of customers who suffer from shortage
        self.network["sfs"] = sf_count / self.network["firm_count"]   # share of firms who suffer from shortage 
    
    #############################################################################################################
    #############################################################################################################

    # do the simulation for one step
    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        self.local_pollution_calculation()
        self.sign_assign_utility()
        self.shortage_signed()
           
        if self.schedule.steps > self.max_steps:
            self.running = False