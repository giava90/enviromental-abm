import numpy as np
from scipy import interpolate
import igraph as ig
from model import SignedNetwork
from multiprocessing import Pool
from tqdm import tqdm

"""
This file contains all the functions we use in this project.
"""

#############################################################################################################
#############################################################################################################

# Functions for generating degree sequence 
def f(x,kappa1,kappa2):
    """
    x: degree 
    kappa1, kappa2: constants for generating the broad degree distribution
    """

    # define the pdf for degree distribution sequence
    return np.power(x, -kappa1) * np.exp(-kappa2*x) # our pfd looks like this


def sample(g,kappa1 = 2.1,kappa2 = 0.004):
    """
    g: probability density function (pdf)
    kappa1, kappa2: constants for generating the broad degree distribution
    """

    # use inversed cdf to generate pdf
    x = np.linspace(1,1001,1000000)
    y = g(x,kappa1,kappa2)                        # probability density function, pdf
    cdf_y = np.cumsum(y)            # cumulative distribution function, cdf
    cdf_y = cdf_y/cdf_y.max()       # takes care of normalizing cdf to 1.0
    inverse_cdf = interpolate.interp1d(cdf_y,x)    # this is a function
    return inverse_cdf

#############################################################################################################
#############################################################################################################

# Functions for generating network with broad distribution
def broad_dist_degrees(N_agents, kappa1, kappa2):
    """
    N_agents: number of agents (nodes) in the network
    kappa1, kappa2: constants for generating the broad degree distribution
    """

    #generate broad degree distribution 
    degree_broad = [1]
    while True :
        inverse_cdf = sample(f, kappa1, kappa2)
        a = inverse_cdf.x[0]
        b = inverse_cdf.x[-1]
        uniform_samples = (b - a) * np.random.random_sample(N_agents) + a
        degree_broad = inverse_cdf(uniform_samples)
        
        # without self-loop, the links from one agent cannot be greater then N_agent - 1
        for i in range(len(degree_broad)):
            if degree_broad[i] > (N_agents - 1):
                degree_broad[i] = N_agents - 1
        
        degree_broad = list(np.intc(degree_broad))  # convert every items to integers
        
        # the degree sequence should have three characteristics:
        # 1) it is graphic, 
        # 2) the largest firm links to more than half of the nodes,
        # 3) number of firms should be greater than 7 (average of degree sequence generation) and less than 10 (10% of total)
        if ig.is_graphical(degree_broad):
            if max(degree_broad) >= N_agents / 2:
                #if ((len(np.where(np.array(degree_broad)>= 10)[0]) >7) and (len(np.where(np.array(degree_broad)>= 10)[0]) < 10)):
                break

    return degree_broad

#############################################################################################################
#############################################################################################################

# Function for assigning the types of agents (nodes)

def type_assignment(network):
    """
    network: the network in this model
    """
    # assign attribute "type" to vertex
    for vertex in range(len(network.vs)):
        if network.degree(vertex) > 10:
            network.vs[vertex]["type"] = "firm" # degree >10 are firms
        else:
            network.vs[vertex]["type"] = "customer" # degree <= 10 are customers
            
    return network

#############################################################################################################
#############################################################################################################

# Functions for assigning the sign of links (edges)
def sign_assignment(network):
    """ 
    assign attribute "sign" to edges
    edges between two nodes of same type are positive (+1)
    edges between two nodes of different types are negative (-1)
    
    Args:
        network: the network in this model
    """
    
    for i in range(len(network.es)):
        if network.vs[network.es[i].tuple[0]]["type"] == network.vs[network.es[i].tuple[1]]["type"]:
            network.es[i]["sign"] = -1
        elif network.vs[network.es[i].tuple[0]]["type"] != network.vs[network.es[i].tuple[1]]["type"]:
            network.es[i]["sign"] = 1
        else:
            network.es[i]["sign"] = 0
    
    return network

#############################################################################################################
#############################################################################################################

# Function for calculating the number of links between firms and customers
def c_f_links(network):
    """
    network: the network in this model
    """

    # For every node, calculate its links to nodes of different type
    for i in range(len(network.vs)):
        c_f_num = 0
        
        for neighbor in network.neighbors(i):
            if network.vs[i]["type"] != network.vs[neighbor]["type"]:
                c_f_num += 1
        
        network.vs[i]["cf_num"] = c_f_num
    
    return network

#############################################################################################################
#############################################################################################################

# The function to calculate some basic characteristics of this customer-firm network
def network_conclusion(network):
    """
    network: the network in this model
    """
    
    # assign five attributes to network:
    # 1) number of firms, 2) number of customers,
    # 3) number of firm-firm links, 4) number of customer-customer links, 5) number of firm-customer links
    # first, we count the numbers of firms and customers in the network
   
    firm_count = 0         # number of firms
    customer_count = 0     # number of customers
    
    for i in range(len(network.vs)):
        if network.vs[i]["type"] == "firm":
            firm_count += 1
        elif network.vs[i]["type"] == "customer":
            customer_count += 1
           
    network["firm_count"] = firm_count
    network["customer_count"] = customer_count
    
    # second, we count the number of links between different types of vertex
    f_f_links = 0         # number of firm-firm links
    c_c_links = 0         # number of customer-customer links
    f_c_links = 0         # number of firm-customer links
    
    for i in range(len(network.es)):
        
        if ((network.vs[network.es[i].tuple[0]]["type"] == "firm") and (network.vs[network.es[i].tuple[1]]["type"] == "firm")):
            f_f_links += 1
        elif ((network.vs[network.es[i].tuple[0]]["type"] == "customer") and (network.vs[network.es[i].tuple[1]]["type"] == "customer")):
            c_c_links += 1
        else:
            f_c_links += 1
    
    network["f_f_links"] = f_f_links
    network["c_c_links"] = c_c_links
    network["f_c_links"] = f_c_links
    
    return network

#############################################################################################################
#############################################################################################################

# Function to assign firms' production and customers' demand, as constant
def production_demand_const(network, production, demand):
    """
    network: the network in this model
    production: the production of firms
    demand: the demand of customers
    """
    # we start with the simplest scenario: every customer has the same demand and every firm has the same production
    for i in range(len(network.vs)):
        if network.vs[i]["type"] == "firm":
            network.vs[i]["production"] = production 
            
            network.vs[i]["demand"] = 0              # firms do not have demand
                
        
        elif network.vs[i]["type"] == "customer":
            
            network.vs[i]["demand"] = demand  
            
            network.vs[i]["production"] = 0         #customers do not have production
     
    return network

#############################################################################################################
#############################################################################################################

# Function to assign firms' production and customers' demand, as heterogeneous
def production_demand_hetero(network, production, demand, p_dev = 0.45, d_dev =0.05):
    """
    demand / production will be heterogeneous among customers / firms
    the mean of total demand and total supply will be equal
    but for every customer/firm, its demand/production will be normally distributed aroud the mean value
    
    network: the network in this model
    production: the mean value of production of firms
    demand: the mean value of demand of customers
    p_dev: the standard deviation of production of firms
    d_dev: the standard deviation of demand of customers
    """
    for i in range(len(network.vs)):
        if network.vs[i]["type"] == "firm":

            network.vs[i]["production"] = np.random.normal(production, p_dev)
            
            network.vs[i]["demand"] = 0              # firms do not have demand
    
        
        elif network.vs[i]["type"] == "customer":
            
            # customers' demand is normally distributed with the mean value "demand"
            network.vs[i]["demand"] = np.random.normal(demand, d_dev)      
            
            network.vs[i]["production"] = 0         #customers do not have production
     
    return network

#############################################################################################################
#############################################################################################################

# Function to calculate customers and firms' satisfaction levels without considering environmental sensitivity (consciousness)

def non_satisfaction(network):
    """
    network: the network in this model
    """

    # calculate nodes' states without considering environmental sensitivity and signs of links
    sc_count = 0   # count the number of customers who suffer from shortage
    sf_count = 0   # count the number of firms who is over-producing
    
    for i in range(len(network.vs)):
        
        sd = 0.0 # count the actual supply of firm or actual demand supplied of customer
        
        if network.vs[i]["type"] == "firm":
            
            for j in network.neighbors(i):
                
                if network.vs[j]["type"] == "customer":
                    
                    # simplified version of fomula (3) because only firm-customer links are positive now
                    sd += network.vs[j]["demand"] / network.vs[j]["cf_num"]
           
            
            network.vs[i]["local_demand"] = sd      # assign the attribute "local_demand" to firms  
            network.vs[i]["local_supply"] = 0       # firms do not have "local_supply"
            network.vs[i]["demand_rec"] = 0         # firms do not have "demand_rec"
            
            # compare local_demand and average production
            if network.vs[i]["local_demand"] < 0.999 * network.vs[i]["production"]:
                network.vs[i]["state"] = 1 # state = 1 for a firm means overproduction
                sf_count += 1
            else:
                network.vs[i]["state"] = 0 # state = 0 for a firm means production shortage
        
        if network.vs[i]["type"] == "customer":
            
            for j in network.neighbors(i):
                
                if network.vs[j]["type"] == "firm":
                    
                    # simplified version of fomula (4) because only firm-customer links are positive now
                    sd += network.vs[j]["production"] / network.vs[j]["cf_num"]
            
            network.vs[i]["local_supply"] = sd   # assign the attribute "local_supply" to customers    
            network.vs[i]["local_demand"] = 0    # customers do not have "local_demand"
            
            # products received by customer is the minimum of local supply and the customer's demand
            network.vs[i]["demand_rec"] = min(sd, network.vs[i]["demand"])     
            
            # compare the products received by customer and the customer's threshold
            if network.vs[i]["demand_rec"] < 0.999 * network.vs[i]["demand"]:
                network.vs[i]["state"] = 0   # state = 0 for a customer means demand shortage
                sc_count += 1
            else:
                network.vs[i]["state"] = 1    # state = 1 for a customer means demand surplus 
    
    # assign new attributes to network 
    network["sc_count"] = sc_count   # number of customers who suffer from shortage
    network["sf_count"] = sf_count   # number of firms who suffer from shortage
    network["scs"] = sc_count / network["customer_count"]   # share of customers who suffer from shortage
    network["sfs"] = sf_count / network["firm_count"]   # share of firms who suffer from shortage
   
    return network

#############################################################################################################
#############################################################################################################

# Function to assign customers and firms' environmental sensitivity (consciousness) and the pollution generated by firms
def env_assign(network, c_lower = 1, c_upper = 2, f_lower = 1, f_upper = 2):
    """
    We write a function to:
    1) assign uniformally distributed environmental sensitivity (consciousness) to nodes (both firms and customers)
    2) calculate pollution per item per firm
    3) calculate pollution generated by each firm

    Customers' and firms' environmental sensitivities follow uniform distribution.
    customers' sensitivities are sampled between c_lower and c_upper
    firms' sensitivities are sampled between f_lower and f_upper
    """
    
    for i in range(len(network.vs)):
        # assign a uniformly distributed environmental sensitivity to nodes (both firms and customers)
        if network.vs[i]["type"] == "customer":
            
            network.vs[i]["sensitivity"] = np.random.uniform(c_lower,c_upper)
            
            # customers do not generate pollution
            network.vs[i]["unit_pollution"] = 0
            network.vs[i]["pollution"] = 0
        
        if network.vs[i]["type"] == "firm":
            network.vs[i]["sensitivity"] = np.random.uniform(f_lower,f_upper)
            
            # environmental pollution per item and per firm
            network.vs[i]["unit_pollution"] = 1 / network.vs[i]["sensitivity"]
            # total pollution generated by firm K
            network.vs[i]["pollution"] = network.vs[i]["unit_pollution"] * network.vs[i]["production"]
      
    return network

#############################################################################################################
#############################################################################################################

# Function to calculate the "local pollution" of every customer in the network
# "Local pollution" is the pollution generated from the customer's neighboring firms

def local_pollution(network):
    """
    network: the network in this model
    """
    for i in range(len(network.vs)):
        
        if network.vs[i]["type"] == "customer":
            # calculate the total pollution of customer i's neighboring firms
            lp = 0
            
            for j in network.neighbors(i):
                if network.vs[j]["type"] == "firm":
                    lp += network.vs[j]["pollution"]
            
            network.vs[i]["local_pollution"] = lp
        
        if network.vs[i]["type"] == "firm":
            network.vs[i]["local_pollution"] = 0  # firms do not have local pollution
    return network

#############################################################################################################
#############################################################################################################

# Function to calculate the states of firms and customers when environmental sensitivity (consciousness) is included
# In this case, all the links between firms and customers are positive
def shortage_with_env(network):
    """ 
    function to claculate the states of firms and customers when environmental sensitivity is included
    customer i' "demand_rec" is now the minimum of 
    1) the amount of products neighboring firms can supply and 2) the amount customer i wants to receive 

    network: the network in this model
    """
    sc_count = 0   # count the number of customers who suffer from shortage
    sf_count = 0   # count the number of firms who are over-producing
    
    for i in range(len(network.vs)):
        
        # for firms, "sd" counts for their local demand
        # for customers, "sd" counts for the amount of products they actually receives
        sd = 0.0   
        
        # figure out in which condition firm will be over-producing or not
        
        if network.vs[i]["type"] == "firm":
            
            for j in network.neighbors(i):
                
                if network.vs[j]["type"] == "customer":
                    
                    sk_t = 0  # calculate the denominator part of distribution factor
                    for k in network.neighbors(j):
                        if network.vs[k]["type"] == "firm":
                            
                            sk_t += np.exp(- network.vs[j]["sensitivity"] * (network.vs[k]["pollution"]/network.vs[j]["local_pollution"]))
                    
                    # calculate the numerator part of distribution factor
                    sk = np.exp(- network.vs[j]["sensitivity"] * (network.vs[i]["pollution"]/network.vs[j]["local_pollution"]))
                    
                    # calculate the demand of customer j from firm i, add it to total demand requirement of firm i
                    sd += network.vs[j]["demand"] *  sk / sk_t
                    
            network.vs[i]["local_demand"] = sd   # local demand of the firm
            network.vs[i]["local_supply"] = 0    # firms do not have local supply
            network.vs[i]["demand_rec"] = 0      # firms do not have demand received
            network.vs[i]["pollution_rec"] = 0   # firms do not have pollution received
            # the amount of products the firm has sold
            network.vs[i]["production_sold"] = min(network.vs[i]["local_demand"], network.vs[i]["production"])
            
            # compare local_demand and production
            
            # local demand is smaller than production
            # firm is over-producing, state = 1
            if network.vs[i]["local_demand"] < 0.999 * network.vs[i]["production"]:
                network.vs[i]["state"] = 1     
                
                # the amount of products the firm has not sold
                network.vs[i]["production_unsold"] = network.vs[i]["production"] - network.vs[i]["production_sold"]
                
                sf_count += 1
                
            # local demand is greater than or equal to production
            # firm sells out all the products, state = 0
            else:
                network.vs[i]["state"] = 0    
                network.vs[i]["production_unsold"] = 0   # all products have been sold
            
           
        # figure out in which condition customer will suffer from shortage or not
        if network.vs[i]["type"] == "customer":
            
            ls = 0    # calculate local supply
            sk_t = 0  # calculate the denominator part of distribution factor
            P_rec = 0    # calculate the customer's pollution received
            
            for k in network.neighbors(i):
                if network.vs[k]["type"] == "firm":
                            
                    sk_t += np.exp(- network.vs[i]["sensitivity"] * (network.vs[k]["pollution"]/network.vs[i]["local_pollution"]))
                        
            for j in network.neighbors(i):
                
                if network.vs[j]["type"] == "firm":
                    
                    # the max. amount production firm j can offer customer i
                    max_supply = network.vs[j]["production"] / network.vs[j]["cf_num"]
                    
                    # the max. amount production customer i want to gain from firm j
                    sk = np.exp(- network.vs[i]["sensitivity"] * (network.vs[j]["pollution"]/network.vs[i]["local_pollution"]))
                    max_want = network.vs[i]["demand"] *  sk / sk_t
                    
                    # production customer i can really gain from firm j is the min. value of max_supply and max_want
                    sd += min(max_supply, max_want)
                    ls += max_supply
                    P_rec += min(max_supply, max_want) * network.vs[j]["unit_pollution"]
                    
            network.vs[i]["local_demand"] = 0         # customers do not have local demand 
            network.vs[i]["local_supply"] = ls        # customer's local supply
            network.vs[i]["demand_rec"] = min(sd, network.vs[i]["demand"])          # customer's demand received
            network.vs[i]["pollution_rec"] = P_rec    # customer's pollution received
            network.vs[i]["production_sold"] = 0      # customers do not have this attribute
            network.vs[i]["production_unsold"] = 0    # customers do not have this attribute
            
            # compare the products received by customer and the customer's demand
            
            # the amount of products received by customer is smaller than customer's demand
            # customer suffers from shortage, state = 0
            if network.vs[i]["demand_rec"] < 0.999 * network.vs[i]["demand"]:
                network.vs[i]["state"] = 0   
                sc_count += 1
            
            # the amount of products received by customer is greater than or equal to customer's demand
            # customer is satisfied, state = 1
            else:
                network.vs[i]["state"] = 1   
    
    # assign new attributes to network 
    network["sc_count"] = sc_count   # number of customers who suffer from shortage
    network["sf_count"] = sf_count   # number of firms who suffer from shortage
    network["scs"] = sc_count / network["customer_count"]   # share of customers who suffer from shortage
    network["sfs"] = sf_count / network["firm_count"]   # share of firms who suffer from shortage
   
    return network

##############################################################################################################
##############################################################################################################

# assign the signs of links between firms and customers according to utility
def sign_assign_utility(network, alpha = 1, uthr = 0):
    """
    assign the signs of links according to utility
    
    Args:
        alpha: a parameter to scale up the pollution customer i expects to receive
        uthr: the threshold of utility, if the utility is greater than uthr, the link is positive, otherwise negative
    """
    for i in range(len(network.vs)):
        
        if network.vs[i]["type"] == "customer":
            
            f_neighbor = 0       # number of neighboring firms
            pollution_i = 0      # the amount of pollution customer i expects to receive from a neighboring firm
            
            for j in network.neighbors(i):
                
                if network.vs[j]["type"] == "firm":
                    
                    f_neighbor += 1
            
            if f_neighbor > 0:  
                pollution_i = network.vs[i]["demand"] / f_neighbor
            else:
                pollution_i = network.vs[i]["demand"] # in case there's isolated customers
                
            pollution_i /= network.vs[i]["sensitivity"]
            
            # the amount of pollution customer expects to receive from a neighboring firm
            network.vs[i]["pollution_exp"] = pollution_i       
            
        if network.vs[i]["type"] == "firm":
            
            c_neighbor = 0         # number of neighboring customers
            pollution_k = 0        # the amount of pollution firm K expects to give to a neighboring customer
            
            for j in network.neighbors(i):
                
                if network.vs[j]["type"] == "customer":
                    
                    c_neighbor += 1
            
            if c_neighbor > 0:
                pollution_k = network.vs[i]["production"] / c_neighbor # in case there's isolated firms
            else:
                pollution_k = network.vs[i]["production"]
                
            pollution_k /= network.vs[i]["sensitivity"]
            
            # the amount of pollution firm K expects to give to a neighboring customer 
            network.vs[i]["pollution_exp"] = pollution_k
    
    # assign the sign of links between customers and firms
    # utility = customer i's expected pollution - firm K's expected pollution
    # if utility > 0, this link is positive (sign = 1), otherwise negative (sign = -1)
    
    for i in range(len(network.es)):
        if (network.vs[network.es[i].tuple[0]]["type"] == "customer") & (network.vs[network.es[i].tuple[1]]["type"] == "firm"):
            if (network.vs[network.es[i].tuple[0]]["pollution_exp"] * alpha - network.vs[network.es[i].tuple[1]]["pollution_exp"]) > uthr:
                network.es[i]["sign"] = 1
            else:
                network.es[i]["sign"] = -1
                
        if (network.vs[network.es[i].tuple[0]]["type"] == "firm") & (network.vs[network.es[i].tuple[1]]["type"] == "customer"):
            if (network.vs[network.es[i].tuple[1]]["pollution_exp"] * alpha - network.vs[network.es[i].tuple[0]]["pollution_exp"]) > uthr:
                network.es[i]["sign"] = 1
            else:
                network.es[i]["sign"] = -1
    
    return network

##############################################################################################################
##############################################################################################################

# calculate the states of customers and firms when 1) environmental sensitivity (consciousness) and 2) signs of links are considered
def shortage_signed(network):
    """
    This functions calculates the states of the customers and firms with the signs of the links
    """
    sc_count = 0   # count the number of customers who suffer from demand shortage
    sf_count = 0   # count the number of firms wo suffer from overproduction
    
    for i in range(len(network.vs)):
        
        sd = 0.0   # count the actual supply of firm or actual demand supplied of customer
        
        # figure out in which condition firm will suffer from shortage
        
        if network.vs[i]["type"] == "firm":
            
            for j in network.neighbors(i):
                
                if network.vs[j]["type"] == "customer":
                    
                    sk_t = 0  # calculate the denominator part of distribution factor
                    for k in network.neighbors(j):
                        if network.vs[k]["type"] == "firm":
                            
                            sk_t += np.exp(- network.vs[j]["sensitivity"] * (network.vs[k]["pollution"]/network.vs[j]["local_pollution"]))
                    
                    # calculate the numerator part of distribution factor
                    sk = np.exp(- network.vs[j]["sensitivity"] * (network.vs[i]["pollution"]/network.vs[j]["local_pollution"]))
                    
                    # to get the sign of the link between node i and node j
                    link_ij = network.get_eid(i,j)
                    sign_ij = network.es[link_ij]["sign"]
                    
                    # calculate the demand of customer j from firm i, add it to total demand requirement of firm i
                    sd += network.vs[j]["demand"] *  sk / sk_t * ((1 + sign_ij) / 2)
                    
            network.vs[i]["local_demand"] = sd   # local demand of the firm
            network.vs[i]["local_supply"] = 0    # firms do not have local supply
            network.vs[i]["demand_rec"] = 0      # firms do not have demand received
            network.vs[i]["pollution_rec"] = 0   # firms do not have pollution received
            # the amount of products the firm has sold
            network.vs[i]["production_sold"] = min(network.vs[i]["local_demand"], network.vs[i]["production"])
            
            # firm state is determined by local_demand and average production
            if network.vs[i]["local_demand"] < 0.999 * network.vs[i]["production"]:
                network.vs[i]["state"] = 1     # firms with overproduction have a state of 1
                
                # the amount of products the firm has not sold
                network.vs[i]["production_unsold"] = network.vs[i]["production"] -  network.vs[i]["production_sold"]
                
                sf_count += 1
            else:
                network.vs[i]["state"] = 0    # firms without overproduction have a state of 0
                network.vs[i]["production_unsold"] = 0   # all products have been sold
            
           
        # figure out in which condition customer will suffer from shortage
        if network.vs[i]["type"] == "customer":
            ls = 0    # calculate local supply
            sk_t = 0  # calculate the denominator part of distribution factor
            P_rec = 0    # calculate the customer's pollution received
            for k in network.neighbors(i):
                if network.vs[k]["type"] == "firm":
                            
                    sk_t += np.exp(- network.vs[i]["sensitivity"] * (network.vs[k]["pollution"]/network.vs[i]["local_pollution"]))
                        
            for j in network.neighbors(i):
                
                if network.vs[j]["type"] == "firm":
                    
                    # to get the sign of the link between node i and node j
                    link_ij = network.get_eid(i,j)
                    sign_ij = network.es[link_ij]["sign"]
                    
                    # the max. amount production firm j can offer customer i
                    max_supply = network.vs[j]["production"] / network.vs[j]["cf_num"]* ((1 + sign_ij) / 2)
                    
                    # the max. amount production customer i want to gain from firm j
                    sk = np.exp(- network.vs[i]["sensitivity"] * (network.vs[j]["pollution"]/network.vs[i]["local_pollution"]))

                    max_want = network.vs[i]["demand"] *  sk / sk_t * ((1 + sign_ij) / 2)
                    
                    # production customer i can really gain from firm j is the min. value of max_supply and max_want
                    sd += min(max_supply, max_want)
                    ls += max_supply
                    P_rec += min(max_supply, max_want) * network.vs[j]["unit_pollution"]
                    
            network.vs[i]["local_demand"] = 0         # customers do not have local demand 
            network.vs[i]["local_supply"] = ls        # customer's local supply
            network.vs[i]["demand_rec"] = min(sd, network.vs[i]["demand"])          # customer's demand received
            network.vs[i]["pollution_rec"] = P_rec    # customer's pollution received
            network.vs[i]["production_sold"] = 0      # customers do not have this attribute
            network.vs[i]["production_unsold"] = 0    # customers do not have this attribute
            
            # customer state is determined by demand_rec and the threshold of satisfaction
            if network.vs[i]["demand_rec"] < 0.999 * network.vs[i]["demand"]:
                network.vs[i]["state"] = 0   # customer suffer from shortage has a state of 0
                sc_count += 1
            else:
                network.vs[i]["state"] = 1   # custoemr do not suffer from shortage has a state of 1
    
    # assign new attributes to network 
    network["sc_count"] = sc_count   # number of customers who suffer from shortage
    network["sf_count"] = sf_count   # number of firms who suffer from shortage
    network["scs"] = sc_count / network["customer_count"]   # share of customers who suffer from shortage
    network["sfs"] = sf_count / network["firm_count"]   # share of firms who suffer from shortage
   
    return network

#############################################################################################################
#############################################################################################################

# A function to generate a list of all the firms in the network
def firm_list(network):
    # the list of all the firms of in the network
    fl = []
    for i in range(len(network.vs)):
        if network.vs[i]["type"] == "firm":
            fl.append(i)
    
    network["firm_list"] = fl
    
    return network

#############################################################################################################
#############################################################################################################

# A function to generate a list of all the customers in the network
def customer_list(network):
    # the list of all the customers in the network
    cl = []
    
    for i in range(len(network.vs)):
        if network.vs[i]["type"] == "customer":
            cl.append(i)
    network["customer_list"] = cl
    
    return network

#############################################################################################################
#############################################################################################################

# A function to find the firms with which a customer can create a new relation
def potential_firm(network):
    # finds the firms with which a customer can create a new relation
    for i in range(len(network.vs)):
        
        pf = []        # store the firms that the customer can build new link with
            
        if network.vs[i]["type"] == "customer":
            for firm in network["firm_list"]:
                # customer can only build new link with firms that are not its neighbors
                if firm not in network.neighbors(i):
                    pf.append(firm)
        network.vs[i]["potential_firm"] = pf
    
    return network        

#############################################################################################################
#############################################################################################################

# A function to calculate the gini coefficient
def gini_calculation(demand_list): 
    """
    Calculating the gini coefficient
    """
    cum_wealths = np.cumsum(sorted(np.append(demand_list,0)))
    sum_wealths = cum_wealths[-1]
    xarray = np.array(range(0, len(cum_wealths))) / np.float(len(cum_wealths) - 1)
    yarray = cum_wealths / sum_wealths
    B = np.trapz(yarray, x = xarray)
    A = 0.5 - B
    gini = A / (A + B)
    if gini <= 0.001:
        gini = 0
    elif gini >= 0.999:
        gini = 1
    return gini

#############################################################################################################
#############################################################################################################

# A function to calcualte the gini coefficient when every one is satisfied
def gini_satisfied(sample =100):
    """
    this function is to general gini coefficient when every one is satisfied
    output is a list of gini coefficient, we will use this sample data to calculate 95% confident interval
    """
    gini_s = []
    
    for i in range(sample):
        
        # the same distribution as customers' demand
        demand_list = np.random.normal(1, 0.05, 45)
        
        # calculate the coefficient of the all-satisfied demand
        gini = gini_calculation(demand_list)
        
        gini_s.append(gini)
    
    return gini_s
    
    
    # a function calcualte the three criteria with unilateral,bilateral and smart rule of rewiring
def criteria_calculation(network, c_adapt_l, c_adapt_u, n=11,  max_steps = 5, sample = 10, parameters = {"f_adapt": 0.2,"c_rate": 0.31, "f_rate" :0.6}):
    
    # store the data we need in unilateral case
    gini_list_u = []    # store the final value of gini coefficients
    ds_list_u = []     # store the final value of demand satisfied (DS)
    scs_list_u = []    # store the final value of gini coefficients
    
    # store the data we need in bilateral case
    gini_list_b = []    # store the final value of gini coefficients
    ds_list_b = []     # store the final value of demand satisfied (DS)
    scs_list_b = []    # store the final value of gini coefficients
    
    # store the data we need in smart case
    gini_list_s = []    # store the final value of gini coefficients
    ds_list_s = []     # store the final value of demand satisfied (DS)
    scs_list_s = []    # store the final value of gini coefficients
    
    # the range of c_adapt
    c_adapt_range = np.linspace( c_adapt_l, c_adapt_u, num = n)
    f_adapt = parameters["f_adapt"]
    c_rate = parameters["c_rate"]
    f_rate = parameters["f_rate"]
   
    for c_adapt in range(len(c_adapt_range)):
        
        # parameters for "unilateral" case
        gini_u = 0
        scs_u = 0
        ds_u = 0
        
        # parameters for "bilateral" case
        gini_b = 0
        scs_b = 0
        ds_b = 0
        
        # parameters for "bilateral" case
        gini_s = 0
        scs_s = 0
        ds_s = 0
        
        # simulate several times and calculate the average values
        for i in range(sample):
            
            # case 1: unilateral rule for rewiring
            # copy the input network and assign some attributes required for model
            g_u = network.copy()
            g_u = potential_firm(g_u)
        
            # run model in unilateral case
            model_u = SignedNetwork(g_u, alpha = 1, uthr = 0, 
                                           c_adapt = c_adapt_range[c_adapt], 
                                           f_adapt = f_adapt,
                                           c_rate = c_rate, 
                                           f_rate = f_rate, 
                                           unilateral = True, bilateral = False, smart = False, 
                                           max_steps = max_steps,
                                           verbose = False)
                                        
            model_u.run_model()
        
            
            df_u = model_u.datacollector.get_model_vars_dataframe()
            gini_u += df_u["Gini"][max_steps +1]
            scs_u += df_u["SCS"][max_steps+1]
            ds_u += df_u["Demand satisfied"][max_steps+1]
            
            # case 2: bilateral rule for rewiring
            # copy the input network and assign some attributes required for model
            g_b = network.copy()
            g_b = potential_firm(g_b)
        
            # run model in unilateral case
            model_b = SignedNetwork(g_b, alpha = 1, uthr = 0, 
                                           c_adapt = c_adapt_range[c_adapt], 
                                           f_adapt = f_adapt,
                                           c_rate = c_rate, 
                                           f_rate = f_rate, 
                                           unilateral = False, bilateral = True, 
                                           smart = False, max_steps = max_steps,
                                           verbose = False)
                                        
            model_b.run_model()
        
        
            df_b = model_b.datacollector.get_model_vars_dataframe()
            gini_b += df_b["Gini"][max_steps+1]
            scs_b += df_b["SCS"][max_steps+1]
            ds_b += df_b["Demand satisfied"][max_steps+1]
            
            # case 3: smart rule for rewiring
            # copy the input network and assign some attributes required for model
            g_s = network.copy()
            g_s = potential_firm(g_s)
        
            # run model in smart case
            model_s = SignedNetwork(g_s, alpha = 1, uthr = 0, 
                                           c_adapt = c_adapt_range[c_adapt], 
                                           f_adapt = f_adapt,
                                           c_rate = c_rate, 
                                           f_rate = f_rate, 
                                           unilateral = False, bilateral = False, smart = True,
                                           max_steps = max_steps,
                                           verbose = False)
                                        
            model_s.run_model()
        
        
            df_s = model_s.datacollector.get_model_vars_dataframe()
            gini_s += df_s["Gini"][max_steps+1]
            scs_s += df_s["SCS"][max_steps+1]
            ds_s += df_s["Demand satisfied"][max_steps+1]
            
        # add the average values of unilateral case
        gini_list_u.append(gini_u / sample)
        scs_list_u.append(scs_u / sample)
        ds_list_u.append(ds_u / sample)
        
        # add average values of bilateral case
        gini_list_b.append(gini_b / sample)
        scs_list_b.append(scs_b / sample)
        ds_list_b.append(ds_b / sample)
        
        # add average values of smart case
        gini_list_s.append(gini_s / sample)
        scs_list_s.append(scs_s / sample)
        ds_list_s.append(ds_s / sample)
        #scs will not be returned!
    return gini_list_u, gini_list_b, gini_list_s, ds_list_u, ds_list_b, ds_list_s


def create_heatmap(signed_graph, n_cores=25, n=11, sample=100):
    """
    This function creates a heatmap of the criteria for the different values of f_adapt, c_rate and f_rate
    Args:
        n_cores: number of cores to use
        n: number of values to use for f_adapt, c_rate and f_rate
        sample: number of samples to use for each combination of f_adapt, c_rate and f_rate
    """
    gu_all = []
    dsu_all = []
    gb_all = []
    dsb_all = []
    gs_all = []
    dss_all = []
    with Pool(n_cores) as p:
        parameters_list = []
        c_adapt_l = 0
        c_adapt_u = 1.0
        max_steps = 5
        for f_adapt in np.linspace(0, 1, num = n):
            for f_rate in np.linspace(0, 1, num = n):
                for c_rate in np.linspace(0, 1, num = n):
                        parameters_list.append((signed_graph, c_adapt_l, c_adapt_u, n,  max_steps, sample, {"f_adapt": f_adapt,"c_rate": c_rate, "f_rate" :f_rate}))
        cs = int((len(parameters_list)/n_cores))
        cs = max(1, cs)
        for result in p.starmap(criteria_calculation, parameters_list, chunksize=cs):
            gu_single_p, gb_single_p, gs_single_p, dsu_single_p, dsb_single_p, dss_single_p = result
            gu_all.append(gu_single_p)
            gb_all.append(gb_single_p)
            gs_all.append(gs_single_p)
            dsu_all.append(dsu_single_p)
            dsb_all.append(dsb_single_p)
            dss_all.append(dss_single_p)
    gu_all, dsu_all, gb_all, dsb_all, gs_all, dss_all = reshape_lists(gu_all, dsu_all, gb_all, dsb_all, gs_all, dss_all, n=n)
    return gu_all, dsu_all, gb_all, dsb_all, gs_all, dss_all    


def reshape_lists(gu_all, dsu_all, gb_all, dsb_all, gs_all, dss_all, n=11):
    """
    This function reshapes the lists of the criteria to a 4-dimensional array
    Args:
        gu_all: list of gini coefficients for unilateral rule
        dsu_all: list of demand satisfied for unilateral rule
        gb_all: list of gini coefficients for bilateral rule
        dsb_all: list of demand satisfied for bilateral rule
        gs_all: list of gini coefficients for smart rule
        dss_all: list of demand satisfied for smart rule
        n: number of values used for f_adapt, c_rate and f_rate
    """
    gu_all = np.array(gu_all)
    gb_all = np.array(gb_all)
    gs_all = np.array(gs_all)
    gu_all = gu_all.reshape((n,n,n,n))
    gb_all = gb_all.reshape((n,n,n,n))
    gs_all = gs_all.reshape((n,n,n,n))
    dsu_all = np.array(dsu_all)
    dsb_all = np.array(dsb_all)
    dss_all = np.array(dss_all)
    dsu_all = dsu_all.reshape((n,n,n,n))
    dsb_all = dsb_all.reshape((n,n,n,n))
    dss_all = dss_all.reshape((n,n,n,n))
    return gu_all, dsu_all, gb_all, dsb_all, gs_all, dss_all
