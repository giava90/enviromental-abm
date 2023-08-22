import igraph as ig # redundant, but for memory
import matplotlib.pyplot as plt
import matplotlib 
import numpy as np

"""
This file contains functions for visualizing the network and the simulation results.
"""

# A function to define the shapes and the initial colors of the nodes
def network_visualization(network, min_size = 6, max_size =80):
    # network should be an igraph object or similar
    # first, set node color and shape : firm as blue diamond and customer as green circle
    # min_size and max_size are the minimum and maximum size of nodes
    for i in range(len(network.vs)):
        
        # set the colors and shapes of firms and customers
        if network.vs[i]["type"] == "firm":
            network.vs[i]["color"] = "#e9a3c9"
            network.vs[i]["shape"] = "diamond"
        else:
            network.vs[i]["color"] = "#a1d76a"
            network.vs[i]["shape"] = "circle"
    
    # second, set the size of nodes
    m = max(network.vs.degree())
    
    # set the size of nodes according to their degree
    for i in range(len(network.vs)):
        network.vs[i]["size"] = min_size + (max_size - min_size) * (network.vs[i].degree() / m)
    
    # third, set the color of links 
    for i in range(len(network.es)):
        if network.es[i]["sign"] == -1:
            network.es[i]["color"] = "red"
        else:
            network.es[i]["color"] = "black"
    
    return network

############################################################################################################
############################################################################################################

# A function to visualize the network with the simulation results
def shortage_color(network):
    # make nodes with shortage red and no shortage blue
    for i in range(len(network.vs)):
        
        if network.vs[i]["type"] == "firm":
            if network.vs[i]["state"] == 1:
                network.vs[i]["color"] = "tomato"  # firms with overproduction are colored red
            else:
                network.vs[i]["color"] = "dodgerblue" # firms without overproduction are colored blue
        
        if network.vs[i]["type"] == "customer":
            if network.vs[i]["state"] == 0:
                network.vs[i]["color"] = "tomato"  # customers suffer from shortage are colored red
            else:
                network.vs[i]["color"] = "dodgerblue" # customers with no shortage are colored blue
    
    return network

############################################################################################################
############################################################################################################

# A function to visualize the network with the simulation results
def demandsatisfied_color(network, n_c = 255, mid = 0.85):
    # make nodes with shortage red and no shortage blue
    #pal = ig.GradientPalette('#67000d', '#fff5f0', n_c+1)
    pal = {0: "#b2182b", 1 :"#ef8a62", 2 :"#fddbc7", 3 :"#d1e5f0", 4 :"#67a9cf", 5 :"#2166ac"}
    step = (1-mid)/3
    for i in range(len(network.vs)):
        if network.vs[i]["type"] == "firm":
            ps = network.vs[i]['production_sold'] / network.vs[i]['production']
            if ps <mid:
                if ps < mid - 2*step:
                    network.vs[i]["color"] = pal[0]
                elif ps < mid - step:
                    network.vs[i]["color"] = pal[1]
                else:
                    network.vs[i]["color"] = pal[2]
            elif ps > mid + step*2:
                network.vs[i]["color"] = pal[5]
            elif ps > mid + step:
                network.vs[i]["color"] = pal[4]
            else:
                network.vs[i]["color"] = pal[3]
        
        if network.vs[i]["type"] == "customer":
            ds = network.vs[i]['demand_rec'] / network.vs[i]['demand']
            if ds <mid:
                step = (1-mid)/3
                if ds < mid - 2*step:
                    network.vs[i]["color"] = pal[0]
                elif ds < mid - step:
                    network.vs[i]["color"] = pal[1]
                else:
                    network.vs[i]["color"] = pal[2]
            elif ds > mid + 2*step:
                network.vs[i]["color"] = pal[5]
            elif ds > mid + step:
                network.vs[i]["color"] = pal[4]
            else:
                network.vs[i]["color"] = pal[3]
            #print(ds, network.vs[i]['demand_rec'] / network.vs[i]['demand'])
            #ds = round(ds,0)
            #ds = int(ds)
            #network.vs[i]["color"] = pal.get(ds) # customers suffer from shortage are colored red
    return network

cm = 1/2.54  

############################################################################################################
############################################################################################################

# A function to generate a colorbar for the simulation results
def colorbar(cmap, v_min = 0.6, v_max = 1.0, v_mid = 0.8, show = True, filename=False, title = "", rounding=1, ticks=[]):
    matplotlib.rcParams["font.size"]=12
    a = np.array([[0,1]])
    plt.figure(figsize=(2*cm, 9*cm))
    img = plt.imshow(a, cmap=cmap, vmin=v_min, vmax=v_max)
    plt.gca().set_visible(False)
    cax = plt.axes([0.1, 0.1, 0.2, 0.8])
    if ticks ==[]:
        ticks = [v_min, (v_min + v_mid)/2, v_mid, (v_max + v_mid)/2, v_max]
        ticks = np.round(ticks,rounding)
        ticks = np.unique(ticks)
        ticks = np.sort(ticks)
    cbar = plt.colorbar(orientation="vertical", cax=cax, ticks=ticks)
    cbar.ax.set_yticklabels(ticks)
    cbar.ax.set_title(title)
    if filename:
        plt.savefig(filename)
    if show:
        plt.show()
    else:
        plt.clf()
#colorbar(cmap = my_colormap, v_min = 0.6, v_max = 1.0, v_mid = 0.8, show=True, filename = "figures/colorbar.pdf")

import matplotlib.colors as colors
class MidpointNormalize(colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)
	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
    
cm = 1/2.54
my_colormap = matplotlib.colors.LinearSegmentedColormap.from_list("my_colors",["#EA8300","#F7F7F7","#0D84BF"])

############################################################################################################
############################################################################################################

# A function to generate a heat map for the simulation results (SCS: share of customers suffering from shortage)
def heat_map_scs(matrix, show=True, filename= False, xticks= True, xlabel=True, yticks = True, ylabel=True, v_min = 0.6, v_max = 1.0, v_mid = 0.8, my_colormap = my_colormap):
    matplotlib.rcParams["font.size"]=13
    plt.figure(figsize=(9*cm,9*cm))
    plt.imshow(matrix, cmap=my_colormap, interpolation='none', origin='lower', norm=MidpointNormalize(v_min, v_max, v_mid))
    if xticks==True:
        xticks = [0.0, 0.3, 0.6, 0.9]
    if yticks==True:
        yticks = [0.0, 0.3, 0.6, 0.9]
    if type(xticks) is list:
        xticks = np.array(xticks)
        plt.xticks(xticks*10,xticks)
    else:
        plt.xticks([],[])
    if xlabel:
        plt.xlabel(r"$c_{\mathrm{adapt}}$")
    if type(yticks) is list:
        yticks = np.array(yticks)
        plt.yticks(xticks*10,xticks)
    else:
        plt.yticks([],[])
    if ylabel:
        plt.ylabel(r"$r_{\mathrm{c}}$")
    #plt.tight_layout()
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
    if filename:
        plt.savefig(filename, pad_inches=0)
    if show:
        plt.show()
    else:
        plt.clf()
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)

############################################################################################################
############################################################################################################

# A function to generate a heat map for the simulation results (DS: demand satisfied)
def heat_map_ds(matrix, show=True, filename= False, xticks= True, xlabel=True, yticks = True, ylabel=True, v_min = 0.6, v_max = 1.0, v_mid = 0.8, my_colormap = my_colormap):
    matplotlib.rcParams["font.size"]=13
    plt.figure(figsize=(9*cm,9*cm))
    plt.imshow(matrix, cmap=my_colormap, interpolation='none', origin='lower', norm=MidpointNormalize(v_min, v_max, v_mid))
    if xticks==True:
        xticks = [0.0, 0.3, 0.6, 0.9]
    if yticks==True:
        yticks = [0.0, 0.3, 0.6, 0.9]
    if type(xticks) is list:
        xticks = np.array(xticks)
        plt.xticks(xticks*10,xticks)
    else:
        plt.xticks([],[])
    if xlabel:
        if xlabel != True:
            plt.xlabel(xlabel)
        else:
            plt.xlabel(r"$c_{\mathrm{adapt}}$")
    if type(yticks) is list:
        yticks = np.array(yticks)
        plt.yticks(xticks*10,xticks)
    else:
        plt.yticks([],[])
    if ylabel:
        if ylabel != True:
            plt.ylabel(ylabel)
        else:
            plt.ylabel(r"$r_{\mathrm{c}}$")
    #plt.tight_layout()
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
    if filename:
        plt.savefig(filename, pad_inches=0)
    if show:
        plt.show()
    else:
        plt.clf()
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)