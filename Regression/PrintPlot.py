from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np

def print_graph(Y, X, q, m):

    #https://matplotlib.org/tutorials/introductory/pyplot.html
    style.use ('classic')

    plt.plot(Y, X, color='r', linewidth=1.0)
    plt.show()


    '''
#disegna subplots
fig_1 = plt.figure(1, figsize=(6.4,4.8))
chart_1 = fig_1.add_subplot(121)
chart_2 = fig_1.add_subplot(122)
chart_1.plot(crtes,sergio)
chart_2.scatter(alberto,crtes)   #stampare i punti non la retta intera
                                                                            

plt.axis([-2,2,-2,2])    #definire min e max sugli assi                                             
label = plt.plot(alberto, crtes, color ='r', linewidth=8.0, )         
plt.show()                                                            
                                                                            
'''

#Plot a line from slope and intercept
def mqline(slope, intercept):
    axes = plt.gca()
    plt.xlabel('alberto')
    plt.ylabel('crtes')
    x_vals = np.array(axes.get_xlim())
    y_vals = slope + intercept * x_vals
    plt.plot(x_vals, y_vals, color = 'b')

#mqline(20,3)

#label = plt.plot(alberto, crtes, color ='r', linewidth=1.0)
#plt.show()




