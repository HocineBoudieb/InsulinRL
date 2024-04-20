
import functions.PAmodel as GIM
import functions.plot_fct as plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


''' Define basal values '''

Gb = 120                                    # basal glucose level [mg/dl]
Ib = 12                                     # basal insulin level [muU/ml]
weight = 70                                 # weight [kg]


''' Define scenarios '''


intensity = [4317]                          # intensity in AC count (60% VO2max)
duration = [60]                             # PA duration [min]
pa_start = [240]                            # timing of PA start [min]

t_meal = [60,360,720]                               # timing of meals [min]
D = [4e4]  
''' Parameters '''

params = pd.read_csv('parameters/params_standard.csv', header=None,
                     dtype={0: str}, delimiter=';').set_index(0).squeeze().to_dict()


''' Define model input '''

t_sim = 1441                                # simulation duration [min]
t = np.arange(t_sim)                        # create time points

meal = np.zeros(t_sim)                      # define meal input [mg/min]
meal[t_meal] = D


''' Compute glucose profile '''

G = []

u = np.zeros(t_sim)         # define insulin input [muU/min]

AC = np.zeros(t_sim)        # define AC input [counts/min]
AC[pa_start[0]:pa_start[0]+duration[0]] = intensity[0]

model = GIM.compute_glucose_simulation(params, [Gb, Ib], weight, AC, meal, u=u,
                                        bolus_calc=True)

glucose = model['G']
G.append(model['G'])
print(G)

''' Plot results '''
plot.plot_simulation(G, labels=['60%'],
                        colors=[plt.cm.PuBu(0.85)] + list(plt.cm.Reds(np.linspace(0.3, 0.9, 3))),
                        pa=[pa_start[0], pa_start[0]+duration[0]])

plt.show()
plt.close()

