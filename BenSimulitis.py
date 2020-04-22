# -*- coding: utf-8 -*-
"""
Benjamin's Simulitis
by Benjamin Krawciw

This program performs simulations of COVID-19 using a model heavily inspired
by https://www.washingtonpost.com/graphics/2020/world/corona-simulator/, 
borrowing heavily from the code on https://github.com/mnfienen/pysimulitis.

Note, the extra feature of reactionary isolation has been added, completely
originally.
"""

#Module imports
import numpy as np
import matplotlib.pyplot as plt
import os,shutil
import imageio
import imageio_ffmpeg

#Creating a space for figures
if os.path.exists('figures'):
    shutil.rmtree('figures')
os.mkdir('figures')


#Define reactionary isolation parameter
def reactionaryIso(infPercent):
    minim = 0.05
    if infPercent <= minim:
        return(0.0)
    output = 5 * (infPercent - minim)
    if output > 0.90:
        return(0.90)
    else:
        return(output)
    

#Creating a class for disease simulations
class Simulitis():
    #On initialization, set default parameters
    def __init__(self):
        #Box size
        self.lim= 200
        
        #Number of agents
        self.n = 200
        
        #Disease infection radius
        self.rad = 2.5
        
        #Speed of particles
        self.speed = 10
        
        #Probability an agent is socially isolating
        self.iso =  0.5
        
        #Total days of simulation
        self.t_tot = 70
        
        #Size of timestep
        self.delT = 0.5
        
        #Recovery time
        self.t_recovery = 14 
        
        #Probability a carrier starts out infected
        self.p_init = 0.015
        
        #Probability of transmission within radius
        self.p_trans = 0.99
        
        #Probability of dying
        self.p_mort =0.02
        
        #Are we running an attempted quarantine?
        self.wallExists = False
        self.x1 = 30 * self.lim / 120
        self.x2 = 50 * self.lim / 120
        
        #Are we doing reactionary isolation?
        self.reactionary = False

    #Function for setting up simulation
    def setup(self):
        #Gathering up the variables from self
        lim = self.lim
        n = self.n
        rad = self.rad
        speed = self.speed
        iso = self.iso
        t_tot = self.t_tot
        delT = self.delT
        t_recovery = self.t_recovery
        p_init = self.p_init
        p_trans = self.p_trans
        p_mort = self.p_mort
        
        #List of positions
        self.pos = np.random.random((n,2))*lim
        
        #List of isolating agents
        self.isolate = np.random.random((n,1))<iso

        #List of traveling directions                
        self.th = np.random.random((n,1))*2*np.pi

        #List of velocity vectors                  
        self.v = np.hstack((speed*np.cos(self.th), speed*np.sin(self.th)))   
        
        #List of infected agents
        self.infected = np.random.random((n,1))<p_init
        
        #Make sure there is at least one infected agent
        if sum(self.infected==0):
            self.infected[int((np.round(np.random.random(1)*n))[0])] = True
        
        #Make a list of the healthy agents
        self.healthy = ~self.infected
        
        #Make a list of the recovered agents (starting with zeroes)
        self.recovered = np.zeros((n,1))
        
        #Make a list of the dead (starting with zeroes)
        self.dead = np.zeros((n,1))
        
        #List of death probabilities (calculating each individual's risk)
        self.p_death = np.random.random(n)<p_mort
        
        #List of each agent's recovery time
        self.t_rec = np.ceil(t_recovery * np.ones(n) + 
                        np.random.normal(loc=0, scale=4, size=n))/delT
        
        #Tracking time
        self.t = np.linspace(0,t_tot,np.int(t_tot/delT))
        
        #Collisions matrix
        self.collision = np.zeros((n,n))
        
        #Collision delay
        self.n_delay = 10

        # set up solution vectors
        # Percentage of infected carriers
        self.inf_sum = np.zeros(np.int(t_tot/delT))      
        # Percentage of unaffected carriers
        self.hea_sum = self.inf_sum.copy()  
        # Percentage of recovered carriers                    
        self.rec_sum = self.inf_sum.copy()  
        # Percentage of dead carriers                    
        self.dead_sum = self.inf_sum.copy()
        # Percentage of cumulative disease cases                     
        self.cumulative_sum = self.inf_sum.copy()   

        #In the attempted quarantine, put all infected agents in the quarantine
        if self.wallExists:
            self.pos = self.healthy * self.pos

    #Function for running the simulation
    def simulate(self):
        #Gathering up the variables from self
        lim = self.lim
        n = self.n
        rad = self.rad
        speed = self.speed
        iso = self.iso
        t_tot = self.t_tot
        delT = self.delT
        t_recovery = self.t_recovery
        p_init = self.p_init
        p_trans = self.p_trans
        p_mort = self.p_mort
        pos = self.pos
        isolate = self.isolate    
        th = self.th
        v = self.v
        infected = self.infected
        healthy = self.healthy
        recovered = self.recovered
        dead = self.dead
        p_death = self.p_death
        t_rec = self.t_rec
        t = self.t
        collision = self.collision
        n_delay = self.n_delay
        inf_sum = self.inf_sum    
        hea_sum = self.hea_sum                   
        rec_sum = self.rec_sum                
        dead_sum = self.dead_sum                  
        cumulative_sum = self.cumulative_sum 
            
        
        for i in range(np.int(t_tot/delT)):
            print('Timestep {0} out of {1}'.format((i+1), np.int(t_tot/delT)))
            
            #Performing reactionary isolation
            if self.reactionary:
                #Calculate new isolation parameter
                newIso = reactionaryIso(inf_sum[i-1]/100)
                
                #Add or remove isolating people
                if newIso > iso:
                    isolate = isolate + ~isolate * np.random.random((n,1)) < (newIso - iso)
                
                if newIso < iso:
                    isolate = isolate - isolate * np.random.random((n,1)) < (iso - newIso)
                
                self.isolate = isolate
                iso = newIso
                self.iso = iso
    
            
            #Remove quarantine at half time
            if i > np.int(t_tot/delT)/2:
                self.wallExists = False
            
            # Decrement collision delay
            collision = collision-np.ones((n,n))
            collision[collision<0]=0
            
            # Update carrier position
            pos_new = pos + v*(~isolate)*delT
            
            # Step through each carrier
            for k in range(n):
                
                # If recovery time is up, carrier is either recovered or dead
                if infected[k] and t_rec[k]<=0:
                    
                    # If recovery time is up and carrier is dead, well, it's dead.
                    # Zero it's velocity
                    if p_death[k]:
                        dead[k] = 1;
                        v[k,:] = [0, 0]
                        recovered[k]=0
                        infected[k]=0
                        healthy[k]=0
                    else:
                        # If recovery time is up and carrier is not dead, recover
                        # that carrier
                        recovered[k]=1
                        infected[k]=0
                        healthy[k]=0
                        dead[k]=0
        
                    
                # If carrier is infected and not recovered, decrement recovery time
                elif infected[k]:
                    t_rec[k] -= 1
        
                
                # Step through all other carriers, looking for collisions, and if
                # so, transmit disease and recalculate trajectory
                for j in range(n):
                    
                    if j != k:
                        
                        # Get positions of carriers j and k
                        pos0 = pos_new[k,:]
                        pos1 = pos_new[j,:]
                        
                        # If collision between two living specimens, re-calcuate
                        # direction and transmit virus (but don't check the same
                        # two carriers twice)
                        if (np.linalg.norm(pos0-pos1)<=(2*rad)) and (1 - collision[k,j]) and (1 - collision[j,k]):
                             
                            # Create collision delay (i.e. if carrier j and k have
                            # recently collided, don't recompute collisions for a
                            # n_delay time steps in case they're still close in proximity, 
                            # otherwise they might just keep orbiting eachother)
                            collision[k,j] = n_delay
                            collision[j,k] = n_delay
                            
                            # Compute New Velocities
                            phi = np.arctan2((pos1[1]-pos0[1]),(pos1[0]-pos0[0]))
                            
                            # if one carrier is isolated, treat it like a wall and
                            # bounce the other carrier off it
                            if isolate[j] or dead[j]:
                                
                                # Get normal direction vector of 'virtual wall'
                                phi_wall = -phi+np.pi/2;
                                n_wall = [np.sin(phi_wall), np.cos(phi_wall)];
                                dot = v[k,:].dot(np.array(n_wall).T)
                                
                                # Redirect non-isolated carrier
                                v[k,0] = v[k,0]-2*dot*n_wall[0]
                                v[k,1] = v[k,1]-2*dot*n_wall[1]
                                v[j,0] = 0
                                v[j,1] = 0
                                
                            elif isolate[k] or dead[k]:
                                
                                # Get normal direction vector of 'virtual wall'
                                phi_wall = -phi+np.pi/2
                                n_wall = [np.sin(phi_wall), np.cos(phi_wall)]
                                dot = v[j,:].dot(np.array(n_wall).T)
                                
                                # Redirect non-isolated carrier
                                v[j,0] = v[j,0]-2*dot*n_wall[0]
                                v[j,1] = v[j,1]-2*dot*n_wall[1]
                                v[k,0] = 0
                                v[k,1] = 0
                                
                            # Otherwise, transfer momentum between carriers
                            else:                        
                                # Get velocity magnitudes
                                v0_mag = np.sqrt(v[k,0]**2+v[k,1]**2)
                                v1_mag = np.sqrt(v[j,0]**2+v[j,1]**2)
                                
                                # Get directions
                                th0 = np.arctan2(v[k,1],v[k,0]);
                                th1 = np.arctan2(v[j,1],v[j,0]);
                                
                                # Compute new velocities
                                v[k,0] = v1_mag*np.cos(th1-phi)*np.cos(phi)+v0_mag*np.sin(th0-phi)*np.cos(phi+np.pi/2)
                                v[k,1] = v1_mag*np.cos(th1-phi)*np.sin(phi)+v0_mag*np.sin(th0-phi)*np.sin(phi+np.pi/2)
                                v[j,0] = v0_mag*np.cos(th0-phi)*np.cos(phi)+v1_mag*np.sin(th1-phi)*np.cos(phi+np.pi/2)
                                v[j,1] = v0_mag*np.cos(th0-phi)*np.sin(phi)+v1_mag*np.sin(th1-phi)*np.sin(phi+np.pi/2)
                                                    
                            # If either is infected and not dead...
                            if (infected[j] or infected[k]) and ((1-dead[k]) or (1-dead[j])):
                                
                                # If either is recovered, no transmission
                                if recovered[k]:
                                    infected[k]=0
                                elif recovered[k]:
                                    infected[j]=0
                                    
                                # Otherwise, transmit virus
                                else:
                                    transmission = np.random.random(1)[0]<p_trans
                                    if transmission:
                                        infected[j]=1
                                        infected[k]=1
                                        healthy[j]=0
                                        healthy[k]=0
                # Look for collisions with walls and re-direct
                
                # Left Wall
                if pos_new[k,0]<=rad:
                    if v[k,0]<0:
                        v[k,0]=-v[k,0]            
                # Right wall
                elif pos_new[k,0]>=lim-rad:
                    if v[k,0]>0:
                        v[k,0]=-v[k,0]
                
                # Bottom Wall
                if pos_new[k,1] <=rad:
                    if v[k,1]<0:
                        v[k,1]=-v[k,1]
    
                # Top Wall
                elif pos_new[k,1] >=(lim-rad):
                    if v[k,1]>0:
                        v[k,1]=-v[k,1]
                
                #Wall for attempted quarantine
                x1 = self.x1
                x2 = self.x2
                mid = (x1 + x2) / 2
                if self.wallExists:
                    if (pos_new[k,0] >= x1) and  (pos_new[k,0] <= mid):
                        if v[k,0]>0:
                            v[k,0]=-v[k,0]
                    elif (pos_new[k, 0] <= x2) and (pos_new[k, 0] > mid):
                        if v[k,0]<0:
                            v[k,0]=-v[k,0]
                        
                        
            # Update color vector
            color=np.hstack([infected, healthy, recovered]*(1-dead))    
            color = np.hstack((color,dead))
            # Update solution vectors
            inf_sum[i] = sum(infected)*100/n
            hea_sum[i] = sum(healthy)*100/n
            rec_sum[i] = sum(recovered)*100/n
            dead_sum[i] = sum(dead)*100/n
            cumulative_sum[i] = 100-hea_sum[i]
            
            
            # make plots
            # first plot positions
            fig, ax = plt.subplots(2,1,figsize=(5,10))
            
        
            infidx = np.where(color[:,0]==1)[0]
            ax[0].scatter(pos_new[infidx,0],pos_new[infidx,1], c='red')
            healthyidx = np.where(color[:,1]==1)[0]
            ax[0].scatter(pos_new[healthyidx,0],pos_new[healthyidx,1], c='blue')
            deadidx = np.where(color[:,3]==1)[0]
            ax[0].scatter(pos_new[deadidx,0],pos_new[deadidx,1], c='black')
        
            recovidx = np.where(color[:,2]==1)[0]
            ax[0].scatter(pos_new[recovidx,0],pos_new[recovidx,1], c='blue', alpha=.5)
            ax[0].axis('square')    
            ax[0].set_xticks([])
            ax[0].set_yticks([])
            ax[0].set_xlim([0,lim])
            ax[0].set_ylim([0,lim])
            
            if self.wallExists:
                ax[0].axvline(self.x1, 0, lim)
                ax[0].axvline(self.x2, 0, lim)
            
            ax[0].set_title('Agent-based simulation')
            plt.suptitle('Isolation = {}%\n'.format(round(iso*100, 1)) + 'Time = {:.2f} days'.format(i*delT))
            
            # now plot the distributions evolving over time
            
            ax[1].fill_between(t[:i],(inf_sum[:i]+dead_sum[:i]+hea_sum[:i]),
                               (inf_sum[:i]+dead_sum[:i]+hea_sum[:i]+rec_sum[:i]),color='blue', alpha=.5)
            ax[1].fill_between(t[:i],(inf_sum[:i]+dead_sum[:i]),
                               (inf_sum[:i]+dead_sum[:i]+hea_sum[:i]),color='blue')
            ax[1].fill_between(t[:i],dead_sum[:i],
                               (inf_sum[:i]+ dead_sum[:i]),color='r')
            ax[1].fill_between(t[:i],np.zeros(i),dead_sum[:i], color='k')
            
            
            ax[1].set_xlim([0,t_tot])
            ax[1].set_ylim([0,100])
            ax[1].set_title('Epidemic Breakdown')
            ax[1].set_xlabel('Days')
            ax[1].set_label('Percent of population')
            plt.legend(['recovered', 'healthy','infected','dead'])
            plt.savefig('figures/fig{0}.png'.format(i))
            
            #print(inf_sum[i]+dead_sum[i]+hea_sum[i]+rec_sum[i])
            plt.close('all')
            
            pos=pos_new
            
    #Function that outputs gifs and mp4s
    def animations(self, title = None):
        #Gathering up the variables from self
        iso = self.iso
        t_tot = self.t_tot
        delT = self.delT
        
        #Setting the output title
        if title == None:
            title = 'model{}'.format(iso)
        
        #Make the gif
        frames_path = 'figures/fig{i}.png'
        with imageio.get_writer((title+'.gif'), mode='I') as writer:
            for i in range(np.int(t_tot/delT)):
                writer.append_data(imageio.imread(frames_path.format(i=i)))
        
        #Make the mp4
        frames_path = 'figures/fig{i}.png'
        with imageio.get_writer((title+'.mp4'), mode='I') as writer:
            for i in range(np.int(t_tot/delT)):
                writer.append_data(imageio.imread(frames_path.format(i=i)))

'''        
#Running a free-for-all simulation
sim = Simulitis()
sim.iso = 0.0
sim.setup()
sim.simulate()
sim.animations(title = 'Free for All')

#Running an attempted quarantine
sim = Simulitis()
sim.iso = 0.0
sim.wallExists = True
sim.setup()
sim.simulate()
sim.animations(title = 'Attempted quarantine')
'''

#Running a simulation with moderate isolation
sim = Simulitis()
sim.iso = 0.5
sim.setup()
sim.simulate()
sim.animations(title = 'Moderate Isolation fifty percent')

#Running a simulation with major isolation
sim = Simulitis()
sim.iso = 0.9
sim.setup()
sim.simulate()
sim.animations(title = 'Major Isolation ninety percent')
 
#Running a simulation with reactionary isolation
sim = Simulitis()
sim.reactionary = True
sim.setup()
sim.simulate()
sim.animations(title = 'Reactionary Isolation')      