import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.utils import *

class ThreeOnOne:
    
    def __init__(self, accel_defender, accel_attacker1, accel_attacker2, accel_attacker3, accel_ball=3, \
                 mass_defender=1,  mass_attacker1=1, mass_attacker2=1, mass_attacker3=1, mass_ball=0.3, \
                 damping=0.25, dt=0.1, max_step=300):
        
        self.accel_d  = accel_defender
        self.accel_a1 = accel_attacker1
        self.accel_a2 = accel_attacker2
        self.accel_a3 = accel_attacker3
        self.accel_b  = accel_ball

        self.mass_d = mass_defender     
        self.mass_a1 = mass_attacker1
        self.mass_a2 = mass_attacker2
        self.mass_a3 = mass_attacker3
        self.mass_b = mass_ball
        self.damping = damping
        self.dt = dt
        self.max_step = max_step
    
    def reset(self):
        
        self.pos_d = np.random.uniform(-0.3, 0.3, 2)
        self.vel_d = np.zeros(2)
        
        rand = np.random.randint(0, 12)
        ang1 = rand * -np.pi / 6
        self.pos_a1 = np.array([np.cos(ang1)*0.5, np.sin(ang1)*0.5])  
        ang2 = (rand+4) * -np.pi / 6
        self.pos_a2 = np.array([np.cos(ang2)*0.5, np.sin(ang2)*0.5])
        ang3 = (rand+8) * -np.pi / 6
        self.pos_a3 = np.array([np.cos(ang3)*0.5, np.sin(ang3)*0.5])
        self.vel_a1 = np.zeros(2)
        self.vel_a2 = np.zeros(2)
        self.vel_a3 = np.zeros(2)
        self.vel_b = np.zeros(2)
        self.pass_a1, self.pass_a2, self.pass_a3, self.pass_d = False, False, False, False
        self.to_a1, self.to_a2, self.to_a3 = False, False, False
        self.catch_d = False
        self.with_b_d = False
        self.obs_tmp = None
        self.action_tmp = None
        
        r = np.random.random()
        if  r < 0.3333:        
            self.with_b_a1, self.with_b_a2, self.with_b_a3 = True, False, False 
            self.pos_b = self.pos_a1.copy()
        elif r >= 0.3333 and r < 0.6666:
            self.with_b_a1, self.with_b_a2, self.with_b_a3 = False, True, False
            self.pos_b = self.pos_a2.copy()        
        else:
            self.with_b_a1, self.with_b_a2, self.with_b_a3 = False, False, True
            self.pos_b = self.pos_a3.copy()
        
        obs_d = get_obs_d(self.pos_d, self.vel_d, self.pos_a1, self.vel_a1, self.pos_a2, self.vel_a2, \
                          self.pos_a3, self.vel_a3, self.pos_b, self.vel_b)
        
        obs_a1 = get_obs_a(self.pos_a1, self.vel_a1, self.pos_a2, self.vel_a2, self.pos_a3, self.vel_a3, \
                           self.pos_d, self.vel_d, self.pos_b, self.vel_b)
        
        obs_a2 = get_obs_a(self.pos_a2, self.vel_a2, self.pos_a3, self.vel_a3, self.pos_a1, self.vel_a1, \
                           self.pos_d, self.vel_d, self.pos_b, self.vel_b)
        
        obs_a3 = get_obs_a(self.pos_a3, self.vel_a3, self.pos_a1, self.vel_a1, self.pos_a2, self.vel_a2, \
                           self.pos_d, self.vel_d, self.pos_b, self.vel_b)
        
        return obs_d, obs_a1, obs_a2, obs_a3, self.with_b_a1, self.with_b_a2, self.with_b_a3
    
    
    def step(self, obs_d, obs_a1, obs_a2, obs_a3, action_d, action_a1, action_a2, action_a3, num_step):
        
        abs_u_d = get_abs_u(action_d, self.pos_d, self.pos_b)
             
        abs_u_a1, abs_u_b_a1, next_pass_a1, next_with_b_a1, action_a1 = get_abs_ball_u(action_a1, self.pos_a1, self.pos_a2, self.pos_a3, self.pos_d, \
                                                                                       self.with_b_a1, self.pass_a1, \
                                                                                       self.to_a1, self.to_a2, self.to_a3, \
                                                                                       self.pos_b, self.vel_b)
        
        abs_u_a2, abs_u_b_a2, next_pass_a2, next_with_b_a2, action_a2 = get_abs_ball_u(action_a2, self.pos_a2, self.pos_a3, self.pos_a1, self.pos_d, \
                                                                                       self.with_b_a2, self.pass_a2, \
                                                                                       self.to_a2, self.to_a3, self.to_a1, \
                                                                                       self.pos_b, self.vel_b)
        
        abs_u_a3, abs_u_b_a3, next_pass_a3, next_with_b_a3, action_a3 = get_abs_ball_u(action_a3, self.pos_a3, self.pos_a1, self.pos_a2, self.pos_d, \
                                                                                       self.with_b_a3, self.pass_a3, \
                                                                                       self.to_a3, self.to_a1, self.to_a2, \
                                                                                       self.pos_b, self.vel_b)
               
        next_pos_d, next_vel_d = get_next_own_state(self.pos_d, self.vel_d, abs_u_d, self.mass_d, self.accel_d, self.damping, self.dt)
        
        next_pos_a1, next_vel_a1 = get_next_own_state_circ(self.pos_a1, self.vel_a1, abs_u_a1, self.mass_a1, self.accel_a1, self.damping, self.dt)  
        
        next_pos_a2, next_vel_a2 = get_next_own_state_circ(self.pos_a2, self.vel_a2, abs_u_a2, self.mass_a2, self.accel_a2, self.damping, self.dt) 
        
        next_pos_a3, next_vel_a3 = get_next_own_state_circ(self.pos_a3, self.vel_a3, abs_u_a3, self.mass_a3, self.accel_a3, self.damping, self.dt) 
        
        next_pos_b, next_vel_b = get_next_ball_state(self.pos_b, self.vel_b, abs_u_b_a1 , abs_u_b_a2, abs_u_b_a3, self.mass_b, self.accel_b, self.dt, \
                                                     next_pass_a1, next_pass_a2, next_pass_a3, self.pass_a1, self.pass_a2, self.pass_a3, \
                                                     self.with_b_a1, self.with_b_a2, self.with_b_a3, next_vel_a1, next_vel_a2, next_vel_a3)         
        
        #---- To ------
        next_to_a1, next_to_a2, next_to_a3 = get_next_to(action_a1, action_a2, action_a3, self.with_b_a1, self.with_b_a2, self.with_b_a3, \
                                                         self.to_a1, self.to_a2, self.to_a3)
        
        #---- Hold ----
        catch_a1, next_with_b_a1, next_pos_b, next_vel_b = get_catch(next_pos_a1, next_vel_a1, next_pos_b, next_vel_b, next_pass_a1, self.pass_a1, \
                                                                     next_with_b_a1, self.with_b_a1)
        catch_a2, next_with_b_a2, next_pos_b, next_vel_b = get_catch(next_pos_a2, next_vel_a2, next_pos_b, next_vel_b, next_pass_a2, self.pass_a2, \
                                                                     next_with_b_a2, self.with_b_a2)
        catch_a3, next_with_b_a3, next_pos_b, next_vel_b = get_catch(next_pos_a3, next_vel_a3, next_pos_b, next_vel_b, next_pass_a3, self.pass_a3, \
                                                                     next_with_b_a3, self.with_b_a3)
        catch_d, next_with_b_d, next_pos_b, next_vel_b = get_catch(next_pos_d, next_vel_d, next_pos_b, next_vel_b, self.pass_d, self.pass_d, \
                                                                   self.with_b_d, self.with_b_d)
        next_pass_a1, next_pass_a2, next_pass_a3 = reset_pass(catch_a1, catch_a2, catch_a3, next_pass_a1, next_pass_a2, next_pass_a3)
        
        obs_tmp, action_tmp = hold_obs_action(obs_a1, obs_a2, obs_a3, action_a1, action_a2, action_a3, next_pass_a1, next_pass_a2, next_pass_a3, \
                                              self.pass_a1, self.pass_a2, self.pass_a3, next_with_b_a1, next_with_b_a2, next_with_b_a3, catch_d, \
                                              self.obs_tmp, self.action_tmp)
        #---------------
        
        reward_d = get_reward_defender(next_pos_d, next_pos_b)
        reward_a1_w = get_reward_attacker(self.pass_a1, catch_a2, catch_a3, catch_d, next_with_b_a1)
        reward_a2_w = get_reward_attacker(self.pass_a2, catch_a3, catch_a1, catch_d, next_with_b_a2)
        reward_a3_w = get_reward_attacker(self.pass_a3, catch_a1, catch_a2, catch_d, next_with_b_a3)
        
        reward_a1_wo = get_reward_attacker_wo(catch_a1, self.to_a1, next_pos_b)
        reward_a2_wo = get_reward_attacker_wo(catch_a2, self.to_a2, next_pos_b)
        reward_a3_wo = get_reward_attacker_wo(catch_a3, self.to_a3, next_pos_b)

        next_obs_d = get_obs_d(next_pos_d, next_vel_d, next_pos_a1, next_vel_a1, next_pos_a2, next_vel_a2, \
                               next_pos_a3, next_vel_a3, next_pos_b, next_vel_b)
        
        next_obs_a1 = get_obs_a(next_pos_a1, next_vel_a1, next_pos_a2, next_vel_a2, next_pos_a3, next_vel_a3, \
                                next_pos_d, next_vel_d, next_pos_b, next_vel_b)
        
        next_obs_a2 = get_obs_a(next_pos_a2, next_vel_a2, next_pos_a3, next_vel_a3, next_pos_a1, next_vel_a1, \
                                next_pos_d, next_vel_d, next_pos_b, next_vel_b)
        
        next_obs_a3 = get_obs_a(next_pos_a3, next_vel_a3, next_pos_a1, next_vel_a1, next_pos_a2, next_vel_a2,\
                                next_pos_d, next_vel_d, next_pos_b, next_vel_b)        
                                        
        #---- merge ----
        comp_a1 = get_complete(self.pass_a1, catch_a2, catch_a3, catch_d, next_with_b_a1)
        comp_a2 = get_complete(self.pass_a2, catch_a3, catch_a1, catch_d, next_with_b_a2)
        comp_a3 = get_complete(self.pass_a3, catch_a1, catch_a2, catch_d, next_with_b_a3)
        
        obs_a, action_a, reward_a, next_obs_a, push_a = get_merge(comp_a1, comp_a2, comp_a3, obs_tmp, action_tmp, \
                                                                  reward_a1_w, reward_a2_w, reward_a3_w, next_obs_a1, next_obs_a2, next_obs_a3)
        # ----------------
        
        done = get_done(next_pos_d, next_pos_a1, next_pos_a2, next_pos_a3, next_pos_b, num_step, self.max_step)       
        
        self.pos_d = next_pos_d
        self.vel_d = next_vel_d
        self.pos_a1 = next_pos_a1
        self.vel_a1 = next_vel_a1
        self.pos_a2 = next_pos_a2
        self.vel_a2 = next_vel_a2
        self.pos_a3 = next_pos_a3
        self.vel_a3 = next_vel_a3
        self.pos_b = next_pos_b
        self.vel_b = next_vel_b
        self.with_b_a1 = next_with_b_a1
        self.with_b_a2 = next_with_b_a2
        self.with_b_a3 = next_with_b_a3
        self.pass_a1 = next_pass_a1
        self.pass_a2 = next_pass_a2
        self.pass_a3 = next_pass_a3
        self.obs_tmp = obs_tmp
        self.action_tmp = action_tmp
        self.to_a1 = next_to_a1
        self.to_a2 = next_to_a2
        self.to_a3 = next_to_a3
        
        return next_obs_d, next_obs_a1, next_obs_a2, next_obs_a3, reward_d, reward_a1_wo, reward_a2_wo, reward_a3_wo, done, \
               obs_a, action_a, reward_a, next_obs_a, push_a, next_with_b_a1, next_with_b_a2, next_with_b_a3, next_to_a1, next_to_a2, next_to_a3 


def get_obs_d(pos_d, vel_d, pos_a1_tmp, vel_a1_tmp, pos_a2_tmp, vel_a2_tmp, pos_a3_tmp, vel_a3_tmp, pos_b, vel_b):

    pos_a1, vel_a1, pos_a2, vel_a2, pos_a3, vel_a3 = get_order_adv3(pos_d, pos_a1_tmp, vel_a1_tmp, pos_a2_tmp, vel_a2_tmp, pos_a3_tmp, vel_a3_tmp)
    
    sub_vel_own_adv1 = get_sub_vel(pos_d, pos_a1, vel_d)
    sub_vel_own_adv2 = get_sub_vel(pos_d, pos_a2, vel_d)
    sub_vel_own_adv3 = get_sub_vel(pos_d, pos_a3, vel_d)
    sub_vel_own_ball = get_sub_vel(pos_d, pos_b, vel_d)
        
    sub_pos_adv1 = get_sub_pos(pos_d, pos_a1)
    sub_pos_adv2 = get_sub_pos(pos_d, pos_a2)
    sub_pos_adv3 = get_sub_pos(pos_d, pos_a3)
    sub_pos_ball = get_sub_pos(pos_d, pos_b)
    
    sub_vel_ball = get_sub_vel(pos_d, pos_b, vel_b)
               
    obs_d = np.concatenate([pos_d] + [sub_vel_own_adv1] + [sub_vel_own_adv2] + [sub_vel_own_adv3] + [sub_vel_own_ball] + \
                           [pos_a1] + [sub_pos_adv1] + \
                           [pos_a2] + [sub_pos_adv2] + \
                           [pos_a3] + [sub_pos_adv3] + \
                           [pos_b] + [sub_pos_ball] + [sub_vel_ball]).reshape(1,28)
    return obs_d


def get_obs_a(pos_a1, vel_a1, pos_a2, vel_a2, pos_a3, vel_a3, pos_d, vel_d, pos_b, vel_b):
        
    sub_pos_mate1 = get_sub_pos(pos_a1, pos_a2)
    sub_pos_mate2 = get_sub_pos(pos_a1, pos_a3)
    sub_pos_adv = get_sub_pos(pos_a1, pos_d)
    sub_pos_ball = get_sub_pos(pos_a1, pos_b)
    
    sub_vel_ball = get_sub_vel(pos_d, pos_b, vel_b)
                 
    obs_a = np.concatenate([pos_a1] + \
                           [pos_a2] + [sub_pos_mate1] + \
                           [pos_a3] + [sub_pos_mate2] + \
                           [pos_d] + [sub_pos_adv] + \
                           [pos_b] + [sub_pos_ball] + [sub_vel_ball]).reshape(1,20)
    return obs_a
    


def get_abs_ball_u(action, abs_own_pos, abs_mate1_pos, abs_mate2_pos, abs_adv_pos, \
                   with_b_own, pass_own, to_own, to_mate1, to_mate2, \
                   abs_ball_pos, abs_ball_vel):
    
    if with_b_own == True:
        
        th0 = 1.0
        th2 = 10
        
        dist = get_dist(abs_own_pos, abs_adv_pos)
        
        if dist < th0:
            a, b = get_angle_pass(abs_own_pos, abs_mate1_pos, abs_mate2_pos, abs_adv_pos)
            a, b = a*180/np.pi, b*180/np.pi
            
            if a < th2 and b < th2: 
                abs_u = [0, 0]
                abs_u_b = [0, 0]
                next_pass_own = False
                next_with_b_own = True
                action = 0
            else:
                if a > b:
                    abs_u = [0, 0]
                    sub_u_b = [1, 0]
                    noise = np.random.normal(0, 0.05, 2)
                    sub_u_b += noise
                    abs_u_b = rotate_u(sub_u_b, abs_own_pos, abs_mate1_pos)
                    next_pass_own = True
                    next_with_b_own = False
                    action = 1

                elif a <= b:
                    abs_u = [0, 0]
                    sub_u_b = [1, 0]
                    noise = np.random.normal(0, 0.05, 2)
                    sub_u_b += noise
                    abs_u_b = rotate_u(sub_u_b, abs_own_pos, abs_mate2_pos)
                    next_pass_own = True
                    next_with_b_own = False
                    action = 2
                
        else:
            abs_u = [0, 0]
            abs_u_b = [0, 0]
            next_pass_own = False
            next_with_b_own = True
            action = 0
                        
    elif with_b_own == False:
        
        th1 = 30
        th3 = 0.8
        
        if to_own == True:
            
            sign, val = get_dir_catch(abs_own_pos, abs_ball_pos, abs_ball_vel)
            if val > 0.05:            
                if sign == 1:
                    origin = [0,0]
                    ang = -np.pi / 2
                    sub_u = [np.cos(ang), np.sin(ang)]
                    abs_u = rotate_u(sub_u, abs_own_pos, origin)
                    abs_u_b = [0, 0]
                    next_pass_own = pass_own
                    next_with_b_own = False
                    action = 2
                elif sign == -1:
                    origin = [0,0]
                    ang = np.pi / 2
                    sub_u = [np.cos(ang), np.sin(ang)]
                    abs_u = rotate_u(sub_u, abs_own_pos, origin)
                    abs_u_b = [0, 0]
                    next_pass_own = pass_own
                    next_with_b_own = False
                    action = 1
            else:
                abs_u = [0, 0]
                abs_u_b = [0, 0]
                next_pass_own = pass_own
                next_with_b_own = False
                action = 0
        
        elif to_mate1 == True:
            
            dist = get_dist(abs_own_pos, abs_mate1_pos)
        
            if dist > th3:

                a, b = get_angle_pass(abs_mate1_pos, abs_mate2_pos, abs_own_pos, abs_adv_pos)
                a, b = a*180/np.pi, b*180/np.pi
#                 print('a', a, 'b', b)

                if  b < th1:
                    origin = [0,0]
                    ang = np.pi / 2
                    sub_u = [np.cos(ang), np.sin(ang)]
                    abs_u = rotate_u(sub_u, abs_own_pos, origin)
                    abs_u_b = [0, 0]
                    next_pass_own = pass_own
                    next_with_b_own = False
                    action = 1

                else:
                    abs_u = [0, 0]
                    abs_u_b = [0, 0]
                    next_pass_own = pass_own
                    next_with_b_own = False
                    action = 0
                    
            else:
                abs_u = [0, 0]
                abs_u_b = [0, 0]
                next_pass_own = pass_own
                next_with_b_own = False
                action = 0
            
                
        elif to_mate2 == True:
            
            dist = get_dist(abs_own_pos, abs_mate2_pos)
        
            if dist > th3:
                
                a, b = get_angle_pass(abs_mate2_pos, abs_own_pos, abs_mate1_pos, abs_adv_pos)
                a, b = a*180/np.pi, b*180/np.pi
            
                if a < th1:
                    origin = [0,0]
                    ang = -np.pi / 2
                    sub_u = [np.cos(ang), np.sin(ang)]
                    abs_u = rotate_u(sub_u, abs_own_pos, origin)
                    abs_u_b = [0, 0]
                    next_pass_own = pass_own
                    next_with_b_own = False
                    action = 2
            
                else:
                    abs_u = [0, 0]
                    abs_u_b = [0, 0]
                    next_pass_own = pass_own
                    next_with_b_own = False
                    action = 0
                    
            else:
                    abs_u = [0, 0]
                    abs_u_b = [0, 0]
                    next_pass_own = pass_own
                    next_with_b_own = False
                    action = 0
                
        else:
            abs_u = [0, 0]
            abs_u_b = [0, 0]
            next_pass_own = pass_own
            next_with_b_own = False
            action = 0
            

    return abs_u, abs_u_b, next_pass_own, next_with_b_own, action


def hold_obs_action(obs_a1, obs_a2, obs_a3, action_a1, action_a2, action_a3, next_pass_a1, next_pass_a2, next_pass_a3, pass_a1, pass_a2, pass_a3, \
                    next_with_b_a1, next_with_b_a2, next_with_b_a3, catch_d, pre_obs_tmp, pre_action_tmp):
    
    if next_pass_a1==True and pass_a1==False:
        obs_tmp = obs_a1
        action_tmp = action_a1
    elif next_pass_a2==True and pass_a2==False:
        obs_tmp = obs_a2
        action_tmp = action_a2
    elif next_pass_a3==True and pass_a3==False:
        obs_tmp = obs_a3
        action_tmp = action_a3
    
    elif next_with_b_a1==True and catch_d==True:
        obs_tmp = obs_a1
        action_tmp = action_a1
    elif next_with_b_a2==True and catch_d==True:
        obs_tmp = obs_a2
        action_tmp = action_a2
    elif next_with_b_a3==True and catch_d==True:
        obs_tmp = obs_a3
        action_tmp = action_a3
        
    else:
        obs_tmp = pre_obs_tmp
        action_tmp = pre_action_tmp

    return obs_tmp, action_tmp


def get_next_ball_state(abs_pos_ball, abs_vel_ball, abs_u_b_a1 , abs_u_b_a2, abs_u_b_a3, mass, accel, dt, \
                        next_pass_a1, next_pass_a2, next_pass_a3, pass_a1, pass_a2, pass_a3, with_b_a1, with_b_a2, with_b_a3, 
                        next_vel_a1, next_vel_a2, next_vel_a3):

    if next_pass_a1 == True and pass_a1 == False:
        abs_acc_ball = np.array(abs_u_b_a1) / mass
        next_abs_vel_ball = abs_acc_ball * accel * dt

    elif next_pass_a2 == True and pass_a2 == False:
        abs_acc_ball = np.array(abs_u_b_a2) / mass
        next_abs_vel_ball = abs_acc_ball * accel * dt

    elif next_pass_a3 == True and pass_a3 == False:
        abs_acc_ball = np.array(abs_u_b_a3) / mass
        next_abs_vel_ball = abs_acc_ball * accel * dt
        
    elif with_b_a1 == True and next_pass_a1 == False:
        next_abs_vel_ball = next_vel_a1
    
    elif with_b_a2 == True and next_pass_a2 == False:
        next_abs_vel_ball = next_vel_a2
        
    elif with_b_a3 == True and next_pass_a3 == False:
        next_abs_vel_ball = next_vel_a3
                    
    else:        
        next_abs_vel_ball = abs_vel_ball

    next_abs_pos_ball = abs_pos_ball + next_abs_vel_ball * dt

    return next_abs_pos_ball, next_abs_vel_ball


def get_next_to(action_a1, action_a2, action_a3, with_b_a1, with_b_a2, with_b_a3, to_a1, to_a2, to_a3):
    
    if with_b_a1 == True:
        if action_a1 == 0:
            next_to_a1, next_to_a2, next_to_a3 = True, False, False
        elif action_a1 == 1:
            next_to_a1, next_to_a2, next_to_a3 = False, True, False
        elif action_a1 == 2:
            next_to_a1, next_to_a2, next_to_a3 = False, False, True
            
    elif with_b_a2 == True:
        if action_a2 == 0:
            next_to_a1, next_to_a2, next_to_a3 = False, True, False
        elif action_a2 == 1:
            next_to_a1, next_to_a2, next_to_a3 = False, False, True
        elif action_a2 == 2:
            next_to_a1, next_to_a2, next_to_a3 = True, False, False
            
    elif with_b_a3 == True:
        if action_a3 == 0:
            next_to_a1, next_to_a2, next_to_a3 = False, False, True
        elif action_a3 == 1:
            next_to_a1, next_to_a2, next_to_a3 = True, False, False
        elif action_a3 == 2:
            next_to_a1, next_to_a2, next_to_a3 = False, True, False
        
    else:
        next_to_a1, next_to_a2, next_to_a3 = to_a1, to_a2, to_a3
    
    return next_to_a1, next_to_a2, next_to_a3


def get_catch(next_pos_own, next_vel_own, next_pos_ball, next_vel_ball, next_pass_own, pass_own, next_with_b_own, with_b_own):
    dist = get_dist(next_pos_own, next_pos_ball)

    if dist < 0.1 and next_pass_own == False and pass_own == False and with_b_own == False:
        catch_own = True
        next_with_b_own = True
        next_pos_ball, next_vel_ball = next_pos_own, next_vel_own
    else:
        catch_own = False
        next_with_b_own = next_with_b_own 
        next_pos_ball, next_vel_ball = next_pos_ball, next_vel_ball

    return catch_own, next_with_b_own, next_pos_ball, next_vel_ball


def reset_pass(catch_a1, catch_a2, catch_a3, next_pass_a1, next_pass_a2, next_pass_a3):
    if catch_a1 == True or catch_a2 == True or catch_a3 == True:
        next_pass_a1, next_pass_a2, next_pass_a3 = False, False, False
    else:
        next_pass_a1, next_pass_a2, next_pass_a3 = next_pass_a1, next_pass_a2, next_pass_a3
    
    return next_pass_a1, next_pass_a2, next_pass_a3


def get_reward_defender(abs_pos_own, abs_pos_adv):
    dist = get_dist(abs_pos_own, abs_pos_adv)
    reward = 0

    if dist < 0.1:
        reward = 1
    elif abs_pos_own[0] < -1 or abs_pos_own[1] < -1 or abs_pos_own[0] > 1 or abs_pos_own[1] > 1:
        reward = -1

    return reward


def get_reward_attacker(pass_own, catch_mate1, catch_mate2, catch_adv, with_b_own):
    reward = 0

    if pass_own == True and catch_adv == True:
        reward = -1
    elif with_b_own == True and catch_adv == True:
        reward = -1
    elif pass_own == True and catch_mate1 == True:
        reward = 1
    elif pass_own == True and catch_mate2 == True:
        reward = 1
        
    return reward


def get_reward_attacker_wo(catch_own, to_own, abs_pos_ball):
    dist = get_dist(abs_pos_ball)
    reward = 0
    
    if catch_own == True:
        reward = 1
    elif to_own == True and dist > 0.8:
        reward = -1
        
    return reward


def get_complete(pass_own, catch_mate1, catch_mate2, catch_adv, next_with_b_own):
    
    if pass_own == True and catch_mate1 == True:
        comp_own = True
    elif next_with_b_own == True and catch_adv == True:
        comp_own = True
    elif pass_own == True and catch_mate2 == True:    
        comp_own = True
    elif pass_own == True and catch_adv == True:
        comp_own = True        
    else:
        comp_own = False
    
    return comp_own


def get_merge(comp_a1, comp_a2, comp_a3, obs_tmp, action_tmp, reward_a1, reward_a2, reward_a3, next_obs_a1, next_obs_a2, next_obs_a3):
    
    if comp_a1==True:
        obs_a = obs_tmp
        action_a = action_tmp
        reward_a = reward_a1
        next_obs_a = next_obs_a1
        push_a = True
    elif comp_a2==True:
        obs_a = obs_tmp
        action_a = action_tmp
        reward_a = reward_a2
        next_obs_a = next_obs_a2
        push_a = True
    elif comp_a3==True:
        obs_a = obs_tmp
        action_a = action_tmp
        reward_a = reward_a3
        next_obs_a = next_obs_a3
        push_a = True
    else:
        obs_a = None
        action_a = None
        reward_a = None
        next_obs_a = None
        push_a = False
        
    return obs_a, action_a, reward_a, next_obs_a, push_a


def get_done(abs_pos_own, abs_pos_adv1, abs_pos_adv2, abs_pos_adv3, abs_pos_ball, num_step, max_step):
    dist1 = get_dist(abs_pos_ball, abs_pos_own)
    dist2 = get_dist(abs_pos_own)
    dist3 = get_dist(abs_pos_ball)

    if dist1 < 0.1 or dist2 > 0.8 or dist3 > 0.8 or num_step > max_step:
        done = True
    else:
        done = False

    return done


def get_angle_pass(own, mate1, mate2, adv):
    vec0 = adv - own
    vec1 = mate1 - own
    i1 = np.inner(vec0,vec1)
    n1 = np.linalg.norm(vec0) * np.linalg.norm(vec1)
    theta1 = np.arccos(i1/n1)
    
    vec2 = mate2 - own
    i2 = np.inner(vec0,vec2)
    n2 = np.linalg.norm(vec0) * np.linalg.norm(vec2)
    theta2 = np.arccos(i2/n2)
    
    return theta1, theta2

def get_dir_catch(pos_own, pos_ball, vel_ball):
    vec0 = pos_own - pos_ball
    vec1 = vel_ball
    cross = np.cross(vec0, vec1)
    sign = np.sign(cross)
    val = np.abs(cross)
    
    return sign, val
