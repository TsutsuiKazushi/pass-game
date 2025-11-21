import numpy as np


def get_sub_pos(abs_pos_own, abs_pos_adv):
    pos_rel = abs_pos_adv - abs_pos_own
    theta = np.arctan2(abs_pos_own[1], abs_pos_own[0])
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    sub_pos = np.dot(rot, pos_rel)

    return sub_pos


def get_sub_vel(abs_pos_own, abs_pos_adv, abs_vel):
    pos_rel = abs_pos_adv - abs_pos_own
    theta = np.arctan2(abs_pos_own[1], abs_pos_own[0])
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    sub_vel = np.dot(rot, abs_vel)

    return sub_vel


def get_order_adv2(pos_own, pos_adv1_tmp, vel_adv1_tmp, pos_adv2_tmp, vel_adv2_tmp):
    dist1 = get_dist(pos_own, pos_adv1_tmp)
    dist2 = get_dist(pos_own, pos_adv2_tmp)

    d = [dist1, dist2]
    p = [pos_adv1_tmp, pos_adv2_tmp]
    v = [vel_adv1_tmp, vel_adv2_tmp]
    l = list(zip(d, p, v))
    l.sort()
    d, p, v = zip(*l)
    
    pos_adv1, vel_adv1 = p[0], v[0] 
    pos_adv2, vel_adv2 = p[1], v[1]
        
    return pos_adv1, vel_adv1, pos_adv2, vel_adv2


def get_order_adv3(pos_own, pos_adv1_tmp, vel_adv1_tmp, pos_adv2_tmp, vel_adv2_tmp, pos_adv3_tmp, vel_adv3_tmp):
    dist1 = get_dist(pos_own, pos_adv1_tmp)
    dist2 = get_dist(pos_own, pos_adv2_tmp)
    dist3 = get_dist(pos_own, pos_adv3_tmp)

    d = [dist1, dist2, dist3]
    p = [pos_adv1_tmp, pos_adv2_tmp, pos_adv3_tmp]
    v = [vel_adv1_tmp, vel_adv2_tmp, vel_adv3_tmp]
    l = list(zip(d, p, v))
    l.sort()
    d, p, v = zip(*l)
    
    pos_adv1, vel_adv1 = p[0], v[0] 
    pos_adv2, vel_adv2 = p[1], v[1]
    pos_adv3, vel_adv3 = p[2], v[2]
        
    return pos_adv1, vel_adv1, pos_adv2, vel_adv2, pos_adv3, vel_adv3


def get_order_adv4(pos_own, pos_adv1_tmp, vel_adv1_tmp, pos_adv2_tmp, vel_adv2_tmp, pos_adv3_tmp, vel_adv3_tmp, pos_adv4_tmp, vel_adv4_tmp):
    dist1 = get_dist(pos_own, pos_adv1_tmp)
    dist2 = get_dist(pos_own, pos_adv2_tmp)
    dist3 = get_dist(pos_own, pos_adv3_tmp)
    dist4 = get_dist(pos_own, pos_adv4_tmp)

    d = [dist1, dist2, dist3, dist4]
    p = [pos_adv1_tmp, pos_adv2_tmp, pos_adv3_tmp, pos_adv4_tmp]
    v = [vel_adv1_tmp, vel_adv2_tmp, vel_adv3_tmp, vel_adv4_tmp]
    l = list(zip(d, p, v))
    l.sort()
    d, p, v = zip(*l)
    
    pos_adv1, vel_adv1 = p[0], v[0] 
    pos_adv2, vel_adv2 = p[1], v[1]
    pos_adv3, vel_adv3 = p[2], v[2]
    pos_adv4, vel_adv4 = p[3], v[3]
        
    return pos_adv1, vel_adv1, pos_adv2, vel_adv2, pos_adv3, vel_adv3, pos_adv4, vel_adv4


def get_order_adv5(pos_own, pos_adv1_tmp, vel_adv1_tmp, pos_adv2_tmp, vel_adv2_tmp, pos_adv3_tmp, vel_adv3_tmp, pos_adv4_tmp, vel_adv4_tmp, pos_adv5_tmp, vel_adv5_tmp):
    dist1 = get_dist(pos_own, pos_adv1_tmp)
    dist2 = get_dist(pos_own, pos_adv2_tmp)
    dist3 = get_dist(pos_own, pos_adv3_tmp)
    dist4 = get_dist(pos_own, pos_adv4_tmp)
    dist5 = get_dist(pos_own, pos_adv5_tmp)

    d = [dist1, dist2, dist3, dist4, dist5]
    p = [pos_adv1_tmp, pos_adv2_tmp, pos_adv3_tmp, pos_adv4_tmp, pos_adv5_tmp]
    v = [vel_adv1_tmp, vel_adv2_tmp, vel_adv3_tmp, vel_adv4_tmp, vel_adv5_tmp]
    l = list(zip(d, p, v))
    l.sort()
    d, p, v = zip(*l)
    
    pos_adv1, vel_adv1 = p[0], v[0] 
    pos_adv2, vel_adv2 = p[1], v[1]
    pos_adv3, vel_adv3 = p[2], v[2]
    pos_adv4, vel_adv4 = p[3], v[3]
    pos_adv5, vel_adv5 = p[4], v[4]
        
    return pos_adv1, vel_adv1, pos_adv2, vel_adv2, pos_adv3, vel_adv3, pos_adv4, vel_adv4, pos_adv5, vel_adv5


def get_abs_u(action, abs_own_pos, abs_adv_pos):
    if action <= 11:
        ang = action * -np.pi / 6
        sub_u = [np.cos(ang), np.sin(ang)]            
        abs_u = rotate_u(sub_u, abs_own_pos, abs_adv_pos)
    elif action == 12:
        abs_u = [0, 0]
    
    return abs_u


def rotate_u(sub_u, abs_pos_own, abs_pos_adv):
    sub_u = np.array(sub_u)
    pos_rel = abs_pos_adv - abs_pos_own
    theta = np.arctan2(pos_rel[1], pos_rel[0])
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    abs_u = np.dot(rot, sub_u)

    return abs_u


def get_next_own_state(abs_pos_own, abs_vel_own, abs_u, mass, speed, damping, dt):
    abs_acc_own = np.array(abs_u) / mass
    next_abs_vel_own = abs_vel_own * (1 - damping) + abs_acc_own * speed * dt
    next_abs_pos_own = abs_pos_own + next_abs_vel_own * dt

    return next_abs_pos_own, next_abs_vel_own


def get_next_own_state_circ(abs_pos_own, abs_vel_own, abs_u, mass, speed, damping, dt):
    abs_acc_own = np.array(abs_u) / mass
    next_abs_vel_own = abs_vel_own * (1 - damping) + abs_acc_own * speed * dt
    next_abs_pos_own = abs_pos_own + next_abs_vel_own * dt
    next_abs_pos_ang = np.arctan2(next_abs_pos_own[1], next_abs_pos_own[0])
    next_abs_pos_own = np.array([np.cos(next_abs_pos_ang)*0.5, np.sin(next_abs_pos_ang)*0.5])
    
    return next_abs_pos_own, next_abs_vel_own


def get_dist(abs_pos_own, abs_pos_adv=[0,0]):
    pos_rel = abs_pos_adv - abs_pos_own
    dist = np.sqrt(np.sum(np.square(pos_rel)))

    return dist
