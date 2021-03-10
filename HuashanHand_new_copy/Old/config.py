config = dict()
# model
config['latent_dim'] = 20

# data 
config['PIDwidth'] = 3
config['timesteps'] = 235
config['ROINum'] = 15
config['aux_len'] = 11
config['EVA_names'] = ['N','Y']
config['num_class'] = len(config['EVA_names'])

# split data
config['currentFold'] = 1
config['totalFold'] = 6
config['random_state'] = 100 
# config['testNumPer'] = 0.2

# select_brain_region
select_brain_region = dict()
select_brain_region['R']=[0,2,4,6,8,10,12,14,18,22,24,56,58,60,68]
select_brain_region['L']=[1,2,5,7,9,11,13,15,19,23,25,57,59,61,69]
# select_brain_region['R']=[1,3,5,7,9,11,13,15,19,23,25,57,59,61,69]
# select_brain_region['L']=[2,4,6,8,10,12,14,16,20,24,26,58,60,62,70]
# Injury_side = 'R'
# Injury_side = 'L' 
