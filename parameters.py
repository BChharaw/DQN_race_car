class env_parameters:
    gamma = 0.99 # discount factor
    action_repeat = 8 #repeat action in N frames
    img_stack = 4 #stack N image in a state
    seed = 0 #random seed
    delay_between_logs = 1 #how many episodes to run before logging to console

class agent_parameters:
    max_grad_norm = 0.5 #maximum gradient normalization
    clip_param = 0.1  # epsilon in clipped loss
    ppo_epoch = 10 #number of epochs to run PPO for
    buffer_capacity, batch_size = 2000, 128 #buffer and batch size for model

class train_parameters:
    run_episodes = 100 #number of episodes (episode finishes when agent dies)
    t_timesteps = 1000 #number of timesteps to run in playback
    save_weights_as = "param/ppo_net_params.pkl"
    render = True #renders the environment, allows you to view training for debugging, keep off in general to speed up simulation

    
class playback_parameters:
    render = True #renders the environment 
    run_episodes = 10 #number of episodes (episode finishes when agent dies)
    t_timesteps = 1000 #number of timesteps to run in playback
    load_weights_as = "param/ppo_net_params.pkl"

    
