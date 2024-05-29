import numpy as np
from parameters import env_parameters, playback_parameters 
from environment import Env, Net, Agent_playback

if __name__ == "__main__":
    agent = Agent_playback()
    agent.load_param()
    env = Env()

    training_records = []
    running_score = 0
    state = env.reset()
    for i_ep in range(playback_parameters.run_episodes):
        score = 0
        state = env.reset()

        for t in range(playback_parameters.t_timesteps):
            action = agent.select_action(state)
            state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            if playback_parameters.render:
                env.render()
            score += reward
            state = state_
            if done or die:
                break

        print('Ep {}\tScore: {:.2f}\t'.format(i_ep, score))
