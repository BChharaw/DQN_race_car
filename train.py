from environment import Env, Net, Agent
from utils import DrawLine
import numpy as np
from parameters import env_parameters, train_parameters
from tqdm import tqdm

if __name__ == "__main__":
    agent = Agent()
    env = Env()

    training_records = []
    running_score = 0
    state = env.reset()

    # Add tqdm for the outer loop
    for i_ep in tqdm(range(train_parameters.run_episodes), desc='Training Episodes'):
        score = 0
        state = env.reset()

        for t in range(train_parameters.t_timesteps):
            action, a_logp = agent.select_action(state)
            state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            if train_parameters.render:
                env.render()
            if agent.store((state, action, a_logp, reward, state_)):
                print('updating')
                agent.update()
            score += reward
            state = state_
            if done or die:
                break
        running_score = running_score * 0.99 + score * 0.01

        if i_ep % env_parameters.delay_between_logs == 0:
            print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(i_ep, score, running_score))
            agent.save_param()
        if running_score > env.reward_threshold:
            print("Solved! Running reward is now {} and the last episode runs to {}!".format(running_score, score))
            break
