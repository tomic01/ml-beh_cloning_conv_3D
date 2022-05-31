import gym
from gym.wrappers import AtariPreprocessing, Monitor, FrameStack
import torch
from model import BreakoutCnnModelDDQ, Breakout3DConvModel
import numpy as np

outdir = './output'
data_dir = './data'


def createEmptyTensorInput():
    return torch.zeros([4, 84, 84], dtype=torch.float32)


# Create tensor from observation Todo: more efficient way
def createTensor(obs):
    imgT = createEmptyTensorInput()
    # print(type(obs))
    for k in range(4):
        for i in range(84):
            for j in range(84):
                imgT[k, i, j] = obs[k][i][j] / 255
    return imgT


def loadLearnedModel():
    model = Breakout3DConvModel()  # BreakoutCnnModelDDQ()
    model.load_state_dict(torch.load('./saves/model-cnn.pth'))
    return model


def getAction(input, model):
    obsT = torch.unsqueeze(input, 0)  # Add one (batch) dimension, to be compatible
    outputs = model.forward(obsT)
    action = int(torch.argmax(outputs[0]))
    return action


def main():
    env = gym.make('BreakoutNoFrameskip-v0')
    env = AtariPreprocessing(env, frame_skip=4)
    env = FrameStack(env, 4)
    # env = Monitor(env, directory=outdir, force=True)
    env.seed(43)
    the_model = loadLearnedModel()

    actionList = []
    rewardList = []
    beginningList = []

    t = 0
    for episode in range(10):
        obs = env.reset()
        done = False

        ## THE GAME LOOP ##
        t = 0
        while not done and t < 1000:
            env.render()

            # Get the action from the model
            obsT = createTensor(obs)
            action = getAction(obsT, the_model)
            obs, reward, done, info = env.step(action)

            # Add to arrays
            actionList.append(action)
            rewardList.append(reward)
            if t == 0:
                beginningList.append(True)
            else:
                beginningList.append(False)

            t += 1
            if t % 100 == 0:
                print(t)

    print("Environment Close.")
    env.close()

    actions = np.array(actionList)
    episode_starts = np.array(beginningList)
    rewards = np.array(rewardList)

    np.save(data_dir + "/ai_actions", actions)
    np.save(data_dir + "/ai_episode_starts", episode_starts)
    np.save(data_dir + "/ai_rewards", rewards)


if __name__ == '__main__':
    main()
