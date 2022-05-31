import numpy as np
import matplotlib.pyplot as plt

data_dir = './data'
NUM_EPISODES_HUMAN = 10
NUM_EPISODES_AI = 10


def statsAboutHumanPlay():
    assert NUM_EPISODES_HUMAN <= 1000

    # LOAD HUMAN DATA
    actions = np.load(data_dir + '/actionsData.npy')
    actions = actions.reshape(actions.shape[0])
    episode_starts = np.load(data_dir + '/episodeStarts.npy')
    rewards = np.load(data_dir + '/rewards.npy')
    rewards = rewards.reshape(rewards.shape[0])

    # LOAD AI DATA
    ai_actions = np.load(data_dir + '/aiActionsData.npy')
    ai_episode_starts = np.load(data_dir + '/aiEpisodeStarts.npy')
    ai_rewards = np.load(data_dir + '/aiRewards.npy')

    # GET REWARDS INFO
    rewards_eps = getRewardsPerEpisode(rewards, episode_starts, NUM_EPISODES_HUMAN)
    ai_rewards_eps = getRewardsPerEpisode(ai_rewards, ai_episode_starts, NUM_EPISODES_AI)

    print(f"--Max reward--\n \tHuman: {np.max(rewards_eps)}, AI: {np.max(ai_rewards_eps)}")
    print(f"--Mean reward--\n \tHuman: {np.mean(rewards_eps)}, AI: {np.mean(ai_rewards_eps)}")
    print(f"--Std. reward--\n \tHuman: {np.std(rewards_eps)}, AI: {np.std(ai_rewards_eps)}")
    print(f"--Variation reward--\n \tHuman: {np.var(rewards_eps)}, AI: {np.var(ai_rewards_eps)}")

    rewards_mv = movingAverage(rewards_eps)
    ai_rewards_mv = movingAverage(ai_rewards_eps)

    human_plt, = plt.plot(rewards_mv, '-bx', label='human')
    ai_plot, = plt.plot(ai_rewards_mv, '-rx', label='ai')
    plt.legend(handles=[human_plt, ai_plot])
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Human and AI Cumulative Rewards per Episode')
    plt.show()

    # GET REWARDS INFO
    actions0 = len(np.where(actions == 0)[0])
    actions1 = len(np.where(actions == 1)[0])
    actions2 = len(np.where(actions == 2)[0])
    actions3 = len(np.where(actions == 3)[0])

    ai_actions0 = len(np.where(ai_actions == 0)[0])
    ai_actions1 = len(np.where(ai_actions == 1)[0])
    ai_actions2 = len(np.where(ai_actions == 2)[0])
    ai_actions3 = len(np.where(ai_actions == 3)[0])

    action_samples = [actions0, actions1, actions2, actions3]
    ai_action_samples = [ai_actions0, ai_actions1, ai_actions2, ai_actions3]
    action_samples_n = [(x / np.sum(action_samples)) for x in action_samples]
    ai_action_samples_n = [(x / np.sum(ai_action_samples)) for x in ai_action_samples]
    print("Action Probabilities: ", action_samples_n)

    # Plot actions probability distribution #
    x = np.arange(4)
    plt.subplot(111)
    plt.bar(x - 0.15, action_samples_n, color='b', width=0.3)
    plt.bar(x + 0.15, ai_action_samples_n, color='r', width=0.3)
    plt.xticks(x, ['0', '1', '2', '3'])
    plt.xlabel('Actions')
    plt.ylabel('Probability')
    plt.legend(labels=['human', 'ai'])
    plt.show()


def getRewardsPerEpisode(rewards, episode_starts, num_episodes):
    rewards_eps = []
    start_condition = np.where(episode_starts == True)
    start_idxs = start_condition[0][:num_episodes]
    for i in range(num_episodes - 1):
        r = episodeCulReward(start_idxs[i], start_idxs[i + 1] - 1, rewards)
        rewards_eps.append(r)
    return rewards_eps


def episodeCulReward(start_idx, end_idx, rewards):
    return np.sum(rewards[start_idx:end_idx])


def movingAverage(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


if __name__ == '__main__':
    statsAboutHumanPlay()
    pass
