import os
import matplotlib.pyplot as plt

def os_add_pathways():
    os.environ['PATH'] += r";C:\Users\xavier\.mujoco\mjpro150\bin"
    os.add_dll_directory("C://Users//xavier//.mujoco//mjpro150//bin")
    os.environ['PATH'] += r";C:\Users\xavier\.ffmpeg\ffmpeg-2022-02-17-git-2812508086-essentials_build\bin"
    os.add_dll_directory("C://Program Files (x86)//Microsoft SDKs//MPI")


def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
              'max_timesteps': env._max_episode_steps, 'name': env.unwrapped.spec.id}
    return params


def plot_results(success_rate, actor_loss, critic_loss, env_params):

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(24, 10))

    # Plot accuracy values
    ax1.plot(success_rate, label='Success rate', color='green', alpha=0.7)
    ax1.set_title('Success for the {} task'.format(env_params['name']))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Success in %')
    ax1.legend()

    ax2.plot(actor_loss, label='Actor loss', color='black', alpha=0.7)
    ax2.set_title('Actor loss for the {} task'.format(env_params['name']))
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Actor loss')
    ax2.legend()

    ax3.plot(critic_loss, label='Critic loss', color='blue', alpha=0.7)
    ax3.set_title('Critic loss for the {} task'.format(env_params['name']))
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Critic loss')
    ax3.legend()

    fig.suptitle(env_params['name'])
    # save fig
    if not os.path.exists('./Plots'):
        os.mkdir('./Plots')
    plot_path = os.path.join('./Plots', env_params['name'])
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)
    plt.savefig(plot_path + '/' + env_params['name'] + '_plot.png')
    plt.show()



