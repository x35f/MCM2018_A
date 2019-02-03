from make_env import make_env
env=make_env('dragon')
print('action_space:',env.act_space)
print('observation_space:',env.obs_space)
obs=env.reset()
obs,rew,done,info=env.step([0,1,0,0,0,0])
print("obs:",obs,"\trew:",rew)
