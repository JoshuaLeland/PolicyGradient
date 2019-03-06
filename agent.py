import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.normal as Norm
import torch.distributions.categorical as Cat
import numpy as np

#Build this to teach myself some decent policy gradent strategies.
#Followed Berkley's Deep RL, so this follows their A2 very closely.
def build_mlp(inputSize, layerSizes, outputSize, activation = nn.Tanh(), double = False):
	#Make first layer
	layer = [nn.Linear(inputSize,layerSizes[0]), activation]

	#Add the rest.
	for i, num in enumerate(layerSizes[:-1]):
		layer = layer + [nn.Linear(layerSizes[i],layerSizes[i+1]), activation]

	#Make final layer.
	layer = layer + [nn.Linear(layerSizes[-1], outputSize)]

	if double:
		return nn.Sequential(*layer).double()
	else:
		return nn.Sequential(*layer)

def pathlength(path):
    return len(path["reward"])

#TODO: Add CUDA functionality.
class Agent(object):

	def __init__(self, policy_dict, sample_trajectory_args, estimate_return_args):
		super(Agent, self).__init__()
		self.ob_dim = policy_dict['ob_dim']
		self.ac_dim = policy_dict['ac_dim']
		self.discrete = policy_dict['discrete']
		self.size = policy_dict['size']
		self.n_layers = policy_dict['n_layers']
		self.learning_rate = policy_dict['learning_rate']

		self.animate = sample_trajectory_args['animate']
		self.max_path_length = sample_trajectory_args['max_path_length']
		self.min_timesteps_per_batch = sample_trajectory_args['min_timesteps_per_batch']

		self.gamma = estimate_return_args['gamma']
		self.reward_to_go = estimate_return_args['reward_to_go']
		self.nn_baseline = estimate_return_args['nn_baseline']
		self.normalize_advantages = estimate_return_args['normalize_advantages']

	#Sets up the policy given the input.
	def build_policy(self, double = False):
		#Build the history info.
		self.double = double

		#Build feed forward policy
		if self.discrete:
			self.policy = build_mlp(self.ob_dim, [self.size for x in range(self.n_layers)], self.ac_dim, double=double)
		else:
			self.policy = build_mlp(self.ob_dim, [self.size for x in range(self.n_layers)], 2*self.ac_dim, double=double)

		#Initalize the optimizer
		if self.discrete:
			self.optimizer = torch.optim.Adam(self.policy.parameters(), lr = self.learning_rate)
		else:
			self.optimizer = torch.optim.Adam(self.policy.parameters(), lr = self.learning_rate)

		#If we want a baseline do it here.
		if self.nn_baseline:
			self.baseline = build_mlp(self.ob_dim, [self.size for x in range(self.n_layers)], 1, double=double)
			self.optimizer_bl = torch.optim.Adam(self.baseline.parameters(), lr = self.learning_rate)
			self.baseline_loss = torch.nn.MSELoss()

	#Build the sample action function
	def sample_action(self, policy_parameters):
		if self.discrete:
			sy_logits_na = policy_parameters
			sy_sampled_ac = Cat.Categorical(logits = sy_logits_na)
		else:
			if policy_parameters.dim() == 1:
				sy_mean, sy_logstd = policy_parameters[:self.ac_dim], policy_parameters[self.ac_dim:]
			else:
				sy_mean, sy_logstd = policy_parameters[:, :self.ac_dim], policy_parameters[:,self.ac_dim:]
			#print(sy_mean)
			sy_sampled_ac = Norm.Normal(loc=sy_mean, scale=torch.exp(sy_logstd))
		return sy_sampled_ac.sample()

	#Build the get log_prob function
	def get_log_prob(self, policy_parameters, sy_ac_na):
		if self.discrete:
			sy_logits_na = policy_parameters
			sy_sampled_ac = Cat.Categorical(logits = sy_logits_na)
		else:
			if policy_parameters.dim() == 1:
				sy_mean, sy_logstd = policy_parameters[:self.ac_dim], policy_parameters[self.ac_dim:]
			else: 
				sy_mean, sy_logstd = policy_parameters[:, :self.ac_dim], policy_parameters[:,self.ac_dim:]
			sy_sampled_ac = Norm.Normal(loc=sy_mean.view(self.ac_dim,-1), scale=torch.exp(sy_logstd.view(self.ac_dim,-1)))
		return sy_sampled_ac.log_prob(sy_ac_na)

	#Build the loss function. Uses observations actions and advantages to build a loss function.
	# We will calculate the advantaage differently in other functions.
	def calculate_loss(self, param_hist, act_hist, adv_hist):
		#Get Log Prob
		logprobs = self.get_log_prob(param_hist, act_hist)
		logprobs = logprobs.squeeze()

		#discrete only has 1 dim tensor.
		rlogprobs = logprobs*adv_hist
		loss = torch.sum(rlogprobs)
		return loss

	#Helper function for estimate return.
	def sum_of_rewards(self, re_n):
		#Assign the out values.
		gamma = self.gamma
		if self.double:
			q_n = torch.tensor([]).double()
		else:
			q_n = torch.tensor([])
		if self.reward_to_go:
			#Flip & Cumsum
			for re in re_n:
				#Get the size of this.
				tsize = re.shape[0]
				#The discounts don't change so make the vector once.
				for j in range(tsize):
					# Keep adding the rewards to go.
					gammas = torch.tensor([gamma**(i-j) for i in range(j, tsize)])
					if self.double:
						gammas = gammas.double()
					Rtau = torch.sum(gammas*re[j:])
					q_n = torch.cat((q_n,torch.tensor([Rtau])))
		else:
			for re in re_n:
				tsize = re.shape[0]
				gammas = torch.tensor([gamma**i for i in range(0, tsize)])
				if self.double:
					gammas = gammas.double()
				Rtau = torch.sum(gammas*re)
				if self.double:
					ones = torch.ones(tsize).double()
				else:
					ones = torch.ones(tsize)
				q_n = torch.cat((q_n, Rtau*ones))
		return q_n

	#Helper function for estimate return.
	def compute_advantage(self, ob_no, q_n):
		#rescale the output from the nn_baseline to match the statistics
        # (mean and std) of the current batch of Q-values.
        # This trick is from Berkley course.
		if self.nn_baseline:
			b_n = self.baseline(ob_no).view(-1).detach()
			if b_n.std() < 1e-6:
				std = 1
			else:
				std = b_n.std()
			b_n = (b_n - b_n.mean())/std
			b_n = b_n*q_n.std() + q_n.mean()
			adv_n = q_n - b_n
		else:
			adv_n = q_n.clone()
		return adv_n

	#Will normalize the return if advantage is set.
	def estimate_return(self, ob_no, re_n):
		q_n = self.sum_of_rewards(re_n)
		adv_n = self.compute_advantage(ob_no, q_n)

		#If only one trajectory is used, you must use reward to go or this will have a variance of 0 and blow up.
		if self.normalize_advantages:
			adv_n = (adv_n - adv_n.mean())/adv_n.std()
		return q_n, adv_n

	def update_parameters(self, ob_no, ac_na, param_hist,  q_n, adv_n):

		#Update baseline estimator here.
		if self.nn_baseline:
			self.optimizer_bl.zero_grad()
			#Rescale to have mean = 0 and std = 1
			#self.baseline.train()
			target_n = (q_n - q_n.mean())/q_n.std()
			bl = self.baseline(ob_no)
			bl_loss = self.baseline_loss(bl, target_n)
			bl_loss.backward()
			self.optimizer_bl.step()


		#Set policy to train and re-calc distubution parameters.
		#param_hist = self.policy(ob_no)

		#Calc our loss and update the policy.
		loss = -1*self.calculate_loss(param_hist, ac_na, adv_n)
		#print(loss)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		#Turn policy back to evaluation mode.

#Just use this for gym. - Similar to Berkley.
def sample_tracjectory(agent, env, animate_episode):
	ob = env.reset()
	#Make these tensors.
	sampling = True
	steps = 0
	done = False
	while sampling == True:
		if done == True or (steps > agent.min_timesteps_per_batch):
			sampling = False
		else:
			if animate_episode:
				env.render()
				time.sleep(0.1)
			#print(torch.tensor(ob))
			param = agent.policy(torch.tensor(ob))
			#print(param.shape)
			ac = agent.sample_action(param)
			param = param.unsqueeze(0)
			acnp = ac.cpu().numpy() # Get this into numpy
			ob, rew, done, _ = env.step(acnp)
			if steps == 0:
				acs = torch.tensor([ac])
				params = param
				obs = torch.tensor([ob])
				rewards = torch.tensor([rew])
			else:
				#concat the tensors.
				acs = torch.cat((acs,torch.tensor([ac])))
				obs = torch.cat((obs,torch.tensor([ob])))
				params = torch.cat((params,param))
				rewards = torch.cat((rewards,torch.tensor([rew])))
		steps = steps + 1
	if agent.double:
		rewards = rewards.double()
	#print(params)
	path = {"observation" : obs,
			"reward" : rewards,
			"param": params,
			"action" : acs}
	return path

#run the trajectories. -Gym
def sample_trajectories(agent, itr, env):
	time_steps_this_batch = 0
	paths = []
	sampling = True
	while sampling:
		animate_episode = (len(paths)==0 and (itr % 10 == 0) and agent.animate)
		path = sample_tracjectory(agent, env, animate_episode)
		paths.append(path)
		time_steps_this_batch += pathlength(path)
		if time_steps_this_batch > agent.max_path_length:
			sampling = False
	return paths, time_steps_this_batch

#Helper function for GYM
def collect_trajectories(paths):
	obs_no = torch.cat([path["observation"] for path in paths])
	ac_na = torch.cat([path["action"] for path in paths])
	params = torch.cat([path["param"] for path in paths])
	re_n = [path["reward"] for path in paths]

	return obs_no, ac_na, params, re_n