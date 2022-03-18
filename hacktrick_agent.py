from hacktrick_ai_py.agents.agent import Agent, AgentPair
from hacktrick_ai_py.mdp.hacktrick_mdp import HacktrickState, Recipe
from hacktrick_ai_py.mdp.actions import Action
from hacktrick_rl.rllib.rllib import RlLibAgent, load_agent_pair


class MainAgent(Agent):

    def __init__(self):
        super().__init__()

    def action(self, state):
        # Implement your logic here
        # You should change your action value to a compatible Action value from the Action class in Hacktric_ai
        # You do not need to implement the action_probs but it is basically the probability distribution of actions
        action, action_probs = Action.STAY, {}
        return action, action_probs


class OptionalAgent(Agent):

    def __init__(self):
        super().__init__()
        
    def action(self, state):
        # Implement your logic here
        action, action_probs = Action.STAY, {}
        return action, action_probs


class HacktrickAgent(object):
    # Enable this flag if you are using reinforcement learning from the included ppo ray support library
    RL = False
    # Rplace with the directory for the trained agent
    # Note that `agent_dir` is the full path to the checkpoint FILE, not the checkpoint directory
    agent_dir = ''
    # If you do not plan to use the same agent logic for both agents and use the OptionalAgent set it to False
    # Does not matter if you are using RL as this is controlled by the RL agent
    share_agent_logic = True

    def __init__(self):
        Recipe.configure({})

        if self.RL:
            agent_pair = load_agent_pair(self.agent_dir)
            self.agent0 = agent_pair.a0
            self.agent1 = agent_pair.a1
        else:
            self.agent0 = MainAgent()
            self.agent1 = OptionalAgent()
    
    def set_mode(self, mode):
        self.mode = mode

        if "collaborative" in self.mode:
            if self.share_agent_logic and not self.RL:
                self.agent1 = MainAgent()
            self.agent_pair = AgentPair(self.agent0, self.agent1)
        else:
            self.agent1 =None
            self.agent_pair =None
    
    def map_action(self, action):
        action_map = {(0, 0): 'STAY', (0, -1): 'UP', (0, 1): 'DOWN', (1, 0): 'RIGHT', (-1, 0): 'LEFT', 'interact': 'SPACE'}
        action_str = action_map[action[0]]
        return action_str

    def action(self, state_dict):
        state = HacktrickState.from_dict(state_dict['state']['state'])

        if "collaborative" in self.mode:
            (action0, action1) = self.agent_pair.joint_action(state)
            action0 = self.map_action(action0)
            action1 = self.map_action(action1)
            action = [action0, action1]
        else:
            action0 = self.agent0.action(state)
            action0 = self.map_action(action0)
            action = action0

        return action