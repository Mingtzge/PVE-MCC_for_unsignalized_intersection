from collections import deque
import random
import rank_based


class ReplayBuffer(object):

    def __init__(self, buffer_size, batch_size=32, learn_start=2000, steps=100000, rand_s=False):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()
        self.rand_s = rand_s
        conf = {'size': self.buffer_size,
                'learn_start': learn_start,
                'partition_num': 32,
                'steps': steps,
                'batch_size': batch_size}
        self.replay_memory = rank_based.Experience(conf)

    def getBatch(self, batch_size):
        # random draw N
        if self.rand_s:
            return random.sample(self.buffer, batch_size), None, None
        batch, w, e_id = self.replay_memory.sample(self.num_experiences)
        self.e_id = e_id
        self.w_id = w
        '''#state t
        self.state_t_batch = [item[0] for item in batch]
        self.state_t_batch = np.array(self.state_t_batch)
        #state t+1        
        self.state_t_1_batch = [item[1] for item in batch]
        self.state_t_1_batch = np.array( self.state_t_1_batch)
        self.action_batch = [item[2] for item in batch]
        self.action_batch = np.array(self.action_batch)
        self.action_batch = np.reshape(self.action_batch,[len(self.action_batch),self.num_actions])
        self.reward_batch = [item[3] for item in batch]
        self.reward_batch = np.array(self.reward_batch)
        self.done_batch = [item[4] for item in batch]
        self.done_batch = np.array(self.done_batch)'''
        return batch, self.w_id, self.e_id

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, next_state, done):  # add(self, state, next_state, action, reward, done):
        new_experience = (state, action, reward, next_state, done)#(state, action, reward, next_state, done)
        self.num_experiences += 1
        if self.rand_s:
            if self.num_experiences < self.buffer_size:
                self.buffer.append(new_experience)
            else:
                self.buffer.popleft()
                self.buffer.append(new_experience)
        else:
            self.replay_memory.store(new_experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    # def erase(self):
    #  self.buffer = deque()
    #  self.num_experiences = 0
    def rebalance(self):
        self.replay_memory.rebalance()

    def update_priority(self, indices, delta):
        self.replay_memory.update_priority(indices, delta)
