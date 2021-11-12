# 定义了单个Agent的DDPG结构，及一些函数

import tensorflow as tf
import tensorflow.contrib as tc


class MADDPG():
    def __init__(self, name, actor_lr, critic_lr, layer_norm=True, nb_actions=1, nb_other_aciton=3,
                 num_units=64, model="MADDPG"):
        nb_input = 4 * (nb_actions + nb_other_aciton)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.layer_norm = layer_norm
        self.nb_actions = nb_actions
        state_input = tf.placeholder(shape=[None, nb_input], dtype=tf.float32)
        action_input = tf.placeholder(shape=[None, nb_actions], dtype=tf.float32)
        other_action_input = tf.placeholder(shape=[None, nb_other_aciton], dtype=tf.float32)
        if model == "DDPG":
            other_action_input = tf.placeholder(shape=[None, 0], dtype=tf.float32)
        reward = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        # 输入是一个具体的状态state，经过两层的全连接网络输出选择的动作action
        def actor_network(name, state_input):
            with tf.variable_scope(name) as scope:
                x = state_input
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.layers.dense(x, num_units,
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

                x = tf.layers.dense(x, num_units,
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))  # 全连接层
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

                x = tf.layers.dense(x, 1,
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                # x = tf.nn.softmax(x)
                # x = tf.arg_max(x, 1)
                # x = tf.cast(tf.reshape(x, [-1, 1]), dtype=tf.float32)
                # bias = tf.constant(-30, dtype=tf.float32)
                w_ = tf.constant(3, dtype=tf.float32)
                # x = tf.multiply(tf.add(x, bias), w_)
                x = tf.multiply(tf.nn.tanh(x), w_)
            return x

        # 输入时 state，所有Agent当前的action信息
        def critic_network(name, state_input, action_input, reuse=False):
            with tf.variable_scope(name) as scope:
                if reuse:
                    scope.reuse_variables()
                x = state_input
                # x = tf.concat([x, action_input], axis=-1)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.layers.dense(x, num_units,
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

                x = tf.concat([x, action_input], axis=-1)
                x = tf.layers.dense(x, num_units,
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

                x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            return x

        self.state_input = state_input
        self.action_input = action_input
        self.other_action_input = other_action_input
        self.reward = reward
        self.action_output = actor_network(name + "actor", state_input=self.state_input)
        self.critic_output = critic_network(name + '_critic',
                                            action_input=tf.concat([self.action_input, self.other_action_input],
                                                                   axis=1), state_input=self.state_input)

        self.actor_optimizer = tf.train.AdamOptimizer(self.actor_lr)
        self.critic_optimizer = tf.train.AdamOptimizer(self.critic_lr)

        # 最大化Q值
        self.actor_loss = -tf.reduce_mean(
            critic_network(name + '_critic',
                           action_input=tf.concat([self.action_output, self.other_action_input], axis=1),
                           reuse=True, state_input=self.state_input))  # reduce_mean 为求均值，即为期望
        online_var = [i for i in tf.trainable_variables() if name + "actor" in i.name]
        self.actor_train = self.actor_optimizer.minimize(self.actor_loss, var_list=online_var)
        # self.actor_train = self.actor_optimizer.minimize(self.actor_loss)
        self.actor_loss_op = tf.summary.scalar("actor_loss", self.actor_loss)
        self.target_Q = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.critic_loss = tf.reduce_mean(tf.square(self.target_Q - self.critic_output))  # 目标Q 与 真实Q 之间差的平方的均值
        self.critic_loss_op = tf.summary.scalar("critic_loss", self.critic_loss)
        self.critic_train = self.critic_optimizer.minimize(self.critic_loss)
        self.count = 0

    def train_actor(self, state, other_action, sess, summary_writer, lr):
        self.count += 1
        self.actor_lr = lr
        summary_writer.add_summary(
            sess.run(self.actor_loss_op, {self.state_input: state, self.other_action_input: other_action}), self.count)
        sess.run(self.actor_train, {self.state_input: state, self.other_action_input: other_action})

    def train_critic(self, state, action, other_action, target, sess, summary_writer, lr):
        self.critic_lr = lr
        summary_writer.add_summary(
            sess.run(self.critic_loss_op, {self.state_input: state, self.action_input: action,
                                           self.other_action_input: other_action,
                                           self.target_Q: target}), self.count)
        sess.run(self.critic_train,
                 {self.state_input: state, self.action_input: action, self.other_action_input: other_action,
                  self.target_Q: target})

    def action(self, state, sess):
        return sess.run(self.action_output, {self.state_input: state})

    def Q(self, state, action, other_action, sess):
        return sess.run(self.critic_output,
                        {self.state_input: state, self.action_input: action, self.other_action_input: other_action})
