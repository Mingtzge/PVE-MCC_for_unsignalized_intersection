# 模型训练的主代码
import numpy as np
import tensorflow as tf
import os
import scipy.io as scio
import argparse
import cv2
from shutil import copyfile
import matplotlib.pyplot as plt
from traffic_interaction_scene import TrafficInteraction
from traffic_interaction_scene import Visible
import time
from model_agent_maddpg import MADDPG
from replay_buffer import ReplayBuffer
import io
from PIL import Image


def create_init_update(oneline_name, target_name, tau=0.99):
    """
    :param oneline_name: the online model name
    :param target_name: the target model name
    :param tau: The proportion of each transfer from the online model to the target model
    :return:
    """
    online_var = [i for i in tf.trainable_variables() if oneline_name in i.name]
    target_var = [i for i in tf.trainable_variables() if target_name in i.name]

    target_init = [tf.assign(target, online) for online, target in zip(online_var, target_var)]
    target_update = [tf.assign(target, (1 - tau) * online + tau * target) for online, target in
                     zip(online_var, target_var)]  # 按照比例用online更新target

    return target_init, target_update


def get_agents_action(sta, sess, agent, noise_range=0.0):
    """
    :param sta: the state of the agent
    :param sess: the session of tf
    :param agent: the model of the agent
    :param noise_range: the noise range added to the agent model output
    :return: the action of the agent in its current state
    """
    agent1_action = agent.action(state=[sta], sess=sess) + np.random.randn(1) * noise_range
    return agent1_action


def train_agent_seq(agent_ddpg, agent_ddpg_target, agent_memory, agent_actor_target_update,
                    agent_critic_target_update, sess, summary_writer, args):
    batch, w_id, eid = agent_memory.getBatch(
        args.batch_size)
    if not batch:
        return
    agent_num = args.o_agent_num + 1
    total_obs_batch = np.zeros((args.batch_size, agent_num, agent_num * 4))
    rew_batch = np.zeros((args.batch_size,))
    total_act_batch = np.zeros((args.batch_size, agent_num))
    total_next_obs_batch = np.zeros((args.batch_size, agent_num, agent_num * 4))
    next_state_mask = np.zeros((args.batch_size,))
    for k, (s0, a, r, s1, done) in enumerate(batch):
        total_obs_batch[k] = s0
        rew_batch[k] = r
        total_act_batch[k] = a
        if not done:
            total_next_obs_batch[k] = s1
            next_state_mask[k] = 1
    other_act = []
    act_batch = np.array(total_act_batch[:, 0])  # 获取本agent动作集
    act_batch = act_batch.reshape(act_batch.shape[0], 1)
    for n in range(1, agent_num):
        other_act.append(total_act_batch[:, n])
    other_act_batch = np.vstack(other_act).transpose()
    e_id = eid
    obs_batch = total_obs_batch[:, 0, :]  # 获取本agent当前状态集
    target = rew_batch.reshape(-1, 1)
    td_error = abs(agent_ddpg_target.Q(
        state=obs_batch, action=act_batch, other_action=other_act_batch, sess=sess) - target)
    if e_id is not None:
        agent_memory.update_priority(e_id, td_error)
    agent_ddpg.train_critic(state=obs_batch, action=act_batch, other_action=other_act_batch, target=target, sess=sess,
                            summary_writer=summary_writer, lr=args.critic_lr)
    agent_ddpg.train_actor(state=obs_batch, other_action=other_act_batch, sess=sess, summary_writer=summary_writer,
                           lr=args.actor_lr)
    sess.run([agent_actor_target_update, agent_critic_target_update])  # 从online模型更新到target模型


def parse_args():
    parser = argparse.ArgumentParser("MADDPG experiments for multiagent traffic interaction environments")
    parser.add_argument("--num_episodes", type=int, default=1000, help="number of episodes")  # episode次数
    parser.add_argument("--o_agent_num", type=int, default=6, help="other agent numbers")
    parser.add_argument("--seq_max_step", type=int, default=12, help="the step of multi-step learning")

    parser.add_argument("--actor_lr", type=float, default=1e-4, help="learning rate for Adam optimizer")  # 学习率
    parser.add_argument("--critic_lr", type=float, default=1e-3, help="learning rate for Adam optimizer")  # 学习率
    parser.add_argument("--gamma", type=float, default=0.80, help="discount factor")  # 折扣率
    parser.add_argument("--trans_r", type=float, default=0.998, help="transfer rate for online model to target model")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="number of episodes to optimize at the same time")  # 经验采样数目
    parser.add_argument("--learn_start", type=int, default=20000,
                        help="learn start step")  # 经验采样数目
    parser.add_argument("--lane_num", type=int, default=12,
                        help="the num of lane of intersection")  # 车道总数，12表示双向六车道交叉口
    parser.add_argument("--num_units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--collision_thr", type=float, default=2, help="the threshold for collision")
    parser.add_argument("--actual_lane", action="store_true", default=False, help="")
    parser.add_argument("--c_mode", type=str, default="closer",
                        help="the way of choosing closer cars, front ,front-end or closer")

    parser.add_argument("--model", type=str, default="MADDPG",
                        help="the model for training, MADDPG or DDPG")

    parser.add_argument("--exp_name", type=str, default="test ", help="name of the experiment")  # 实验名
    parser.add_argument("--type", type=str, default="test", help="type of experiment train or test")
    parser.add_argument("--mat_path", type=str, default="./data/train/arvTimeNewVeh_for_train.mat",
                        help="the path of mat file")
    parser.add_argument("--save_dir", type=str, default="model_data",
                        help="directory in which training state and model should be saved")  # 模型存储
    parser.add_argument("--save_rate", type=int, default=1,
                        help="save model once every time this many episodes are completed")  # 存储模型的回合间隔
    parser.add_argument("--load_dir", type=str, default="",
                        help="directory in which training state and model are loaded")  # 模型加载目录
    parser.add_argument("--video_name", type=str, default="",
                        help="if it not empty, program will generate a result video (.mp4 format defaultly)with the result imgs")
    parser.add_argument("--visible", action="store_true", default=False, help="visible or not")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)  # 恢复之前的模型，在 load-dir 或 save-dir
    parser.add_argument("--benchmark", action="store_true", default=False)  # 用保存的模型跑测试
    parser.add_argument("--batch_test", action="store_true", default=False)  # 是否批量测试
    parser.add_argument("--benchmark_iters", type=int, default=6000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/",
                        help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/",
                        help="directory where plot data is saved")  # 训练曲线的目录
    return parser.parse_args()


def benchmark(model, arrive_time, sess):
    total_c = 0
    collisions_count = 0
    for mat_file in ["arvTimeNewVeh_300.mat", "arvTimeNewVeh_600.mat", "arvTimeNewVeh_900.mat"]:
        data = scio.loadmat(mat_file)  # 加载.mat数据
        arrive_time = data["arvTimeNewVeh"]
        env = TrafficInteraction(arrive_time, 150, args, vm=6, virtual_l=not args.actual_lane)
        # env = TrafficInteraction(arrive_time, 150, args, vm=6, vM=20, v0=12)
        for i in range(args.benchmark_iters):
            for lane in range(4):
                for ind, veh in enumerate(env.veh_info[lane]):
                    o_n = veh["state"]
                    agent1_action = [[0]]
                    if veh["control"]:
                        agent1_action = get_agents_action(o_n[0], sess, model, noise_range=0)  # 模型根据当前状态进行预测
                    env.step(lane, ind, agent1_action[0][0])  # 环境根据输入的动作返回下一时刻的状态和奖励
                    # env.step(lane, ind, 0)  # 环境根据输入的动作返回下一时刻的状态和奖励
            state_next, reward, actions, collisions, estm_collisions, collisions_per_veh = env.scene_update()
            for k in range(len(actions)):
                if collisions_per_veh[k][0] > 0:
                    collisions_count += 1
            if i % 1000 == 0:
                print("i: %s collisions_rate: %s" % (i, float(collisions_count) / (env.id_seq + total_c)))
            env.delete_vehicle()
        total_c += env.id_seq
        print("vehicle number: %s; collisions occurred number: %s; collisions rate: %s" % (
            total_c, collisions_count, float(collisions_count) / total_c))
    return float(collisions_count) / total_c


def train():
    # 建立Agent，Agent对应两个DDPG结构，一个是eval-net，一个是target-net
    agent1_ddpg = MADDPG('agent1', actor_lr=args.actor_lr, critic_lr=args.critic_lr, nb_other_aciton=args.o_agent_num,
                         num_units=args.num_units, model=args.model)
    agent1_ddpg_target = MADDPG('agent1_target', actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                                nb_other_aciton=args.o_agent_num, num_units=args.num_units, model=args.model)
    saver = tf.train.Saver()  # 为存储模型预备
    agent1_actor_target_init, agent1_actor_target_update = create_init_update('agent1actor', 'agent1_targetactor',
                                                                              tau=args.trans_r)
    agent1_critic_target_init, agent1_critic_target_update = create_init_update('agent1_critic', 'agent1_target_critic',
                                                                                tau=args.trans_r)
    count_n = 0
    col = tf.Variable(0, dtype=tf.int8)
    collisions_op = tf.summary.scalar('collisions', col)
    etsm_col = tf.Variable(0, dtype=tf.int8)
    etsm_collisions_op = tf.summary.scalar('estimate_collisions', etsm_col)
    v_mean = tf.Variable(0, dtype=tf.float32)
    v_mean_op = tf.summary.scalar('v_mean', v_mean)
    collision_rate = tf.Variable(0, dtype=tf.float32)
    collision_rate_op = tf.summary.scalar('collision_rate', collision_rate)
    acc_mean = tf.Variable(0, dtype=tf.float32)
    acc_mean_op = tf.summary.scalar('acc_mean', acc_mean)
    reward_mean = tf.Variable(0, dtype=tf.float32)
    reward_mean_op = tf.summary.scalar('reward_mean', reward_mean)
    collisions_mean = tf.Variable(0, dtype=tf.float32)
    collisions_mean_op = tf.summary.scalar('collisions_mean', collisions_mean)
    estm_collisions_mean = tf.Variable(0, dtype=tf.float32)
    estm_collisions_mean_op = tf.summary.scalar('estm_collisions_mean', estm_collisions_mean)
    collisions_veh_numbers = tf.Variable(0, dtype=tf.int32)
    collisions_veh_numbers_op = tf.summary.scalar('collision_veh_numbers', collisions_veh_numbers)
    vehs_jerk = tf.Variable(0, dtype=tf.int32)
    vehs_jerk_op = tf.summary.scalar('jerk', vehs_jerk)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    config.gpu_options.per_process_gpu_memory_fraction = 0.050
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.run([agent1_actor_target_init, agent1_critic_target_init])
    if args.restore:
        saver.restore(sess, tf.train.latest_checkpoint(os.path.join(args.save_dir, args.exp_name)))
        print("load cptk file from " + tf.train.latest_checkpoint(os.path.join(args.save_dir, args.exp_name)))

    summary_writer = tf.summary.FileWriter(os.path.join(args.save_dir, args.exp_name), graph=tf.get_default_graph())

    # 设置经验池最大空间
    agent1_memory_seq = ReplayBuffer(500000, args.batch_size, args.learn_start, 50000, rand_s=True)
    reward_list = []
    jerk_list = []
    collisions_list = []
    estm_collisions_list = []
    statistic_count = 0
    mean_window_length = 50
    state_now = []
    collisions_count = 0
    rate_latest = 1.0
    test_rate_latest = 1.0
    time_total = []
    seq_max_step = args.seq_max_step
    for epoch in range(args.num_episodes):
        collisions_count_last = collisions_count
        args.gamma = np.tanh(float(epoch + 6) / 12.0) * 0.90
        data = scio.loadmat("./data/train/arvTimeNewVeh_for_train.mat")  # 加载训练.mat数据
        arrive_time = data["arvTimeNewVeh"]
        env = TrafficInteraction(arrive_time, 150, args, vm=6, virtual_l=not args.actual_lane, lane_num=args.lane_num)
        for i in range(6000):
            state_now.clear()
            for lane in range(args.lane_num):
                for ind, veh in enumerate(env.veh_info[lane]):
                    o_n = veh["state"]
                    agent1_action = [[0]]
                    if veh["control"]:
                        count_n += 1
                        agent1_action = get_agents_action(o_n[0], sess, agent1_ddpg, noise_range=0.2)  # 模型根据当前状态进行预测
                        state_now.append(o_n)
                    env.step(lane, ind, agent1_action[0][0])
            ids, state_next, reward, actions, collisions, estm_collisions, collisions_per_veh, jerks, lock = env.scene_update()
            for seq, car_index in enumerate(ids):
                env.veh_info[car_index[0]][car_index[1]]["buffer"].append(
                    [state_now[seq], actions[seq], reward[seq], state_next[seq],
                     env.veh_info[car_index[0]][car_index[1]]["Done"]])
                if env.veh_info[car_index[0]][car_index[1]]["Done"] or env.veh_info[car_index[0]][car_index[1]][
                    "count"] > seq_max_step:
                    seq_data = env.veh_info[car_index[0]][car_index[1]]["buffer"]
                    if env.veh_info[car_index[0]][car_index[1]]["Done"]:
                        r_target = seq_data[-1][2]
                    else:
                        other_act_next = []
                        for n in range(1, args.o_agent_num + 1):
                            other_act_next.append(agent1_ddpg_target.action([seq_data[-1][3][n]], sess)[0][0])
                        r_target = seq_data[-1][2] + args.gamma * agent1_ddpg_target.Q(state=[seq_data[-1][3][0]],
                                                                                       action=agent1_ddpg_target.action(
                                                                                           [seq_data[-1][3][0]],
                                                                                           sess), other_action=[
                                other_act_next], sess=sess)[0][0]
                    for cur_data in reversed(seq_data[:-1]):
                        r_target = cur_data[2] + args.gamma * r_target
                    agent1_memory_seq.add(np.array(seq_data[0][0]), np.array(seq_data[0][1]), r_target,
                                          np.array(seq_data[0][3]), False)
                    env.veh_info[car_index[0]][car_index[1]]["buffer"].pop(0)
                    env.veh_info[car_index[0]][car_index[1]]["count"] -= 1
            reward_list += reward
            jerk_list += jerks
            if len(collisions_per_veh) > 0:
                collisions_list += list(np.array(collisions_per_veh)[:, 0])
                estm_collisions_list += list(np.array(collisions_per_veh)[:, 1])
            reward_list = reward_list[-mean_window_length:]
            jerk_list = jerk_list[-mean_window_length:]
            collisions_list = collisions_list[-mean_window_length:]
            estm_collisions_list = estm_collisions_list[-mean_window_length:]
            for k in range(len(actions)):
                if collisions_per_veh[k][0] > 0:
                    collisions_count += 1
            if count_n > 10000:
                statistic_count += 1
                time_t = time.time()
                train_agent_seq(agent1_ddpg, agent1_ddpg_target, agent1_memory_seq,
                                agent1_actor_target_update, agent1_critic_target_update, sess, summary_writer, args)
                time_total.append(time.time() - time_t)
                a = tf.trainable_variables
                if len(actions) > 0:
                    summary_writer.add_summary(sess.run(collisions_op, {col: collisions}), statistic_count)
                    summary_writer.add_summary(sess.run(etsm_collisions_op, {etsm_col: estm_collisions}),
                                               statistic_count)
                    summary_writer.add_summary(sess.run(v_mean_op, {v_mean: np.mean(np.array(state_next)[:, 0, 1])}),
                                               statistic_count)
                    summary_writer.add_summary(sess.run(vehs_jerk_op, {vehs_jerk: np.mean(jerk_list)}), statistic_count)
                    summary_writer.add_summary(
                        sess.run(acc_mean_op, {acc_mean: np.mean(np.array(state_next)[:, 0, 2])}),
                        statistic_count)
                summary_writer.add_summary(sess.run(reward_mean_op, {reward_mean: np.mean(reward_list)}),
                                           statistic_count)
                summary_writer.add_summary(sess.run(collisions_mean_op, {collisions_mean: np.mean(collisions_list)}),
                                           statistic_count)
                summary_writer.add_summary(
                    sess.run(estm_collisions_mean_op, {estm_collisions_mean: np.mean(estm_collisions_list)}),
                    statistic_count)
                summary_writer.add_summary(
                    sess.run(collisions_veh_numbers_op, {collisions_veh_numbers: collisions_count}), statistic_count)
                if i % 100 == 0:
                    print(
                        "reward mean: %s;epoch: %s;i: %s;count: %s;collisions_count: %s latest_c_rate: %s;"
                        "test best c_rate: %s;a-lr: %0.6f; c-lr: %0.6f; time_mean: %s" % (
                            np.mean(reward_list), epoch, i, count_n, collisions_count, rate_latest, test_rate_latest,
                            args.actor_lr, args.critic_lr, np.mean(time_total)))
            env.delete_vehicle()
        if epoch % args.save_rate == 0:
            print('update model to ' + os.path.join(args.save_dir, args.exp_name, str(epoch) + '.cptk'))
            saver.save(sess, os.path.join(args.save_dir, args.exp_name, str(epoch) + '.cptk'))
            if rate_latest > (collisions_count - collisions_count_last) / float(env.id_seq):
                rate_latest = (collisions_count - collisions_count_last) / float(env.id_seq)
                copyfile(
                    os.path.join(args.save_dir, args.exp_name, str(epoch) + '.cptk.data-00000-of-00001'),
                    os.path.join(args.save_dir, args.exp_name, 'best.cptk.data-00000-of-00001'))
                copyfile(
                    os.path.join(args.save_dir, args.exp_name, str(epoch) + '.cptk.index'),
                    os.path.join(args.save_dir, args.exp_name, 'best.cptk.index'))
                copyfile(
                    os.path.join(args.save_dir, args.exp_name, str(epoch) + '.cptk.meta'),
                    os.path.join(args.save_dir, args.exp_name, 'best.cptk.meta'))
            summary_writer.add_summary(sess.run(collision_rate_op, {
                collision_rate: (collisions_count - collisions_count_last) / float(env.id_seq)}),
                                       epoch)
        if epoch % 2 == 0 and args.benchmark:
            c_rate = benchmark(agent1_ddpg, arrive_time, sess)
            if c_rate < test_rate_latest:
                test_rate_latest = c_rate
                copyfile(
                    os.path.join(args.save_dir, args.exp_name, str(epoch) + '.cptk.data-00000-of-00001'),
                    os.path.join(args.save_dir, args.exp_name, 'test_best.cptk.data-00000-of-00001'))
                copyfile(
                    os.path.join(args.save_dir, args.exp_name, str(epoch) + '.cptk.index'),
                    os.path.join(args.save_dir, args.exp_name, 'test_best.cptk.index'))
                copyfile(
                    os.path.join(args.save_dir, args.exp_name, str(epoch) + '.cptk.meta'),
                    os.path.join(args.save_dir, args.exp_name, 'test_best.cptk.meta'))
        if epoch % 5 == 4:
            args.actor_lr = args.actor_lr * 0.9
            args.critic_lr = args.critic_lr * 0.9
    sess.close()


# 特征重要性分析工具
def actor_feature_importance_analyze(state, model, sess, idx=0):
    plt.figure(0)
    imps = np.zeros(state.shape[0])
    base = get_agents_action(state, sess, model)[0]
    for j in range(imps.shape[0]):
        fes = []
        for i in range(100):
            tmp = state.copy()
            tmp[j] += np.random.rand(1) * 10
            fes.append(tmp)
        imps[j] = np.mean(abs((model.action(state=fes, sess=sess).reshape(100) - base[0])))
    if sum(imps) > 1:
        print(state, imps)
    plt.bar([i for i in range(len(imps))], imps)
    plt.savefig("result_img/feature_importance_curve_%s.png" % idx)
    plt.close()


def test():
    agent1_ddpg_test = MADDPG('agent1', actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                              nb_other_aciton=args.o_agent_num, num_units=args.num_units)
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    model_path = os.path.join(args.save_dir, args.exp_name, "test_best.cptk")
    if not os.path.exists(model_path + ".meta"):
        model_path = tf.train.latest_checkpoint(os.path.join(args.save_dir, args.exp_name))
    saver.restore(sess, model_path)
    print("load cptk file from " + model_path)
    visible = Visible(lane_w=2.5, control_dis=150, l_mode="actual", c_mode=args.c_mode, lane_num=args.lane_num)
    size = (960, 960)
    fps = 20
    video_writer = cv2.VideoWriter()
    if args.video_name != "":
        video_writer = cv2.VideoWriter(os.path.join("result_imgs", args.video_name + ".avi"),
                                       cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)
    mat_path = os.path.join("./data/test", args.mat_path)
    data = scio.loadmat(mat_path)  # 加载.mat数据
    arrive_time = data["arvTimeNewVeh"]
    print("mat_path: ", mat_path)
    lock_total = 0
    collisions_count = 0
    time_total = []
    env = TrafficInteraction(arrive_time, 150, args, show_col=False, virtual_l=not args.actual_lane,
                             lane_num=args.lane_num)
    jerk_total = 0
    for i in range(1000):
        for lane in range(args.lane_num):
            for ind, veh in enumerate(env.veh_info[lane]):
                o_n = veh["state"]
                agent1_action = [[0]]
                if veh["control"]:
                    temp_t = time.time()
                    agent1_action = get_agents_action(o_n[0], sess, agent1_ddpg_test, noise_range=0)  # 模型根据当前状态进行预测
                    time_total.append(time.time() - temp_t)
                env.step(lane, ind, agent1_action[0][0])  # 环境根据输入的动作返回下一时刻的状态和奖励
        ids, state_next, reward, actions, collisions, estm_collisions, collisions_per_veh, jerks, lock = env.scene_update()
        jerk_total += sum(jerks)
        lock_total += lock
        for k in range(len(actions)):
            if collisions_per_veh[k][0] > 0:
                collisions_count += 1
        if i % 50 == 0:
            print("i: %s collisions_rate: %s reward std: %s reward mean: %s lock_num: %s" % (
                i, float(collisions_count) / env.id_seq, np.std(reward), np.mean(reward), lock_total))
        if (args.visible or args.video_name != ""):
            visible.show(env, i)
            img = cv2.imread("result_imgs/%s.png" % i)
            # cv2.putText(img, "density: " + str(args.mat_pa), (200, 160), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(img, "frame: " + str(i), (200, 200), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(img, "veh: " + str(env.id_seq), (200, 240), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(img, "c-veh: %s" % collisions_count, (200, 280), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (0, 0, 255),
                        1)
            cv2.putText(img, "c-r: %0.4f" % (float(collisions_count) / env.id_seq), (200, 320),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.5, (0, 0, 255), 1)
            cv2.putText(img, "p_veh: " + str(env.passed_veh), (200, 360), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (0, 0, 0),
                        1)
            cv2.putText(img,
                        "pT-m: %0.4f s" % (
                                float(env.passed_veh_step_total) / (env.passed_veh + 0.0001) * env.deltaT),
                        (200, 400), cv2.FONT_HERSHEY_COMPLEX,
                        0.5, (0, 0, 0), 1)
            if args.visible:
                cv2.imshow("unsignalized intersection", img)
                cv2.waitKey(1)
            if args.video_name != "":
                video_writer.write(img)
        env.delete_vehicle()
        # if i < 2000:
        #     scio.savemat("test_mat.mat", {"veh_info": env.veh_info_record})
    video_writer.release()
    cv2.destroyAllWindows()
    choose_veh_visible = False
    veh_route = False
    if veh_route:
        n = 0
        color = {"0": 'darksalmon', "3": 'orchid', "7": 'b', "10": 'mediumslateblue', "9": "mediumseagreen"}
        plt.figure(0, figsize=(6.4, 3.2))
        plt.rcParams['font.family'] = ['SimHei']
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        # 绘制轨迹
        t_l = 85
        leg = {"0": '目标车道-车辆', "3": '冲突车道1-车辆', "7": '冲突车道2-车辆', "10": '冲突车道3-车辆', "9": "冲突车道4-车辆"}
        idx = ["0", "3", "7", "10", "9"]
        for veh in env.virtual_data:
            n += 1
            x = [t[0] for t in env.virtual_data[veh] if t_l - 30 < t[0] < t_l]
            y = [t[1] for t in env.virtual_data[veh] if t_l - 30 < t[0] < t_l]
            if len(idx) > 0 and veh.split("_")[0] == idx[0]:
                plt.plot(x, y, color[veh.split("_")[0]], label=leg[veh.split("_")[0]])
                plt.legend()
                leg.pop(idx[0])
                idx.pop(0)
            else:
                plt.plot(x, y, color[veh.split("_")[0]])
        # plt.legend()
        plt.xlabel("时间/s")
        plt.ylabel("车辆与冲突点的距离/m")
        # plt.savefig("exp_result_imgs/route.png")
        png1 = io.BytesIO()
        plt.savefig(png1, format="png", dpi=500, pad_inches=.1, bbox_inches='tight')
        png2 = Image.open(png1)
        png2.save("exp_result_imgs/route.tiff")
        png1.close()
        # plt.savefig("result_imgs/efficiency.png")
        plt.close()
        plt.close()
        plt.figure(1, figsize=(6.4, 3.2))
        plt.rcParams['font.family'] = ['SimHei']
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        # 绘制速度
        t_l = 85
        leg = {"0": '目标车道-车辆', "3": '冲突车道1-车辆', "7": '冲突车道2-车辆', "10": '冲突车道3-车辆', "9": "冲突车道4-车辆"}
        idx = ["0", "3", "7", "10", "9"]
        for veh in env.virtual_data:
            n += 1
            x = [t[0] for t in env.virtual_data[veh] if t_l - 30 < t[0] < t_l]
            y = [t[2] for t in env.virtual_data[veh] if t_l - 30 < t[0] < t_l]
            if len(idx) > 0 and veh.split("_")[0] == idx[0]:
                plt.plot(x, y, color[veh.split("_")[0]], lw=2, label=leg[veh.split("_")[0]])
                plt.legend()
                leg.pop(idx[0])
                idx.pop(0)
            else:
                plt.plot(x, y, color[veh.split("_")[0]], lw=2)
        # plt.legend()
        plt.xlabel("时间 [s]")
        plt.ylabel("距离冲突点距离 [m]")
        plt.savefig("exp_result_imgs/velocity.png")
        plt.close()
    if choose_veh_visible:
        choose_veh_info = [np.array(item) for item in env.choose_veh_info]
        plt.figure(0)
        color = ['r', 'g', 'b', 'y']
        y_units = ['distance [m]', 'velocity [m/s]', 'accelerate speed [m/s^2]']
        titles = ["The distance of the vehicle varies with the time",
                  "The velocity of the vehicle varies with the time",
                  "The accelerate spped of the vehicle varies with the time"]
        for m in range(len(y_units)):
            for n in range(4):
                plt.plot(choose_veh_info[n][:, 0], choose_veh_info[n][:, m + 1], color[n])
            plt.legend(["lane-0", "lane-1", "lane-2", "lane-3"])
            plt.xlabel("time [s]")
            plt.ylabel(y_units[m])
            plt.title(titles[m], fontsize='small')
            plt.savefig("exp_result_imgs/%s.png" % (y_units[m].split(" ")[0]), dpi=600)
            plt.close()
    print(
        "vehicle number: %s; collisions occurred number: %s; collisions rate: %s, time_mean: %s, pT-m: %0.4f s jerks: %s" % (
            env.id_seq, collisions_count, float(collisions_count) / env.id_seq, np.mean(time_total),
            float(env.passed_veh_step_total) / (env.passed_veh + 0.0001) * env.deltaT, jerk_total / env.passed_veh))
    sess.close()


def batch_test():
    agent1_ddpg_test = MADDPG('agent1', actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                              nb_other_aciton=args.o_agent_num, num_units=args.num_units)
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    model_path = os.path.join(args.save_dir, args.exp_name, "test_best.cptk")
    if not os.path.exists(model_path + ".meta"):
        model_path = tf.train.latest_checkpoint(os.path.join(args.save_dir, args.exp_name))
    saver.restore(sess, model_path)
    print("load cptk file from " + model_path)
    dens = [1200, 1000, 900, 800, 600, 400, 200]
    tw = open(args.exp_name + "_batch_test_result_12_v1.txt", "w")
    for d in dens:
        dens_f = "arvTimeNewVeh_new_%s_%s.mat" % (d, args.lane_num)
        mat_path = os.path.join("./data/test", dens_f)
        print(mat_path)
        tw.write(mat_path + "\n")
        data = scio.loadmat(mat_path)  # 加载.mat数据
        arrive_time = data["arvTimeNewVeh"]
        env = TrafficInteraction(arrive_time, 150, args, show_col=False, virtual_l=not args.actual_lane,
                                 lane_num=args.lane_num)
        jerk_total = 0
        collisions_count = 0
        lock_total = 0
        for i in range(36000):
            for lane in range(args.lane_num):
                for ind, veh in enumerate(env.veh_info[lane]):
                    o_n = veh["state"]
                    agent1_action = [[0]]
                    if veh["control"]:
                        agent1_action = get_agents_action(o_n[0], sess, agent1_ddpg_test,
                                                          noise_range=0)  # 模型根据当前状态进行预测
                    env.step(lane, ind, agent1_action[0][0])  # 环境根据输入的动作返回下一时刻的状态和奖励
            ids, state_next, reward, actions, collisions, estm_collisions, collisions_per_veh, jerks, lock = env.scene_update()
            jerk_total += sum(jerks)
            lock_total += lock
            for k in range(len(actions)):
                if collisions_per_veh[k][0] > 0:
                    collisions_count += 1
            if i % 1000 == 0:
                print("i: %s collisions_rate: %s reward std: %s reward mean: %s lock_num: %s" % (
                    i, float(collisions_count) / env.id_seq, np.std(reward), np.mean(reward), lock_total))
            env.delete_vehicle()
        result_txt = "vehicle number %s  collisions occurred number %s collisions rate %s pT-m %0.4f s jerks %s " \
                     "lock_num %s" % (
                         env.id_seq, collisions_count, float(collisions_count) / env.id_seq,
                         float(env.passed_veh_step_total) / (env.passed_veh + 0.0001) * env.deltaT,
                         jerk_total / env.passed_veh,
                         lock_total)
        print(result_txt)
        tw.write(result_txt + "\n")
    tw.close()
    sess.close()


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists("result_imgs"):
        os.makedirs("result_imgs")
    if not os.path.exists("exp_result_imgs"):
        os.makedirs("exp_result_imgs")
    if not os.path.exists(os.path.join(args.save_dir, args.exp_name)):
        os.makedirs(os.path.join(args.save_dir, args.exp_name))
    if args.type == "train":
        with open(os.path.join(args.save_dir, args.exp_name, "args.txt"), "w") as fw:
            fw.write(str(args))
        train()
    else:
        if args.batch_test:
            batch_test()
        else:
            test()
