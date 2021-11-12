import numpy as np
import matplotlib.pyplot as plt
import os
import copy as cp
import random

random.seed(0)
import math


class TrafficInteraction:
    # vm = 0; % minimum
    # velocity
    # v0 = 10; % initial
    # velocity
    # vM = 13; % maximum
    # velocity
    # am = -3; % minimum
    # acceleration
    # aM = 3; % maximum
    def __init__(self, arrive_time, dis_ctl, args, deltaT=0.1, vm=5, vM=13, am=-3, aM=3, v0=10, diff_max=220,
                 lane_cw=2.5,
                 loc_con=True, show_col=False, virtual_l=True, lane_num=12):
        # 坐标轴,车道0 从左到右, 车道1,从右到左 车道2,从下到上 车道3 从上到下
        #           dis_ctl
        #  -dis_ctl    0     dis_ctl
        #           -dis_ctl
        self.virtual_l = virtual_l
        self.virtual_data = {}
        self.show_col = show_col
        self.loc_con = loc_con
        self.collision_thr = args.collision_thr
        self.choose_veh = 15
        self.safe_distance = 20
        self.vm = vm
        self.vM = vM
        self.am = am
        self.aM = aM
        self.v0 = v0
        self.lane_cw = lane_cw
        self.lane_num = lane_num
        self.intention_re = 0
        self.thr = pow(self.vM - self.vm, 2) / 4 / self.aM + 2.2
        self.choose_veh_info = [[] for i in range(self.lane_num)]
        self.veh_info_record = [[] for i in range(self.lane_num)]
        if self.lane_num == 3:
            # T字形
            self.lane_info = [
                [dis_ctl - 2 * lane_cw, 3.1415 / 2 * 3 * lane_cw, -(dis_ctl - 2 * lane_cw)],  # 左转
                [dis_ctl - 2 * lane_cw, 4 * lane_cw, -(dis_ctl - 2 * lane_cw)],  # 直行
                [dis_ctl - 2 * lane_cw, 3.1415 / 2 * lane_cw, -(dis_ctl - 2 * lane_cw)]  # 右转
            ]
            self.lane2lane = [
                [2, 4, 5],
                [2],
                [4, 0, 1],
                [4],
                [0, 2, 3],
                [0]
            ]
            self.intention = [
                [1, 2],
                [0, 1],
                [0, 2]
            ]
        elif self.lane_num == 4:
            # 单车道
            self.lane_info = [
                [dis_ctl - 2 * lane_cw, 3.1415 / 2 * 3 * lane_cw, -(dis_ctl - 2 * lane_cw)],  # 左转
                [dis_ctl - 2 * lane_cw, 4 * lane_cw, -(dis_ctl - 2 * lane_cw)],  # 直行
                [dis_ctl - 2 * lane_cw, 3.1415 / 2 * lane_cw, -(dis_ctl - 2 * lane_cw)]  # 右转
            ]
            # 顺序：如果左转直行和右转同时交汇，按照左转，直行，右转的顺序排列
            self.lane2lane = [
                [10, 6, 9, 3, 7, 4, 8],  # 0
                [10, 6, 3, 4, 9, 5],  # 1
                [6, 10],  # 2
                [1, 9, 0, 6, 10, 7, 11],  # 3
                [1, 9, 6, 7, 0, 8],  # 4
                [9, 1],  # 5
                [4, 0, 3, 9, 1, 10, 2],  # 6
                [4, 0, 9, 10, 3, 11],  # 7
                [0, 4],  # 8
                [7, 3, 6, 0, 4, 1, 5],  # 9
                [7, 3, 0, 1, 6, 2],  # 10
                [3, 7]  # 11
            ]
            self.direction_num = 12
            self.direction = [
                [6, 7, 8],
                [0, 1, 2],
                [9, 10, 11],
                [3, 4, 5]
            ]
            self.alpha = math.atan((4 - math.sqrt(2)) / (4 + math.sqrt(2)))  # 图中alpha的值
            self._alpha = math.atan((4 + math.sqrt(2)) / (4 - math.sqrt(2)))
            self.beta = math.atan(2 / math.sqrt(5))  # 图中beta的值
            self._beta = math.atan(math.sqrt(5) / 2)
            self.gama = math.atan(1 / 2 * math.sqrt(2))  # 图中gamma的值
        elif self.lane_num == 8:
            # 两车道
            self.lane_info = [
                [dis_ctl - 4 * lane_cw, 3.1415 / 2 * 5 * lane_cw, -(dis_ctl - 4 * lane_cw)],
                [dis_ctl - 4 * lane_cw, 8 * lane_cw, -(dis_ctl - 4 * lane_cw)],
                [dis_ctl - 4 * lane_cw, 3.1415 / 2 * lane_cw, -(dis_ctl - 4 * lane_cw)]
            ]
            self.lane2lane = [
                [14, 4, 13, 12, 9, 10, 5],  # 0
                [14, 13, 8, 4, 5, 6, 12],  # 1
                [14, 13, 8, 4, 5, 6, 7],  # 2
                [14],  # 3
                [2, 8, 1, 0, 13, 14, 9],  # 4
                [2, 1, 12, 8, 9, 10, 0],  # 5
                [2, 1, 12, 8, 9, 10, 11],  # 6
                [2],  # 7
                [6, 12, 5, 4, 1, 2, 13],  # 8
                [6, 5, 0, 12, 13, 14, 4],  # 9
                [6, 5, 0, 12, 13, 14, 15],  # 10
                [6],  # 11
                [10, 0, 9, 8, 5, 6, 1],  # 12
                [10, 9, 4, 0, 1, 2, 8],  # 13
                [10, 9, 4, 0, 1, 2, 3],  # 14
                [10]  # 15
            ]
            self.intention = [
                [0, 1],
                [1, 2],
                [0, 1],
                [1, 2],
                [0, 1],
                [1, 2],
                [0, 1],
                [1, 2]
            ]
            self.direction_num = 16
            self.direction = [
                [0, 1, -1],
                [-1, 2, 3],
                [4, 5, -1],
                [-1, 6, 7],
                [8, 9, -1],
                [-1, 10, 11],
                [12, 13, -1],
                [-1, 14, 15]
            ]
        elif self.lane_num == 12:
            # 三车道
            self.lane_info = [
                [dis_ctl - 6 * lane_cw, 3.1415 / 2 * 7 * lane_cw, -(dis_ctl - 6 * lane_cw)],
                [dis_ctl - 6 * lane_cw, 12 * lane_cw, -(dis_ctl - 6 * lane_cw)],
                [dis_ctl - 6 * lane_cw, 3.1415 / 2 * lane_cw, -(dis_ctl - 6 * lane_cw)]
            ]
            self.lane2lane = [
                [10, 3, 9, 7],
                [10, 6, 3, 4],
                [],
                [1, 6, 0, 10],
                [1, 9, 6, 7],
                [],
                [4, 9, 3, 1],
                [4, 0, 9, 10],
                [],
                [7, 0, 6, 4],
                [7, 3, 0, 1],
                []
            ]
            self.direction_num = 12
            self.direction = [
                [0, -1, -1],
                [-1, 1, -1],
                [-1, -1, 2],
                [3, -1, -1],
                [-1, 4, -1],
                [-1, -1, 5],
                [6, -1, -1],
                [-1, 7, -1],
                [-1, -1, 8],
                [9, -1, -1],
                [-1, 10, -1],
                [-1, -1, 11]
            ]
            self.cita = (2 * math.sqrt(10) - 6) * self.lane_cw  # 曲线交点据x轴或y轴的距离
            self.alpha = math.atan((6 * self.lane_cw + self.cita) / (3 * self.lane_cw))  # 交点与在半圆中中的角度（大的一个）
            self.beta = math.pi / 2 - self.alpha  # 交点与在半圆中中的角度（小的一个）
            self.gama = math.atan((math.sqrt(13) * self.lane_cw) / (6 * self.lane_cw))  # 两圆交点在半角中的角度（小的）
            self._gama = math.pi / 2 - self.gama
        self.closer_veh_num = args.o_agent_num
        self.c_mode = args.c_mode
        self.merge_p = [
            [0, 0, self.lane_cw, -self.lane_cw],
            [0, 0, -self.lane_cw, self.lane_cw],
            [-self.lane_cw, self.lane_cw, 0, 0],
            [self.lane_cw, -self.lane_cw, 0, 0]
        ]
        self.arrive_time = arrive_time
        self.current_time = 0
        self.passed_veh = 0
        self.passed_veh_step_total = 0
        self.virtual_lane = []
        self.virtual_lane_4 = [[] for i in range(self.direction_num)]
        self.virtual_lane_real_p = [[] for i in range(self.direction_num)]
        self.closer_cars = []
        self.closer_same_l_car = [-1, -1]
        self.deltaT = deltaT
        self.dis_control = dis_ctl
        self.veh_num = [0 for i in range(self.lane_num)]  # 每个车道车的数量
        self.veh_rec = [0 for i in range(self.lane_num)]  # 每个车道车的总数量
        self.input = [0 for i in range(4)]  # 每个入口车的总数量
        self.veh_info = [[] for i in range(self.lane_num)]
        self.diff_max = diff_max
        self.collision = False
        self.id_seq = 0
        self.delete_veh = []
        init = True
        while init:
            for i in range(self.lane_num):
                if self.veh_num[i] > 0:
                    init = False
            if init:
                self.scene_update()

    def scene_update(self):
        self.current_time += self.deltaT
        collisions = 0
        estm_collisions = 0
        re_state = []
        reward = []
        collisions_per_veh = []
        actions = []
        ids = []
        jerks = []
        self.delete_veh.clear()
        for i in range(self.lane_num):
            if len(self.veh_info[i]) > 0:
                for index, direction in enumerate(self.direction[i]):
                    if direction == -1:
                        continue
                    self.virtual_lane_4[direction].clear()
                    self.virtual_lane_real_p[direction].clear()
                    for _itr in self.virtual_lane:
                        # 目标车道
                        if _itr[1] == i:
                            self.virtual_lane_real_p[direction].append([_itr[0], _itr[1], _itr[2],
                                                                        self.veh_info[_itr[1]][_itr[2]]["v"],
                                                                        direction])
                            if self.direction[_itr[1]][_itr[3]] == direction:
                                # 同一车道直接添加  p, i, j, v
                                self.virtual_lane_4[direction].append([_itr[0], _itr[1], _itr[2],
                                                                       self.veh_info[_itr[1]][_itr[2]]["v"], direction])
                            else:
                                if self.veh_info[_itr[1]][_itr[2]]["p"] - \
                                        self.lane_info[self.veh_info[_itr[1]][_itr[2]]['intention']][1] > 0:
                                    virtual_dis = self.veh_info[_itr[1]][_itr[2]]["p"] - \
                                                  self.lane_info[self.veh_info[_itr[1]][_itr[2]]['intention']][1] + \
                                                  self.lane_info[index][1]
                                    self.virtual_lane_4[direction].append(
                                        [virtual_dis, _itr[1], _itr[2], self.veh_info[_itr[1]][_itr[2]]["v"],
                                         direction])
                        elif self.direction[_itr[1]][_itr[3]] in self.lane2lane[direction]:
                            # 与之相交的车道
                            virtual_d, choose = self.get_virtual_distance(self.direction[_itr[1]][_itr[3]], direction,
                                                                          _itr[0])
                            if choose:
                                self.virtual_lane_real_p[direction].append([_itr[0], _itr[1], _itr[2],
                                                                            self.veh_info[_itr[1]][_itr[2]]["v"],
                                                                            direction])
                                for virtual_temp in range(len(virtual_d)):
                                    self.virtual_lane_4[direction].append([virtual_d[virtual_temp], _itr[1], _itr[2],
                                                                           self.veh_info[_itr[1]][_itr[2]]["v"],
                                                                           self.direction[_itr[1]][_itr[3]]])
                    self.virtual_lane_4[direction] = sorted(self.virtual_lane_4[direction], key=lambda item: item[0])
                    self.virtual_lane_real_p[direction] = sorted(self.virtual_lane_real_p[direction],
                                                                 key=lambda item: item[0])
                    for j, item in enumerate(self.veh_info[i]):
                        if self.veh_info[i][j]["intention"] == index:
                            if self.veh_info[i][j]["seq_in_lane"] == self.choose_veh:
                                self.choose_veh_info[i].append(
                                    [self.current_time, self.veh_info[i][j]["p"], self.veh_info[i][j]["v"],
                                     self.veh_info[i][j]["action"]])
                            t_distance = 2
                            d_distance = 10
                            if self.veh_info[i][j]["control"]:
                                self.veh_info_record[i][item["seq_in_lane"]].append(
                                    [self.current_time, item["p"], item["v"], item["a"]]
                                )
                                sta, virtual_lane = self.get_state(i, j, self.virtual_lane_4[direction], direction)
                                self.virtual_lane_4[direction] = virtual_lane
                                self.veh_info[i][j]["state"] = cp.deepcopy(sta)
                                re_state.append(np.array(sta))
                                actions.append([state[2] for state in sta])
                                ids.append([i, j])
                                self.veh_info[i][j]["count"] += 1
                                closer_car = self.closer_cars[0]
                                if closer_car[0] >= 0:
                                    id_seq_temp = [temp_item[1:3] for temp_item in self.virtual_lane_4[direction]]
                                    if [closer_car[0], closer_car[1]] not in id_seq_temp:
                                        index_closer = -1
                                    else:
                                        index_closer = id_seq_temp.index([closer_car[0], closer_car[1]])
                                    d_distance = abs(
                                        self.veh_info[i][j]["p"] - self.virtual_lane_4[direction][index_closer][0])
                                    self.veh_info[i][j]["closer_p"] = self.virtual_lane_4[direction][index_closer][0]
                                    if d_distance != 0:
                                        t_distance = (self.veh_info[i][j]["p"] -
                                                      self.virtual_lane_4[direction][index_closer][0]) / \
                                                     (self.veh_info[i][j]["v"] -
                                                      self.veh_info[closer_car[0]][closer_car[1]]["v"] +
                                                      0.0001)
                                else:
                                    self.veh_info[i][j]["closer_p"] = 150
                                vw = 2.0
                                cw = 3.0
                                r_ = 0
                                if 0 < t_distance < 4:
                                    r_ += 1 / np.tanh(-t_distance / 4.0)
                                r_ -= pow(self.veh_info[i][j]["jerk"] / self.deltaT, 2) / 3600.0 * cw
                                if d_distance < 10:
                                    r_ += np.log(pow(d_distance / 10, 5) + 0.00001)
                                r_ += (self.veh_info[i][j]["v"] - self.vm) / float(self.aM - self.am) * vw
                                reward.append(min(20, max(-20, r_)))
                                self.veh_info[i][j]["jerk_sum"] += abs(self.veh_info[i][j]["jerk"] / self.deltaT)
                                if 0 <= closer_car[0]:
                                    veh_choose = self.veh_info[i][j]
                                    veh_closer = self.veh_info[closer_car[0]][closer_car[1]]
                                    p_choose = self.get_p(veh_choose["p"], i, self.veh_info[i][j]["intention"])
                                    p_closer = self.get_p(veh_closer["p"], closer_car[0],
                                                          self.veh_info[closer_car[0]][closer_car[1]]["intention"])
                                    d_distance = np.sqrt(
                                        np.power((p_closer[0] - p_choose[0]), 2) + np.power((p_closer[1] - p_choose[1]),
                                                                                            2)
                                    )
                                if abs(d_distance) < self.collision_thr:
                                    self.veh_info[i][j]["collision"] += 1  # 发生碰撞
                                    self.veh_info[closer_car[0]][closer_car[1]]["collision"] += 1  # 发生碰撞
                                if self.veh_info[i][j]["finish"]:
                                    self.veh_info[i][j]["control"] = False
                                collisions += self.veh_info[i][j]["collision"]
                                estm_collisions += self.veh_info[i][j]["estm_collision"]
                                collisions_per_veh.append(
                                    [self.veh_info[i][j]["collision"], self.veh_info[i][j]["estm_collision"]])
                            if self.veh_info[i][j]["p"] < -self.dis_control + int(
                                    (self.lane_num + 1) / 2) * self.lane_cw or self.veh_info[i][j][
                                "collision"] > 0:
                                # 驶出交通路口, 删除该车辆
                                if self.veh_info[i][j]["collision"] > 0:
                                    reward[-1] = -10
                                self.veh_info[i][j]["Done"] = True
                                self.delete_veh.append([i, j])
                                self.veh_info[i][j]["vir_header"] = [-1, -1]
                            elif self.veh_info[i][j]["p"] < 0 and self.veh_info[i][j]["control"]:
                                self.veh_info[i][j]["Done"] = True
                                self.veh_info[i][j]["finish"] = True
                                self.veh_info[i][j]["control"] = False
                                self.veh_info[i][j]["vir_header"] = [-1, -1]
                                self.veh_info[i][j]["lock"] = False
                                self.passed_veh += 1
                                reward[-1] = 5
                                jerks.append(self.veh_info[i][j]["jerk_sum"])
                                self.passed_veh_step_total += self.veh_info[i][j]["step"]
            # 添加新车
            self.add_new_veh(i)
            # if self.show_col:
            #     print("add new car:", i, self.veh_num[i] - 1)
        self.virtual_lane.clear()
        lock = 0
        for i in range(self.lane_num):
            for j, veh in enumerate(self.veh_info[i]):
                if veh["control"] and not self.veh_info[i][j]["lock"]:
                    if self.check_lock(i, j):
                        lock += 1
        for v in self.virtual_lane_4[0]:
            v_name = "%s_%s" % (v[1], self.veh_info[v[1]][v[2]]["seq_in_lane"])
            if v_name not in self.virtual_data:
                self.virtual_data[v_name] = []
            self.virtual_data[v_name].append([self.current_time, v[0], v[3]])
        return ids, re_state, reward, actions, collisions, estm_collisions, collisions_per_veh, jerks, lock

    def add_new_veh(self, i):
        if self.current_time >= self.arrive_time[self.veh_rec[i]][i]:
            state_total = np.zeros((self.closer_veh_num + 1, (self.closer_veh_num + 1) * 4))
            intention = 1  # 默认
            random.seed()
            if self.lane_num == 3:
                intention = self.intention[i][random.randint(0, 1)]
            elif self.lane_num == 4:
                # intention = random.randint(0, 2)
                intention = self.intention_re % 3
                self.intention_re += 1
            elif self.lane_num == 8:
                intention = self.intention[i][random.randint(0, 1)]
                # intention = self.intention[i][self.intention_re % 2]
                self.intention_re += 1
            elif self.lane_num == 12:
                intention = i % 3
            p = sum(self.lane_info[intention][0:2])
            self.veh_info[i].append(
                {
                    "intention": intention,  # 随机生成意向0~2分别表示左转，直行和右转
                    "buffer": [],
                    "route": self.direction[i][intention],
                    "count": 0,
                    "Done": False,
                    "p": p,
                    "jerk": 0,
                    "jerk_sum": 0,
                    "lock_a": 0,
                    "lock": False,
                    "vir_header": [-1, -1],
                    "vir_dis": 100,
                    "v": self.v0,
                    "a": 0,
                    "action": 0,
                    "closer_p": 150,
                    "lane": i,
                    "header": False,
                    "reward": 10,
                    "dis_front": 50,
                    "seq_in_lane": self.veh_rec[i],
                    "control": True,
                    "state": state_total,
                    "step": 0,
                    "collision": 0,
                    "finish": False,
                    "estm_collision": 0,
                    "estm_arrive_time": abs(p / self.v0),
                    "id_info": [self.id_seq, self.veh_num[i]]
                })
            # "id_info":[在所有车中的出现次序,在当前车道中的出现次序]
            self.veh_num[i] += 1
            self.veh_rec[i] += 1
            self.input[i % 4] += 1
            self.veh_info_record[i].append([])
            self.id_seq += 1

    def delete_vehicle(self):
        # 删除旧车
        self.delete_veh = sorted(self.delete_veh, key=lambda item: -item[1])
        for d_i in self.delete_veh:
            if len(self.veh_info[d_i[0]]) > d_i[1]:
                self.veh_info[d_i[0]].pop(d_i[1])
                if self.veh_num[d_i[0]] > 0:
                    self.veh_num[d_i[0]] -= 1
            else:
                print("except!!!")

    # 返回两车在按照碰撞点为原点的情况下的距离,lane1和p1为遍历到的车辆,lane2为每次遍历固定的lane
    def get_virtual_distance(self, lane1, lane2, p1):
        virtual_d = []
        thr = 0
        # if self.lane_num==4:
        #     thr = -5
        choose = False
        if self.lane_num == 4:
            if lane2 in [0, 3, 6, 9]:
                if lane1 == self.lane2lane[lane2][0]:
                    delta_d1 = p1 - (4 * self.lane_cw - 3 * self.lane_cw * math.cos(self.gama))
                    if delta_d1 > thr:
                        virtual_d.append(abs(delta_d1) + (3 * self.lane_cw * (0.5 * 3.1415 - self.gama)))
                        choose = True
                # if lane1 == self.lane2lane[lane2][1]:
                #     delta_d1 = p1 - (1.5 * 3.1415) * self.lane_cw * (self.alpha / (0.5 * 3.1415))
                #     if delta_d1 > thr:
                #         virtual_d.append(abs(delta_d1) + (1.5 * 3.1415 * self.lane_cw * (self._alpha / (0.5 * 3.1415))))
                #         choose = True
                if lane1 == self.lane2lane[lane2][2]:
                    delta_d1 = p1 - 1.5 * 3.1415 * self.lane_cw * self.beta / (0.5 * 3.1415)
                    if delta_d1 > thr:
                        virtual_d.append(abs(delta_d1) + (1.5 * 3.1415 * self.lane_cw * self._beta / (0.5 * 3.1415)))
                        choose = True
                if lane1 == self.lane2lane[lane2][3]:
                    delta_d1 = p1 - 1.5 * 3.1415 * self.lane_cw * self._beta / (0.5 * 3.1415)
                    if delta_d1 > thr:
                        virtual_d.append(abs(delta_d1) + 1.5 * 3.1415 * self.lane_cw * self.beta / (0.5 * 3.1415))
                        choose = True
                if lane1 == self.lane2lane[lane2][1]:
                    delta_d1 = p1 - (1.5 * 3.1415) * self.lane_cw * (self._alpha / (0.5 * 3.1415))
                    if delta_d1 > thr:
                        virtual_d.append(abs(delta_d1) + (1.5 * 3.1415) * self.lane_cw * (self.alpha / (0.5 * 3.1415)))
                        choose = True
                if lane1 == self.lane2lane[lane2][4]:
                    delta_d1 = p1 - 3 * self.lane_cw * math.cos(self.gama)
                    if delta_d1 > thr:
                        virtual_d.append(abs(delta_d1) + (1.5 * 3.1415 * self.lane_cw * (self.gama / (0.5 * 3.1415))))
                        choose = True
                if lane1 == self.lane2lane[lane2][5]:
                    delta_d1 = p1
                    if delta_d1 > thr:
                        virtual_d.append(p1)
                        choose = True
                if lane1 == self.lane2lane[lane2][6]:
                    delta_d1 = p1
                    if delta_d1 > thr:
                        virtual_d.append(p1)
                        choose = True
            elif lane2 in [1, 4, 7, 10]:
                if lane1 == self.lane2lane[lane2][0]:
                    delta_d1 = p1 - self.lane_cw
                    if delta_d1 > thr:
                        virtual_d.append(abs(delta_d1) + 3 * self.lane_cw)
                        choose = True
                elif lane1 == self.lane2lane[lane2][1]:
                    delta_d1 = p1 - 1.5 * 3.1415 * self.lane_cw * self.gama / (0.5 * 3.1415)
                    if delta_d1 > thr:
                        virtual_d.append(abs(delta_d1) + 3 * self.lane_cw * math.cos(self.gama))
                        choose = True
                elif lane1 == self.lane2lane[lane2][2]:
                    delta_d1 = p1 - 1.5 * 3.1415 * self.lane_cw * (0.5 * 3.1415 - self.gama) / (0.5 * 3.1415)
                    if delta_d1 > thr:
                        virtual_d.append(abs(delta_d1) + (4 * self.lane_cw - 3 * self.lane_cw * math.cos(self.gama)))
                        choose = True
                elif lane1 == self.lane2lane[lane2][3]:
                    delta_d1 = p1 - 3 * self.lane_cw
                    if delta_d1 > thr:
                        virtual_d.append(abs(delta_d1) + self.lane_cw)
                        choose = True
                elif lane1 == self.lane2lane[lane2][4]:
                    delta_d1 = p1
                    if delta_d1 > thr:
                        virtual_d.append(p1)
                        choose = True
                elif lane1 == self.lane2lane[lane2][5]:
                    delta_d1 = p1
                    if delta_d1 > thr:
                        virtual_d.append(p1)
                        choose = True
            elif lane2 in [2, 5, 8, 11]:
                if lane1 == self.lane2lane[lane2][0]:
                    delta_d1 = p1
                    if delta_d1 > thr:
                        virtual_d.append(p1)
                        choose = True
                if lane1 == self.lane2lane[lane2][1]:
                    delta_d1 = p1
                    if delta_d1 > thr:
                        virtual_d.append(p1)
                        choose = True
        elif self.lane_num == 8:
            # 左转车道
            # [14, 4, 13, 12, 9, 10, 5]
            if lane2 in [0, 4, 8, 12]:
                if lane1 == self.lane2lane[lane2][0]:
                    delta_d1 = p1 - (8 * self.lane_cw - math.sqrt(24) * self.lane_cw)
                    if delta_d1 > thr:
                        virtual_d.append(abs(delta_d1) + math.atan(math.sqrt(24)) * 5 * self.lane_cw)
                        choose = True
                if lane1 == self.lane2lane[lane2][1]:
                    delta_d1 = p1 - math.atan(3 / 4) * 5 * self.lane_cw
                    if delta_d1 > thr:
                        virtual_d.append(abs(delta_d1) + math.atan(4 / 3) * 5 * self.lane_cw)
                        choose = True
                if lane1 == self.lane2lane[lane2][2]:
                    delta_d1 = p1 - 4 * self.lane_cw
                    if delta_d1 > thr:
                        virtual_d.append(abs(delta_d1) + math.atan(4 / 3) * 5 * self.lane_cw)
                        choose = True
                if lane1 == self.lane2lane[lane2][3]:  # lane2==0 lane1==12
                    # ! delta_d1 = p1 - 5 * self.lane_cw
                    delta_d1 = p1 - math.atan(4 / 3) * 5 * self.lane_cw
                    if delta_d1 > thr:
                        virtual_d.append(abs(delta_d1) + math.atan(3 / 4) * 5 * self.lane_cw)
                        choose = True
                if lane1 == self.lane2lane[lane2][4]:
                    delta_d1 = p1 - 4 * self.lane_cw
                    if delta_d1 > thr:
                        virtual_d.append(abs(delta_d1) + math.atan(3 / 4) * 5 * self.lane_cw)
                        choose = True
                if lane1 == self.lane2lane[lane2][5]:
                    delta_d1 = p1 - math.sqrt(24) * self.lane_cw
                    if delta_d1 > thr:
                        virtual_d.append(abs(delta_d1) + math.atan(1 / math.sqrt(24)) * 5 * self.lane_cw)
                        choose = True
                if lane1 == self.lane2lane[lane2][6]:
                    delta_d1 = p1
                    if delta_d1 > thr:
                        virtual_d.append(p1)
                        choose = True
            # 左边车道直行车道
            # [14, 13, 8, 4, 5, 6, 12],  # 1
            elif lane2 in [1, 5, 9, 13]:
                if lane1 == self.lane2lane[lane2][0]:
                    delta_d1 = p1 - 3 * self.lane_cw
                    if delta_d1 > thr:
                        virtual_d.append(delta_d1 + 7 * self.lane_cw)
                        choose = True
                if lane1 == self.lane2lane[lane2][1]:
                    delta_d1 = p1 - 3 * self.lane_cw
                    if delta_d1 > thr:
                        virtual_d.append(abs(delta_d1) + 5 * self.lane_cw)
                        choose = True
                if lane1 == self.lane2lane[lane2][2]:
                    delta_d1 = p1 - math.atan(3 / 4) * 5 * self.lane_cw
                    if delta_d1 > thr:
                        virtual_d.append(abs(delta_d1) + 4 * self.lane_cw)
                        choose = True
                if lane1 == self.lane2lane[lane2][3]:
                    delta_d1 = p1 - math.atan(4 / 3) * 5 * self.lane_cw
                    if delta_d1 > thr:
                        virtual_d.append(abs(delta_d1) + 4 * self.lane_cw)
                        choose = True
                if lane1 == self.lane2lane[lane2][4]:
                    delta_d1 = p1 - 5 * self.lane_cw
                    if delta_d1 > thr:
                        virtual_d.append(delta_d1 + 3 * self.lane_cw)
                        choose = True
                if lane1 == self.lane2lane[lane2][5]:
                    delta_d1 = p1 - 5 * self.lane_cw
                    if delta_d1 > thr:
                        virtual_d.append(delta_d1 + self.lane_cw)
                        choose = True
                if lane1 == self.lane2lane[lane2][6]:
                    delta_d1 = p1
                    if delta_d1 > thr:
                        virtual_d.append(p1)
                        choose = True
            # 右边车道直行车道
            elif lane2 in [2, 6, 10, 14]:
                # [14, 13, 8, 4, 5, 6, 7],  # 2
                # if lane2 in [0, 4, 8, 12]:
                if lane1 == self.lane2lane[lane2][0]:
                    delta_d1 = p1 - self.lane_cw
                    if delta_d1 > thr:
                        virtual_d.append(delta_d1 + 7 * self.lane_cw)
                        choose = True
                if lane1 == self.lane2lane[lane2][1]:
                    delta_d1 = p1 - self.lane_cw
                    if delta_d1 > thr:
                        virtual_d.append(delta_d1 + 5 * self.lane_cw)
                        choose = True
                if lane1 == self.lane2lane[lane2][2]:
                    delta_d1 = p1 - math.atan(1 / math.sqrt(24)) * 5 * self.lane_cw
                    if delta_d1 > thr:
                        virtual_d.append(abs(delta_d1) + math.sqrt(24) * self.lane_cw)
                        choose = True
                if lane1 == self.lane2lane[lane2][3]:
                    delta_d1 = p1 - math.atan(math.sqrt(24)) * 5 * self.lane_cw
                    if delta_d1 > thr:
                        virtual_d.append(abs(delta_d1) + 8 * self.lane_cw - math.sqrt(24) * self.lane_cw)
                        choose = True
                if lane1 == self.lane2lane[lane2][4]:
                    delta_d1 = p1 - 7 * self.lane_cw
                    if delta_d1 > thr:
                        virtual_d.append(delta_d1 + 3 * self.lane_cw)
                        choose = True
                if lane1 == self.lane2lane[lane2][5]:
                    delta_d1 = p1 - 7 * self.lane_cw
                    if delta_d1 > thr:
                        virtual_d.append(delta_d1 + self.lane_cw)
                        choose = True
                if lane1 == self.lane2lane[lane2][6]:
                    delta_d1 = p1
                    if delta_d1 > thr:
                        virtual_d.append(p1)
                        choose = True
            # 右转车道
            elif lane2 in [3, 7, 11, 15]:
                if lane1 == self.lane2lane[lane2][0]:
                    delta_d1 = p1
                    if delta_d1 > thr:
                        virtual_d.append(p1)
                        choose = True
        elif self.lane_num == 12:
            # if lane2 in [1, 4, 7, 10]:
            #     if lane1 == self.lane2lane[lane2][0]:
            #         # 据碰撞点的距离
            #         delta_d1 = p1 - 3 * self.lane_cw
            #         # delta_d2 = p2 - 9 * self.lane_cw
            #         # delta_d = delta_d1 - delta_d2
            #         if delta_d1 > thr:
            #             virtual_d.append(9 * self.lane_cw + delta_d1)
            #             choose = True
            #     elif lane1 == self.lane2lane[lane2][1]:
            #         # 角度为alpha段的圆弧长度
            #         beta_d = self.beta * 7 * self.lane_cw
            #         delta_d1 = p1 - beta_d
            #         # delta_d2 = p2 - 6 * self.lane_cw - self.cita
            #         # delta_d = delta_d1 - delta_d2
            #         if delta_d1 > thr:
            #             virtual_d.append(6 * self.lane_cw - self.cita + delta_d1)
            #             choose = True
            #     elif lane1 == self.lane2lane[lane2][2]:
            #         # 角度为beta段的圆弧长度
            #         alpha_d = self.alpha * 7 * self.lane_cw
            #         delta_d1 = p1 - alpha_d
            #         # delta_d2 = p2 - 6 * self.lane_cw + self.cita
            #         # delta_d = delta_d1 - delta_d2
            #         if delta_d1 > thr:
            #             virtual_d.append(6 * self.lane_cw - self.cita + delta_d1)
            #             choose = True
            #     elif lane1 == self.lane2lane[lane2][3]:
            #         delta_d1 = p1 - 9 * self.lane_cw
            #         # delta_d2 = p2 - 3 * self.lane_cw
            #         # delta_d = delta_d1 - delta_d2
            #         if delta_d1 > thr:
            #             virtual_d.append(3 * self.lane_cw + delta_d1)
            #             choose = True
            #     else:
            #         if p1 > thr:
            #             virtual_d.append(p1)
            #             choose = True
            # elif lane2 in [0, 3, 6, 9]:
            #     if lane1 == self.lane2lane[lane2][0]:
            #         delta_d1 = p1 - 6 * self.lane_cw + self.cita
            #         # delta_d2 = p2 - self.alpha * 7 * self.lane_cw
            #         # delta_d = delta_d1 - delta_d2
            #         if delta_d1 > thr:
            #             virtual_d.append(self.alpha * 7 * self.lane_cw + delta_d1)
            #             choose = True
            #     elif lane1 == self.lane2lane[lane2][1]:
            #         delta_d1 = p1 - self.gama * 7 * self.lane_cw
            #         # delta_d2 = p2 - self._gama * 7 * self.lane_cw
            #         # delta_d = delta_d1 - delta_d2
            #         if delta_d1 > thr:
            #             virtual_d.append(self._gama * 7 * self.lane_cw + delta_d1)
            #             choose = True
            #     elif lane1 == self.lane2lane[lane2][2]:
            #         delta_d1 = p1 - self._gama * 7 * self.lane_cw
            #         # delta_d2 = p2 - self.gama * 7 * self.lane_cw
            #         # delta_d = delta_d1 - delta_d2
            #         if delta_d1 > thr:
            #             virtual_d.append(self.gama * 7 * self.lane_cw + delta_d1)
            #             choose = True
            #     else:
            #         delta_d1 = p1 - 6 * self.lane_cw - self.cita
            #         # delta_d2 = p2 - self.beta * 7 * self.lane_cw
            #         # delta_d = delta_d1 - delta_d2
            #         if delta_d1 > thr:
            #             virtual_d.append(self.beta * 7 * self.lane_cw + delta_d1)
            #             choose = True
            # else:
            #     if p1 > thr:
            #         virtual_d.append(p1)
            #         choose = True
            if lane2 in [1, 4, 7, 10]:
                if lane1 == self.lane2lane[lane2][0]:
                    # 据碰撞点的距离
                    delta_d1 = p1 - 3 * self.lane_cw
                    # delta_d2 = p2 - 9 * self.lane_cw
                    # delta_d = delta_d1 - delta_d2
                    if delta_d1 > thr:
                        virtual_d.append(9 * self.lane_cw + delta_d1)
                        choose = True
                elif lane1 == self.lane2lane[lane2][1]:
                    # 角度为alpha段的圆弧长度
                    beta_d = self.beta * 7 * self.lane_cw
                    delta_d1 = p1 - beta_d
                    # delta_d2 = p2 - 6 * self.lane_cw - self.cita
                    # delta_d = delta_d1 - delta_d2
                    if delta_d1 > thr:
                        virtual_d.append(6 * self.lane_cw + self.cita + delta_d1)
                        choose = True
                elif lane1 == self.lane2lane[lane2][2]:
                    # 角度为beta段的圆弧长度
                    alpha_d = self.alpha * 7 * self.lane_cw
                    delta_d1 = p1 - alpha_d
                    # delta_d2 = p2 - 6 * self.lane_cw + self.cita
                    # delta_d = delta_d1 - delta_d2
                    if delta_d1 > thr:
                        virtual_d.append(6 * self.lane_cw - self.cita + delta_d1)
                        choose = True
                elif lane1 == self.lane2lane[lane2][3]:
                    delta_d1 = p1 - 9 * self.lane_cw
                    # delta_d2 = p2 - 3 * self.lane_cw
                    # delta_d = delta_d1 - delta_d2
                    if delta_d1 > thr:
                        virtual_d.append(3 * self.lane_cw + delta_d1)
                        choose = True
                else:
                    if p1 > 0:
                        virtual_d.append(p1)
                        choose = True
            elif lane2 in [0, 3, 6, 9]:
                if lane1 == self.lane2lane[lane2][0]:
                    delta_d1 = p1 - 6 * self.lane_cw + self.cita
                    # delta_d2 = p2 - self.alpha * 7 * self.lane_cw
                    # delta_d = delta_d1 - delta_d2
                    if delta_d1 > thr:
                        virtual_d.append(self.alpha * 7 * self.lane_cw + delta_d1)
                        choose = True
                elif lane1 == self.lane2lane[lane2][1]:
                    delta_d1 = p1 - self.gama * 7 * self.lane_cw
                    # delta_d2 = p2 - self._gama * 7 * self.lane_cw
                    # delta_d = delta_d1 - delta_d2
                    if delta_d1 > thr:
                        virtual_d.append(self._gama * 7 * self.lane_cw + delta_d1)
                        choose = True
                elif lane1 == self.lane2lane[lane2][2]:
                    delta_d1 = p1 - self._gama * 7 * self.lane_cw
                    # delta_d2 = p2 - self.gama * 7 * self.lane_cw
                    # delta_d = delta_d1 - delta_d2
                    if delta_d1 > thr:
                        virtual_d.append(self.gama * 7 * self.lane_cw + delta_d1)
                        choose = True
                else:
                    delta_d1 = p1 - 6 * self.lane_cw - self.cita
                    # delta_d2 = p2 - self.beta * 7 * self.lane_cw
                    # delta_d = delta_d1 - delta_d2
                    if delta_d1 > thr:
                        virtual_d.append(self.beta * 7 * self.lane_cw + delta_d1)
                        choose = True
            else:
                if p1 > 0:
                    virtual_d.append(p1)
                    choose = True
        return virtual_d, choose

    # 根据车道的位置计算其真实位置
    def get_p(self, p, i, intention):
        # x, y, yaw(与x轴正方向夹角)
        new_p = [0, 0, 0]
        # car_info = self.veh_info[i][j]
        intention_info = intention
        if self.lane_num == 3:
            if i == 0:
                # 直行
                if intention_info == 1:
                    new_p[0] = -1 * p + 2 * self.lane_cw
                    new_p[1] = -1 * self.lane_cw
                    new_p[2] = 0
                # 右转
                else:
                    if p > self.lane_info[2][1]:
                        new_p[0] = -1 * (p - self.lane_info[2][1] + 2 * self.lane_cw)
                        new_p[1] = -1 * self.lane_cw
                        new_p[2] = 0
                    elif p > 0:
                        beta_temp = p / self.lane_cw
                        delta_y = math.sin(beta_temp) * self.lane_cw
                        delta_x = math.cos(beta_temp) * self.lane_cw
                        new_p[0] = -1 * (2 * self.lane_cw - delta_x)
                        new_p[1] = -1 * (2 * self.lane_cw - delta_y)
                        new_p[2] = beta_temp + 1.5 * 3.1415
                    else:
                        new_p[0] = -1 * self.lane_cw
                        new_p[1] = -1 * (-1 * p + 2 * self.lane_cw)
                        new_p[2] = 1.5 * 3.1415
            elif i == 1:
                # 左转
                if intention_info == 0:
                    # 没到交叉口中
                    if p > self.lane_info[0][1]:
                        new_p[0] = 1 * (p - self.lane_info[0][1] + 2 * self.lane_cw)
                        new_p[1] = 1 * self.lane_cw
                        new_p[2] = 3.1415
                    elif p > 0:
                        beta_temp = p / (3 * self.lane_cw)  # rad
                        delta_y = math.sin(beta_temp) * 3 * self.lane_cw
                        delta_x = math.cos(beta_temp) * 3 * self.lane_cw
                        new_p[0] = -1 * (delta_x - 2 * self.lane_cw)
                        new_p[1] = -1 * (2 * self.lane_cw - delta_y)
                        new_p[2] = 3.1415 / 2 - beta_temp + 3.1415
                    else:
                        new_p[0] = -1 * self.lane_cw
                        new_p[1] = -1 * (-1 * p + 2 * self.lane_cw)
                        new_p[2] = 1.5 * 3.1415
                # 直行
                else:
                    new_p[0] = p - 2 * self.lane_cw
                    new_p[1] = 1 * self.lane_cw
                    new_p[2] = 3.1415
            else:
                # 左转
                if intention_info == 0:
                    # 没到交叉口中
                    if p > self.lane_info[0][1]:
                        new_p[0] = 1 * self.lane_cw
                        new_p[1] = -1 * (p - self.lane_info[0][1] + 2 * self.lane_cw)
                        new_p[2] = 3.1415 / 2
                    elif p > 0:
                        beta_temp = p / (3 * self.lane_cw)  # rad
                        delta_x = math.sin(beta_temp) * 3 * self.lane_cw
                        delta_y = math.cos(beta_temp) * 3 * self.lane_cw
                        new_p[0] = 1 * (delta_x - 2 * self.lane_cw)
                        new_p[1] = -1 * (2 * self.lane_cw - delta_y)
                        new_p[2] = 3.1415 / 2 - beta_temp + 3.1415 / 2
                    else:
                        new_p[0] = -1 * (-1 * p + 2 * self.lane_cw)
                        new_p[1] = self.lane_cw
                        new_p[2] = 3.1415
                # 右转
                else:
                    if p > self.lane_info[2][1]:
                        new_p[0] = 1 * self.lane_cw
                        new_p[1] = -1 * (p - self.lane_info[2][1] + 2 * self.lane_cw)
                        new_p[2] = 3.1415 / 2
                    elif p > 0:
                        beta_temp = p / self.lane_cw
                        delta_x = math.sin(beta_temp) * self.lane_cw
                        delta_y = math.cos(beta_temp) * self.lane_cw
                        new_p[0] = 1 * (2 * self.lane_cw - delta_x)
                        new_p[1] = -1 * (2 * self.lane_cw - delta_y)
                        new_p[2] = beta_temp
                    else:
                        new_p[0] = -1 * p + 2 * self.lane_cw
                        new_p[1] = -1 * self.lane_cw
                        new_p[2] = 0
        elif self.lane_num == 4:
            if i == 0:
                # 左转
                if intention_info == 0:
                    # 没到交叉口中
                    if p > self.lane_info[0][1]:
                        new_p[0] = -1 * (p - self.lane_info[0][1] + 2 * self.lane_cw)
                        new_p[1] = -1 * self.lane_cw
                        new_p[2] = 0
                    elif p > 0:
                        beta_temp = p / (3 * self.lane_cw)  # rad
                        delta_y = math.sin(beta_temp) * 3 * self.lane_cw
                        delta_x = math.cos(beta_temp) * 3 * self.lane_cw
                        new_p[0] = 1 * (delta_x - 2 * self.lane_cw)
                        new_p[1] = 1 * (2 * self.lane_cw - delta_y)
                        new_p[2] = 3.1415 / 2 - beta_temp
                    else:
                        new_p[0] = self.lane_cw
                        new_p[1] = -1 * p + 2 * self.lane_cw
                        new_p[2] = 3.1415 / 2
                # 直行
                elif intention_info == 1:
                    new_p[0] = -1 * p + 2 * self.lane_cw
                    new_p[1] = -1 * self.lane_cw
                    new_p[2] = 0
                # 右转
                else:
                    if p > self.lane_info[2][1]:
                        new_p[0] = -1 * (p - self.lane_info[2][1] + 2 * self.lane_cw)
                        new_p[1] = -1 * self.lane_cw
                        new_p[2] = 0
                    elif p > 0:
                        beta_temp = p / self.lane_cw
                        delta_y = math.sin(beta_temp) * self.lane_cw
                        delta_x = math.cos(beta_temp) * self.lane_cw
                        new_p[0] = -1 * (2 * self.lane_cw - delta_x)
                        new_p[1] = -1 * (2 * self.lane_cw - delta_y)
                        new_p[2] = beta_temp + 1.5 * 3.1415
                    else:
                        new_p[0] = -1 * self.lane_cw
                        new_p[1] = -1 * (-1 * p + 2 * self.lane_cw)
                        new_p[2] = 1.5 * 3.1415
            elif i == 1:
                # 左转
                if intention_info == 0:
                    # 没到交叉口中
                    if p > self.lane_info[0][1]:
                        new_p[0] = 1 * (p - self.lane_info[0][1] + 2 * self.lane_cw)
                        new_p[1] = 1 * self.lane_cw
                        new_p[2] = 3.1415
                    elif p > 0:
                        beta_temp = p / (3 * self.lane_cw)  # rad
                        delta_y = math.sin(beta_temp) * 3 * self.lane_cw
                        delta_x = math.cos(beta_temp) * 3 * self.lane_cw
                        new_p[0] = -1 * (delta_x - 2 * self.lane_cw)
                        new_p[1] = -1 * (2 * self.lane_cw - delta_y)
                        new_p[2] = 3.1415 / 2 - beta_temp + 3.1415
                    else:
                        new_p[0] = -1 * self.lane_cw
                        new_p[1] = -1 * (-1 * p + 2 * self.lane_cw)
                        new_p[2] = 1.5 * 3.1415
                # 直行
                elif intention_info == 1:
                    new_p[0] = p - 2 * self.lane_cw
                    new_p[1] = 1 * self.lane_cw
                    new_p[2] = 3.1415
                # 右转
                else:
                    if p > self.lane_info[2][1]:
                        new_p[0] = 1 * (p - self.lane_info[2][1] + 2 * self.lane_cw)
                        new_p[1] = 1 * self.lane_cw
                        new_p[2] = 3.1415
                    elif p > 0:
                        beta_temp = p / self.lane_cw
                        delta_y = math.sin(beta_temp) * self.lane_cw
                        delta_x = math.cos(beta_temp) * self.lane_cw
                        new_p[0] = 1 * (2 * self.lane_cw - delta_x)
                        new_p[1] = 1 * (2 * self.lane_cw - delta_y)
                        new_p[2] = beta_temp + 3.1415 / 2
                    else:
                        new_p[0] = 1 * self.lane_cw
                        new_p[1] = -1 * p + 2 * self.lane_cw
                        new_p[2] = 3.1415 / 2
            elif i == 2:
                # 左转
                if intention_info == 0:
                    # 没到交叉口中
                    if p > self.lane_info[0][1]:
                        new_p[0] = 1 * self.lane_cw
                        new_p[1] = -1 * (p - self.lane_info[0][1] + 2 * self.lane_cw)
                        new_p[2] = 3.1415 / 2
                    elif p > 0:
                        beta_temp = p / (3 * self.lane_cw)  # rad
                        delta_x = math.sin(beta_temp) * 3 * self.lane_cw
                        delta_y = math.cos(beta_temp) * 3 * self.lane_cw
                        new_p[0] = 1 * (delta_x - 2 * self.lane_cw)
                        new_p[1] = -1 * (2 * self.lane_cw - delta_y)
                        new_p[2] = 3.1415 / 2 - beta_temp + 3.1415 / 2
                    else:
                        new_p[0] = -1 * (-1 * p + 2 * self.lane_cw)
                        new_p[1] = self.lane_cw
                        new_p[2] = 3.1415
                # 直行
                elif intention_info == 1:
                    new_p[0] = self.lane_cw
                    new_p[1] = -1 * p + 2 * self.lane_cw
                    new_p[2] = 3.1415 / 2
                # 右转
                else:
                    if p > self.lane_info[2][1]:
                        new_p[0] = 1 * self.lane_cw
                        new_p[1] = -1 * (p - self.lane_info[2][1] + 2 * self.lane_cw)
                        new_p[2] = 3.1415 / 2
                    elif p > 0:
                        beta_temp = p / self.lane_cw
                        delta_x = math.sin(beta_temp) * self.lane_cw
                        delta_y = math.cos(beta_temp) * self.lane_cw
                        new_p[0] = 1 * (2 * self.lane_cw - delta_x)
                        new_p[1] = -1 * (2 * self.lane_cw - delta_y)
                        new_p[2] = beta_temp
                    else:
                        new_p[0] = -1 * p + 2 * self.lane_cw
                        new_p[1] = -1 * self.lane_cw
                        new_p[2] = 0
            else:
                # 左转
                if intention_info == 0:
                    # 没到交叉口中
                    if p > self.lane_info[0][1]:
                        new_p[0] = -1 * self.lane_cw
                        new_p[1] = 1 * (p - self.lane_info[0][1] + 2 * self.lane_cw)
                        new_p[2] = 1.5 * 3.1415
                    elif p > 0:
                        beta_temp = p / (3 * self.lane_cw)  # rad
                        delta_x = math.sin(beta_temp) * 3 * self.lane_cw
                        delta_y = math.cos(beta_temp) * 3 * self.lane_cw
                        new_p[0] = -1 * (delta_x - 2 * self.lane_cw)
                        new_p[1] = 1 * (2 * self.lane_cw - delta_y)
                        new_p[2] = 3.1415 / 2 - beta_temp + 1.5 * 3.1415
                    else:
                        new_p[0] = 1 * (-1 * p + 2 * self.lane_cw)
                        new_p[1] = -1 * self.lane_cw
                        new_p[2] = 0
                # 直行
                elif intention_info == 1:
                    new_p[0] = -1 * self.lane_cw
                    new_p[1] = p - 2 * self.lane_cw
                    new_p[2] = 1.5 * 3.1415
                # 右转
                else:
                    if p > self.lane_info[2][1]:
                        new_p[0] = -1 * self.lane_cw
                        new_p[1] = 1 * (p - self.lane_info[2][1] + 2 * self.lane_cw)
                        new_p[2] = 1.5 * 3.1415
                    elif p > 0:
                        beta_temp = p / self.lane_cw
                        delta_x = math.sin(beta_temp) * self.lane_cw
                        delta_y = math.cos(beta_temp) * self.lane_cw
                        new_p[0] = -1 * (2 * self.lane_cw - delta_x)
                        new_p[1] = 1 * (2 * self.lane_cw - delta_y)
                        new_p[2] = beta_temp + 3.1415
                    else:
                        new_p[0] = -1 * (-1 * p + 2 * self.lane_cw)
                        new_p[1] = 1 * self.lane_cw
                        new_p[2] = 3.1415
        elif self.lane_num == 8:
            if i == 0:
                # 左转
                if intention_info == 0:
                    # 没到交叉口中
                    if p > self.lane_info[0][1]:
                        new_p[0] = 1 * (p - self.lane_info[0][1] + 4 * self.lane_cw)
                        new_p[1] = 1 * self.lane_cw
                        new_p[2] = 3.1415
                    elif p > 0:
                        beta_temp = p / (5 * self.lane_cw)  # rad
                        delta_y = math.sin(beta_temp) * 5 * self.lane_cw
                        delta_x = math.cos(beta_temp) * 5 * self.lane_cw
                        new_p[0] = -1 * (delta_x - 4 * self.lane_cw)
                        new_p[1] = -1 * (4 * self.lane_cw - delta_y)
                        new_p[2] = 3.1415 / 2 - beta_temp + 3.1415
                    else:
                        new_p[0] = -1 * self.lane_cw
                        new_p[1] = -1 * (-1 * p + 4 * self.lane_cw)
                        new_p[2] = 1.5 * 3.1415
                # 直行
                elif intention_info == 1:
                    new_p[0] = p - 4 * self.lane_cw
                    new_p[1] = 1 * self.lane_cw
                    new_p[2] = 3.1415
            elif i == 1:
                # 直行
                if intention_info == 1:
                    new_p[0] = p - 4 * self.lane_cw
                    new_p[1] = 3 * self.lane_cw
                    new_p[2] = 3.1415
                # 右转
                elif intention_info == 2:
                    if p > self.lane_info[2][1]:
                        new_p[0] = 1 * (p - self.lane_info[2][1] + 4 * self.lane_cw)
                        new_p[1] = 3 * self.lane_cw
                        new_p[2] = 3.1415
                    elif p > 0:
                        beta_temp = p / self.lane_cw
                        delta_y = math.sin(beta_temp) * self.lane_cw
                        delta_x = math.cos(beta_temp) * self.lane_cw
                        new_p[0] = 1 * (4 * self.lane_cw - delta_x)
                        new_p[1] = 1 * (4 * self.lane_cw - delta_y)
                        new_p[2] = beta_temp + 3.1415 / 2
                    else:
                        new_p[0] = 3 * self.lane_cw
                        new_p[1] = -1 * p + 4 * self.lane_cw
                        new_p[2] = 3.1415 / 2
            elif i == 2:
                # 左转
                if intention_info == 0:
                    # 没到交叉口中
                    if p > self.lane_info[0][1]:
                        new_p[0] = -1 * self.lane_cw
                        new_p[1] = 1 * (p - self.lane_info[0][1] + 4 * self.lane_cw)
                        new_p[2] = 1.5 * 3.1415
                    elif p > 0:
                        beta_temp = p / (5 * self.lane_cw)  # rad
                        delta_x = math.sin(beta_temp) * 5 * self.lane_cw
                        delta_y = math.cos(beta_temp) * 5 * self.lane_cw
                        new_p[0] = -1 * (delta_x - 4 * self.lane_cw)
                        new_p[1] = 1 * (4 * self.lane_cw - delta_y)
                        new_p[2] = 3.1415 / 2 - beta_temp + 1.5 * 3.1415
                    else:
                        new_p[0] = 1 * (-1 * p + 4 * self.lane_cw)
                        new_p[1] = -1 * self.lane_cw
                        new_p[2] = 0
                # 直行
                elif intention_info == 1:
                    new_p[0] = -1 * self.lane_cw
                    new_p[1] = p - 4 * self.lane_cw
                    new_p[2] = 1.5 * 3.1415
            elif i == 3:
                # 直行
                if intention_info == 1:
                    new_p[0] = -3 * self.lane_cw
                    new_p[1] = p - 4 * self.lane_cw
                    new_p[2] = 1.5 * 3.1415
                # 右转
                elif intention_info == 2:
                    if p > self.lane_info[2][1]:
                        new_p[0] = -3 * self.lane_cw
                        new_p[1] = 1 * (p - self.lane_info[2][1] + 4 * self.lane_cw)
                        new_p[2] = 1.5 * 3.1415
                    elif p > 0:
                        beta_temp = p / self.lane_cw
                        delta_x = math.sin(beta_temp) * self.lane_cw
                        delta_y = math.cos(beta_temp) * self.lane_cw
                        new_p[0] = -1 * (4 * self.lane_cw - delta_x)
                        new_p[1] = 1 * (4 * self.lane_cw - delta_y)
                        new_p[2] = beta_temp + 3.1415
                    else:
                        new_p[0] = -1 * (-1 * p + 4 * self.lane_cw)
                        new_p[1] = 3 * self.lane_cw
                        new_p[2] = 3.1415
            elif i == 4:
                # 左转
                if intention_info == 0:
                    # 没到交叉口中
                    if p > self.lane_info[0][1]:
                        new_p[0] = -1 * (p - self.lane_info[0][1] + 4 * self.lane_cw)
                        new_p[1] = -1 * self.lane_cw
                        new_p[2] = 0
                    elif p > 0:
                        beta_temp = p / (5 * self.lane_cw)  # rad
                        delta_y = math.sin(beta_temp) * 5 * self.lane_cw
                        delta_x = math.cos(beta_temp) * 5 * self.lane_cw
                        new_p[0] = 1 * (delta_x - 4 * self.lane_cw)
                        new_p[1] = 1 * (4 * self.lane_cw - delta_y)
                        new_p[2] = 3.1415 / 2 - beta_temp
                    else:
                        new_p[0] = self.lane_cw
                        new_p[1] = -1 * p + 4 * self.lane_cw
                        new_p[2] = 3.1415 / 2
                # 直行
                elif intention_info == 1:
                    new_p[0] = -1 * p + 4 * self.lane_cw
                    new_p[1] = -1 * self.lane_cw
                    new_p[2] = 0
            elif i == 5:
                # 直行
                if intention_info == 1:
                    new_p[0] = -1 * p + 4 * self.lane_cw
                    new_p[1] = -3 * self.lane_cw
                    new_p[2] = 0
                # 右转
                elif intention_info == 2:
                    if p > self.lane_info[2][1]:
                        new_p[0] = -1 * (p - self.lane_info[2][1] + 4 * self.lane_cw)
                        new_p[1] = -3 * self.lane_cw
                        new_p[2] = 0
                    elif p > 0:
                        beta_temp = p / self.lane_cw
                        delta_y = math.sin(beta_temp) * self.lane_cw
                        delta_x = math.cos(beta_temp) * self.lane_cw
                        new_p[0] = -1 * (4 * self.lane_cw - delta_x)
                        new_p[1] = -1 * (4 * self.lane_cw - delta_y)
                        new_p[2] = beta_temp + 1.5 * 3.1415
                    else:
                        new_p[0] = -3 * self.lane_cw
                        new_p[1] = -1 * (-1 * p + 4 * self.lane_cw)
                        new_p[2] = 1.5 * 3.1415
            elif i == 6:
                # 左转
                if intention_info == 0:
                    # 没到交叉口中
                    if p > self.lane_info[0][1]:
                        new_p[0] = 1 * self.lane_cw
                        new_p[1] = -1 * (p - self.lane_info[0][1] + 4 * self.lane_cw)
                        new_p[2] = 3.1415 / 2
                    elif p > 0:
                        beta_temp = p / (5 * self.lane_cw)  # rad
                        delta_x = math.sin(beta_temp) * 5 * self.lane_cw
                        delta_y = math.cos(beta_temp) * 5 * self.lane_cw
                        new_p[0] = 1 * (delta_x - 4 * self.lane_cw)
                        new_p[1] = -1 * (4 * self.lane_cw - delta_y)
                        new_p[2] = 3.1415 / 2 - beta_temp + 3.1415 / 2
                    else:
                        new_p[0] = -1 * (-1 * p + 4 * self.lane_cw)
                        new_p[1] = 1 * self.lane_cw
                        new_p[2] = 3.1415
                # 直行
                elif intention_info == 1:
                    new_p[0] = 1 * self.lane_cw
                    new_p[1] = -1 * p + 4 * self.lane_cw
                    new_p[2] = 3.1415 / 2
            elif i == 7:
                # 直行
                if intention_info == 1:
                    new_p[0] = 3 * self.lane_cw
                    new_p[1] = -1 * p + 4 * self.lane_cw
                    new_p[2] = 3.1415 / 2
                # 右转
                elif intention_info == 2:
                    if p > self.lane_info[2][1]:
                        new_p[0] = 3 * self.lane_cw
                        new_p[1] = -1 * (p - self.lane_info[2][1] + 4 * self.lane_cw)
                        new_p[2] = 3.1415 / 2
                    elif p > 0:
                        beta_temp = p / self.lane_cw
                        delta_x = math.sin(beta_temp) * self.lane_cw
                        delta_y = math.cos(beta_temp) * self.lane_cw
                        new_p[0] = 1 * (4 * self.lane_cw - delta_x)
                        new_p[1] = -1 * (4 * self.lane_cw - delta_y)
                        new_p[2] = beta_temp
                    else:
                        new_p[0] = -1 * p + 4 * self.lane_cw
                        new_p[1] = -3 * self.lane_cw
                        new_p[2] = 0
        elif self.lane_num == 12:
            rotation_angle = 3.141593 / 2 * int(i / 3)
            p_temp = [0, 0, 0, 0]
            if i % 3 == 0:
                if p > self.lane_info[0][1]:
                    yaw = (3.1415 + i * (3.1415 / 6)) % (2 * 3.1415)
                    p_temp = [p - self.lane_info[0][1] + 6 * self.lane_cw, self.lane_cw, yaw]
                elif p > 0:
                    yaw = (3.1415 + i * (3.1415 / 6)) % (2 * 3.1415)
                    r_a = (self.lane_info[0][1] - p) / self.lane_info[0][1] * 3.141593 / 2
                    p0 = [6 * self.lane_cw, self.lane_cw]
                    p_r = [6 * self.lane_cw, -6 * self.lane_cw]
                    p_temp[0] = p_r[0] + (p0[0] - p_r[0]) * np.cos(r_a) - (p0[1] - p_r[1]) * np.sin(r_a)
                    p_temp[1] = p_r[1] + (p0[1] - p_r[1]) * np.cos(r_a) + (p0[0] - p_r[0]) * np.sin(r_a)
                    p_temp[2] = yaw + r_a
                else:
                    yaw = (1.5 * 3.1415 + i * (3.1415 / 6)) % (2 * 3.1415)
                    p_temp = [-self.lane_cw, -6 * self.lane_cw + p, yaw]
            elif i % 3 == 1:
                yaw = (3.1415 + (i - 1) * (3.1415 / 6)) % (2 * 3.1415)
                p_temp = [p - 6 * self.lane_cw, 3 * self.lane_cw, yaw]
            else:
                if p > self.lane_info[2][1]:
                    yaw = (3.1415 + (i - 2) * (3.1415 / 6)) % (2 * 3.1415)
                    p_temp = [p - self.lane_info[2][1] + 6 * self.lane_cw, 5 * self.lane_cw, yaw]
                elif p > 0:
                    yaw = (0.5 * 3.1415 + (i - 2) * (3.1415 / 6)) % (2 * 3.1415)
                    r_a = (self.lane_info[2][1] - p) / self.lane_info[2][1] * 3.141593 / 2
                    p0 = [6 * self.lane_cw, 5 * self.lane_cw]
                    p_r = [6 * self.lane_cw, 6 * self.lane_cw]
                    # 绕点p_r逆时顺时针旋转r_a角度
                    p_temp[0] = p_r[0] + (p0[0] - p_r[0]) * np.cos(r_a) + (p0[1] - p_r[1]) * np.sin(r_a)
                    p_temp[1] = p_r[1] + (p0[1] - p_r[1]) * np.cos(r_a) - (p0[0] - p_r[0]) * np.sin(r_a)
                    p_temp[2] = 3.1415 / 2 - r_a + yaw
                else:
                    yaw = (0.5 * 3.1415 + (i - 2) * (3.1415 / 6)) % (2 * 3.1415)
                    p_temp = [5 * self.lane_cw, 6 * self.lane_cw - p, yaw]
            new_p[0] = ((p_temp[0]) * np.cos(rotation_angle) - (p_temp[1]) * np.sin(rotation_angle))
            new_p[1] = ((p_temp[1]) * np.cos(rotation_angle) + (p_temp[0]) * np.sin(rotation_angle))
            new_p[2] = p_temp[2]
        return new_p

    def get_state(self, i, j, virtual_lane_4_ori, direction):
        virtual_lane_4 = cp.deepcopy(virtual_lane_4_ori)
        state = []
        state_total = np.zeros((self.closer_veh_num + 1, (self.closer_veh_num + 1) * 4))
        id_seq = [item[1:3] for item in virtual_lane_4]
        if [i, j] not in id_seq:
            index = -1
        else:
            index = id_seq.index([i, j])
        if self.lane_num == 4 and direction in [0, 3, 6, 9]:
            for seq_, item in enumerate(virtual_lane_4_ori):
                if item[4] == self.lane2lane[direction][1]:
                    ori_p = item[0] + (self._alpha - self.alpha) * 3 * self.lane_cw
                    if virtual_lane_4_ori[index][0] < ori_p:
                        # 这种情况只会在较远的碰撞点发生碰撞
                        # 此时要保证本车先通过碰撞点，但是较远碰撞点构建的虚拟车有可能会在本车前面，导致本车减速，这时就要强制修改虚拟距离
                        remote_p1 = virtual_lane_4_ori[index][0]
                        remote_p2 = ori_p - self._alpha * 3 * self.lane_cw + self.alpha * 3 * self.lane_cw
                        virtual_lane_4[seq_][0] = remote_p2
                        if remote_p2 < remote_p1:
                            virtual_lane_4[seq_][0] = remote_p1 + 1
                    else:
                        # 与上一种情况相反
                        remote_p1 = virtual_lane_4_ori[index][0]
                        remote_p2 = ori_p + self._alpha * 3 * self.lane_cw - self.alpha * 3 * self.lane_cw
                        virtual_lane_4[seq_][0] = remote_p2
                        if remote_p2 > remote_p1:
                            virtual_lane_4[seq_][0] = remote_p1 - 1
        p = virtual_lane_4[index][0]
        # if self.lane_num == 4 and direction in [0, 4, 8, 12]:

        # self.virtual_lane_search_closer(i, j, virtual_lane_4, mode="closer", veh_num=6)
        self.virtual_lane_search_closer(i, j, virtual_lane_4, mode="closer", veh_num=6)
        for num, car in enumerate(self.closer_cars):
            if car[0] != -1:
                car_index = id_seq.index([car[0], car[1]])
                vir_car_temp = virtual_lane_4[car_index]  # 虚拟属性
                car_temp = self.veh_info[car[0]][car[1]]  # 真实属性
                state += [vir_car_temp[0], vir_car_temp[3], car_temp["a"],
                          car_temp["route"]]  # [vir_poisition,v,a,lane]
                state_total[num + 1] = np.array(self.veh_info[car[0]][car[1]]["state"][0][:])
            else:
                state += [0, 0, 0, 0]
                state_total[num + 1] = np.array([0 for m in range(28)])
        state = [p, virtual_lane_4[index][3], self.veh_info[i][j]["a"], self.veh_info[i][j]["route"]] + state[:]
        state_total[0] = np.array(state[:])
        return state_total, virtual_lane_4

    def virtual_lane_search_closer(self, i, j, virtual_lane_4, mode="front", veh_num=3):
        id_seq = [item[1:3] for item in virtual_lane_4]
        if [i, j] not in id_seq:
            index = -1
        else:
            index = id_seq.index([i, j])
        self.closer_cars.clear()
        self.closer_same_l_car = [-1, -1]
        if index >= 0:
            if index == 0:
                self.veh_info[i][j]["vir_header"] = [-1, -1]
                self.veh_info[i][j]["vir_dis"] = 100
            else:
                self.veh_info[i][j]["vir_header"] = id_seq[index - 1]
                self.veh_info[i][j]["vir_dis"] = virtual_lane_4[index][0] - virtual_lane_4[index - 1][0]
            if mode == "front":  # 搜寻前车
                for k in range(index - 1, -1, -1):
                    veh_info = id_seq[k]
                    lane_id = veh_info[0]  # 获取车道id
                    if i + lane_id not in [1, 5]:  # 不添加临近车道
                        self.closer_cars.append(veh_info)
                    if len(self.closer_cars) >= veh_num:
                        break
            elif mode == "front-back":
                for k in range(index - 1, -1, -1):
                    veh_info = id_seq[k]
                    lane_id = veh_info[0]  # 获取车道id
                    if i + lane_id not in [1, 5]:  # 不添加临近车道
                        self.closer_cars.append(veh_info)
                    if len(self.closer_cars) >= veh_num - int(veh_num / 2):
                        break
                for k in range(index + 1, len(id_seq)):
                    veh_info = id_seq[k]
                    lane_id = veh_info[0]  # 获取车道id
                    if i + lane_id not in [1, 5]:  # 不添加临近车道
                        self.closer_cars.append(veh_info)
                    if len(self.closer_cars) >= veh_num / 2:
                        break
            elif mode == "closer":
                thr_ = 5.0
                # if self.lane_num == 4:
                #     thr_ = 10.0
                virtual_lane_abs = []
                for _itr in virtual_lane_4:
                    # virtual_lane_abs.append([abs(_itr[0] - virtual_lane_4[index][0]), _itr[1], _itr[2]])
                    flag = 1
                    # if _itr[1] != i:
                    #     flag += abs(_itr[0] - virtual_lane_4[index][0]) / thr_
                    virtual_lane_abs.append([abs(_itr[0] - virtual_lane_4[index][0]) * flag, _itr[1], _itr[2], _itr[0]])
                virtual_lane_abs = sorted(virtual_lane_abs,
                                          key=lambda item: item[0])  # 对虚拟车道的车辆重新通过距离进行排序
                for _id, _itr in enumerate(virtual_lane_abs):
                    if [_itr[1], _itr[2]] != [i, j] and len(self.closer_cars) < veh_num:
                        # if _itr[1] != i and abs(
                        #         _itr[3] / self.veh_info[_itr[1]][_itr[2]]["v"] - virtual_lane_4[index][0] /
                        #         self.veh_info[i][j]["v"]) > 0.8: # 0.8
                        #     continue
                        self.closer_cars.append([_itr[1], _itr[2]])
                        if _itr[1] == i and self.closer_same_l_car[0] == -1:
                            if self.veh_info[_itr[1]][_itr[2]]["intention"] == self.veh_info[i][j]["intention"]:
                                self.closer_same_l_car = [_itr[1], _itr[2]]
                            elif self.veh_info[_itr[1]][_itr[2]]["p"] > \
                                    self.lane_info[self.veh_info[_itr[1]][_itr[2]]["intention"]][1]:
                                self.closer_same_l_car = [_itr[1], _itr[2]]
        for k in range(veh_num - len(self.closer_cars)):
            self.closer_cars.append([-1, -1])

    def get_other_state(self, i, dis, z_i, z_j):
        """
        获取最密切车辆的状态
        :param i: 车道序号
        :param dis: 本车的距离交叉口中心点的距离
        :param z_i, z_j: 当前车的序号
        :return: 位置(取几何坐标还是距离交叉口中心的距离(绝对值)??, 暂定后者) 速度 加速度 车道序号
        """
        diff = 9999
        seq = -1
        dis_temp = 9999
        estm_collision = False
        for ind, veh in enumerate(self.veh_info[i]):
            if veh["control"]:
                if i != z_i:  # 不是同一车道
                    dis_temp = np.sqrt(np.power(dis, 2) + np.power(veh["p"], 2))
                    if z_j < self.veh_num[z_i]:
                        arv_time_diff = abs(
                            self.veh_info[z_i][z_j]["estm_arrive_time"] - self.veh_info[i][ind]["estm_arrive_time"])
                        if arv_time_diff < 1 * self.deltaT:
                            # 迅速情况下会发生碰撞
                            estm_collision = True
                elif ind != z_j:  # 同一车道但是不是同一辆车
                    dis_temp = abs(abs(dis) - abs(veh["p"]))
                    if z_j < self.veh_num[z_i]:
                        if ind > z_j:  # 后进入
                            if (self.veh_info[z_i][z_j]["estm_arrive_time"] - self.veh_info[i][ind][
                                "estm_arrive_time"]) > -1 * self.deltaT:
                                # 迅速情况下会发生碰撞
                                estm_collision = True
                        else:  # 该车先进入
                            if (self.veh_info[z_i][z_j]["estm_arrive_time"] - self.veh_info[i][ind][
                                "estm_arrive_time"]) < 1 * self.deltaT:
                                # 迅速情况下会发生碰撞
                                estm_collision = True
                if dis_temp < diff:
                    diff = dis_temp
                    seq = ind
                if estm_collision:
                    self.veh_info[z_i][z_j]["estm_collision"] += 1
                    if self.show_col:
                        print("estimate collision occurred!!", [z_i, z_j], [i, ind])
                    estm_collision = False
                if z_j < self.veh_num[z_i]:
                    if diff < 2:
                        # 发生碰撞!!!!
                        self.veh_info[z_i][z_j]["collision"] += 1
                        if self.show_col:
                            print("collision occurred!!", [z_i, z_j], [i, seq])
        if diff < self.diff_max and seq != -1:
            return [i, seq]
        else:
            return [-1, -1]

    def judge_fb(self, i, j):
        #  函数功能：判断最邻近车辆在后面还是前面
        back = True
        closer_p = self.veh_info[i][j]["closer_p"]
        if closer_p < self.veh_info[i][j]["p"] or self.veh_info[i][j]["dis_front"] < 10:
            back = False
        return back

    def check_lock(self, i, j):
        N = 10
        thr_d = self.collision_thr
        t_v = [i, j]
        while N:
            N -= 1
            t_v = self.veh_info[t_v[0]][t_v[1]]["vir_header"]
            if t_v[0] == -1:
                break
            if t_v == [i, j]:
                flag = True
                record_ = []
                while flag:  # 更新环内所有车的上锁状态
                    self.veh_info[t_v[0]][t_v[1]]["lock"] = True
                    vir_dis = self.veh_info[t_v[0]][t_v[1]]["vir_dis"]
                    o_v = t_v[:]
                    t_v = self.veh_info[t_v[0]][t_v[1]]["vir_header"]
                    record_.append(
                        [vir_dis - (self.veh_info[o_v[0]][o_v[1]]["v"] - self.veh_info[t_v[0]][t_v[1]]["v"]) * 0,
                         o_v[0], o_v[1], t_v[0], t_v[1]])
                    if t_v == [i, j]:
                        flag = False
                lock_sum = int(sum(record_[:][0]))
                record_.sort()
                dis = [item[0] for item in record_]

                if record_[0][0] < thr_d or sum(dis) / float(len(dis)) < self.collision_thr + 3:
                    self.veh_info[record_[0][1]][record_[0][2]]["lock_a"] = 1
                    self.veh_info[record_[0][3]][record_[0][4]]["lock_a"] = -1
                return True
        return False

    def step(self, i, j, eval_a):
        target_a = min(self.aM, max(self.am, eval_a))
        if self.veh_info[i][j]["lock"] and self.veh_info[i][j]["lock_a"] != 0 and self.veh_info[i][j]["p"] > 70:
            # 死锁破解策略
            target_a = self.veh_info[i][j]["a"] + self.veh_info[i][j]["lock_a"]
        self.veh_info[i][j]["lock"] = False
        self.veh_info[i][j]["lock_a"] = 0
        # 安全干预策略
        if j > 0 and self.veh_info[i][j - 1]["v"] < self.veh_info[i][j]["v"] and self.veh_info[i][j - 1]["control"] and \
                self.veh_info[i][j]["control"]:
            r_t = 0.4  # 反应时间
            d_safe = self.veh_info[i][j]["v"] * r_t + (
                    pow(self.veh_info[i][j]["v"], 2) - pow(self.veh_info[i][j - 1]["v"], 2)) / (2 * abs(self.am)) \
                     - (self.veh_info[i][j]["v"] - self.veh_info[i][j - 1]["v"]) * self.vm / abs(self.am)
            if self.veh_info[i][j]["p"] - self.veh_info[i][j - 1]["p"] < d_safe:
                target_a = self.am
        if len(self.virtual_lane_4[i]) > 0 and self.virtual_lane_4[i][0][1:3] == [i, j]:
            target_a = self.aM
        if i in [2, 5, 8, 11]:
            target_a = self.aM
        target_a = min(self.aM, max(self.am, target_a))
        self.veh_info[i][j]["jerk"] = target_a - self.veh_info[i][j]["a"]
        self.veh_info[i][j]["a"] = target_a
        if (i == 0 or i == 2) and not self.loc_con:
            self.veh_info[i][j]["p"] = self.veh_info[i][j]["p"] + self.veh_info[i][j]["v"] * self.deltaT + 0.5 * \
                                       self.veh_info[i][j]["a"] * pow(self.deltaT, 2)
        else:
            self.veh_info[i][j]["p"] = self.veh_info[i][j]["p"] - self.veh_info[i][j]["v"] * self.deltaT - 0.5 * \
                                       self.veh_info[i][j]["a"] * pow(self.deltaT, 2)
        self.veh_info[i][j]["v"] = min(self.vM,
                                       max(self.veh_info[i][j]["v"] + self.veh_info[i][j]["a"] * self.deltaT, self.vm))
        self.veh_info[i][j]["estm_arrive_time"] = abs(self.veh_info[i][j]["p"] / self.veh_info[i][j]["v"])
        self.veh_info[i][j]["step"] += 1
        if not self.veh_info[i][j]["control"]:
            self.veh_info[i][j]["v"] = self.v0  # 出交叉口之后所有车的速度都变回初始速度
            # todo
            # 将速度突变改为渐变
        else:
            self.virtual_lane.append([self.veh_info[i][j]["p"], i, j, self.veh_info[i][j]["intention"]])


class Visible:
    def __init__(self, lane_w=5, control_dis=150, l_mode="actual", c_mode="front-back", lane_num=8):
        plt.figure(1)
        self.px = [[] for i in range(lane_num)]
        self.py = [[] for i in range(lane_num)]
        self.lane_w = lane_w
        self.color_m = np.zeros((4, 433)) - 1
        self.l_mode = l_mode
        self.c_mode = c_mode
        self.control_dis = control_dis
        self.marker = ["3", "1", "4", "2"]
        if lane_num == 12:
            self.marker = ["3", "3", "3", "1", "1", "1", "4", "4", "4", "2", "2", "2"]
        elif lane_num == 8:
            self.marker = ["3", "3", "1", "1", "4", "4", "2", "2"]
        elif lane_num == 4:
            self.marker = ["4", "3", "2", "1"]

    def get_p(self, p, i, env, intention):
        # x, y, yaw(与x轴正方向夹角)
        new_p = [0, 0, 0]
        if env.lane_num == 4:
            if i == 0:
                # 左转
                if intention == 0:
                    # 没到交叉口中
                    if p > env.lane_info[0][1]:
                        new_p[0] = -1 * (p - env.lane_info[0][1] + 2 * env.lane_cw)
                        new_p[1] = -1 * env.lane_cw
                        new_p[2] = 0
                    elif p > 0:
                        beta_temp = p / (3 * env.lane_cw)  # rad
                        delta_y = math.sin(beta_temp) * 3 * env.lane_cw
                        delta_x = math.cos(beta_temp) * 3 * env.lane_cw
                        new_p[0] = 1 * (delta_x - 2 * env.lane_cw)
                        new_p[1] = 1 * (2 * env.lane_cw - delta_y)
                        new_p[2] = 2
                    else:
                        new_p[0] = env.lane_cw
                        new_p[1] = -1 * p + 2 * env.lane_cw
                        new_p[2] = 2
                # 直行
                elif intention == 1:
                    new_p[0] = -1 * p + 2 * env.lane_cw
                    new_p[1] = -1 * env.lane_cw
                    new_p[2] = 0
                # 右转
                else:
                    if p > env.lane_info[2][1]:
                        new_p[0] = -1 * (p - env.lane_info[2][1] + 2 * env.lane_cw)
                        new_p[1] = -1 * env.lane_cw
                        new_p[2] = 0
                    elif p > 0:
                        beta_temp = p / env.lane_cw
                        delta_y = math.sin(beta_temp) * env.lane_cw
                        delta_x = math.cos(beta_temp) * env.lane_cw
                        new_p[0] = -1 * (2 * env.lane_cw - delta_x)
                        new_p[1] = -1 * (2 * env.lane_cw - delta_y)
                        new_p[2] = 3
                    else:
                        new_p[0] = -1 * env.lane_cw
                        new_p[1] = -1 * (-1 * p + 2 * env.lane_cw)
                        new_p[2] = 3
            elif i == 1:
                # 左转
                if intention == 0:
                    # 没到交叉口中
                    if p > env.lane_info[0][1]:
                        new_p[0] = 1 * (p - env.lane_info[0][1] + 2 * env.lane_cw)
                        new_p[1] = 1 * env.lane_cw
                        new_p[2] = 1
                    elif p > 0:
                        beta_temp = p / (3 * env.lane_cw)  # rad
                        delta_y = math.sin(beta_temp) * 3 * env.lane_cw
                        delta_x = math.cos(beta_temp) * 3 * env.lane_cw
                        new_p[0] = -1 * (delta_x - 2 * env.lane_cw)
                        new_p[1] = -1 * (2 * env.lane_cw - delta_y)
                        new_p[2] = 3
                    else:
                        new_p[0] = -1 * env.lane_cw
                        new_p[1] = -1 * (-1 * p + 2 * env.lane_cw)
                        new_p[2] = 3
                # 直行
                elif intention == 1:
                    new_p[0] = p - 2 * env.lane_cw
                    new_p[1] = 1 * env.lane_cw
                    new_p[2] = 1
                # 右转
                else:
                    if p > env.lane_info[2][1]:
                        new_p[0] = 1 * (p - env.lane_info[2][1] + 2 * env.lane_cw)
                        new_p[1] = 1 * env.lane_cw
                        new_p[2] = 1
                    elif p > 0:
                        beta_temp = p / env.lane_cw
                        delta_y = math.sin(beta_temp) * env.lane_cw
                        delta_x = math.cos(beta_temp) * env.lane_cw
                        new_p[0] = 1 * (2 * env.lane_cw - delta_x)
                        new_p[1] = 1 * (2 * env.lane_cw - delta_y)
                        new_p[2] = 2
                    else:
                        new_p[0] = 1 * env.lane_cw
                        new_p[1] = -1 * p + 2 * env.lane_cw
                        new_p[2] = 2
            elif i == 2:
                # 左转
                if intention == 0:
                    # 没到交叉口中
                    if p > env.lane_info[0][1]:
                        new_p[0] = 1 * env.lane_cw
                        new_p[1] = -1 * (p - env.lane_info[0][1] + 2 * env.lane_cw)
                        new_p[2] = 2
                    elif p > 0:
                        beta_temp = p / (3 * env.lane_cw)  # rad
                        delta_x = math.sin(beta_temp) * 3 * env.lane_cw
                        delta_y = math.cos(beta_temp) * 3 * env.lane_cw
                        new_p[0] = 1 * (delta_x - 2 * env.lane_cw)
                        new_p[1] = -1 * (2 * env.lane_cw - delta_y)
                        new_p[2] = 1
                    else:
                        new_p[0] = -1 * (-1 * p + 2 * env.lane_cw)
                        new_p[1] = env.lane_cw
                        new_p[2] = 1
                # 直行
                elif intention == 1:
                    new_p[0] = env.lane_cw
                    new_p[1] = -1 * p + 2 * env.lane_cw
                    new_p[2] = 2
                # 右转
                else:
                    if p > env.lane_info[2][1]:
                        new_p[0] = 1 * env.lane_cw
                        new_p[1] = -1 * (p - env.lane_info[2][1] + 2 * env.lane_cw)
                        new_p[2] = 2
                    elif p > 0:
                        beta_temp = p / env.lane_cw
                        delta_x = math.sin(beta_temp) * env.lane_cw
                        delta_y = math.cos(beta_temp) * env.lane_cw
                        new_p[0] = 1 * (2 * env.lane_cw - delta_x)
                        new_p[1] = -1 * (2 * env.lane_cw - delta_y)
                        new_p[2] = 0
                    else:
                        new_p[0] = -1 * p + 2 * env.lane_cw
                        new_p[1] = -1 * env.lane_cw
                        new_p[2] = 0
            else:
                # 左转
                if intention == 0:
                    # 没到交叉口中
                    if p > env.lane_info[0][1]:
                        new_p[0] = -1 * env.lane_cw
                        new_p[1] = 1 * (p - env.lane_info[0][1] + 2 * env.lane_cw)
                        new_p[2] = 3
                    elif p > 0:
                        beta_temp = p / (3 * env.lane_cw)  # rad
                        delta_x = math.sin(beta_temp) * 3 * env.lane_cw
                        delta_y = math.cos(beta_temp) * 3 * env.lane_cw
                        new_p[0] = -1 * (delta_x - 2 * env.lane_cw)
                        new_p[1] = 1 * (2 * env.lane_cw - delta_y)
                        new_p[2] = 0
                    else:
                        new_p[0] = 1 * (-1 * p + 2 * env.lane_cw)
                        new_p[1] = -1 * env.lane_cw
                        new_p[2] = 0
                # 直行
                elif intention == 1:
                    new_p[0] = -1 * env.lane_cw
                    new_p[1] = p - 2 * env.lane_cw
                    new_p[2] = 3
                # 右转
                else:
                    if p > env.lane_info[2][1]:
                        new_p[0] = -1 * env.lane_cw
                        new_p[1] = 1 * (p - env.lane_info[2][1] + 2 * env.lane_cw)
                        new_p[2] = 3
                    elif p > 0:
                        beta_temp = p / env.lane_cw
                        delta_x = math.sin(beta_temp) * env.lane_cw
                        delta_y = math.cos(beta_temp) * env.lane_cw
                        new_p[0] = -1 * (2 * env.lane_cw - delta_x)
                        new_p[1] = 1 * (2 * env.lane_cw - delta_y)
                        new_p[2] = 1
                    else:
                        new_p[0] = -1 * (-1 * p + 2 * env.lane_cw)
                        new_p[1] = 1 * env.lane_cw
                        new_p[2] = 1
        elif env.lane_num == 8:
            if i == 0:
                # 左转
                if intention == 0:
                    # 没到交叉口中
                    if p > env.lane_info[0][1]:
                        new_p[0] = 1 * (p - env.lane_info[0][1] + 4 * env.lane_cw)
                        new_p[1] = 1 * env.lane_cw
                        # new_p[2] = 3.1415
                        new_p[2] = 0
                    elif p > 0:
                        beta_temp = p / (5 * env.lane_cw)  # rad
                        delta_y = math.sin(beta_temp) * 5 * env.lane_cw
                        delta_x = math.cos(beta_temp) * 5 * env.lane_cw
                        new_p[0] = -1 * (delta_x - 4 * env.lane_cw)
                        new_p[1] = -1 * (4 * env.lane_cw - delta_y)
                        # new_p[2] = 3.1415 / 2 - beta_temp + 3.1415
                        new_p[2] = 2
                    else:
                        new_p[0] = -1 * env.lane_cw
                        new_p[1] = -1 * (-1 * p + 4 * env.lane_cw)
                        # new_p[2] = 1.5 * 3.1415
                        new_p[2] = 2
                # 直行
                elif intention == 1:
                    new_p[0] = p - 4 * env.lane_cw
                    new_p[1] = 1 * env.lane_cw
                    # new_p[2] = 3.1415
                    new_p[2] = 0
            elif i == 1:
                # 直行
                if intention == 1:
                    new_p[0] = p - 4 * env.lane_cw
                    new_p[1] = 3 * env.lane_cw
                    # new_p[2] = 3.1415
                    new_p[2] = 1
                # 右转
                elif intention == 2:
                    if p > env.lane_info[2][1]:
                        new_p[0] = 1 * (p - env.lane_info[2][1] + 4 * env.lane_cw)
                        new_p[1] = 3 * env.lane_cw
                        # new_p[2] = 3.1415
                        new_p[2] = 1
                    elif p > 0:
                        beta_temp = p / env.lane_cw
                        delta_y = math.sin(beta_temp) * env.lane_cw
                        delta_x = math.cos(beta_temp) * env.lane_cw
                        new_p[0] = 1 * (4 * env.lane_cw - delta_x)
                        new_p[1] = 1 * (4 * env.lane_cw - delta_y)
                        # new_p[2] = beta_temp + 3.1415 / 2
                        new_p[2] = 7
                    else:
                        new_p[0] = 3 * env.lane_cw
                        new_p[1] = -1 * p + 4 * env.lane_cw
                        # new_p[2] = 3.1415 / 2
                        new_p[2] = 7
            elif i == 2:
                # 左转
                if intention == 0:
                    # 没到交叉口中
                    if p > env.lane_info[0][1]:
                        new_p[0] = -1 * env.lane_cw
                        new_p[1] = 1 * (p - env.lane_info[0][1] + 4 * env.lane_cw)
                        # new_p[2] = 1.5 * 3.1415
                        new_p[2] = 2
                    elif p > 0:
                        beta_temp = p / (5 * env.lane_cw)  # rad
                        delta_x = math.sin(beta_temp) * 5 * env.lane_cw
                        delta_y = math.cos(beta_temp) * 5 * env.lane_cw
                        new_p[0] = -1 * (delta_x - 4 * env.lane_cw)
                        new_p[1] = 1 * (4 * env.lane_cw - delta_y)
                        # new_p[2] = 3.1415 / 2 - beta_temp + 1.5 * 3.1415
                        new_p[2] = 4
                    else:
                        new_p[0] = 1 * (-1 * p + 4 * env.lane_cw)
                        new_p[1] = -1 * env.lane_cw
                        # new_p[2] = 0
                        new_p[2] = 4
                # 直行
                elif intention == 1:
                    new_p[0] = -1 * env.lane_cw
                    new_p[1] = p - 4 * env.lane_cw
                    # new_p[2] = 1.5 * 3.1415
                    new_p[2] = 2
            elif i == 3:
                # 直行
                if intention == 1:
                    new_p[0] = -3 * env.lane_cw
                    new_p[1] = p - 4 * env.lane_cw
                    # new_p[2] = 1.5 * 3.1415
                    new_p[2] = 3
                # 右转
                elif intention == 2:
                    if p > env.lane_info[2][1]:
                        new_p[0] = -3 * env.lane_cw
                        new_p[1] = 1 * (p - env.lane_info[2][1] + 4 * env.lane_cw)
                        # new_p[2] = 1.5 * 3.1415
                        new_p[2] = 3
                    elif p > 0:
                        beta_temp = p / env.lane_cw
                        delta_x = math.sin(beta_temp) * env.lane_cw
                        delta_y = math.cos(beta_temp) * env.lane_cw
                        new_p[0] = -1 * (4 * env.lane_cw - delta_x)
                        new_p[1] = 1 * (4 * env.lane_cw - delta_y)
                        # new_p[2] = beta_temp + 3.1415
                        new_p[2] = 1
                    else:
                        new_p[0] = -1 * (-1 * p + 4 * env.lane_cw)
                        new_p[1] = 3 * env.lane_cw
                        # new_p[2] = 3.1415
                        new_p[2] = 1
            elif i == 4:
                # 左转
                if intention == 0:
                    # 没到交叉口中
                    if p > env.lane_info[0][1]:
                        new_p[0] = -1 * (p - env.lane_info[0][1] + 4 * env.lane_cw)
                        new_p[1] = -1 * env.lane_cw
                        # new_p[2] = 0
                        new_p[2] = 4
                    elif p > 0:
                        beta_temp = p / (5 * env.lane_cw)  # rad
                        delta_y = math.sin(beta_temp) * 5 * env.lane_cw
                        delta_x = math.cos(beta_temp) * 5 * env.lane_cw
                        new_p[0] = 1 * (delta_x - 4 * env.lane_cw)
                        new_p[1] = 1 * (4 * env.lane_cw - delta_y)
                        # new_p[2] = 3.1415 / 2 - beta_temp
                        new_p[2] = 6
                    else:
                        new_p[0] = env.lane_cw
                        new_p[1] = -1 * p + 4 * env.lane_cw
                        # new_p[2] = 3.1415 / 2
                        new_p[2] = 6
                # 直行
                elif intention == 1:
                    new_p[0] = -1 * p + 4 * env.lane_cw
                    new_p[1] = -1 * env.lane_cw
                    # new_p[2] = 0
                    new_p[2] = 4
            elif i == 5:
                # 直行
                if intention == 1:
                    new_p[0] = -1 * p + 4 * env.lane_cw
                    new_p[1] = -3 * env.lane_cw
                    # new_p[2] = 0
                    new_p[2] = 5
                # 右转
                elif intention == 2:
                    if p > env.lane_info[2][1]:
                        new_p[0] = -1 * (p - env.lane_info[2][1] + 4 * env.lane_cw)
                        new_p[1] = -3 * env.lane_cw
                        # new_p[2] = 0
                        new_p[2] = 5
                    elif p > 0:
                        beta_temp = p / env.lane_cw
                        delta_y = math.sin(beta_temp) * env.lane_cw
                        delta_x = math.cos(beta_temp) * env.lane_cw
                        new_p[0] = -1 * (4 * env.lane_cw - delta_x)
                        new_p[1] = -1 * (4 * env.lane_cw - delta_y)
                        # new_p[2] = beta_temp + 1.5 * 3.1415
                        new_p[2] = 3
                    else:
                        new_p[0] = -3 * env.lane_cw
                        new_p[1] = -1 * (-1 * p + 4 * env.lane_cw)
                        # new_p[2] = 1.5 * 3.1415
                        new_p[2] = 3
            elif i == 6:
                # 左转
                if intention == 0:
                    # 没到交叉口中
                    if p > env.lane_info[0][1]:
                        new_p[0] = 1 * env.lane_cw
                        new_p[1] = -1 * (p - env.lane_info[0][1] + 4 * env.lane_cw)
                        # new_p[2] = 3.1415 / 2
                        new_p[2] = 6
                    elif p > 0:
                        beta_temp = p / (5 * env.lane_cw)  # rad
                        delta_x = math.sin(beta_temp) * 5 * env.lane_cw
                        delta_y = math.cos(beta_temp) * 5 * env.lane_cw
                        new_p[0] = 1 * (delta_x - 4 * env.lane_cw)
                        new_p[1] = -1 * (4 * env.lane_cw - delta_y)
                        # new_p[2] = 3.1415 / 2 - beta_temp + 3.1415 / 2
                        new_p[2] = 0
                    else:
                        new_p[0] = -1 * (-1 * p + 4 * env.lane_cw)
                        new_p[1] = 1 * env.lane_cw
                        # new_p[2] = 3.1415
                        new_p[2] = 0
                # 直行
                elif intention == 1:
                    new_p[0] = 1 * env.lane_cw
                    new_p[1] = -1 * p + 4 * env.lane_cw
                    # new_p[2] = 3.1415 / 2
                    new_p[2] = 6
            elif i == 7:
                # 直行
                if intention == 1:
                    new_p[0] = 3 * env.lane_cw
                    new_p[1] = -1 * p + 4 * env.lane_cw
                    # new_p[2] = 3.1415 / 2
                    new_p[2] = 7
                # 右转
                elif intention == 2:
                    if p > env.lane_info[2][1]:
                        new_p[0] = 3 * env.lane_cw
                        new_p[1] = -1 * (p - env.lane_info[2][1] + 4 * env.lane_cw)
                        # new_p[2] = 3.1415 / 2
                        new_p[2] = 7
                    elif p > 0:
                        beta_temp = p / env.lane_cw
                        delta_x = math.sin(beta_temp) * env.lane_cw
                        delta_y = math.cos(beta_temp) * env.lane_cw
                        new_p[0] = 1 * (4 * env.lane_cw - delta_x)
                        new_p[1] = -1 * (4 * env.lane_cw - delta_y)
                        # new_p[2] = beta_temp
                        new_p[2] = 5
                    else:
                        new_p[0] = -1 * p + 4 * env.lane_cw
                        new_p[1] = -3 * env.lane_cw
                        # new_p[2] = 0
                        new_p[2] = 5
        elif env.lane_num == 12:
            rotation_angle = 3.141593 / 2 * int(i / 3)
            p_temp = [0, 0, 0]
            lane_map = [3, 1, 10, 6, 4, 1, 9, 7, 4, 0, 10, 7]
            if i % 3 == 0:
                if p > env.lane_info[0][1]:
                    yaw = (3.1415 + i * (3.1415 / 6)) % (2 * 3.1415)
                    p_temp = [p - env.lane_info[0][1] + 6 * env.lane_cw, env.lane_cw, i]
                elif p > 0:
                    yaw = (3.1415 + i * (3.1415 / 6)) % (2 * 3.1415)
                    r_a = (env.lane_info[0][1] - p) / env.lane_info[0][1] * 3.141593 / 2
                    p0 = [6 * env.lane_cw, env.lane_cw]
                    p_r = [6 * env.lane_cw, -6 * env.lane_cw]
                    p_temp[0] = p_r[0] + (p0[0] - p_r[0]) * np.cos(r_a) - (p0[1] - p_r[1]) * np.sin(r_a)
                    p_temp[1] = p_r[1] + (p0[1] - p_r[1]) * np.cos(r_a) + (p0[0] - p_r[0]) * np.sin(r_a)
                    p_temp[2] = lane_map[i]
                else:
                    yaw = (1.5 * 3.1415 + i * (3.1415 / 6)) % (2 * 3.1415)
                    p_temp = [-env.lane_cw, -6 * env.lane_cw + p, lane_map[i]]
            elif i % 3 == 1:
                yaw = (3.1415 + (i - 1) * (3.1415 / 6)) % (2 * 3.1415)
                p_temp = [p - 6 * env.lane_cw, 3 * env.lane_cw, i]
            else:
                if p > env.lane_info[2][1]:
                    yaw = (3.1415 + (i - 2) * (3.1415 / 6)) % (2 * 3.1415)
                    p_temp = [p - env.lane_info[2][1] + 6 * env.lane_cw, 5 * env.lane_cw, i]
                elif p > 0:
                    yaw = (0.5 * 3.1415 + (i - 2) * (3.1415 / 6)) % (2 * 3.1415)
                    r_a = (env.lane_info[2][1] - p) / env.lane_info[2][1] * 3.141593 / 2
                    p0 = [6 * env.lane_cw, 5 * env.lane_cw]
                    p_r = [6 * env.lane_cw, 6 * env.lane_cw]
                    # 绕点p_r逆时顺时针旋转r_a角度
                    p_temp[0] = p_r[0] + (p0[0] - p_r[0]) * np.cos(r_a) + (p0[1] - p_r[1]) * np.sin(r_a)
                    p_temp[1] = p_r[1] + (p0[1] - p_r[1]) * np.cos(r_a) - (p0[0] - p_r[0]) * np.sin(r_a)
                    p_temp[2] = lane_map[i]
                else:
                    yaw = (0.5 * 3.1415 + (i - 2) * (3.1415 / 6)) % (2 * 3.1415)
                    p_temp = [5 * env.lane_cw, 6 * env.lane_cw - p, lane_map[i]]
            new_p[0] = ((p_temp[0]) * np.cos(rotation_angle) - (p_temp[1]) * np.sin(rotation_angle))
            new_p[1] = ((p_temp[1]) * np.cos(rotation_angle) + (p_temp[0]) * np.sin(rotation_angle))
            new_p[2] = p_temp[2]
        return new_p

    def show(self, env, i):
        plt.figure(1, figsize=(9.6, 9.6), dpi=100)
        if env.lane_num == 3:
            plt.plot([-self.control_dis, self.control_dis], [0, 0], c='y', ls='--')
            plt.plot([0, 0], [2 * self.lane_w, -self.control_dis], c='y', ls='--')
            plt.plot([-self.control_dis, self.control_dis], [2 * self.lane_w, 2 * self.lane_w], c='b', ls='-')
            plt.plot([-self.control_dis, self.control_dis], [-2 * self.lane_w, -2 * self.lane_w], c='b', ls='-')
            plt.plot([2 * self.lane_w, 2 * self.lane_w], [2 * self.lane_w, -self.control_dis], c='b', ls='-')
            plt.plot([-2 * self.lane_w, -2 * self.lane_w], [2 * self.lane_w, -self.control_dis], c='b', ls='-')
        if env.lane_num == 4:
            plt.plot([-self.control_dis, self.control_dis], [0, 0], c='y', ls='--')
            plt.plot([0, 0], [self.control_dis, -self.control_dis], c='y', ls='--')
            plt.plot([-self.control_dis, self.control_dis], [2 * self.lane_w, 2 * self.lane_w], c='b', ls='-')
            plt.plot([2 * self.lane_w, 2 * self.lane_w], [self.control_dis, -self.control_dis], c='b', ls='-')
            plt.plot([-self.control_dis, self.control_dis], [-2 * self.lane_w, -2 * self.lane_w], c='b', ls='-')
            plt.plot([-2 * self.lane_w, -2 * self.lane_w], [self.control_dis, -self.control_dis], c='b', ls='-')
        if env.lane_num == 8:
            plt.plot([-self.control_dis, self.control_dis], [0, 0], c='y', ls='--')
            plt.plot([0, 0], [self.control_dis, -self.control_dis], c='y', ls='--')
            plt.plot([-self.control_dis, self.control_dis], [2 * self.lane_w, 2 * self.lane_w], c='b', ls='-')
            plt.plot([2 * self.lane_w, 2 * self.lane_w], [self.control_dis, -self.control_dis], c='b', ls='-')
            plt.plot([-self.control_dis, self.control_dis], [-2 * self.lane_w, -2 * self.lane_w], c='b', ls='-')
            plt.plot([-2 * self.lane_w, -2 * self.lane_w], [self.control_dis, -self.control_dis], c='b', ls='-')
            plt.plot([-self.control_dis, self.control_dis], [4 * self.lane_w, 4 * self.lane_w], c='b', ls='-')
            plt.plot([-self.control_dis, self.control_dis], [-4 * self.lane_w, -4 * self.lane_w], c='b', ls='-')
            plt.plot([4 * self.lane_w, 4 * self.lane_w], [self.control_dis, -self.control_dis], c='b', ls='-')
            plt.plot([-4 * self.lane_w, -4 * self.lane_w], [self.control_dis, -self.control_dis], c='b', ls='-')
        if env.lane_num == 12:
            plt.plot([-self.control_dis, self.control_dis], [0, 0], c='y', ls='--')
            plt.plot([0, 0], [self.control_dis, -self.control_dis], c='y', ls='--')
            plt.plot([-self.control_dis, self.control_dis], [2 * self.lane_w, 2 * self.lane_w], c='b', ls='--',
                     linewidth=1)
            plt.plot([2 * self.lane_w, 2 * self.lane_w], [self.control_dis, -self.control_dis], c='b', ls='--',
                     linewidth=1)
            plt.plot([-self.control_dis, self.control_dis], [-2 * self.lane_w, -2 * self.lane_w], c='b', ls='--',
                     linewidth=1)
            plt.plot([-2 * self.lane_w, -2 * self.lane_w], [self.control_dis, -self.control_dis], c='b', ls='--',
                     linewidth=1)
            plt.plot([-self.control_dis, self.control_dis], [4 * self.lane_w, 4 * self.lane_w], c='b', ls='--',
                     linewidth=1)
            plt.plot([-self.control_dis, self.control_dis], [-4 * self.lane_w, -4 * self.lane_w], c='b', ls='--',
                     linewidth=1)
            plt.plot([4 * self.lane_w, 4 * self.lane_w], [self.control_dis, -self.control_dis], c='b', ls='--',
                     linewidth=1)
            plt.plot([-4 * self.lane_w, -4 * self.lane_w], [self.control_dis, -self.control_dis], c='b', ls='--',
                     linewidth=1)
            plt.plot([-self.control_dis, self.control_dis], [6 * self.lane_w, 6 * self.lane_w], c='b', ls='-')
            plt.plot([-self.control_dis, self.control_dis], [-6 * self.lane_w, -6 * self.lane_w], c='b', ls='-')
            plt.plot([6 * self.lane_w, 6 * self.lane_w], [self.control_dis, -self.control_dis], c='b', ls='-')
            plt.plot([-6 * self.lane_w, -6 * self.lane_w], [self.control_dis, -self.control_dis], c='b', ls='-')
        # plt.plot(self.px[0], self.py[0], c='r', ls='', marker='4')  # 画出当前 ax 列表和 ay 列表中的值的图形
        # plt.plot(self.px[1], self.py[1], c='r', ls='', marker='3')  # 画出当前 ax 列表和 ay 列表中的值的图形
        # plt.plot(self.px[2], self.py[2], c='r', ls='', marker='2')  # 画出当前 ax 列表和 ay 列表中的值的图形
        # plt.plot(self.px[3], self.py[3], c='r', ls='', marker='1')  # 画出当前 ax 列表和 ay 列表中的值的图形
        if self.l_mode == "actual":
            for lane in range(env.lane_num):
                for veh_id, veh in enumerate(env.veh_info[lane]):
                    p = self.get_p(veh["p"], lane, env, env.veh_info[lane][veh_id]["intention"])
                    c_level = [max(float(env.veh_info[lane][veh_id]["v"] - 10) / float(env.vM - 10), 0), 0,
                               0]  # 用颜色体现速度大小
                    plt.plot(p[0], p[1], c=c_level, ls='', marker=self.marker[p[2]],
                             markersize=7)  # 画出当前 ax 列表和 ay 列表中的值的图形
        elif self.l_mode == "virtual":
            for lane in range(env.lane_num):
                for index, direction in enumerate(env.direction[lane]):
                    if direction == -1:
                        continue
                    for item in env.virtual_lane_4[direction]:
                        p_actual = self.get_p(item[0], lane, env, index)
                        c_level = max(float(env.veh_info[item[1]][item[2]]["v"] - 10) / float(env.vM - env.vm), 0)
                        plt.plot(p_actual[0], p_actual[1], c=[c_level, 0, 0], ls='', marker=self.marker[item[1]],
                                 markersize=7)
        plt.xlim((-self.control_dis + 5, self.control_dis + 5))
        plt.ylim((-self.control_dis + 5, self.control_dis + 5))
        if not os.path.exists("result_imgs"):
            os.makedirs("result_imgs")
        plt.savefig("result_imgs/%s.png" % i)
        plt.close()
        # plt.show()
