# encoding:utf-8
# author:WangQiuShi
# data:2020/9/21 16:45
import numpy as np
import math

# 对cifar10使用spld算法 --  loss计算使用的是ResNet18 预测出的类别概率 与真实值的差值
# group_member_ship使用cifar10数据集自带的label 共10类 标签为0-9

# lam2 = 2
# gam2 = 2
# u11 = 0.5
# u22 = 0.5


# def SPLD(loss, group_member_ship, lam, gamma):
#     group_idx = np.array(list(set(group_member_ship)))
#     b = len(group_idx)
#     selected_idx = np.array([0] * len(loss))
#     selected_scores = np.array([0] * len(loss))
#     loss_ingroup = []
#     result = []
#
#     for j in range(b):
#         idx_ingroup = np.where(groupmembership == group_idx[j])[0]
#         for idx in idx_ingroup:
#             loss_ingroup.append(loss[idx])
#         sorted_loss_ingroup = sorted(enumerate(loss_ingroup), key=lambda x: x[1])
#         rank_index = [i[0] for i in sorted_loss_ingroup]
#
#         nj = len(idx_ingroup)
#         for i in range(nj):
#             if loss_ingroup[i] < (lam + gamma / (np.sqrt(i) + np.sqrt(i - 1))):
#                 selected_idx[idx_ingroup[i]] = 1
#             else:
#                 selected_idx[idx_ingroup[i]] = 0
#     selected_idx = np.where(selected_idx == 1)[0]
#     for ii in selected_idx:
#         selected_scores[ii] = loss[ii]
#     sorted_result_ingroup = sorted(enumerate(selected_scores), key=lambda x: x[1])
#     index_result = [i[0] for i in sorted_result_ingroup]
#     for iii in index_result:
#         result.append(selected_idx[iii])
#     return result

def spld(loss, group_member_ship, lam, gamma):
    groups_labels = np.array(list(set(group_member_ship)))
    b = len(groups_labels)
    selected_idx = []
    selected_score = [0] * len(loss)
    for j in range(b):
        idx_in_group = np.where(group_member_ship == groups_labels[j])[0]
        print(idx_in_group.shape)
        print(idx_in_group)
        loss_in_group = []
        # print(type(idx_in_group))
        for idx in idx_in_group:
            loss_in_group.append(loss[idx])
        idx_loss_dict = dict()
        for i in idx_in_group:
            idx_loss_dict[i] = loss[int(i)]
        sorted_idx_in_group = sorted(idx_loss_dict.keys(), key=lambda s: idx_loss_dict[s])
        sorted_idx_in_group_arr = np.array(sorted_idx_in_group)

        for (i, ii) in enumerate(sorted_idx_in_group_arr):
            if loss[ii] < (lam + gamma / (np.sqrt(i + 1) + np.sqrt(i))):
                selected_idx.append(ii)
            else:
                pass
            selected_score[ii] = loss[ii] - (lam + gamma / (np.sqrt(i + 1) + np.sqrt(i)))

    selected_idx_arr = np.array(selected_idx)
    selected_idx_and_new_loss_dict = dict()
    for idx in selected_idx_arr:
        selected_idx_and_new_loss_dict[idx] = selected_score[idx]

    sorted_idx_in_selected_samples = sorted(selected_idx_and_new_loss_dict.keys(),
                                            key=lambda s: selected_idx_and_new_loss_dict[s])

    sorted_idx_in_selected_samples_arr = np.array(sorted_idx_in_selected_samples)
    return sorted_idx_in_selected_samples_arr


def spld_reverse(loss, group_member_ship, lam, gamma):
    groups_labels = np.array(list(set(group_member_ship)))
    b = len(groups_labels)
    selected_idx = []
    selected_score = [0] * len(loss)
    for j in range(b):
        idx_in_group = np.where(group_member_ship == groups_labels[j])[0]
        # print(idx_in_group)
        loss_in_group = []
        # print(type(idx_in_group))
        for idx in idx_in_group:
            loss_in_group.append(loss[idx])
        idx_loss_dict = dict()
        for i in idx_in_group:
            idx_loss_dict[i] = loss[int(i)]
        sorted_idx_in_group = sorted(idx_loss_dict.keys(), key=lambda s: idx_loss_dict[s],reverse=True)
        sorted_idx_in_group_arr = np.array(sorted_idx_in_group)

        for (i, ii) in enumerate(sorted_idx_in_group_arr):
            if loss[ii] > (lam + gamma / (np.sqrt(i + 1) + np.sqrt(i))):
                selected_idx.append(ii)
            else:
                pass
            selected_score[ii] = -loss[ii] + (lam + gamma / (np.sqrt(i + 1) + np.sqrt(i)))

    selected_idx_arr = np.array(selected_idx)
    selected_idx_and_new_loss_dict = dict()
    for idx in selected_idx_arr:
        selected_idx_and_new_loss_dict[idx] = selected_score[idx]

    sorted_idx_in_selected_samples = sorted(selected_idx_and_new_loss_dict.keys(),
                                            key=lambda s: selected_idx_and_new_loss_dict[s])

    sorted_idx_in_selected_samples_arr = np.array(sorted_idx_in_selected_samples)
    # if len(sorted_idx_in_selected_samples_arr) > need_nums:
    return sorted_idx_in_selected_samples_arr
    # else:
    #     lam = u1 * lam
    #     gamma = u2 * gamma
    #     spld_reverse(loss, group_member_ship, lam, gamma, need_nums, u1, u2)


if __name__ == '__main__':
    samples = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n']
    group_member_ship = [1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4]
    loss = [0.5, 0.6, 0.7, 0.8, 1.2, 0.9, 1.0, 2.3, 1.8, 0.7, 3.9, 0.8, 0.9, 0.8]

    ########################################方法一#######################################

    # lam = -0.9
    # gam = -0.1
    # u1 = 0.9
    # u2 = 0.9
    # new_loss = []
    # for lo in loss:
    #     new_loss.append(-1 * lo)
    # print(new_loss)
    selected_samples = []
    new_selected_samples = []

    selected_idx_arr = spld(loss, group_member_ship, 1, 1)
    # new_selected_idx_arr = spld(new_loss, group_member_ship, lam, gam)

    for selected_idx in selected_idx_arr:
        selected_samples.append(samples[selected_idx])
    print("selected samples are:{}".format(",".join(selected_samples)))
    # for _ in range(10):
    #     new_selected_samples = []
    #
    #     new_selected_idx_arr = spld(new_loss, group_member_ship, lam, gam)
    #     for selected_idx in new_selected_idx_arr:
    #         new_selected_samples.append(samples[selected_idx])
    #     print("selected samples are:{}".format(",".join(new_selected_samples)))
    #
    #     lam = u1*lam
    #     gam = u2*gam
    #######################################方法二########################################
    # lam = 10
    # gam = 10
    # u1 = 0.5
    # u2 = 0.5
    # for _ in range(30):
    #     selected_samples = []
    #     selected_idx_arr = spld_reverse(loss, group_member_ship, lam, gam)
    #     for selected_idx in selected_idx_arr:
    #         selected_samples.append(samples[selected_idx])
    #     print("selected samples are:{}".format(",".join(selected_samples)))
    #     lam = u1 * lam
    #     gam = u2 * gam

    # print("=======================================================")
    # lam2 = 2
    # gam2 = 2
    # # u11 = 0.5
    # # u22 = 0.5
    # # selected_samples = []
    # selected_idx_arr = []
    # for i in range(10):
    #     selected_samples = []
    #     selected_idx_arr = spld_reverse(loss, group_member_ship, lam2, gam2)
    #     # if len(selected_idx_arr) > 10:
    #     #     break
    #     for selected_idx in selected_idx_arr:
    #         selected_samples.append(samples[selected_idx])
    #     print("selected samples are:{}".format(",".join(selected_samples)))
    #     # else:
    #     # lam2 = (1 - math.exp(-1 * i)) * lam2
    #     # gam2 = (1 - math.exp(-1 * i)) * gam2
    #
    #     lam2 = (math.exp(-1 * i)) * lam2
    #     gam2 = (math.exp(-1 * i)) * gam2
    # for selected_idx in selected_idx_arr[:10]:
    #     selected_samples.append(samples[selected_idx])
    # print("selected samples are:{}".format(",".join(selected_samples)))
    # print(selected_idx_arr[:10])

    # groups_labels = np.array(list(set(group_member_ship)))
    # b = len(groups_labels)
    # selected_idx = []
    # for j in range(b):
    #     idx_in_group = np.where(group_member_ship == groups_labels[j])[0]
    #     print(idx_in_group)
    #     loss_in_group = []
    #     # print(type(idx_in_group))
    #     for idx in idx_in_group:
    #         loss_in_group.append(loss[idx])
    #     # loss_in_group_arr = np.array(loss_in_group)
    #     idx_loss_dict = dict()
    #     for i in idx_in_group:
    #         idx_loss_dict[i] = loss[int(i)]
    #     sorted_idx_in_group = sorted(idx_loss_dict.keys(), key=lambda s: idx_loss_dict[s])
    #     sorted_idx_in_group_arr = np.array(sorted_idx_in_group)
    #
    #     # print(sorted_idx_in_group_arr)
    #
    #     nj = len(sorted_idx_in_group_arr)
    #     for i in range(nj):
    #         if loss[i] < (0.03 + 0.2 / (np.sqrt(i + 1) + np.sqrt(i))):
    #             selected_idx.append(sorted_idx_in_group_arr[i])
    #         else:
    #             pass
    #
    # selected_idx_arr = np.array(selected_idx)
    # print(selected_idx_arr)
