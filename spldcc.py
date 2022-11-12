# self-paced learning with diversity 简单实现

import numpy as np


# 对于cifar10数据集  我们可以确定的是group_member_ship 一共10类  并且由 0-9 标签表示


def spld(loss, group_member_ship, lam, gamma):
    """
    self-paced learning with diversity
    :param loss: 样本的损失
    :param group_member_ship:样本对应的group类别
    :param lam: 作为简单项的参数
    :param gamma: 作为多样性的参数
    :return: 返回值为  sorted_idx_in_selected_samples_arr  即被选择出来的样本的下标值
    """
    group_labels = np.array(list(set(group_member_ship)))  # 首先确定组别标签
    label_dim = len(group_labels)  # 确定有几个类别
    selected_idx = []
    selected_score = [0] * len(loss)  # 被选择出来的样本的得分
    for j in range(label_dim):  # 在各个类别中
        idx_in_group = np.where(group_member_ship == group_labels[j])[0]  # 根据下标划分组别
        print("idx_in_group：",idx_in_group)
        loss_in_group = []
        # print(type(idx_in_group))
        for idx in idx_in_group:
            loss_in_group.append(loss[idx])  # 得到每一组中样本的loss 存入到loss_in_group中
            print("loss_in_group：",loss_in_group)
        idx_loss_dict = dict()
        for i in idx_in_group:
            idx_loss_dict[i] = loss[int(i)]  # 将不同组别分别存储为 下标：loss的  字典格式
            print("idx_loss_dict:",idx_loss_dict)
        sorted_idx_in_group = sorted(idx_loss_dict.keys(), key=lambda s: idx_loss_dict[s])
        print("sorted_idx_in_group", sorted_idx_in_group)
        sorted_idx_in_group_arr = np.array(sorted_idx_in_group)

        print("sorted_idx_in_group_arr",sorted_idx_in_group_arr)

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


if __name__ == '__main__':
    samples = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n']
    group_member_ship = [1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4]
    loss = [0.5, 0.6, 0.7, 0.8, 1.2, 0.9, 1.0, 2.3, 1.8, 0.7, 3.9, 0.8, 0.9, 0.8]
    select_samples = []

    select_idx_arr = spld(loss, group_member_ship, 2.4, 16)

    print(select_idx_arr)
    for select_idx in select_idx_arr:
        select_samples.append(samples[select_idx])
    print("selected samples are:{}".format(",".join(select_samples)))
