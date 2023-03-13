from node import Node, element_per_packet
from prune_tool import Patch, dump_patches, load_patches
from cnn import SparseCNN
from packet import Packet
from typing import List
import torch


def pack(sender: Node, round_id: int,  mini_model: SparseCNN, patches_list: List[List[Patch]]):
    """
    return
    1. group_len_meta
    2. mini_model_meta
    3. patches_meta_list
    4. pkt_list
    """
    pkt_list: List[Packet] = []
    seg_id = 0
    group_len_meta: List[tuple] = []  # 第 i 组的 (包数量，真实长度)

    # 打包模型
    (mini_model_meta, mini_model_data) = mini_model.dump()
    mini_model_data = mini_model_data.detach().numpy()

    model_data_len = len(mini_model_data)
    mini_model_packet_num = int(model_data_len / element_per_packet)
    mini_model_packet_num += 0 if model_data_len % element_per_packet == 0 else 1
    group_len_meta.append((mini_model_packet_num, model_data_len))
    for i in range(mini_model_packet_num):
        pkt_list.append(
            sender.create_packet(
                round_id,
                seg_id,
                len(group_len_meta),
                False,
                mini_model_data[i * element_per_packet:],
                sender.type == "server"
            )
        )
        seg_id += 1

    # 打包patch
    patches_data_list = []
    patches_meta_list = []
    for patches in reversed(patches_list):
        patches_meta, patches_data = dump_patches(patches)
        patches_meta_list.append(patches_meta)
        patches_data_list.append(patches_data)
    for data in patches_data_list:
        data = data.detach().numpy()
        data_len = len(data)
        packet_num = int(data_len / element_per_packet)
        packet_num += 0 if data_len % element_per_packet == 0 else 1
        group_len_meta.append((packet_num, data_len))
        for i in range(packet_num):
            pkt_list.append(
                sender.create_packet(
                    round_id,
                    seg_id,
                    len(group_len_meta),
                    False,
                    data[i * element_per_packet:],
                    sender.type == "server"
                )
            )
            seg_id += 1

    return group_len_meta, mini_model_meta, patches_meta_list, pkt_list


def unpack(
    group_len_meta: List[int],
    mini_model_meta: dict,
    patches_meta_list: List[dict],
    pkt_list: List[Packet]
):
    """
    return
    1. SparseCNN
    2. patches_list
    """
    cursor = 0
    # 解包 mini_model,
    mini_model_data = torch.cat(
        [(torch.tensor(pkt.tensor) / pkt.aggregate_num) for pkt in pkt_list[cursor: cursor + group_len_meta[0][0]]])
    cursor += group_len_meta[0][0]

    # 解包 patch
    patches_list = []
    for (pkt_len, actual_len), patches_meta in zip(group_len_meta[1:], patches_meta_list):
        if cursor + pkt_len > len(pkt_list):
            break
        t = torch.cat(
            [(torch.tensor(pkt.tensor) / pkt.aggregate_num) for pkt in pkt_list[cursor: cursor + pkt_len]])[:actual_len]
        patches_list.append(load_patches(patches_meta, t))
        cursor += pkt_len
    return SparseCNN(mini_model_meta, mini_model_data), patches_list
