from python_io.server import Server
from python_io.client import Client
from python_io.node import Node
from prune_tool import Patch
from cnn import SparseCNN
from typing import List
from .utils import pack, unpack

USE_SWITCH = True
RECV_BUFFER = 1024 * 10 # 10MB


class EServer(Server):
    def __init__(self, ip_addr: str, rx_port: int, tx_port: int, rpc_addr: str, node_id: int, is_remote_node: bool, iface: str = ""):
        super().__init__(ip_addr, rx_port, tx_port, rpc_addr, node_id, is_remote_node, iface)
        self.round_id = 0
    
    def multicast_model(self, node: Node, mini_model: SparseCNN, patches_list: List[List[Patch]]):
        """
        下发模型，如果 node 为 switch 将会进行多播

        patches_list 必须严格按照剪枝顺序
        """
        group_len_meta, mini_model_meta, patches_meta_list, pkt_list = pack(
            sender=self,
            round_id=self.round_id,
            mini_model=mini_model,
            patches_list=patches_list
        )
        meta = {
            "group_len_meta": group_len_meta,
            "mini_model_meta": mini_model_meta,
            "patches_meta_list": patches_meta_list
        }
        # 保存当前 node 正在跑的模型的元信息，用于接收参数
        node.last_multicast_meta = meta
        self.send(node, self.round_id, pkt_list, meta, [pkt_len for pkt_len, _ in group_len_meta])
        return meta, pkt_list

    def receive_model(self, node: Node) -> SparseCNN:
        """
        receive from client or switch
        """
        pkt_list, _ = self.receive(node, self.round_id, RECV_BUFFER)
        meta = node.last_multicast_meta
        sparse_model, patches_list = unpack(
            meta["group_len_meta"],
            meta["mini_model_meta"],
            meta["patches_meta_list"],
            pkt_list
        )
        for patches in patches_list:
            sparse_model.apply_patches(patches)
        return sparse_model

class EClient(Client):
    def __init__(self, ip_addr: str, rx_port: int, tx_port: int, rpc_addr: str, node_id: int, is_remote_node: bool, max_group_id:int , iface: str = ""):
        super().__init__(ip_addr, rx_port, tx_port, rpc_addr, node_id, is_remote_node, max_group_id, iface)
        self.round_id = 0
        self.last_patches = None

    def reduce_model(self, server: Server, model: SparseCNN):
        """
        本地迭代完成后向 PS 发送模型，不需要传输 meta 因为 ps 维护了当前节点上一次用的 meta

        返回剪枝后的 patches 用于调试
        """
        patches = [model.redo_prune(old_patches)
                   for old_patches in self.last_patches]
        group_len_meta, mini_model_meta, patches_meta_list, pkt_list = pack(
            sender=self, 
            round_id=self.round_id, 
            mini_model=model, 
            patches_list=patches
        )
        self.send(server, self.round_id, pkt_list, {}, USE_SWITCH)
        return patches
    

    def receive_model(self, server: Server) -> SparseCNN:
        pkt_list, meta = self.receive(server, self.round_id, RECV_BUFFER)
        sparse_model, patches_list = unpack(
            meta["group_len_meta"],
            meta["mini_model_meta"],
            meta["patches_meta_list"],
            pkt_list
        )
        for patches in patches_list:
            sparse_model.apply_patches(patches)
        # 保存 patches 为后续 reduce_model 做准备
        self.last_patches = reversed(patches_list)
        return sparse_model