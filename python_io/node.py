import numpy as np
import socket
import math
import packet
import threading
from packet import Packet
from job import Job
import grpc
from io_pb2_grpc import *
from io_pb2 import *
import typing
from grpc_server import GrpcServer
from packet import elemenet_per_packet, pkt_size, switch_pool_size, retranmission_bitmap
import time

add_delay_ms = 10


class Node:
    def __init__(self, ip_addr: str, port: int, rpc_port: int, node_id: int, is_remote_node: bool):
        self.options = {
            "ip_addr": ip_addr,
            "port": port,
            "rpc_port": rpc_port,
            "node_id": node_id,
            "groups": set([0]),  # 所在的分组
            "speed": 100,  # 100 Mbps
        }
        self.children: dict[int, Node] = {}

        self.rx_jobs: dict[(int, int), Job] = {}
        self.rx_jobs_lock = threading.Lock()

        self.rpc_stub: typing.Union[SwitchmlIOStub, None] = None
        self.rpc_server: typing.Union[GrpcServer, None] = None
        self.rx_socket: typing.Union[socket.socket, None] = None
        if not is_remote_node:
            self.rx_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.rx_socket.bind(
                (self.options['ip_addr'], self.options['port']))
            print("成功监听数据端口 %s:%d" %
                  (self.options['ip_addr'], self.options['port']))
            self.__receive_thread = threading.Thread(
                target=self.receive_thread)
            self.__receive_thread.start()
            print("成功启动接收线程 id=%d" % (self.__receive_thread.ident))
            self.rpc_server: GrpcServer = GrpcServer(self)
            print("成功启动 grpc 服务 %s:%d" %
                  (self.options['ip_addr'], self.options['rpc_port']))
        else:
            self._init_as_remote_node()

    # stop the server
    def _close(self):
        print("grpc 服务正在关闭")
        self.rpc_server.stop()

    def _init_as_remote_node(self):
        addr = "%s:%d" % (self.options['ip_addr'], self.options['rpc_port'])
        channel = grpc.insecure_channel(addr)
        self.rpc_stub = SwitchmlIOStub(channel)

    # tensor 长度需要被 elemenet_per_packet 整除
    def send(self, node, group_id, tensor_id, tensor):
        # type: (Node, int, int, np.ndarray)->int
        dst_addr = (node.options['ip_addr'], node.options['port'])
        send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        total_packet_num = math.ceil(tensor.size / elemenet_per_packet)
        current_node_id = self.options['node_id']

        # 等待节点进入接收状态
        node.send_barrier(tensor_id)

        prepare_packet_start = time.time()
        packet_list = []
        for i in range(total_packet_num):
            pkt = Packet()
            offset = i * elemenet_per_packet
            pkt.set_header(
                flow_control=0,
                tensor_id=tensor_id,  # uint32
                segment_id=i,  # uint32
                node_id=current_node_id,  # uint16
                aggregate_num=1,         # uint16
                ucast_grp=group_id,             # uint32
                data_type=packet.DataType.INT32.value,  # uint8
                pool_id=i % switch_pool_size
            )
            pkt.set_tensor(tensor[offset: offset+elemenet_per_packet])
            pkt.deparse_header()
            pkt.deparse_payload()
            packet_list.append(pkt)
        prepare_packet_end = time.time()

        # 一次性发出发送窗口所有包
        finish_cnt = 0
        send_window = []
        send_window_time = []
        for i in range(min(switch_pool_size, total_packet_num)):
            send_window.append(packet_list[i])
            send_window_time.append(time.time())
            send_sock.sendto(send_window[i].buffer, dst_addr)

        rtt = 0.01
        rx_pkt = Packet()
        send_start = time.time()

        while finish_cnt != total_packet_num:
            send_sock.settimeout(rtt)
            try:
                send_sock.recv_into(rx_pkt.buffer)
                rx_pkt.parse_header()
                if rx_pkt.ack:
                    if rx_pkt.tensor_id == send_window[rx_pkt.pool_id].tensor_id and rx_pkt.segment_id == send_window[rx_pkt.pool_id].segment_id:
                        next_pkt_segment_for_this_slot = send_window[rx_pkt.pool_id].segment_id + switch_pool_size
                        finish_cnt += 1
                        # 尝试发出这个窗口下一个包
                        if next_pkt_segment_for_this_slot < total_packet_num:
                            send_window[rx_pkt.pool_id] = packet_list[next_pkt_segment_for_this_slot]
                            send_window_time[rx_pkt.pool_id] = time.time()
                            send_sock.sendto(send_window[rx_pkt.pool_id].buffer, dst_addr)
                if rx_pkt.ecn:
                    # TODO: 如果支持多任务，需要添加 ecn
                    pass
            except:
                # 找出超时的包重发
                now = time.time()
                for i in range(len(send_window)):
                    if now - send_window_time[i] > rtt:
                        send_window[i].flow_control |= retranmission_bitmap
                        send_window[i].deparse_header()
                        send_window_time[i] = now
                        try:
                            send_sock.sendto(send_window[i].buffer, dst_addr)
                        except:
                            pass
        send_end = time.time()

        print("构建包耗时 %f 发送耗时 %f 发送速率 %f Mbps 综合速率 %f Mbps" % (
            prepare_packet_end - prepare_packet_start,
              send_end - send_start,
              tensor.size * 4 / 1024 / 1024 * 8 / (send_end - send_start),
              tensor.size * 4 / 1024 / 1024 * 8 / (send_end - send_start + prepare_packet_end - prepare_packet_start)))

        # TODO: 等待对方完成接收（如果 node 是 switch 则等待 switch 下面所有节点接收完成）
        # 暂时 switch 没有支持可靠传输，所以这里需要端到端检查丢包情况
        missing_slice = node.rpc_read_missing_slice(tensor_id, current_node_id)
        node.rpc_retranmission(tensor_id, current_node_id, {})

        return

    def receive_thread(self) -> None:
        pkt = Packet()
        while True:
            _, client = self.rx_socket.recvfrom_into(pkt.buffer, pkt_size)
            pkt.parse_header()
            if pkt.ack or pkt.ecn:
                # TODO
                continue
            pkt.parse_payload()
            key: tuple = (pkt.tensor_id, pkt.node_id)
            job = self.rx_jobs.get(key)
            if job is None:
                continue
            job.handle_packet(pkt)
            self.rx_socket.sendto(pkt.gen_ack_packet(), client)

    def receive_async(self, node, group_id, tensor_id, tensor):
        # type: (Node, int, int, np.ndarray) -> Job
        key: tuple = (tensor_id, node.options['node_id'])
        job = Job(key, tensor)
        self.rx_jobs[key] = job
        return job

    def receive(self, node, group_id, tensor_id, tensor):
        # type: (Node, int, int, np.ndarray) -> None
        job = self.receive_async(node, group_id, tensor_id, tensor)
        job.wait_until_job_finish()

        received = job.bitmap.sum()
        total = job.bitmap.size
        print("receive %d packet, expect %d, loss %f %%" %
              (received, total, 100 * (total - received) / total))
        key: tuple = (tensor_id, node.options['node_id'])
        del self.rx_jobs[key]
        return

    def add_child(self, node):
        # type: (Node) -> None
        self.children[node.options['node_id']] = node

    def remove_child(self, node) -> bool:
        # type: (Node) -> bool
        if self.children.get(node.options["node_id"]) is None:
            return False
        del self.children[node.options["node_id"]]
        return True

    # 向这个节点重传数据，不可以向当前节点和 Switch 重传
    def rpc_retranmission(self, tensor_id, node_id, data):
        # type: (int, int, dict[int,  str]) -> None
        self.rpc_stub.Retransmission(
            Retransmission.Request(
                tensor_id=tensor_id,
                node_id=node_id,
                data=data
            )
        )

    # 获取这个节点的丢包状态，不可以向当前节点和 Switch 查询
    def rpc_read_missing_slice(self, tensor_id, node_id):
        # type: (int, int) -> list
        return self.rpc_stub.ReadMissingSlice(
            PacketLoss.Request(
                tensor_id=tensor_id,
                node_id=node_id
            )
        )

    # 等待这个节点进入接收状态，不可以向当前节点和 Switch 查询
    def send_barrier(self, tensor_id):
        # type: (int) -> None
        self.rpc_stub.SendBarrier(
            Sync.SendBarrierRequest(
                tensor_id=tensor_id,
                node_id=self.options["node_id"]
            )
        )
