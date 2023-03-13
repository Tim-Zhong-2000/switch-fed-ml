
import sys
sys.path.append('./python_io')
sys.path.append('./prune')

from switch import Switch
from threading import Thread
from exp.comm import EClient, EServer
from prune.cnn import SparseCNN
from prune.main import train_model, test_model
import copy

model = SparseCNN({
    "conv1": (1, 32, 3),
    "conv1_channel_id": list(range(32)),
    "conv2": (32, 64, 3),
    "conv2_channel_id": list(range(64)),
    "fc1": (64 * 5 * 5, 128),
    "fc1_channel_id": list(range(128)),
    "fc2": (128, 10),
})

print(model)

ROUND = 100
node_num = 3

start_port = 50000

server_node_id = 100
server_ip_addr = "127.0.0.1"
server_rx_port = start_port
server_tx_port = start_port + 1
server_rpc_addr = "127.0.0.1:%d" % (start_port + 2)

mock_switch_node_id = 101
mock_switch_ip_addr = "127.0.0.1"
mock_switch_port = 30000

client_configs = [
    {
        "node_id": i+1,
        "ip_addr": "127.0.0.1",
        "rx_port": start_port + (i+1)*3,
        "tx_port": start_port + (i+1)*3 + 1,
        "rpc_addr": "127.0.0.1:%d" % (start_port + (i+1) * 3 + 2),
    } for i in range(node_num)
]
    

def server():
    global model

    server = EServer(
        node_id=server_node_id,
        ip_addr=server_ip_addr,
        rx_port=server_rx_port,
        tx_port=server_tx_port,
        rpc_addr=server_rpc_addr,
        is_remote_node=False
    )
    switch = Switch(
        node_id=mock_switch_node_id,
        ip_addr=mock_switch_ip_addr,
        rx_port=mock_switch_port,
        tx_port=mock_switch_port,
        rpc_addr=""
    )
    for config in client_configs:
        switch.add_child(EClient(
            node_id=config["node_id"],
            ip_addr=config["ip_addr"],
            rx_port=config["rx_port"],
            tx_port=config["tx_port"],
            rpc_addr=config["rpc_addr"],
            max_group_id=3,
            is_remote_node=True,
        ))
    while server.round_id < ROUND:
        print("------------------ ROUND %d START -----------------------" % (server.round_id))
        # do prune
        patches1 = model.prune(0.5)

        # multicast
        server.multicast_model(switch, model, [patches1])

        # receive trained model
        model = server.receive_model(switch)

        # do aggr
        # 目前是单个 switch，不需要 aggr
        
        test_model(model)
        print("------------------ ROUND %d END -----------------------" % (server.round_id))
        
        # next round
        server.round_id += 1


def client(index: int):
    server = EServer(
        node_id=server_node_id,
        ip_addr=server_ip_addr,

        # rx_port=server_rx_port,
        rx_port=mock_switch_port,

        # tx_port=server_tx_port,
        tx_port=mock_switch_port,

        rpc_addr=server_rpc_addr,
        is_remote_node=True
    )
    config = client_configs[index]
    client = EClient(
        node_id=config["node_id"],
        ip_addr=config["ip_addr"],
        rx_port=config["rx_port"],
        tx_port=config["tx_port"],
        rpc_addr=config["rpc_addr"],
        max_group_id=2,
        is_remote_node=False,
    )
    while client.round_id < ROUND:
        model2 = client.receive_model(server)
        # do some training
        train_model(model2, node_num, index)

        # reduce to ps
        client.reduce_model(server, model2)
        
        client.round_id += 1


# diff = (model_cp.fc1.weight - model2.fc1.weight).max()
for i, config in enumerate(client_configs):
    Thread(target=client, args=(i, )).start()

ts = Thread(target=server)
ts.start()
ts.join()

print("finish")
