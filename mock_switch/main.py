from switch import Switch
from node import Node
from group import Group

if __name__ == "__main__":
    node_num = 3
    mock_switch = Switch("127.0.0.1", 30000, 101, debug=True)
    
    # 创建 ps 实例
    ps = Node(100, "127.0.0.1", 50000, 50001, 0)
    mock_switch.nodes[ps.id] = ps

    # 注册组
    group1 = Group(1, ps)
    mock_switch.groups[group1.id] = group1
    group2 = Group(2, ps)
    mock_switch.groups[group2.id] = group2

    # 创建 client 实例
    for i in range(node_num):
        node = Node(i+1, "127.0.0.1", 50000+(i+1)*3, 50000+(i+1)*3 + 1, 1 << i)
        group1.addNode(node)
        group2.addNode(node)
        mock_switch.nodes[node.id] = node

    mock_switch.start()
