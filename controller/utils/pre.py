import p4runtime_sh.shell as sh

# 所有节点添加前需要执行此方法
def insert_mcast_replicas(replicas: list, mcast_grp: int):
    # PRE
    mcge = sh.MulticastGroupEntry(mcast_grp)
    for replica in replicas:
        mcge.add(replica['egress_port'], replica['egress_rid'])
    mcge.insert()

def delete_mcast_replica(egress_port: int, egress_rid: int, mcast_grp: int):
    # PRE
    mcge = sh.MulticastGroupEntry(mcast_grp)
    mcge.delete()