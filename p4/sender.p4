#ifndef _SWITCH_FL_SENDER
#define _SWITCH_FL_SENDER
// !!!
// 令聚合器聚合完成的那一个包，既要完整发给 ps 又要向 client 发送 ack，所以这里使用 PRE 实现包扩增
// 

control Sender(
  inout headers_t hdr,
  inout metadata_t meta,
  inout standard_metadata_t standard_meta
)
{
  action set_dest(
    EthernetAddress src_mac,
    EthernetAddress dst_mac,
    // ip
    IPv4Address src_addr,
    IPv4Address dst_addr,
    // udp
    bit<16> src_port,
    bit<16> dst_port, 
    // switchfl
    NodeId_t node_id
  ) {
    hdr.ethernet.src_addr = src_mac;
    hdr.ethernet.dst_addr = dst_mac;
    hdr.ipv4.src_addr = src_addr;
    hdr.ipv4.dst_addr = dst_addr;
    hdr.udp.src_port = src_port;
    hdr.udp.dst_port = dst_port;
    hdr.switchfl.node_id = node_id;
  }

  // 这里只对 L2 以上层级进行 swap 操作，对于 L1 层的处理在 MyIngress 模块中已经处理过
  action send_back() {
    EthernetAddress temp = hdr.ethernet.dst_addr;
    hdr.ethernet.dst_addr = hdr.ethernet.src_addr;
    hdr.ethernet.src_addr = temp;

    IPv4Address temp_addr = hdr.ipv4.dst_addr;
    hdr.ipv4.dst_addr = hdr.ipv4.src_addr;
    hdr.ipv4.src_addr = temp_addr;

    bit<16> temp_port = hdr.udp.dst_port;
    hdr.udp.dst_port = hdr.udp.src_port;
    hdr.udp.src_port = temp_port;
  }

  action no_action() {}

  // <egress_port, egress_rid> 唯一确认了一个包的目的地
  table switchfl_multicast_address {
    key = {
      standard_meta.egress_port: exact;
      standard_meta.egress_rid: exact;
    }
    actions = {
      set_dest;
      no_action;
    }
    
    #ifndef DEBUG
    size = 128;
    #else 
    const entries = {
      //               src_mac,        dst_mac,        src_addr,   dst_addr,   src_port, dst_port, node_id(of ps)
      (0, 1): set_dest(0x00000b0b0b00, 0x6ac127a9ec4e, 0x0b0b0b00, 0x0b0b0b01, 50000,    50000,    10);
      // 11.11.11.0:50000 (switch) => 11.11.11.1:50000 (client 1 rx_port)

      (0, 2): set_dest(0x00000b0b0b00, 0x6ac127a9ec4e, 0x0b0b0b00, 0x0b0b0b02, 50000,    50000,    10); 
      // 11.11.11.0:50000 (switch) => 11.11.11.2:50000 (client 2 rx_port)
      
      (2, 1): set_dest(0x00000b0b0b00, 0x6ac127a9ec4e, 0x0b0b0b00, 0x0b0b0b03, 50000,    50001,    10);
      // 11.11.11.0:50000 (switch) => 11.11.11.3:50001 (PS tx_port)
    }
    #endif

    const default_action = no_action;
  }

  // <egress_port, egress_rid> 唯一确认了一个包的目的地
  table switchfl_reduce_address {
    key = {
      standard_meta.egress_port: exact;
      standard_meta.egress_rid: exact;
    }

    actions = {
      set_dest;
      no_action;
    }

    #ifndef DEBUG
    size = 128;
    #else 
    const entries = {
      //               src_mac,        dst_mac,        src_addr,   dst_addr,   src_port, dst_port, node_id(of this switch)
      (0, 1): set_dest(0x00000b0b0b00, 0x6ac127a9ec4e, 0x0b0b0b00, 0x0b0b0b01, 50000,    50001,    100); 
      // 11.11.11.0:50000 (switch) => 11.11.11.1:50001 (client 1 tx_port)

      (0, 2): set_dest(0x00000b0b0b00, 0x6ac127a9ec4e, 0x0b0b0b00, 0x0b0b0b02, 50000,    50001,    100); 
      // 11.11.11.0:50000 (switch) => 11.11.11.2:50001 (client 2 tx_port)
      
      (2, 1): set_dest(0x00000b0b0b00, 0x6ac127a9ec4e, 0x0b0b0b00, 0x0b0b0b03, 50000,    50000,    100); 
      // 11.11.11.0:50000 (switch) => 11.11.11.3:50000 (PS rx_port)
    }
    #endif

    const default_action = no_action;
  }


  action set_is_ps() {
    meta.is_ps = true;
  }

  action set_not_ps() {
    meta.is_ps = false;
  }

  table ps_mark {
    key = {
      standard_meta.egress_port: exact;
      standard_meta.egress_rid: exact;
    }

    actions = {
      set_is_ps;
      set_not_ps;
    }

    #ifndef DEBUG
    size = 128;
    #else 
    const entries = {
      (2, 1): set_is_ps();
    }
    #endif

    const default_action = set_not_ps;
  }

  apply {
    // 只有扩增过的包需要还原地址
    if(meta.processor_action == Processor_Action.MCAST) {
      switchfl_multicast_address.apply();
    } else if(meta.processor_action == Processor_Action.FINISH) {
      switchfl_reduce_address.apply();
    } else {
      send_back();
    }
    hdr.switchfl.aggregate_num = meta.total_aggregate_num;
    ps_mark.apply();
  }
}
#endif