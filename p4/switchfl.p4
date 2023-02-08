#ifndef _SWITCH_FL_MAIN
#define _SWITCH_FL_MAIN

#define DEBUG
#define POOL_SIZE 128

#include <core.p4>
#include <v1model.p4>
#include "type.p4"
#include "receiver.p4"
#include "processor.p4"
#include "sender.p4"

// error {
//     IPv4IncorrectVersion,
//     IPv4OptionsNotSupported
//}

parser MyIngressParser(
    packet_in pkt,
    out headers_t hdr,
    inout metadata_t meta,
    inout standard_metadata_t standard_meta)
{
    state start {
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.ether_type) {
            0x0800:  parse_ipv4;
            default: accept;
        }
    }

    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        // verify(hdr.ipv4.version == 4w4, error.IPv4IncorrectVersion);
        // verify(hdr.ipv4.ihl == 4w5, error.IPv4OptionsNotSupported);
        transition select(hdr.ipv4.ihl, hdr.ipv4.frag_offset, hdr.ipv4.protocol) {
            (5, 0, 17): parse_udp;
            default: accept;
        }
    }

    state parse_udp {
        pkt.extract(hdr.udp);
        transition select(hdr.udp.dst_port) {
            SwitchFL_PORT: parse_switchfl;
            default: accept;
        }
    }

    state parse_switchfl {
        pkt.extract(hdr.switchfl);
        pkt.extract(hdr.tensor0);
        pkt.extract(hdr.tensor1);
        pkt.extract(hdr.tensor2);
        pkt.extract(hdr.tensor3);
        pkt.extract(hdr.tensor4);
        pkt.extract(hdr.tensor5);
        pkt.extract(hdr.tensor6);
        pkt.extract(hdr.tensor7);
        transition accept;
    }
}

control MyVerifyChecksum(
    inout headers_t hdr,
    inout metadata_t meta)
{
    apply { }
}

control MyIngress(
    inout headers_t hdr,
    inout metadata_t meta,
    inout standard_metadata_t standard_meta)
{
    Receiver() switchfl_receiver;
    Processor() processor;

    bool dropped = false;

    action drop_action() {
        mark_to_drop(standard_meta);
        dropped = true;
    }

    action to_port_action(bit<9> port) {
        hdr.ipv4.ttl = hdr.ipv4.ttl - 1;
        standard_meta.egress_spec = port;
    }

    table ipv4_match {
        key = {
            hdr.ipv4.dst_addr: lpm;
        }
        actions = {
            drop_action;
            to_port_action;
        }
        #ifndef DEBUG
        size = 128;
        #else
        const entries = {
            (0x0a0a0000 &&& 0xffff0000) : to_port_action(0); // 10.10.0.0 => port 0  
            (0x0b0b0000 &&& 0xffff0000) : to_port_action(1); // 11.11.0.0 => port 1
        }
        #endif
        default_action = drop_action;
    }

    apply {
        if(hdr.ipv4.isValid()) {
            ipv4_match.apply();
        }

        if(hdr.switchfl.isValid()) {
            switchfl_receiver.apply(hdr, meta, standard_meta);
            processor.apply(hdr, meta);
            if(meta.processor_action == Processor_Action.DROP) {
                drop_action();
            }
        }
        if (dropped) return;
    }
}

control MyEgress(
    inout headers_t hdr,
    inout metadata_t meta,
    inout standard_metadata_t standard_meta)
{
    apply { }
}

control MyComputeChecksum(
    inout headers_t hdr,
    inout metadata_t meta)
{
    apply {
        update_checksum(
            hdr.ipv4.isValid(),
            {
                hdr.ipv4.version,
                hdr.ipv4.ihl,
                hdr.ipv4.diffserv,
                hdr.ipv4.total_len,
                hdr.ipv4.identification,
                hdr.ipv4.flags,
                hdr.ipv4.frag_offset,
                hdr.ipv4.ttl,
                hdr.ipv4.protocol,
                hdr.ipv4.src_addr,
                hdr.ipv4.dst_addr
            },
            hdr.ipv4.hdr_checksum,
            HashAlgorithm.csum16
        );
    }
}

control MyEgressDeparser(
    packet_out pkt,
    in headers_t hdr)
{
    apply {
        pkt.emit(hdr);
    }
}

V1Switch(
    MyIngressParser(),
    MyVerifyChecksum(),
    MyIngress(),
    MyEgress(),
    MyComputeChecksum(),
    MyEgressDeparser()
) main;

#endif