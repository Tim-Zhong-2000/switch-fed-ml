# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: io.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x08io.proto\"M\n\x0eRetransmission\x1a;\n\x07Request\x12\x11\n\ttensor_id\x18\x01 \x01(\x05\x12\x0f\n\x07node_id\x18\x02 \x01(\x05\x12\x0c\n\x04\x64\x61ta\x18\x63 \x03(\x0c\"d\n\nPacketLoss\x1a-\n\x07Request\x12\x11\n\ttensor_id\x18\x01 \x01(\x05\x12\x0f\n\x07node_id\x18\x02 \x01(\x05\x1a\'\n\x08Response\x12\x1b\n\x13missing_packet_list\x18\x01 \x03(\x05\"\x06\n\x04Null2}\n\nSwitchmlIO\x12\x30\n\x0eRetransmission\x12\x17.Retransmission.Request\x1a\x05.Null\x12=\n\x10ReadMissingSlice\x12\x13.PacketLoss.Request\x1a\x14.PacketLoss.Responseb\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'io_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _RETRANSMISSION._serialized_start=12
  _RETRANSMISSION._serialized_end=89
  _RETRANSMISSION_REQUEST._serialized_start=30
  _RETRANSMISSION_REQUEST._serialized_end=89
  _PACKETLOSS._serialized_start=91
  _PACKETLOSS._serialized_end=191
  _PACKETLOSS_REQUEST._serialized_start=30
  _PACKETLOSS_REQUEST._serialized_end=75
  _PACKETLOSS_RESPONSE._serialized_start=152
  _PACKETLOSS_RESPONSE._serialized_end=191
  _NULL._serialized_start=193
  _NULL._serialized_end=199
  _SWITCHMLIO._serialized_start=201
  _SWITCHMLIO._serialized_end=326
# @@protoc_insertion_point(module_scope)
