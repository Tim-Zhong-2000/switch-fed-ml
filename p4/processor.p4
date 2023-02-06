#ifndef _SWITCH_FL_PROCESSOR
#define _SWITCH_FL_PROCESSOR

#include "type.p4"

#define POOL_SIZE 1048576

control Processor (
  inout switchfl_h hdr,
  inout metadata_t metadata,
  inout tensor_t tensor
)
{
  register<bit<32>>(POOL_SIZE) aggregate_bitmap;
  register<TensorId_t>(POOL_SIZE) current_tensor_id;
  register<SegmentId_t>(POOL_SIZE) current_segment_id;
  register<bit<1>>(POOL_SIZE) is_free;

  // 128 bytes per pool
  register<bit<32>>(POOL_SIZE) r0;
  register<bit<32>>(POOL_SIZE) r1;
  register<bit<32>>(POOL_SIZE) r2;
  register<bit<32>>(POOL_SIZE) r3;
  register<bit<32>>(POOL_SIZE) r4;
  register<bit<32>>(POOL_SIZE) r5;
  register<bit<32>>(POOL_SIZE) r6;
  register<bit<32>>(POOL_SIZE) r7;
  register<bit<32>>(POOL_SIZE) r8;
  register<bit<32>>(POOL_SIZE) r9;
  register<bit<32>>(POOL_SIZE) r10;
  register<bit<32>>(POOL_SIZE) r11;
  register<bit<32>>(POOL_SIZE) r12;
  register<bit<32>>(POOL_SIZE) r13;
  register<bit<32>>(POOL_SIZE) r14;
  register<bit<32>>(POOL_SIZE) r15;
  register<bit<32>>(POOL_SIZE) r16;
  register<bit<32>>(POOL_SIZE) r17;
  register<bit<32>>(POOL_SIZE) r18;
  register<bit<32>>(POOL_SIZE) r19;
  register<bit<32>>(POOL_SIZE) r20;
  register<bit<32>>(POOL_SIZE) r21;
  register<bit<32>>(POOL_SIZE) r22;
  register<bit<32>>(POOL_SIZE) r23;
  register<bit<32>>(POOL_SIZE) r24;
  register<bit<32>>(POOL_SIZE) r25;
  register<bit<32>>(POOL_SIZE) r26;
  register<bit<32>>(POOL_SIZE) r27;
  register<bit<32>>(POOL_SIZE) r28;
  register<bit<32>>(POOL_SIZE) r29;
  register<bit<32>>(POOL_SIZE) r30;
  register<bit<32>>(POOL_SIZE) r31;


  action release(bit<32> pool_id) {
    is_free.write(pool_id, 1);
    // tensor_id 和 segment_id 不需要清理
  }

  action acquire(bit<32> pool_id) {
    current_tensor_id.write(pool_id, hdr.tensor_id);
    current_segment_id.write(pool_id, hdr.segment_id);
    is_free.write(pool_id, 0);
    aggregate_bitmap.write(pool_id, 0);
  }

  action aggregate_to(bit<32> pool_id) {
    bit<32> temp;

    // loop start
    r0.read(temp, pool_id);
    temp = temp + tensor.d0;
    tensor.d0 = temp;
    r0.write(pool_id, temp);

    r1.read(temp, pool_id);
    temp = temp + tensor.d1;
    tensor.d1 = temp;
    r1.write(pool_id, temp);
    
    r2.read(temp, pool_id);
    temp = temp + tensor.d2;
    tensor.d2 = temp;
    r2.write(pool_id, temp);
    
    r3.read(temp, pool_id);
    temp = temp + tensor.d3;
    tensor.d3 = temp;
    r3.write(pool_id, temp);
    
    r4.read(temp, pool_id);
    temp = temp + tensor.d4;
    tensor.d4 = temp;
    r4.write(pool_id, temp);
    
    r5.read(temp, pool_id);
    temp = temp + tensor.d5;
    tensor.d5 = temp;
    r5.write(pool_id, temp);
    
    r6.read(temp, pool_id);
    temp = temp + tensor.d6;
    tensor.d6 = temp;
    r6.write(pool_id, temp);
    
    r7.read(temp, pool_id);
    temp = temp + tensor.d7;
    tensor.d7 = temp;
    r7.write(pool_id, temp);
    
    r8.read(temp, pool_id);
    temp = temp + tensor.d8;
    tensor.d8 = temp;
    r8.write(pool_id, temp);
    
    r9.read(temp, pool_id);
    temp = temp + tensor.d9;
    tensor.d9 = temp;
    r9.write(pool_id, temp);
    
    r10.read(temp, pool_id);
    temp = temp + tensor.d10;
    tensor.d10 = temp;
    r10.write(pool_id, temp);

    r11.read(temp, pool_id);
    temp = temp + tensor.d11;
    tensor.d11 = temp;
    r11.write(pool_id, temp);

    r12.read(temp, pool_id);
    temp = temp + tensor.d12;
    tensor.d12 = temp;
    r12.write(pool_id, temp);
    
    r13.read(temp, pool_id);
    temp = temp + tensor.d13;
    tensor.d13 = temp;
    r13.write(pool_id, temp);
    
    r14.read(temp, pool_id);
    temp = temp + tensor.d14;
    tensor.d14 = temp;
    r14.write(pool_id, temp);
    
    r15.read(temp, pool_id);
    temp = temp + tensor.d15;
    tensor.d15 = temp;
    r15.write(pool_id, temp);
    
    r16.read(temp, pool_id);
    temp = temp + tensor.d16;
    tensor.d16 = temp;
    r16.write(pool_id, temp);
    
    r17.read(temp, pool_id);
    temp = temp + tensor.d17;
    tensor.d17 = temp;
    r17.write(pool_id, temp);
    
    r18.read(temp, pool_id);
    temp = temp + tensor.d18;
    tensor.d18 = temp;
    r18.write(pool_id, temp);
    
    r19.read(temp, pool_id);
    temp = temp + tensor.d19;
    tensor.d19 = temp;
    r19.write(pool_id, temp);
    
    r20.read(temp, pool_id);
    temp = temp + tensor.d20;
    tensor.d20 = temp;
    r20.write(pool_id, temp);

    r21.read(temp, pool_id);
    temp = temp + tensor.d21;
    tensor.d21 = temp;
    r21.write(pool_id, temp);

    r22.read(temp, pool_id);
    temp = temp + tensor.d22;
    tensor.d22 = temp;
    r22.write(pool_id, temp);
    
    r23.read(temp, pool_id);
    temp = temp + tensor.d23;
    tensor.d23 = temp;
    r23.write(pool_id, temp);
    
    r24.read(temp, pool_id);
    temp = temp + tensor.d24;
    tensor.d24 = temp;
    r24.write(pool_id, temp);
    
    r25.read(temp, pool_id);
    temp = temp + tensor.d25;
    tensor.d25 = temp;
    r25.write(pool_id, temp);
    
    r26.read(temp, pool_id);
    temp = temp + tensor.d26;
    tensor.d26 = temp;
    r26.write(pool_id, temp);
    
    r27.read(temp, pool_id);
    temp = temp + tensor.d27;
    tensor.d27 = temp;
    r27.write(pool_id, temp);
    
    r28.read(temp, pool_id);
    temp = temp + tensor.d28;
    tensor.d28 = temp;
    r28.write(pool_id, temp);
    
    r29.read(temp, pool_id);
    temp = temp + tensor.d29;
    tensor.d29 = temp;
    r29.write(pool_id, temp);
    
    r30.read(temp, pool_id);
    temp = temp + tensor.d30;
    tensor.d30 = temp;
    r30.write(pool_id, temp);
    
    r31.read(temp, pool_id);
    temp = temp + tensor.d31;
    tensor.d31 = temp;
    r31.write(pool_id, temp);
    // loop end
  }


  apply {
    bit<1> current_pool_is_free;
    bit<32> pool_id = (bit<32>)hdr.pool_id;

    // 多任务时可能会出现竞争
    is_free.read(current_pool_is_free, pool_id);
    if(current_pool_is_free == 1) {
      acquire(pool_id);
    } else {
      bit<32> current_pool_segment_id;
      bit<32> current_pool_tensor_id;
      current_segment_id.read(current_pool_segment_id, pool_id);
      current_tensor_id.read(current_pool_tensor_id, pool_id);
      if(hdr.segment_id != current_pool_segment_id || hdr.tensor_id != current_pool_tensor_id) {
        metadata.processor_action = Processor_Action.ECN;
        return;
      }
    }


    bit<32> current_pool_bitmap;
    // 处理重传包，然后检查聚合完成状态
    aggregate_bitmap.read(current_pool_bitmap, pool_id);
    // 检查当前 worker_bitmap 在当前聚合器是否聚合过，如果聚合过说明是重传包，ecn 降低节点发包速度
    if(current_pool_bitmap & metadata.bitmap > 0) {
      metadata.processor_action = Processor_Action.ECN;
      return;
    }
    current_pool_bitmap = current_pool_bitmap | metadata.bitmap;
    aggregate_bitmap.write(pool_id, current_pool_bitmap);
    
    if(current_pool_bitmap == metadata.aggregate_finish_bitmap) { // 聚合完成
      metadata.processor_action = Processor_Action.MCAST;
      release(pool_id);
    }
    aggregate_to(pool_id);
  }
}

#endif