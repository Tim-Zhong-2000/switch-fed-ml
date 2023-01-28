#ifndef SWITCH_FED_ML_TENSOR_H
#define SWITCH_FED_ML_TENSOR_H

#include <cstdint>
#include "packet.h"

namespace switchml
{
  typedef uint32_t TensorId;

  class Tensor
  {
    void *buffer;
    uint64_t len;
    DataType data_type;
    TensorId tensor_id;

    /** create a new tensor*/
    Tensor(uint64_t len, DataType data_type, TensorId tensor_id);

    /** do not support tensor copy */
    Tensor(Tensor &tensor) = delete;

    /** support tensor move*/
    Tensor(Tensor &&tensor);

    ~Tensor();

    /**
     * copy buf to tensor.
     * 
     * the element size is determined by data_type
     * 
     * @param len number of element
     * @return -1 if tensor is moved, -2 if exceed max length;
     */
    int write(void *buf, uint64_t offset, uint64_t len);

    void* seek(uint64_t offset);
  };
}
#endif