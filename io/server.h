#pragma once

#include <vector>
#include <memory>
#include "node.h"
#include "tensor.h"

namespace switchfl
{
  class Server : public Node
  {
  public:
    Server(NodeOptions options) : Node(options) {}

    /** multicast to node_list */
    int multicast(std::vector<std::shared_ptr<Node>> &node_list, GroupId group_id, std::shared_ptr<Tensor> &tensor);

    /** reduce from node_list */
    int reduce(std::vector<std::shared_ptr<Node>> &node_list, GroupId group_id, std::shared_ptr<Tensor> &tensor);

  private:
    /** 广播和聚合过程，可以直接对最近的 switch 进行操作而不是对节点操作 */
    std::vector<std::shared_ptr<Node>> optimize(std::vector<std::shared_ptr<Node>> &node_list);
  };
}