#ifndef SWITCH_FED_ML_NODE_H
#define SWITCH_FED_ML_NODE_H

#include <string>
#include <vector>
#include <memory>
#include "tensor.h"

namespace switchml
{

  enum NodeType
  {
    Server,
    Switch,
    Client,
  };

  struct NodeOption
  {
    std::string ip_addr;
    uint16_t port;
    uint16_t rpc_port;
  };

  class Node
  {
  public:
    uint16_t port_;
    std::string ip_addr_;
    uint16_t rpc_port_;

    Node(NodeOption option);

    ~Node();

    /** send tensor to node */
    int send(Node &node, Tensor &tensor);

    /** receive tensor from node */
    int receive(Node &node, Tensor &tensor);

    /** return -1 when failed */
    int addChild(std::shared_ptr<Node> node);

    /** return -1 when failed */
    int removeChild(std::shared_ptr<Node> node);

    inline NodeType getNodeType()
    {
      if (this->children.size() == 0 && this->parent)
        return NodeType::Client;
      if (!this->parent)
        return NodeType::Server;
      return NodeType::Switch;
    }

  private:
    std::vector<std::shared_ptr<Node>> children;
    std::shared_ptr<Node> parent;
  };
}
#endif