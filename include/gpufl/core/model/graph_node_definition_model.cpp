#include "gpufl/core/model/graph_node_definition_model.hpp"

#include <sstream>

#include "gpufl/core/model/model_utils.hpp"

namespace gpufl::model {

std::string GraphNodeDefinitionModel::buildJson() const {
    std::ostringstream oss;
    oss << "{\"type\":\"graph_node_definition\""
        << ",\"session_id\":\"" << jsonEscape(event_.session_id) << "\""
        << ",\"graph_exec_key\":\"0x" << std::hex
        << event_.graph_exec_key << std::dec << "\""
        << ",\"node_index\":" << event_.node_index
        << ",\"node_type\":" << event_.node_type
        << ",\"dependency_count\":" << event_.dependency_count
        << "}";
    return oss.str();
}

}  // namespace gpufl::model
