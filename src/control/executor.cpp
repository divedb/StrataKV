#include "stratakv/control/executor.h"

#include <chrono>
#include <thread>

namespace stratakv::control {

void InlineExecutor::Execute(const TransferGraph& graph,
                             common::TaskHandle /*owner*/,
                             CompletionSink& sink) {
  for (const TransferOp& op : graph.ops()) {
    if (per_op_delay_ > std::chrono::milliseconds(0)) {
      std::this_thread::sleep_for(per_op_delay_);
    }
    sink.OnOpCompleted(op.op_id, /*ok=*/true);
  }
}

}  // namespace stratakv::control
