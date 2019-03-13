#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

int main()
{
    using namespace tensorflow;
    using namespace tensorflow::ops;
    Scope root = Scope::NewRootScope();
    auto a = Placeholder(root, DT_INT32);
    auto c = Add(root, a, {41});

    ClientSession session(root);
    std::vector<Tensor> outputs;

    Status s = session.Run({{a, {1}}}, {c}, &outputs);
    if (!s.ok())
    {
        LOG(INFO) << "Shit";
    }
    else
    {
        LOG(INFO) << "You finally made it!!!";
    }

    return 0;
}