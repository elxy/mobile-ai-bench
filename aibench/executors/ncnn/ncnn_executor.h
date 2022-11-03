// Copyright 2018 The MobileAIBench Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef AIBENCH_EXECUTORS_NCNN_NCNN_EXECUTOR_H_
#define AIBENCH_EXECUTORS_NCNN_NCNN_EXECUTOR_H_

#include <map>
#include <string>

#include "aibench/executors/base_executor.h"
#include "ncnn/include/ncnn/datareader.h"
#include "ncnn/include/ncnn/layer.h"
#include "ncnn/include/ncnn/mat.h"
#include "ncnn/include/ncnn/modelbin.h"
#include "ncnn/include/ncnn/net.h"

namespace ncnn {
class DataReaderFromEmpty : public DataReader
{
public:
    virtual int scan(const char* format, void* p) const
    {
        (void)format;
        (void)p;
        return 0;
    }
    virtual size_t read(void* buf, size_t size) const
    {
        memset(buf, 0, size);
        return size;
    }
};
}  // namespace ncnn

namespace aibench {

class NcnnExecutor : public BaseExecutor {
 public:
  explicit NcnnExecutor(const std::string &model_file)
      : BaseExecutor(NCNN, CPU, model_file, "") {}

  virtual Status Init(int num_threads);

  virtual Status Prepare();

  virtual Status Run(const std::map<std::string, BaseTensor> &inputs,
                     std::map<std::string, BaseTensor> *outputs);

  virtual void Finish();
 private:
  ncnn::Net net;
};

}  // namespace aibench

#endif  // AIBENCH_EXECUTORS_NCNN_NCNN_EXECUTOR_H_
