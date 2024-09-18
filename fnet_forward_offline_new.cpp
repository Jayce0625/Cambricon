#include <sys/time.h>
// #include "glog/logging.h"
#include <cnrt.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cnml.h>
#include <assert.h>
#include <torch/torch.h>
#include <chrono>
using namespace std;

// 定义结构体
struct myData{
  void ** outputCpuPtrS;
  void ** outputMluPtrS;
};

void linspace(float start, float stop, int num, float * data){
  float step = (stop - start)/(num - 1);
  for (int i = 0 ; i < num ; ++i){
    data[i] = start + step * i ;
  }
}

void readData(float* data, int length, std::string filename) {
  std::ifstream file;
  file.open(filename);
  float temp = 0.0;
  int j = 0;
  while (file >> temp) {
    data[j] = temp;
    j++;
  }
  file.close();
  assert(j == length);
  std::cout << " Read data done! " << std::endl;
  // CHECK_EQ(j, length)<< "plase check the input data dim";
}

void writeData(float * data, int length, std::string filename){
  std::ofstream fout(filename, std::ios::out);
  fout << std::flush;
  for (int i = 0; i < length; i++) {
    // std::cout<< i<<" "<<(reinterpret_cast<float*>(data))[i]<<std::endl;
    fout << (reinterpret_cast<float*>(data))[i] << std::endl;
  }
  fout << std::flush;
  fout.close();
  std::cout << " Write data done! " << std::endl;
}

// 送进来的数据和索引全部是NCHW排布的
float * gather_cpu(void * data, vector<int> &data_shape, void * index, vector<int> &index_shape, int dim, std::string fname, int num){
  at::Tensor gather_data, gather_index;
  assert(data != nullptr && index != nullptr);
  if (data_shape.size() == 5) {
    // 输出前五个元素的数值，作为形状
    // std::cout << " Data shape is: " << data_shape[0] << " " << data_shape[1] << " " << data_shape[2] << " " << data_shape[3] << " " << data_shape[4] << std::endl; 
    gather_data = torch::from_blob(reinterpret_cast<float*>(data), {data_shape[0], data_shape[1], data_shape[2], data_shape[3], data_shape[4]}, torch::kFloat32);
    // 按照前五个元素计算个数来存储
    // writeData(reinterpret_cast<float*>(data), accumulate(data_shape.begin(), data_shape.begin()+5, 1, multiplies<int>()), "./output_new/" + fname + "_gather" + std::to_string(num) + "_data_5_wjc.txt");
  } else if (data_shape.size() == 4) {
    // std::cout << " Data shape is: " << data_shape[0] << " " << data_shape[1] << " " << data_shape[2] << " " << data_shape[3] << std::endl; 
    gather_data = torch::from_blob(reinterpret_cast<float*>(data), {data_shape[0], data_shape[1], data_shape[2], data_shape[3]}, torch::kFloat32);
    // writeData(reinterpret_cast<float*>(data), accumulate(data_shape.begin(), data_shape.begin()+4, 1, multiplies<int>()), "./output_new/" + fname + "_gather" + std::to_string(num) + "_data_4_wjc.txt");
  } else {
    throw " data_shape.size() is not 4 or 5! ";
  }
  if (index_shape.size() == 5) {
    // std::cout << " Index shape is: "<< index_shape[0] << " " << index_shape[1] << " " << index_shape[2] << " " << index_shape[3] << " " << index_shape[4] << std::endl;
    gather_index = torch::from_blob(reinterpret_cast<float*>(index), {index_shape[0], index_shape[1], index_shape[2], index_shape[3], index_shape[4]}, torch::kFloat32);  
    // writeData(reinterpret_cast<float*>(index), accumulate(index_shape.begin(), index_shape.begin()+5, 1, multiplies<int>()), "./output_new/" + fname + "_gather" + std::to_string(num) + "_index_5_wjc.txt");
  } else if (index_shape.size() == 4) {
    // std::cout << " Index shape is: "<< index_shape[0] << " " << index_shape[1] << " " << index_shape[2] << " " << index_shape[3] << std::endl;
    gather_index = torch::from_blob(reinterpret_cast<float*>(index), {index_shape[0], index_shape[1], index_shape[2], index_shape[3]}, torch::kFloat32);
    // writeData(reinterpret_cast<float*>(index), accumulate(index_shape.begin(), index_shape.begin()+4, 1, multiplies<int>()), "./output_new/" + fname + "_gather" + std::to_string(num) + "_index_4_wjc.txt");
  } else {
    throw " index_shape.size() is not 4 or 5! ";
  }
  auto gather_output_NCHW = torch::gather(gather_data, dim, gather_index.to(torch::kInt64));
  // std::cout << " gather_output_NCHW shape is: " << gather_output_NCHW.sizes() << std::endl;
  auto flattened1 = gather_output_NCHW.flatten();
  auto size_NCHW = flattened1.numel();
  float * gather_output_NCHW_data = flattened1.data_ptr<float>();
  // writeData(gather_output_NCHW_data, size_NCHW, "./output_new/" + fname + "_gather" + std::to_string(num) + "_output_NCHW_wjc.txt");
  
  auto gather_output_NHWC = gather_output_NCHW.permute({0, 2, 3, 4, 1}).contiguous();
  // std::cout << " gather_output_NHWC shape is: " << gather_output_NHWC.sizes() << std::endl;
  auto flattened2 = gather_output_NHWC.flatten();
  auto size_NHWC = flattened2.numel();
  // std::cout << " NHWC data length: " << size_NHWC << std::endl;
  float * gather_output_NHWC_data = flattened2.data_ptr<float>();  // 获取tensor的指针
  float * ans_NHWC = new float[size_NHWC];
  for (int i = 0; i < size_NHWC; i++) ans_NHWC[i] = gather_output_NHWC_data[i];
  // writeData(ans_NHWC, size_NHWC, "./output_new/" + fname + "_gather" + std::to_string(num) + "_output_NHWC_wjc.txt");

  return ans_NHWC;
}

myData output_align(std::string fname, int outputNum, int64_t* outputSizeS, cnrtDataType_t* output_data_type, cnrtFunction_t function, myData computeData){
  auto start = std::chrono::high_resolution_clock::now();
  std::string output_path = "./output_new/";
  for(int i = 0; i < outputNum; i++){
   if(((fname == "cfnet1.cambricon" || fname == "cfnet2.cambricon") && (i == 1 || i == 3 || i == 5 || i == 7)) || (fname == "cfnet3.cambricon")){  // 只有用于gather的输出才需要保存回主机端，分别对应索引是1, 3, 5, 7
      // Step 1: copy data from MLU to CPU
      void ** outPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * outputNum));
      void * temp_output_cpu_data = (void*)malloc(outputSizeS[i]);
      cnrtMemcpy(temp_output_cpu_data, computeData.outputMluPtrS[i], outputSizeS[i], CNRT_MEM_TRANS_DIR_DEV2HOST);
      // Step 2: cast cpu_data type to CNRT_FLOAT32
      int output_count = outputSizeS[i] / cnrtDataTypeSize(output_data_type[i]);
      computeData.outputCpuPtrS[i] = reinterpret_cast<void*>(reinterpret_cast<float*>(malloc(sizeof(float) * output_count)));
      if (output_data_type[i] != CNRT_FLOAT32) {
        cnrtCastDataType(temp_output_cpu_data, output_data_type[i], computeData.outputCpuPtrS[i], CNRT_FLOAT32, output_count, nullptr);
      } else {
        memcpy(computeData.outputCpuPtrS[i], temp_output_cpu_data, outputSizeS[i]);
      }
      if (fname == "cfnet3.cambricon") {  // cfnet3唯一一个输出是4维
        // Step 3: trans data order from NHWC to NCHW
        outPtrS[i] = reinterpret_cast<void*>(reinterpret_cast<float*>(malloc(sizeof(float) * output_count)));
        std::vector<int> shape_for_output_cast(4, 1);  // 1,3,5,7全部是维度为5的输出
        int dimNum_for_output_cast = 4;
        cnrtGetOutputDataShape((int**)&shape_for_output_cast, &dimNum_for_output_cast, i, function);
        // std::cout << " " + fname + "_outputdata " << i << " NHWC shape is: " << shape_for_output_cast[0] << " " << 
        //     shape_for_output_cast[1] << " " << shape_for_output_cast[2] << " " << shape_for_output_cast[3] << std::endl;
        int dim_order_for_output_cast[4] = {0, 3, 1, 2};
        int dim_shape_for_output_cast[4] = {shape_for_output_cast[0], shape_for_output_cast[1], shape_for_output_cast[2], shape_for_output_cast[3]};  // NHWC
        cnrtRet_t ret = cnrtTransDataOrder(computeData.outputCpuPtrS[i], CNRT_FLOAT32, outPtrS[i], 4, dim_shape_for_output_cast, dim_order_for_output_cast);
        // std::cout << " " << fname << " TransDataOrder End!" << std::endl;
        // Step 4: save 4 datas to file
        memcpy(computeData.outputCpuPtrS[i], outPtrS[i], sizeof(float) * output_count);
        // writeData(reinterpret_cast<float*>(computeData.outputCpuPtrS[i]), output_count, output_path + fname + "_forward_output_" + std::to_string(i));
      } else {
        // Step 3: trans data order from NHWC to NCHW
        outPtrS[i] = reinterpret_cast<void*>(reinterpret_cast<float*>(malloc(sizeof(float) * output_count)));
        std::vector<int> shape_for_output_cast(5, 1);  // 1,3,5,7全部是维度为5的输出
        int dimNum_for_output_cast = 5;
        cnrtGetOutputDataShape((int**)&shape_for_output_cast, &dimNum_for_output_cast, i, function);
        // std::cout << " " + fname + "_outputdata " << i << " NHWC shape is: " << shape_for_output_cast[0] << " " << 
        //     shape_for_output_cast[1] << " " << shape_for_output_cast[2] << " " << shape_for_output_cast[3] << " " << shape_for_output_cast[4] << std::endl;
        int dim_order_for_output_cast[4] = {3, 0, 1, 2};  // (1) 16 128 256 12 -> (1) 12 16 128 256 所以索引顺序为 3,0,1,2
        int dim_shape_for_output_cast[4] = {shape_for_output_cast[1], shape_for_output_cast[2], shape_for_output_cast[3], shape_for_output_cast[4]};  // NHWC
        auto start2 = std::chrono::high_resolution_clock::now();
        cnrtRet_t ret = cnrtTransDataOrder(computeData.outputCpuPtrS[i], CNRT_FLOAT32, outPtrS[i], 4, dim_shape_for_output_cast, dim_order_for_output_cast);
        auto end2 = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
        std::cout << " " << fname << " cnrtTransDataOrder time is: " << duration2.count() << "us. " << std::endl;
        // std::cout << " " << fname << " TransDataOrder End!" << std::endl;
        // Step 4: save 4 datas to file
        memcpy(computeData.outputCpuPtrS[i], outPtrS[i], sizeof(float) * output_count);
        // writeData(reinterpret_cast<float*>(computeData.outputCpuPtrS[i]), output_count, output_path + fname + "_forward_output_" + std::to_string(i));
      }
      free(outPtrS[i]);
      free(temp_output_cpu_data);
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << " " << fname << " output_align time is: " << duration.count() << "us. " << std::endl;

  return computeData;
}

void** input_align(void** inputMluPtrS, std::string fname, void** this_input_mlu, void** this_input_mlu2, void** this_input_cpu, int inputNum, int64_t* inputSizeS, cnrtDataType_t* input_data_type, cnrtFunction_t function){
  auto start = std::chrono::high_resolution_clock::now();
  if(fname == "cfnet1.cambricon"){
    void** inputCpuPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * inputNum));
    void** tempPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * inputNum));
    void* temp_input_cpu_data = nullptr;
    for (int i = 0; i < inputNum; i++) {
      int ip = inputSizeS[i] / cnrtDataTypeSize(input_data_type[i]); 
      auto databuf = reinterpret_cast<float*>(malloc(sizeof(float) * ip)); 
      switch(i) {
        case 3: readData(databuf, ip, "./data_6_5/imgl_data.txt"); break;
        case 4: readData(databuf, ip, "./data_6_5/imgR_data.txt"); break;
        case 2: readData(databuf, ip, "./data_6_5/disp_sparse_data.txt"); break;
        case 1: readData(databuf, ip, "./data_6_5/sparse_mask_data.txt"); break;
        case 0: linspace(0.0, 255, 256, databuf); break;
      }
      inputCpuPtrS[i] = reinterpret_cast<void*>(databuf);  // NCHW
    }
    // 更换维度
    for (int i = 0; i < inputNum; i++) {
      int ip = inputSizeS[i] / cnrtDataTypeSize(input_data_type[i]);
      tempPtrS[i] = reinterpret_cast<void*>(reinterpret_cast<float*>(malloc(sizeof(float) * ip)));
      std::vector<int> shape(4, 1);
      int dimNum = 4;
      cnrtGetInputDataShape((int**)&shape, &dimNum, i, function); // NHWC
      int dim_order[4] = {0, 2, 3, 1}; // 0 1 2 3-> NCHW  
      int dim_shape[4] = {shape[0], shape[3], shape[1], shape[2]};  // NCHW-> NHWC
      cnrtTransDataOrder(inputCpuPtrS[i], CNRT_FLOAT32, tempPtrS[i], 4, dim_shape, dim_order);
      temp_input_cpu_data = (void*)malloc(inputSizeS[i]);
      int input_count = inputSizeS[i] / cnrtDataTypeSize(input_data_type[i]);
      if (input_data_type[i] != CNRT_FLOAT32) {
        cnrtCastDataType(tempPtrS[i], CNRT_FLOAT32, temp_input_cpu_data, input_data_type[i], input_count, nullptr);
      } else {
        temp_input_cpu_data = tempPtrS[i];
      }
      cnrtMemcpy(inputMluPtrS[i], temp_input_cpu_data, inputSizeS[i], CNRT_MEM_TRANS_DIR_HOST2DEV);
      if (temp_input_cpu_data) {
        free(temp_input_cpu_data);
        temp_input_cpu_data = nullptr;
      }
    }
    free(inputCpuPtrS);
    free(tempPtrS);
  } else if (fname == "cfnet2.cambricon") {
    void* temp_input_cpu_data = nullptr;
    for (int i = 0; i < 2; i++) {  // handle cpu data from gather
	    int j = -1;
	    if (i == 0) j = 14;  // TODO modify index
	    else if (i == 1) j = 12;  // TODO modify index
      temp_input_cpu_data = (void*)malloc(inputSizeS[j]);
      int input_count = inputSizeS[j] / cnrtDataTypeSize(input_data_type[j]);
      if (input_data_type[j] != CNRT_FLOAT32)
        cnrtCastDataType(this_input_cpu[i], CNRT_FLOAT32, temp_input_cpu_data, input_data_type[j], input_count, nullptr);
      else
        temp_input_cpu_data = this_input_cpu[i];
	    // this_input_cpu[i] = temp_input_cpu_data;  错误写法
	    cnrtMemcpy(inputMluPtrS[j], temp_input_cpu_data, inputSizeS[j], CNRT_MEM_TRANS_DIR_HOST2DEV);
    }
    if (temp_input_cpu_data) {
      free(temp_input_cpu_data);
      temp_input_cpu_data = nullptr;
    }
    
    for (int i = 0; i < inputNum; i++) {
      switch(i) {
        case 0: cnrtMemcpy(inputMluPtrS[i], this_input_mlu[15], inputSizeS[i], CNRT_MEM_TRANS_DIR_DEV2DEV); break;
        case 1: cnrtMemcpy(inputMluPtrS[i], this_input_mlu[14], inputSizeS[i], CNRT_MEM_TRANS_DIR_DEV2DEV); break;
        case 2:
        {
          int lin_ip = inputSizeS[i] / cnrtDataTypeSize(input_data_type[i]);
          assert(lin_ip == 512);
          auto databuf = reinterpret_cast<float*>(malloc(sizeof(float) * lin_ip));
          linspace(0.0, 511, 512, databuf);
          void * templin = reinterpret_cast<void*>(databuf);
          void * temp_lin_cpu_data = (void*)malloc(inputSizeS[i]);
          if (input_data_type[i] != CNRT_FLOAT32)
            cnrtCastDataType(templin, CNRT_FLOAT32, temp_lin_cpu_data, input_data_type[i], lin_ip, nullptr);
          else
            temp_lin_cpu_data = templin;
          free(templin);
          cnrtMemcpy(inputMluPtrS[i], temp_lin_cpu_data, inputSizeS[i], CNRT_MEM_TRANS_DIR_HOST2DEV);
          free(temp_lin_cpu_data);
          break;
        }
        case 3: cnrtMemcpy(inputMluPtrS[i], this_input_mlu[13], inputSizeS[i], CNRT_MEM_TRANS_DIR_DEV2DEV); break;
        case 4: cnrtMemcpy(inputMluPtrS[i], this_input_mlu[12], inputSizeS[i], CNRT_MEM_TRANS_DIR_DEV2DEV); break;
        case 5: cnrtMemcpy(inputMluPtrS[i], this_input_mlu[11], inputSizeS[i], CNRT_MEM_TRANS_DIR_DEV2DEV); break;
        case 6: cnrtMemcpy(inputMluPtrS[i], this_input_mlu[10], inputSizeS[i], CNRT_MEM_TRANS_DIR_DEV2DEV); break;
        case 7: cnrtMemcpy(inputMluPtrS[i], this_input_mlu[9], inputSizeS[i], CNRT_MEM_TRANS_DIR_DEV2DEV); break;
        case 8: cnrtMemcpy(inputMluPtrS[i], this_input_mlu[8], inputSizeS[i], CNRT_MEM_TRANS_DIR_DEV2DEV); break;
        case 9: cnrtMemcpy(inputMluPtrS[i], this_input_mlu[6], inputSizeS[i], CNRT_MEM_TRANS_DIR_DEV2DEV); break;
        case 10: cnrtMemcpy(inputMluPtrS[i], this_input_mlu[2], inputSizeS[i], CNRT_MEM_TRANS_DIR_DEV2DEV); break;
        case 11: cnrtMemcpy(inputMluPtrS[i], this_input_mlu[4], inputSizeS[i], CNRT_MEM_TRANS_DIR_DEV2DEV); break;
        case 12: break;
        case 13: cnrtMemcpy(inputMluPtrS[i], this_input_mlu[0], inputSizeS[i], CNRT_MEM_TRANS_DIR_DEV2DEV); break;
        case 14: break;
      }
    }
  } else {
    void* temp_input_cpu_data = nullptr;
    for (int i = 0; i < 2; i++) {  // handle cpu data from gather
	    int j = -1;
	    if (i == 0) j = 9;  // TODO modify index
	    else if (i == 1) j = 7;  // TODO modify index
      temp_input_cpu_data = (void*)malloc(inputSizeS[j]);
      int input_count = inputSizeS[j] / cnrtDataTypeSize(input_data_type[j]);
      if (input_data_type[j] != CNRT_FLOAT32)
        cnrtCastDataType(this_input_cpu[i], CNRT_FLOAT32, temp_input_cpu_data, input_data_type[j], input_count, nullptr);
      else
        temp_input_cpu_data = this_input_cpu[i];
	    // this_input_cpu[i] = temp_input_cpu_data;  错误写法
	    cnrtMemcpy(inputMluPtrS[j], temp_input_cpu_data, inputSizeS[j], CNRT_MEM_TRANS_DIR_HOST2DEV);
    }
    if (temp_input_cpu_data) {
      free(temp_input_cpu_data);
      temp_input_cpu_data = nullptr;
    }
    
    for (int i = 0; i < inputNum; i++) {
      switch(i) {
        // this_input_mlu2保存cfnet1的未使用输出
        case 0: cnrtMemcpy(inputMluPtrS[i], this_input_mlu2[17], inputSizeS[i], CNRT_MEM_TRANS_DIR_DEV2DEV); break;
        case 1: cnrtMemcpy(inputMluPtrS[i], this_input_mlu2[18], inputSizeS[i], CNRT_MEM_TRANS_DIR_DEV2DEV); break;
        case 2: cnrtMemcpy(inputMluPtrS[i], this_input_mlu2[16], inputSizeS[i], CNRT_MEM_TRANS_DIR_DEV2DEV); break;
        case 3: cnrtMemcpy(inputMluPtrS[i], this_input_mlu[8], inputSizeS[i], CNRT_MEM_TRANS_DIR_DEV2DEV); break;
        case 4: cnrtMemcpy(inputMluPtrS[i], this_input_mlu[6], inputSizeS[i], CNRT_MEM_TRANS_DIR_DEV2DEV); break;
        case 5: cnrtMemcpy(inputMluPtrS[i], this_input_mlu[2], inputSizeS[i], CNRT_MEM_TRANS_DIR_DEV2DEV); break;
        case 6: cnrtMemcpy(inputMluPtrS[i], this_input_mlu[4], inputSizeS[i], CNRT_MEM_TRANS_DIR_DEV2DEV); break;
        case 7: break;
        case 8: cnrtMemcpy(inputMluPtrS[i], this_input_mlu[0], inputSizeS[i], CNRT_MEM_TRANS_DIR_DEV2DEV); break;
        case 9: break;
      }
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << " " << fname << " input_align time is: " << duration.count() << "us. " << std::endl;

  return inputMluPtrS;
}

// void ** forward(std::string fname, int Dev_use, void ** this_input_mlu, void ** this_input_cpu){
myData forward(std::string fname, int Dev_use, void ** this_input_mlu, void ** this_input_mlu2, void ** this_input_cpu){
  // begin function
  std::string name = (std::string)"subnet0";
  cnrtModel_t model;
  cnrtLoadModel(&model, fname.c_str());
  cnrtFunction_t function;
  cnrtRuntimeContext_t rt_ctx_;
  cnrtCreateFunction(&function);
  assert(cnrtExtractFunction(&function, model, name.c_str())==CNRT_RET_SUCCESS);
  cnrtCreateRuntimeContext(&rt_ctx_, function, NULL);
  cnrtQueue_t cnrt_queue;
  cnrtCreateQueue(&cnrt_queue);
  cnrtSetRuntimeContextDeviceId(rt_ctx_, Dev_use);
  cnrtInitRuntimeContext(rt_ctx_, NULL);

  // get network settings
  int inputNum, outputNum;
  int64_t* inputSizeS = nullptr;
  int64_t* outputSizeS = nullptr;
  cnrtGetInputDataSize(&inputSizeS, &inputNum, function);
  cnrtGetOutputDataSize(&outputSizeS, &outputNum, function);
  cnrtDataType_t* input_data_type = nullptr;
  cnrtDataType_t* output_data_type = nullptr;
  cnrtGetInputDataType(&input_data_type, &inputNum, function);
  cnrtGetOutputDataType(&output_data_type, &outputNum, function);
  cnrtNotifier_t notifierBeginning, notifierEnd;
  cnrtCreateNotifier(&notifierBeginning);
  cnrtCreateNotifier(&notifierEnd);
  float event_time_use;

  // get buffer
  void** param = reinterpret_cast<void**>(malloc(sizeof(void*) * (inputNum + outputNum)));
  void** inputMluPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * inputNum));
  // void** outputMluPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * outputNum));
  struct myData computeData{};  // 实例化结构体
  computeData.outputMluPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * outputNum));
  computeData.outputCpuPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * outputNum));
  for (int i = 0; i < inputNum; i++) {
    cnrtMalloc(&inputMluPtrS[i], inputSizeS[i]);
    param[i] = inputMluPtrS[i];  // 赋相同地址
  }
  for (int i = 0; i < outputNum; i++) {
    cnrtMalloc(&computeData.outputMluPtrS[i], outputSizeS[i]);
    param[inputNum + i] = computeData.outputMluPtrS[i];
  }
  // param -> inputMluPtrS -> this_input_mlu
  // get input data
  // 此处尽管把真实地input data存到了inputMluPtrs中，但因为param[i]和inputMluPtrS[i]指向同一块内存区域，实际上param也可以访问到这些input data
  inputMluPtrS = input_align(inputMluPtrS, fname, this_input_mlu, this_input_mlu2, this_input_cpu, inputNum, inputSizeS, input_data_type, function);

  // run MLU
  auto start = std::chrono::high_resolution_clock::now();
  cnrtPlaceNotifier(notifierBeginning, cnrt_queue);
  CNRT_CHECK(cnrtInvokeRuntimeContext(rt_ctx_, param, cnrt_queue, nullptr));
  cnrtPlaceNotifier(notifierEnd, cnrt_queue);
  assert(cnrtSyncQueue(cnrt_queue) == CNRT_RET_SUCCESS);
  cnrtNotifierDuration(notifierBeginning, notifierEnd, &event_time_use);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << " " << fname << " forward time is: " << duration.count() << "us. " << std::endl;

  for (int i = 0; i < inputNum; i++) {
    cnrtFree(inputMluPtrS[i]);
  }

  computeData = output_align(fname, outputNum, outputSizeS, output_data_type, function, computeData);
  
  cnrtDestroyNotifier(&notifierBeginning);
  cnrtDestroyNotifier(&notifierEnd);
  cnrtDestroyRuntimeContext(rt_ctx_);
  cnrtDestroyQueue(cnrt_queue);
  cnrtUnloadModel(model);
  cnrtDestroyFunction(function);
  
  return computeData;
}
int main(int argc, char* argv[]) {
  cnrtInit(0);
  // unsigned devNum;
  // cnrtGetDeviceCount(&devNum);

  cnrtDev_t dev;
  int Dev_use = 1;
  cnrtGetDeviceHandle(&dev, Dev_use); 
  cnrtSetCurrentDevice(dev);

  auto start2 = std::chrono::high_resolution_clock::now();
  std::string fname1 = (std::string)"cfnet1.cambricon";
  struct myData computeData1{};  // 实例化结构体
  computeData1 = forward(fname1, Dev_use, NULL, NULL, NULL);
  // 因为forward返回的computeData.outputCpuPtrS[i]已经都转换成了NCHW的四维数据，所以对应shape也应变为NCHW排布的四维数据
  vector<int> data_index_shape1 = {1, 12, 16, 128, 256};
  vector<int> data_index_shape2 = {1, 160, 16, 128, 256};
  // two gather
  // 此时computeData.outputCpuPtrS[i]中已经都是转换回NCHW的4维数据，入参的1, 2, ...代表第几个gather用于命名，无实际含义
  auto start = std::chrono::high_resolution_clock::now();
  float* gather_output_forward1_1 = gather_cpu(computeData1.outputCpuPtrS[3], data_index_shape1, 
                                                computeData1.outputCpuPtrS[1], data_index_shape1, 4, fname1, 1);
  float* gather_output_forward1_2 = gather_cpu(computeData1.outputCpuPtrS[7], data_index_shape2, 
                                                computeData1.outputCpuPtrS[5], data_index_shape2, 4, fname1, 2);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << " " << fname1 << " gather execution time is: " << duration.count() << "us. " << std::endl;                                              
  void ** this_input_cpu = reinterpret_cast<void**>(malloc(sizeof(float*) * 2));
  this_input_cpu[0] = gather_output_forward1_1;
  this_input_cpu[1] = gather_output_forward1_2;

  std::string fname2 = (std::string)"cfnet2.cambricon";
  struct myData computeData2{};  // 实例化结构体
  computeData2 = forward(fname2, Dev_use, computeData1.outputMluPtrS, NULL, this_input_cpu);
  // 因为forward返回的computeData.outputCpuPtrS[i]已经都转换成了NCHW的四维数据，所以对应shape也应变为NCHW排布的四维数据
  vector<int> data_index_shape3 = {1, 6, 12, 256, 512};
  vector<int> data_index_shape4 = {1, 80, 12, 256, 512};
  // two gather
  // 此时computeData.outputCpuPtrS[i]中已经都是转换回NCHW的4维数据，入参的1, 2, ...代表第几个gather用于命名，无实际含义
  start = std::chrono::high_resolution_clock::now();
  float* gather_output_forward2_1 = gather_cpu(computeData2.outputCpuPtrS[3], data_index_shape3, 
                                                computeData2.outputCpuPtrS[1], data_index_shape3, 4, fname2, 1);
  float* gather_output_forward2_2 = gather_cpu(computeData2.outputCpuPtrS[7], data_index_shape4, 
                                                computeData2.outputCpuPtrS[5], data_index_shape4, 4, fname2, 2);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << " " << fname2 << " gather execution time is: " << duration.count() << "us. " << std::endl;                                               
  this_input_cpu[0] = gather_output_forward2_1;
  this_input_cpu[1] = gather_output_forward2_2;

  std::string fname3 = (std::string)"cfnet3.cambricon";
  struct myData computeData3{};  // 实例化结构体
  computeData3 = forward(fname3, Dev_use, computeData2.outputMluPtrS, computeData1.outputMluPtrS, this_input_cpu);

  auto end2 = std::chrono::high_resolution_clock::now();
  auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
  std::cout << " Total execution time is: " << duration2.count() << "us. " << std::endl;

  exit(0);
}
  /*
  float* gather_output_1 = gather_cpu(outputCpuPtrS[3], shape_input, outputCpuPtrS[1],shape_index,2);

  std::stringstream ss;
  ss << output_path << fname << "gather_output" ;
  std::string output_name = ss.str();
  std::ofstream fout(output_name, std::ios::out);
  int output_gather_count = accumulate(shape_index.begin(), shape_index.end(),1,multiplies<int>());
  fout << std::flush;
  for(int j = 0 ; j < output_gather_count; j++){
    fout << ((float*)gather_output_1)[j] << std::endl;
  }
  fout<<std::flush;
  fout.close();

  cnrtDestroyRuntimeContext(rt_ctx_);
  cnrtUnloadModel(model);
  cnrtDestroyFunction(function);

  gettimeofday(&tpend,NULL);
  execTime = 1000000 * (tpend.tv_sec - tpstart.tv_sec) +
    tpend.tv_usec - tpstart.tv_usec;
   */
  // LOG(INFO) << " cfnet1_gather time: " << execTime << " us";

  // gettimeofday(&tpstart,NULL);

  // cnrtModel_t model2;
  // cnrtRuntimeContext_t ctx2;
  // cnrtQueue_t queue2;
  // std::string fname2 = (std::string)"cfnet2.cambricon";
  // int size2;
  // cnrtGetModelSize(fname2.c_str(), &size2);
  // assert(cnrtLoadModel(&model2, fname2.c_str())==CNRT_RET_SUCCESS);
  // cnrtFunction_t function2;

  // gettimeofday(&tpstart,NULL);

  // cnrtModel_t model2;
  // cnrtRuntimeContext_t ctx2;
  // cnrtQueue_t queue2;
  // std::string fname2 = (std::string)"cfnet2.cambricon";
  // int size2;
  // cnrtGetModelSize(fname2.c_str(), &size2);
  // assert(cnrtLoadModel(&model2, fname2.c_str())==CNRT_RET_SUCCESS);
  // cnrtFunction_t function2;
  // std::string name2 = (std::string)"subnet0";
  // assert(cnrtCreateFunction(&function2)==CNRT_RET_SUCCESS);
  // assert(cnrtExtractFunction(&function2, model2, name2.c_str())==CNRT_RET_SUCCESS);
  // assert(cnrtCreateRuntimeContext(&ctx2, function2, NULL)==CNRT_RET_SUCCESS);
  // assert(cnrtSetRuntimeContextDeviceId(ctx2, 0)==CNRT_RET_SUCCESS);
  // cnrtRet_t rett = cnrtInitRuntimeContext(ctx2, NULL);
  // if (rett != CNRT_RET_SUCCESS){
  //   // LOG(FATAL)<<"Failed to initialize runtime context";
  // }

  // cnrtRuntimeContextCreateQueue(ctx2,&queue2);

  // int inputNum2;
  // int outputNum2;
  // int64_t* inputSizeS2 = nullptr;
  // int64_t* outputSizeS2 = nullptr;
  
  // cnrtGetInputDataSize(&inputSizeS2, &inputNum2, function2);
  // cnrtGetOutputDataSize(&outputSizeS2, &outputNum2, function2);
  // cnrtDataType_t* input_data_type2 = nullptr;
  // cnrtDataType_t * output_data_type2 = nullptr;
  // cnrtGetInputDataType(&input_data_type2, &inputNum2, function2);
  // cnrtGetOutputDataType(&output_data_type2, &outputNum2, function2);

  // void ** param2 = reinterpret_cast<void**>(malloc(sizeof(void*) * (inputNum2 + outputNum2)));
  // void ** inputMluPtrS2 = reinterpret_cast<void**>(malloc(sizeof(void*) * inputNum2));
  // void ** outputMluPtrS2 = reinterpret_cast<void**>(malloc(sizeof(void*) * outputNum2));


  // cnrtFree(outputMluPtrS[3]);
  // cnrtFree(outputMluPtrS[5]);
  // cnrtFree(outputMluPtrS[7]);
  // cnrtFree(outputMluPtrS[1]);

  // for(int i = 0 ; i< inputNum2 ; i++){
  //   switch(i){
  //     case 0:
  //       inputMluPtrS2[14] = gather_output_mlu1;
  //     break;
  //     case 1:
  //       inputMluPtrS2[13] = outputMluPtrS[0];
  //     break;
  //     case 2:
  //       inputMluPtrS2[10] = outputMluPtrS[2];
  //     break;
  //     case 3:
  //       inputMluPtrS2[12] = gather_output_mlu2;
  //     break;
  //     case 4:
  //       inputMluPtrS2[11] = outputMluPtrS[4];
  //     break;
  //     case 5:
  //       inputMluPtrS2[9] = outputMluPtrS[6];
  //       break;
  //     case 6:
  //       inputMluPtrS2[8] =outputMluPtrS[8] ;
  //       break;
  //     case 7:
  //       inputMluPtrS2[7] = outputMluPtrS[9] ;
  //       break;
  //     case 8:
  //       inputMluPtrS2[5] = outputMluPtrS[10] ;
  //       break;
  //     case 9:
  //       inputMluPtrS2[6] = outputMluPtrS[11];
  //       break;
  //     case 10:
  //       inputMluPtrS2[4] = outputMluPtrS[12];
  //       break;
  //     case 11:
  //       inputMluPtrS2[3] = outputMluPtrS[13];
  //       break;
  //       int tempdim_shape[4] = {sshape[0], sshape[3],
  //                       sshape[1], sshape[2]};  // NCHW
  //       cnrtTransDataOrder(templin, CNRT_FLOAT32, tempshape,4,tempdim_shape, tempdim_order);
  //       free(templin);
  //       cnrtMalloc(&inputMluPtrS2[2], inputSizeS2[2]);
  //       void * temp_lin_cpu_data = (void*)malloc(inputSizeS2[2]);
  //       if(input_data_type2[2] != CNRT_FLOAT32){
  //         cnrtCastDataType(tempshape,
  //                       CNRT_FLOAT32,
  //                       temp_lin_cpu_data,
  //                       input_data_type2[2],
  //                       lin_ip,
  //                       nullptr);
  //       }
  //       else{
  //                         temp_lin_cpu_data = tempshape;
  //                       }
  //       free(tempshape);
  //       cnrtMemcpy(inputMluPtrS2[2],temp_lin_cpu_data,inputSizeS2[2],CNRT_MEM_TRANS_DIR_HOST2DEV);
  //       break;
  //   }
  // }

  // for(int i = 0; i < outputNum2; i++){
  //   assert(cnrtMalloc(&outputMluPtrS2[i], outputSizeS2[i])==CNRT_RET_SUCCESS);
  // }
  // for (int i = 0; i < inputNum2; i++) {
  //   param2[i] = inputMluPtrS2[i];
  // }
  // for (int i = 0; i < outputNum2; i++) {
  //   param2[inputNum2 + i] = outputMluPtrS2[i];
  // }

  // float event_time_use2;
  // CNRT_CHECK(cnrtInvokeRuntimeContext(ctx2, param2, queue2, nullptr));
  // if (cnrtSyncQueue(queue2) == CNRT_RET_SUCCESS) {
  // } else {
  //   // LOG(INFO) << " SyncQueue Error ";
  // }

  // for (int i = 0 ; i< inputNum2 ; i++){
  //   cnrtFree(inputMluPtrS2[i]);
  // }


  // gettimeofday(&tpend,NULL);
  // execTime = 1000000 * (tpend.tv_sec - tpstart.tv_sec) +
  //   tpend.tv_usec - tpstart.tv_usec;
  // // LOG(INFO) << " cfnet2 time: " << execTime << " us";

  // gettimeofday(&tpstart,NULL);

  // std::vector<int> shape_input2(5,1); int dimNum2 = 5;
  // cnrtGetOutputDataShape((int **)&shape_input2, &dimNum2, 3, function2);
  // // LOG(INFO)<<"the shape of right_feature_map: "<<" "<<shape_input[1]<<" "<<shape_input[2]<<" "<<shape_input[3]<<" "<<shape_input[4];
  // int input_shape2[] = {shape_input2[1], shape_input2[2], shape_input2[3], shape_input2[4]};
  // std::vector<int> shape_index2(5,1); int dimNum_index2 = 5;
  // cnrtGetOutputDataShape((int **)&shape_index2, &dimNum_index2, 1, function2);
  // int index_shape2[] = {shape_index2[1], shape_index2[2], shape_index2[3], shape_index2[4]};
  // void* gather_output1_mlu_cfnet2 = gather_mlu(outputMluPtrS2[3], input_shape2, outputMluPtrS2[1],index_shape2,CNML_DIM_C,queue2);
  
  // cnrtGetOutputDataShape((int **)&shape_input2, &dimNum2, 7, function2);
  // input_shape2[0] = shape_input2[1]; input_shape2[1] = shape_input2[2]; input_shape2[2] = shape_input2[3]; input_shape2[3] = shape_input2[4];
  // cnrtGetOutputDataShape((int **)&shape_index2, &dimNum_index2, 5, function2);
  // index_shape2[0] = shape_index2[1]; index_shape2[1] = shape_index2[2]; index_shape2[2] = shape_index2[3]; index_shape2[3] = shape_index2[4];
  // void* gather_output2_mlu_cfnet2 = gather_mlu(outputMluPtrS2[7], input_shape2, outputMluPtrS2[5],index_shape2,CNML_DIM_C,queue2);
  
  // cnrtDestroyRuntimeContext(ctx2);
  // cnrtUnloadModel(model2);
  // cnrtDestroyFunction(function2);
  // cnrtDestroyQueue(cnrt_queue);
  // cnrtDestroyQueue(queue2);

  // cnrtFree(outputMluPtrS2[3]);
  // cnrtFree(outputMluPtrS2[5]);
  // cnrtFree(outputMluPtrS2[7]);
  // cnrtFree(outputMluPtrS2[1]);

  // gettimeofday(&tpend,NULL);
  // execTime = 1000000 * (tpend.tv_sec - tpstart.tv_sec) +
  //   tpend.tv_usec - tpstart.tv_usec;
  // // LOG(INFO) << " cfnet2_gather time: " << execTime << " us";

  // gettimeofday(&tpstart,NULL);

  // cnrtModel_t model3;
  // cnrtRuntimeContext_t ctx3;
  // cnrtQueue_t queue3 ;
  // std::string fname3= (std::string)"cfnet3.cambricon";
  // // LOG(INFO)<<"load file: "<<fname3;
  // int size3;
  // cnrtGetModelSize(fname3.c_str(),&size3);
  // // LOG(INFO)<<"model size"<<size3;
  // assert(cnrtLoadModel(&model3,fname3.c_str())==CNRT_RET_SUCCESS);
  // cnrtFunction_t function3;
  // std::string name3 = (std::string)"subnet0";
  // assert(cnrtCreateFunction(&function3)==CNRT_RET_SUCCESS);
  // assert(cnrtExtractFunction(&function3, model3, name3.c_str())==CNRT_RET_SUCCESS);
  // assert(cnrtCreateRuntimeContext(&ctx3, function3, NULL)==CNRT_RET_SUCCESS);
  // assert(cnrtSetRuntimeContextDeviceId(ctx3, 0)==CNRT_RET_SUCCESS);
  // cnrtRet_t initret = cnrtInitRuntimeContext(ctx3, NULL);
  // if (initret != CNRT_RET_SUCCESS){
  //   // LOG(FATAL)<<"Failed to initialize runtime context";

  // cnrtModel_t model3;
  // cnrtRuntimeContext_t ctx3;
  // cnrtQueue_t queue3 ;
  // std::string fname3= (std::string)"cfnet3.cambricon";
  // // LOG(INFO)<<"load file: "<<fname3;
  // int size3;
  // cnrtGetModelSize(fname3.c_str(),&size3);
  // // LOG(INFO)<<"model size"<<size3;
  // assert(cnrtLoadModel(&model3,fname3.c_str())==CNRT_RET_SUCCESS);
  // cnrtFunction_t function3;
  // std::string name3 = (std::string)"subnet0";
  // assert(cnrtCreateFunction(&function3)==CNRT_RET_SUCCESS);
  // assert(cnrtExtractFunction(&function3, model3, name3.c_str())==CNRT_RET_SUCCESS);
  // assert(cnrtCreateRuntimeContext(&ctx3, function3, NULL)==CNRT_RET_SUCCESS);
  // assert(cnrtSetRuntimeContextDeviceId(ctx3, 0)==CNRT_RET_SUCCESS);
  // cnrtRet_t initret = cnrtInitRuntimeContext(ctx3, NULL);
