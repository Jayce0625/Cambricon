/*************************************************************************
 * Copyright (C) [2019] by Cambricon, Inc.
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *************************************************************************/

/************************************************************************
 *
 *  @file cnrt.h
 *
 *  @brief Runtime APIs provide programmable interfaces for users to develop
 *  their-owned programs, which includes device management, context
 *  management, memory management of both sides (devices and hosts), etc.
 *
 **************************************************************************/

#ifndef __CNRT_H
#define __CNRT_H

#define CNRT_MAJOR_VERSION 4
#define CNRT_MINOR_VERSION 10
#define CNRT_PATCH_VERSION 1

#define CNRT_VERSION (CNRT_MAJOR_VERSION * 10000 + CNRT_MINOR_VERSION * 100 + CNRT_PATCH_VERSION)

/************************************************************************
 *  Include files
 ************************************************************************/
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#if defined(__cplusplus)
extern "C" {
#endif /*__cplusplus*/

/************************************************************************
 *  Definitions
 ************************************************************************/
/**< DLL exports controller. */
#if defined(WIN32) || defined(WINDOWS)
#ifdef USE_CNRT_DLL
#ifdef CNRT_DLL_EXPORTS
#define CNRT_DLL_API __declspec(dllexport)
#else /*CNRT_DLL_EXPORTS*/
#define CNRT_DLL_API __declspec(dllimport)
#endif /*CNRT_DLL_EXPORTS*/
#else
#define CNRT_DLL_API
#endif /*USE_CNRT_DLL*/
#else  /*WIN32 || WINDOWS*/
#define CNRT_DLL_API
#endif /*WIN32 || WINDOWS*/

/**< struct tailed */
#define CNRT_PARAM_END (void *)0xFFFFFFFF

/************************************************************************
 *  Data type declaration
 ************************************************************************/

#ifndef __CAMB_TYPES_H
#define __CAMB_TYPES_H
#if defined(WIN32) || defined(WINDOWS)
typedef unsigned __int64 u64_t;
typedef __int64 i64_t;
typedef unsigned __int32 u32_t;
typedef unsigned __int16 u16_t;
typedef unsigned __int8 u8_t;
typedef signed __int32 i32_t;
typedef signed __int16 i16_t;
typedef signed __int8 i8_t;

#else /*!WIN32 || WINDOWS*/

typedef uint64_t u64_t;
typedef int64_t i64_t;
typedef uint32_t u32_t;
typedef uint16_t u16_t;
typedef uint8_t u8_t;
typedef int32_t i32_t;
typedef int16_t i16_t;
typedef int8_t i8_t;

#endif /*WIN32||WINDOWS*/
#endif /*__CAMB_TYPES*/

/*!
 * @brief An enum.
 *
 * Enumeration variables describing the return values of CNRT APIs.
 */
typedef enum {
  CNRT_RET_SUCCESS = 0,
  /*!< The function call is completed with no errors. In case of
   *   ::cnrtQueryNotifier and ::cnrtQueryQueue calls,
   *   this also represents that the operation being queried is completed,
   */
  CNRT_RET_WARNING_FAKE_DEVICE = 1,
  /*!< A fake device is used currently.*/
  CNRT_RET_ERR_NOT_READY = 632006,
  /*!< Asynchronous operations issued previously
   *   have not been completed yet.*/
  CNRT_RET_ERR_INVALID = 632007,
  /*!< The value of one or more the parameters passed to this function
   *   call is not in the supported range. */
  CNRT_RET_ERR_NOMEM = 632008,
  /*!< Unable to allocate enough
   *   memory to perform the requested operation.*/
  CNRT_RET_ERR_NODEV = 632009,
  /*!< No MLU device is detected. */
  CNRT_RET_ERR_SYS = 632011,
  /*!< A system function call returns error. */
  // reserve
  CNRT_RET_ERR_ACCES = 632012,
  CNRT_RET_ERR_ACCESS = 632012,
  /*!< The current device cannot directly access the memory
   *   from the peer device. */
  CNRT_RET_ERR_FAULT = 632013,
  /*!< The memory address cannot be found. */
  CNRT_RET_ERR_BUSY = 632014,
  /*!< Deprecated, device or resource busy. */
  CNRT_RET_ERR_NORES = 632019,
  /*!< Insufficient resources for the input parameters. */
  CNRT_RET_ERR_UNSUPPORTED = 632020,
  /*!< The current operation is not supported. */
  CNRT_RET_ERR_INVALID_POINTER = 632021,
  /*!< The input pointer is invalid. */
  CNRT_RET_ERR_NO_EXIST = 632022,
  /*!< The resource cannot be found. */
  CNRT_RET_ERR_BROKEN = 632023,
  /*!< Data transmission is broken. */
  CNRT_RET_ERR_INIT = 632024,
  /*!< The current environment has not been
   *   initialized or an error occurred during initialization. */
  CNRT_RET_ERR_QUEUE = 632025,
  /*!< Errors occurred during retrieving the default queue. */
  CNRT_RET_ERR_OUT_RANGE = 632026,
  /*!< The input value is out of range. */
  CNRT_RET_ERR_FUNC_CALL = 632028,
  /*!< Errors occurred during calling runtime functions. */
  CNRT_RET_ERR_UNHANDLED = 632029,
  /*!< Errors occurred during calling the ::cnrtInit function. */
  CNRT_RET_ERR_INVALID_TYPE = 632030,
  /*!< The data type of the input parameter is invalid. */
  CNRT_RET_ERR_INVALID_OP = 632031,
  /*!< The current operation is invalid. */
  CNRT_RET_ERR_RESHAPE = 632035,
  /*!< Errors occurred during the reshape operation. */
  CNRT_RET_ERR_MEMCPY = 632036,
  /*!< Errors occurred during transmitting data. */
  CNRT_RET_ERR_ENCRYPT = 632037,
  /*!< Errors occurred during encrypting data. */
  CNRT_RET_ERR_INVALID_DATADESC = 632038,
  /*!< The data descriptor is invalid. */
  CNRT_RET_ERR_MAP = 632039,
  /*!< A mapping error occurred when calling the ::cnrtMap or
   *   ::cnrtMapRange function.*/
  CNRT_RET_ERR_UNMAP = 632040,
  /*!< A unmapping error occurred when calling the ::cnrtUnmap function. */
  CNRT_RET_ERR_CACHE = 632041,
  /*!< A cache operation error occurred when calling the
   *   ::cnrtCacheOperation or ::cnrtCacheOperationRange function. */
  CNRT_RET_ERR_FIND_DEV_ADDR = 632042,
  /*!< An error occurred during finding the device address from
   *   the mapped address with the
   *   ::cnrtFindDevAddrByMappedAddr or ::cnrtFindDevAddrWithOffsetByMappedAddr function. */
  CNRT_RET_ERR_KERNEL_VERSION_TOO_HIGH = 632043,
  /*!< The version of kernel bin compiled by CNCC is too high. */
  CNRT_RET_ERR_CNDRV_FUNC_CALL = 632046,
  /*!< Errors occurred from driver functions that are returned via CNRT. See the detailed
   *   error messages on the screen.
   */
  CNRT_RET_ERR_CNDEV_FUNC_CALL = 632047,
  /*!< Errors occurred from CNDEV functions that are returned via CNRT.*/
  CNRT_RET_ERR_IO,
  /*!< Deprecated, I/O error. */
  CNRT_RET_ERR_TIMEOUT,
  /*!< Deprecated, time expired. */
  CNRT_RET_ERR_NOSYS,
  /*!< Deprecated, function not implemented. */
  CNRT_RET_ERR_AGAIN,
  /*!< Deprecated, try again later. */
  CNRT_RET_ERR_EXIST,
  /*!< Deprecated, a resource or file already exists. */
  CNRT_RET_ERR_MATH_OVERFLOW,
  /*!< Deprecated, math results not representable. */
  CNRT_RET_ERR_MLU,
  /*!< Deprecated, MLU error. */
  CNRT_RET_ERR_ONCHIP_CORE,
  /*!< Deprecated, on-chip core error. */
  CNRT_RET_ERR_GET_MEM_RANGE,
  /*!< Deprecated, failure on get mem range. */
  CNRT_RET_ERR_GET_MEM_INFO,
  /*!< Deprecated, failure on get mem info. */
  CNRT_RET_ERR_EVENT,
  /*!< Deprecated, failure on event. */
  CNRT_RET_ERR_NOTIFIER,
  /*!< Deprecated, failure on notifier. */
  CNRT_RET_ERR_GET_IMP_DEV,
  /*!< Deprecated, get device handle error. */
  CNRT_RET_ERR_LOAD_MODULE,
  /*!< Ununsed, load module error. */
  CNRT_RET_ERR_UNKNOWN = 999991,
  /*!< An unknown error. */
  CNRT_RET_ERR_MAX
  /*!< The error on upper boundary. */
} cnrtRet_t;

/*!
 * @brief An enum.
 *
 * Enumeration variables describing memory types available
 * for allocating memory on the host.
 */
typedef enum {
  CNRT_MEMTYPE_DEFAULT = 0,
  /*!< The page-locked type of the host memory with extra support for IPC. */
  CNRT_MEMTYPE_LOCKED,
  /*!< Deprecated. Do not use. The page-locked type of the host memory. */
  CNRT_MEMTYPE_DEV
  /*!< Deprecated. Do not use. The device memory. */
} cnrtMemType_t;

/*!
 * @brief
 *
 * Enumeration variables describing malloc types available for cnrtMallocBufferEx.
 * Internal used. */
typedef enum { CNRT_MALLOC_EX_PARALLEL_FRAMEBUFFER = 1 } cnrtMallocExType_t;

/**< Execution modes of tasks on MLU. */
/*! @brief An enum.
 *
 *  Enumeration variables describing the number of cores used
 *  for computation on the MLU devices.
 */
typedef enum {
  CNRT_FUNC_TYPE_BLOCK = 1,
  /*!< One MLU core is used to execute tasks. */
  CNRT_FUNC_TYPE_BLOCK0 = CNRT_FUNC_TYPE_BLOCK,
  /*!< The IP core 0 is used to execute tasks. */
  CNRT_FUNC_TYPE_BLOCK1 = CNRT_FUNC_TYPE_BLOCK0 + 1,
  /*!< The IP heterogeneous core 1 is used to execute tasks. */
  CNRT_FUNC_TYPE_UNION1 = 4,
  /*!< Four MLU cores is used to execute tasks. */
  CNRT_FUNC_TYPE_UNION2 = 8,
  /*!< Eight MLU cores is used to execute tasks. */
  CNRT_FUNC_TYPE_UNION4 = 16,
  /*!< 16 MLU cores is used to execute tasks. */
  CNRT_FUNC_TYPE_UNION8 = 32,
  /*!< 32 MLU cores is used to execute tasks. */
  CNRT_FUNC_TYPE_UNION16 = 64,
  /*!< 64 MLU cores is used to execute tasks. */
  CNRT_FUNC_TYPE_MUTABLE = -1,
  /*!< Deprecated. Do not use. */
  CNRT_JOB_TYPE_BLOCK = CNRT_FUNC_TYPE_BLOCK,
  /*!< One MLU core is used to execute tasks.
   *   This is only used for tensor dimension mutable function.*/
  CNRT_JOB_TYPE_UNION1 = CNRT_FUNC_TYPE_UNION1,
  /*!< Four MLU cores is used to execute tasks.
   *   This is only used for tensor dimension mutable function.*/
  CNRT_JOB_TYPE_UNION2 = CNRT_FUNC_TYPE_UNION2,
  /*!< Eight MLU cores is used to execute tasks.
   *   This is only used for tensor dimension mutable function.*/
  CNRT_JOB_TYPE_UNION4 = CNRT_FUNC_TYPE_UNION4,
  /*!< 16 MLU cores is used to execute tasks.
   *   This is only used for tensor dimension mutable function.*/
} cnrtFunctionType_t,
    cnrtJobType_t;

/*!
 * @brief An enum.
 *
 * Enumeration variables describing which memory channel of a MLU device
 * to be used.
 * MLU220 only has one memory channel, so only
 * CNRT_CHANNEL_TYPE_DUPLICATE, CNRT_CHANNEL_TYPE_NONE, and
 * CNRT_CHANNEL_TYPE_0 are supported for MLU220. MLU270 has
 * four memory channels, so all the values are supported for MLU270.
 */
typedef enum {
  CNRT_CHANNEL_TYPE_DUPLICATE = -2,
  /*!< Interleave access to all memory channels on the device. */
  CNRT_CHANNEL_TYPE_NONE = -1,
  /*!< A random memory channel is used. */
  CNRT_CHANNEL_TYPE_0 = 0,
  /*!< Memory channel 0 is used. */
  CNRT_CHANNEL_TYPE_1,
  /*!< Memory channel 1 is used. */
  CNRT_CHANNEL_TYPE_2,
  /*!< Memory channel 2 is used. */
  CNRT_CHANNEL_TYPE_3
  /*!< Memory channel 3 is used. */
} cnrtChannelType_t;

/*! @brief An enum.
 *
 * Enumeration variables describing the direction of data copy.
 */
typedef enum {
  CNRT_MEM_TRANS_DIR_HOST2DEV = 0,
  /*!< From host to device. */
  CNRT_MEM_TRANS_DIR_DEV2DEV,
  /*!< Data copy in a single device. */
  CNRT_MEM_TRANS_DIR_DEV2HOST,
  /*!< From device to host. */
  CNRT_MEM_TRANS_DIR_HOST2HOST,
  /*!< From host to host. */
  CNRT_MEM_TRANS_DIR_PEER2PEER,
  /*!< From a device to another device for p2p. */
  CNRT_MEM_TRANS_DIR_NODIR,
  /*!< Data copy without the specified direction. */
} cnrtMemTransDir_t;

/*! @brief An enum.
 *
 *  Enumeration variables describing the operations on cache.
 *  This enum is only used on Edge platform.
 */
typedef enum {
  CNRT_FLUSH_CACHE = 1,
  /*!< Flush dcache of the host CPU. */
  CNRT_INVALID_CACHE = 2,
  /*!< Invalidate dcache of the host CPU, currently reserved. */
} cnrtCacheOps_t;

/**< Parameter for function call. */
/*!
 * @struct cnrtDim3_t
 * @brief A struct.
 *
 * cnrtDim3_t is a structure describing grid dimensions used for
 * task execution.
 *
 */
typedef struct {
  unsigned int x;
  /*!< The X axis. The value of x equals to the number of tasks
   *   you want to run on each core multiplies 4.
   */
  unsigned int y;
  /*!< The Y axis. Each task is run ``y*z`` times.*/
  unsigned int z;
  /*!< The Z axis. Each task is run ``y*z`` times.*/
} cnrtDim3_t;

/**< Parameter for invoke function call. */
/*!
 *  @struct cnrtInvokeFuncParam_t
 *  @brief A struct.
 *
 *  Deprecated. Parameters which need to be invoked by the user. */
typedef struct {
  int *data_parallelism;  /*!< Data parallelism.*/
  unsigned int *affinity; /*!< Affinity.*/
  void *end;              /*!< End of struct.*/
} cnrtInvokeFuncParam_t;

/**< Type of cnrtInvokeParam. */
/*! @brief An enum
 * Enumeration variables describing a mode for launching a kernel or a CNRT Runtime Context, such as
 * enabling the cluster affinity function.
 *
 * This struct is used in the ::cnrtInvokeParam_t structure. */
typedef enum {
  CNRT_INVOKE_PARAM_TYPE_0 = 0,
  /*!< Enable the cluster affinity function used in launching a kernel or a CNRT Runtime Context.
   * This parameter is used together with the ::cnrtClusterAffinity_t struct that specifies the
   * clusters to be used.
   */
} cnrtInvokeParamType_t;

/*!
 *  @struct cnrtClusterAffinity_t
 *  @brief The cnrtClusterAffinity_t is a structure describing the MLU cluster affinity used in
 * launching a kernel or a CNRT Runtime Context.
 *
 * This structure is used on MLU hardware platforms with mlutiple clusters. Each MLU cluster
 * corresponds to a DDR channel with the lowest latency, so accessing to other channels has a
 * relatively higher latency. For example, the MLU cluster 0/1/2/3 accesses the corresponding DDR
 * 0/1/2/3 with the lowest latency. You can specify which cluster to be used to improve the
 * performance.
 *
 * If the cluster you specified is busy, the calculation task will wait until the cluster can be
 * used.
 *
 * This structure is only used in ::cnrtInvokeParam_t structure. To enable this function, you need
 * to set ``CNRT_INVOKE_PARAM_TYPE_0`` in the ::cnrtInvokeParamType_t enum.
 */
typedef struct {
  unsigned int *affinity;
  /*!< The number of clusters you can use is based on the MLU hardware platform. Each cluster is
   *   declared with bitwise mark.
   *
   *   - 0x01 represents cluster 0.
   *   - 0x02 represents cluster 1.
   *   - 0x04 represents cluster 2.
   *   - 0x08 represents cluster 3.
   *
   *   For example, if you set affinity to 0x03, the cluster 0 and cluster 1 are used in the kernel
   *   launch.
   */
} cnrtClusterAffinity_t;

/**< Parameter for function call. */
/*!
 *  @struct cnrtInvokeParam_t
 *  @brief The cnrtInvokeParam_t is a structure describing additional functions, such as cluster
 * affinity, used in launching a kernel or a CNRT Runtime Context.
 *
 * This struct is used in the ::cnrtInvokeRuntimeContext or ::cnrtInvokeRuntimeContext_V2 function
 * to
 * invoke a CNRT Runtime Context, or in the ::cnrtInvokeKernel_V3 to launch a kernel.
 */
typedef struct {
  cnrtInvokeParamType_t invoke_param_type;
  /*!< Enable or disable the cluster affinity function. */
  cnrtClusterAffinity_t cluster_affinity;
  /*!< Specify which clusters to be used for cluster affinity in launching a kernel or a CNRT
   *   Runtime Context. */
} cnrtInvokeParam_t;

/**< Data type and data order.*/
/*! @brief An enum.
 *
 * Enumeration variables describing the data types supported by CNRT.
 */
typedef enum cnrtDataType {
  CNRT_INVALID = 0x0,
  /*!< Invalid data type. */
  CNRT_FLOAT16 = 0x12,
  /*!< The 16-bit floating-point data type is used. */
  CNRT_FLOAT32 = 0x13,
  /*!< The 32-bit floating-point data type is used. */
  CNRT_FLOAT64 = 0x14,
  /*!< The 64-bit floating-point data type is used. */
  CNRT_INT4 = 0x20,
  /*!< The 4-bit integer data type is used.*/
  CNRT_INT8 = 0x21,
  /*!< The 8-bit integer data type is used. */
  CNRT_INT16 = 0x22,
  /*!< The 16-bit integer data type is used. */
  CNRT_INT32 = 0x23,
  /*!< The 32-bit integer data type is used. */
  CNRT_INT64 = 0x24,
  /*!< The 64-bit integer data type is used. */
  CNRT_AUTO = 0x25,
  /*!< The bit-width integer data type, such as int8, int6 that is automatically selected for
     best performance and precision in computing. */
  CNRT_UINT8 = 0x31,
  /*!< The 8-bit unsigned integer data type is used. */
  CNRT_UINT16 = 0x32,
  /*!< The 16-bit unsigned integer data type is used. */
  CNRT_UINT32 = 0x33,
  /*!< The 32-bit unsigned integer data type is used. */
  CNRT_FIX8 = 0x41,
  /*!< Deprecated. The 8-bit fixed-point data type is used. */
  CNRT_QUANT8 = 0x51,
  /*!< Deprecated. The 8-bit quantized data type is used. */
  CNRT_BOOL = 0x61,
  /*!< The Boolean data type is used. */
} cnrtDataType_t;

/*! @brief An enum.
 *
 * Enumeration variables describing the format of the data layout.
 *
 * The data can be defined in three, four, or five dimensions.
 *
 * Take images for example, the format of the data layout can be NCHW:
 *
 * - N: The number of images.
 * - C: The number of image channels.
 * - H: The height of the images.
 * - W: The width of the images.
 */
typedef enum cnrtDimOrder {
  CNRT_NCHW = 0x0123,
  /*!< The data layout is in the following order: batch size, channel, height, and width. */
  CNRT_NHWC = 0x0231,
  /*!< The data layout is in the following order: batch size, height, width, and channel. */
  CNRT_HWCN = 0x2310,
  /*!< The data layout is in the following order: height, width, channel, and batch size. */
  CNRT_TNC = 0x401,
  /*!< Deprecated. The data layout is in the following order for RNN exclusive: time sequence,
     batch size, and vector length. */
  CNRT_NTC = 0x041,
  /*!< Deprecated. The data layout is in the following order for RNN exclusive: batch size, time
     sequence, and vector length. */
  CNRT_NCDHW = 0x01523,
  /*!< The data layout is in the following order for 3-dimension image with depth information:
     batch size, channel, depth, height, and width. */
  CNRT_NDHWC = 0x05231,
  /*!< The data layout is in the following order for 3-dimension image with depth information:
     batch size, depth, height, width, and channel. */
  CNRT_DHWCN = 0x52310,
  /*!< The data layout is in the following order for 3-dimension image with depth information:
    depth, height, width, channel, and batch size. */
} cnrtDimOrder_t;

/*! @brief An enum.
 *
 *  Enumeration variables describing CNRT Runtime Context information.
 *
 *  This struct is used in the ::cnrtGetRuntimeContextInfo function to get information from a CNRT
 * Runtime Context.
 */
typedef enum cnrtRuntimeContextInfo {
  CNRT_RT_CTX_FUNCTION = 1,
  /*!< The CNRT Function information in the CNRT Runtime Context. */
  CNRT_RT_CTX_DEV_ORDINAL = 2,
  /*!< The MLU device ID used for running the CNRT Runtime Context. */
  CNRT_RT_CTX_CORE_NUMBER = 3,
  /*!< Deprecated. The core number set by compile time.*/
  CNRT_RT_CTX_MODEL_PARALLEL = 4,
  /*!< The core number used for computing in the CNRT Runtime Context. */
  CNRT_RT_CTX_CHANNEL = 5,
  /*!< Deprecated. The channel of the device memory. */
  CNRT_RT_CTX_MAX_BATCH_NUM = 6,
  /*!< Deprecated. The maximum batch number. */
} cnrtRuntimeContextInfo_t;

/*!
 * @brief An enum.
 *
 * Enumeration variables describing the core version of MLU device
 * that is used in the ::cnrtDeviceInfo_t struct.
 *
 */
typedef enum cnrtCoreVersion {
  CNRT_1H8 = 0,     /*!< 1H8 hardware. */
  CNRT_1H16 = 1,    /*!< 1H16 hardware. */
  CNRT_1H8MINI = 4, /*!< 1H8MINI hardware. */
  CNRT_MLU100 = 3,  /*!< MLU100 hardware. */
  CNRT_MLU270 = 5,  /*!< MLU270 hardware. */
  CNRT_MLU220 = 6,  /*!< MLU220 hardware, namely the Edge platform. */
  CNRT_MLU290 = 7,  /*!< MLU290 hardware. */
  CNRT_1M20 = 8,    /*!< 1M20 product. */
  CNRT_1M70 = 9,    /*!< 1M70 product. */
  CNRT_CISS = 10,   /*!< Reserved. */
  CNRT_MLU370 = 11, /*!< MLU370 hardware. */
} cnrtCoreVersion_t;

/*!
 * @struct cnrtDeviceInfo_t
 * @brief A struct.
 *
 * The cnrtDeviceInfo_t is a structure describing the information about
 * MLU device.
 *
 * This struct is used in the ::cnrtGetDeviceInfo function.
 */
typedef struct {
  char device_name[64];           /*!< The device name. */
  cnrtCoreVersion_t core_version; /*!< The core version of MLU device. */
  int core_num;                   /*!< The total number of device cores. */
} cnrtDeviceInfo_t;

/**< Device affinity information. */
/*!
 *  @struct cnrtDeviceAffinity_t
 *  @brief A struct.
 *
 *  The cnrtDeviceAffinity_t is a structure describing the information about the
 *  device affinity.
 *
 *  The ::cnrtGetDeviceAffinity function is used to retrieve the device affinity.
 */
typedef struct {
  uint32_t cpu_count;
  /*!< The total number of CPUs that can set the affinity for the MLU devices. */
  uint32_t cpu_affinity_bitmap[1024];
  /*!< CPU affinity bit mask. Each cpu_affinity_bitmap can store
  *    32-bit CPU bit masks based on the total
  *    number of CPU cores in reverse order. For example, if you
  *    have 80 CPU cores in the system, CPU affinity bit masks are stored
  *    in the following format:
  *
  *   - cpuAffinityBitMap[0]: 15-0 bit: CPU79 - CPU64. 31-16 bit: unused.
  *   - cpuAffinityBitMap[1]: 31-0 bit: CPU63 - CPU32.
  *   - cpuAffinityBitMap[2]: 31-0 bit: CPU31 - CPU0.
  */
} cnrtDeviceAffinity_t;

/*!
 *  @brief An enum.
 *
 *  Enumeration variables describing the topology relationship
 *  of multiple MLU devices.
 *
 *  The ::cnrtTopologyGetRelationship function is used to retrieve
 *  the relationship between two MLU devices. The
 *  ::cnrtTopologyGetNearestDevices function is used to retrieve all
 *  the devices that have the topology relationship with a specified
 *  device.
 */
typedef enum {
  CNRT_TOPO_SELF = 0,
  /*!< The device to be retrieved itself.*/
  CNRT_TOPO_INTERNAL = 1,
  /*!< Devices that are on the same board.*/
  CNRT_TOPO_SINGLE = 2,
  /*!< All devices that traverse all devices only through a PCIe switch.*/
  CNRT_TOPO_MULTIPLE = 3,
  /*!< All devices that traverse all devices through multiple PCIe switches,
   *   but not through the host bridge.*/
  CNRT_TOPO_HOST_BRIDGE = 4,
  /*!< All devices that are connected to the same host bridge.*/
  CNRT_TOPO_CPU = 5,
  /*!< All devices that are connected to the same CPU.*/
  CNRT_TOPO_SYSTEM = 6
  /*!< All devices in the system.*/
} cnrtTopologyRelationshipEnum_t;

/*!
 *  @brief An enum.
 *
 *  Enumeration variables describing the queue flag used for the
 *  current process executions on the current device.
 *
 *  The ::cnrtSetDeviceFlag function is used to set the flag.
 *  The ::cnrtGetDeviceFlag function is used to retrieve the flag you set.
 *
 */
typedef enum {
  CNRT_QUEUE_SYNC_SPIN = 0,
  /*!< CPU actively spins when waiting for the device execution result.*/
  CNRT_QUEUE_SYNC_BLOCK = 1,
  /*!< CPU thread is blocked on a synchronization primitive when
   *   waiting for the device execution results.*/
  CNRT_QUEUE_SYNC_YIELD = 2,
  /*!< CPU thread yields when waiting for the device execution results.*/
  CNRT_QUEUE_SYNC_MAX
  /*!< The error on upper bondary.*/
} cnrt_queue_sync_type;

/*!
 *  @struct cnrtQuantizedParam
 *  @brief A struct.
 *
 *  The cnrtQuantizedParam is a structure describing the parameters that are used for data
 *  quantization.
 *
 *  The ::cnrtCreateQuantizedParam or ::cnrtCreateQuantizedParamByChannel function is used to create
 *  a cnrtQuantizedParam instance for casting data type via the ::cnrtCastDataType or
 *  ::cnrtTransOrderAndCast function. The ::cnrtDestroyQuantizedParam function is used to destroy an
 *  instance.
 */
struct cnrtQuantizedParam;
/*! Pointer to the ::cnrtQuantizedParam struct that holds the parameters used for data
 *  quantization. */
typedef struct cnrtQuantizedParam *cnrtQuantizedParam_t;

/**< Model and function. */
/*!
 *  @struct cnrtModel
 *  @brief A struct.
 *
 *  The cnrtModel is a structure describing information about offline models.
 *
 *  A cnrtModel structure instance is created by loading an offline model form a file via the
 *  ::cnrtLoadModel function or from the memory via the ::cnrtLoadModelFromMem function. The
 *  ::cnrtUnloadModel function is used to destroy an instance. */
struct cnrtModel;
/*! Pointer to the ::cnrtModel structure that holds the information about an offline model file. */
typedef struct cnrtModel *cnrtModel_t;

/*! The IPC memory handle. Pointer to void by default. */
typedef void *cnrtIpcMemHandle;

/*!
 *  @struct cnrtFunction
 *  @brief A struct.
 *
 *  cnrtFunction is a structure describing CNRT Functions that holds the information used in
 *  computing.
 *
 *  The CNRT Function is used to create ::cnrtRuntimeContext via the ::cnrtCreateRuntimeContext
 *  function. If a ::cnrtModel instance is destroyed, the CNRT Functions extracted from the
 *  corresponding model file will be invalid.
 *
 *  A cnrtFunction structure instance is created via the ::cnrtCreateFunction function. The
 *  information of the CNRT Function is extracted from a ::cnrtModel instance via the
 *  ::cnrtExtractFunction function. The cnrtFunction structure instance is destroyed via the
 *  ::cnrtDestroyFunction function.
 *
 */
struct cnrtFunction;
/*! Pointer to the ::cnrtFunction structure that holds the CNRT Function information. */
typedef struct cnrtFunction *cnrtFunction_t;

/*!
 *  @struct cnrtParamDesc
 *  @brief A struct
 *
 *  The cnrtParamDesc is a structure describing the shape, order, and data type of input or output
 *  parameters.
 *
 *  The structure instance is created via the ::cnrtCreateParamDesc function. The
 *  ::cnrtSetShapeToParamDesc, ::cnrtSetNameToParamDesc, and ::cnrtSetDataTypeToParamDesc functions
 *  are used to specify shape, order, and data type of the parameters. The ::cnrtDestroyParamDesc is
 *  used to destroy the structure instance.
 */
struct cnrtParamDesc;
/*! Pointer to the ::cnrtParamDesc structure that holds information about input and output
 *  parameters. */
typedef struct cnrtParamDesc *cnrtParamDesc_t;
/*! Pointer to a set of ::cnrtParamDesc_t that points to the information about input and output
 *  parameters defined in the ::cnrtParamDesc struct. */
typedef struct cnrtParamDesc **cnrtParamDescArray_t;

/*!
 *  @struct cnrtQueue
 *  @brief A struct.
 *
 *  The cnrtQueue is a structure describing queue types.
 *
 */
struct cnrtQueue;
/*! A pointer to the ::cnrtQueue struct holding the information about a queue.
 *
 *  The ::cnrtCreateQueue and ::cnrtDestroyQueue function are used to create
 *  and destroy an instance of cnrtQueue_t respectively.
 */
typedef struct cnrtQueue *cnrtQueue_t;

/*!
 *  @struct cnrtNotifier
 *  @brief A struct.
 *
 *  The cnrtNotifier is a structure describing notifier types.
 *
 */
struct cnrtNotifier;
/*! A pointer to the ::cnrtNotifier struct holding the information
 *  about a notifier.
 *
 *  The ::cnrtCreateNotifier and ::cnrtDestroyNotifier functions are
 *  used to create and destroy an instance of cnrtNotifier_t respectively.
 *
 */
typedef struct cnrtNotifier *cnrtNotifier_t;

/*!
 *  @struct cnrtRuntimeContext
 *  @brief A struct.
 *
 *  The cnrtRuntimeContext is a structure describing CNRT Runtime Context created based on CNRT
 *  Function.
 *
 *  The structure instance that can be created from CNRT Function or another CNRT Runtime Context
 *  via the ::cnrtCreateRuntimeContext or ::cnrtForkRuntimeContext functions accordingly, and is
 *  destroyed via ::cnrtDestroyRuntimeContext function. The instance is invoked via the
 *  ::cnrtInvokeRuntimeContext or ::cnrtInvokeRuntimeContext_V2 function.
 */
struct cnrtRuntimeContext;
/*! Pointer to ::cnrtRuntimeContext structure that holds information about CNRT Runtime Context. */
typedef struct cnrtRuntimeContext *cnrtRuntimeContext_t;

/*! The cnrtDev_t is the unsigned int64 data type. */
typedef u64_t cnrtDev_t;

/*! The MLUDev_t is the unsigned int64 data type. */
typedef u64_t MLUdev_t;

/*!
 *  @struct cnrtPluginOpDimInfo
 *  @brief A struct.
 *
 *  A semi-internal structure describing the dimension of the plugin operator. */
struct cnrtPluginOpDimInfo;
/*! A pointer to the structure holding the information about plugin operator dimension. */
typedef struct cnrtPluginOpDimInfo *cnrtPluginOpDimInfo_t;

/*!
 * @struct cnrtKernelParamsBuffer.
 * @brief A struct
 *
 * A structure describing the information about parameter buffer used for kernel launch.
 *
 * The ::cnrtGetKernelParamsBuffer function is used to retrieve the information about the
 * parameter buffer. The ::cnrtKernelParamsBufferAddParam function is used to add a parameter
 * to the buffer. The ::cnrtKernelParamsBufferMarkInput function is used to add an input
 * parameter to the buffer. The ::cnrtKernelParamsBufferMarkOutput function is used to add
 * an output parameter to the buffer. The ::cnrtKernelParamsBufferMarkStatic function is used
 * to add a static parameter to the buffer. The ::cnrtDestroyKernelParamsBuffer function is
 * used to destroy a parameter buffer.
 */
typedef struct cnrtKernelParamsBuffer {
  void *host_ptr;
  /*!< Pointer to the base address of the parameters in the buffer. */
  unsigned int max_param;
  /*!< The maximum number of parameters that can be stored in the buffer.
   *   The value should not be greater than 3072.
   */
  unsigned int cur_param;
  /*!< The number of parameters that are stored in the buffer.*/
  int *input_index;
  /*!< Pointer to the base address of the input parameters in the buffer. */
  int num_input;
  /*!< The number of input parameters that are stored in the buffer.*/
  int *output_index;
  /*!< Pointer to the base address of the output parameters in the buffer. */
  int num_output;
  /*!< The number of output parameters that are stored in the buffer.*/
  int *static_index;
  /*!< Pointer to the base address of the static parameters in the buffer. */
  int num_static;
  /*!< The number of output parameters that are stored in the buffer.*/
  cnrtPluginOpDimInfo_t dim_info;
  /*!< Only reserved for further usage.*/
  int num_dim_info;
  /*!< Only reserved for further usage.*/
} * cnrtKernelParamsBuffer_t;

/**< Compiler. */
/*!
 * @struct cnrtKernelInitParam.
 * @brief A struct.
 *
 * cnrtKernelInitParam is a structure describing the information used for initializing
 * kernel parameters in a buffer.
 */
struct cnrtKernelInitParam;

/*! cnrtKernelInitParam_t is a pointer to the ::cnrtKernelInitParam structure that holds
 *  the information used for initializing kernel parameters in a buffer.
 *
 *  The ::cnrtCreateKernelInitParam function is used to create one instance. The
 *  ::cnrtInitKernelMemory function is used to initialize the kernel parameters. The
 *  ::cnrtDestroyKernelInitParamAndMemory function is used to destroy the instance.
 */
typedef struct cnrtKernelInitParam *cnrtKernelInitParam_t;

/************************************************************************
 * Function prototype declaration
 ************************************************************************/

/************************************************************************
 * Error handling
 ************************************************************************/

/**
 * @brief Retrieves the error message string by an error code.
 *
 * @param  err_code[in] The error code returned from the previous function call.
 * @return A pointer to a constant string.
 *
 * @details Returns the error message string corresponding to the given error code
 * ``err_code``.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 @verbatim
 int main () {
   // Initialize device
   cnrtInit(0);

   // Get device handle
   cnrtDev_t dev;
   cnrtGetDeviceHandle(&dev, 0);

   // Set the device handle for current thread execution context
   cnrtSetCurrentDevice(dev);

   cnrtRet_t ret = CNRT_RET_SUCCESS;
   size_t size = sizeof(int) * N;
   void *dev_addr = NULL;

   ret = cnrtMalloc(&dev_addr, size);
   ret = cnrtFree(dev_addr);
   ret = cnrtFree(dev_addr);
   const char *error_name = cnrtGetErrorName(ret);
   const char *error_str = cnrtGetErrorStr(ret);
   printf("error_name: %s\n", error_name);
   printf("error_str: %s\n", error_str);

   ret = cnrtGetLastErr();
   printf("error_code: %d\n", ret);

   return 0;
 }
 @endverbatim
 */
extern CNRT_DLL_API const char *cnrtGetErrorStr(cnrtRet_t err_code);

/**
 * @brief Retrieves the error name string by an error code.
 *
 * @param  err_code[in] The error code returned from the previous function call.
 * @return A pointer to an error code name string.
 *
 * @details Returns the error name string corresponding to the given error code
 * ``err_code``.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtGetErrorStr function.
 */
extern CNRT_DLL_API const char *cnrtGetErrorName(cnrtRet_t err_code);

/**
 * @brief Retrieves the last error code returned from CNRT function call.
 *
 *
 * @return An error code of the CNRT function call.
 *
 * @details Returns the last error code returned from the CNRT function call in the
 *          same host thread.
 *
 * @note
 * - Once called ::cnrtGetLastErr, the error code will be reset to CNRT_RET_SUCCESS.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 @verbatim
 // __mlu_global__ function must be written in a single file named *.mlu
 __mlu_global__ void bangKernelAdd(int *a, int *b) {
   *a = *a + *b;
 }

 // Hybrid programming
 int main () {
   // Initialize device
   cnrtInit(0);

   // Get device handle
   cnrtDev_t dev;
   cnrtGetDeviceHandle(&dev, 0);

   // Set the device handle for current thread execution context
   cnrtSetCurrentDevice(dev);

   int a, b;
   uint32_t seed = 0x123;
   a = (int)rand_r(&seed) % 10000;
   b = (int)rand_r(&seed) % 10000;

   cnrtDim3_t dim;
   dim.x = 1;
   dim.y = 1;
   dim.z = 1;
   cnrtFunctionType_t type = CNRT_FUNC_TYPE_BLOCK;

   cnrtQueue_t queue;
   cnrtCreateQueue(&queue);

   int *k_a, *k_b;
   cnrtMalloc((void **)&k_a, sizeof(int));
   cnrtMalloc((void **)&k_b, sizeof(int));
   cnrtMemcpy(k_a, &a, sizeof(int), CNRT_MEM_TRANS_DIR_HOST2DEV);
   cnrtMemcpy(k_b, &b, sizeof(int), CNRT_MEM_TRANS_DIR_HOST2DEV);

   // Launch kernel
   bangKernelAdd(k_a, k_b, dim, type, queue);
   printf("cnrtPeekLastErr, %d\n", cnrtPeekLastErr());
   CNRT_CHECK(cnrtGetLastErr());
   cnrtSyncQueue(queue);

   // Free resources
   cnrtFree(k_a);
   cnrtFree(k_b);
   cnrtDestroyQueue(queue);
   cnrtDestroy();

   return 0;
 }
 @endverbatim
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetLastErr(void);

/**
 * @brief Retrieves the last error code returned from the CNRT function call without resetting.
 *
 * @return An error code of the CNRT function call.
 *
 * @details Returns the last error code returned from the CNRT function call.
 *
 * @note
 * - The error code will not be reset to CNRT_RET_SUCCESS after ::cnrtPeekLastErr.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtGetLastErr function.
*/
extern CNRT_DLL_API cnrtRet_t cnrtPeekLastErr(void);

/**
 *
 * @brief Checks the error code and prints the error message.
 *
 * @param result[out]: The error code of the function call.
 * @param func[in]: The name of CNRT function to be checked.
 * @param file[out]: The __FILE__ macro that expands the name of the
 *              input file.
 * @param line[out]: The __LINE__ macro that expands to the line number
 *              in which the error occurred.
 * @return Void.
 *
 * @details Returns the error code ``result`` and macro __FILE__
 *          and __LINE__ for the given function ``func``.
 *          This function is used in macro CNRT_CHECK() for debugging.
 *
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
extern CNRT_DLL_API void cnrtCheck(cnrtRet_t result,
                                   const char *const func,
                                   const char *const file,
                                   int const line);

#define CNRT_CHECK(val) cnrtCheck((val), #val, __FILE__, __LINE__)

/****************************************************************************
 * Runtime Device Management
 *
 * This section describes the device management functions of the runtime
 * application programming interface.
 *
 * @{
 *****************************************************************************/

/**
 * @brief Initializes the runtime environment in the current process space.
 *
 * @param flags[in] If you set the value of this parameter to 0, the real
          device is initialized. If you set the value of this parameter to 1,
          the fake device is initialized.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because the
 *         value of the ``flags`` parameter is invalid.
 * @retval CNRT_RET_ERR_INVALID_TYPE This function call failed because the
 *         ``flags`` parameter is set to different values if calling the
 *         ::cnrtInit function more than once within a process.
 * @retval CNRT_RET_ERR_NODEV This function call failed because probing the
 *         real devices failed.
 * @retval CNRT_RET_ERR_FUNC_CALL This function call failed because errors
 *         from driver functions are returned via CNRT. See the detailed
 *         error messages on the screen.
 * @retval CNRT_RET_ERR_INIT This function call failed because the value of
 *         the CNRT_DEFAULT_DEVICE environment variable, which specifies the
 *         device to be used for the current thread execution context is invalid.
 *         or errors occurred when checking the version of the driver functions.
 * @retval CNRT_RET_ERR_NOMEM This function call failed because not enough memory
 *         can be allocated.
 *
 * @details Initializes either a real device or a fake device with ``flags``
 *          in the current process space.
 *
 *
 * @note
 * - You need to call this function before calling any runtime functions to ensure
 *   the thread security.
 * - You need to call the ::cnrtDestroy function to destroy the device resources.
 * - If check the version of driver functions failed, it will return CNRT_RET_ERR_INIT.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
@verbatim
int main(void) {
  cnrtInit(0);

  // user programming code
  // ......

  cnrtDestroy();
}
@endverbatim
 *
 */
extern CNRT_DLL_API cnrtRet_t cnrtInit(unsigned int flags);

/**
 * @brief Destroys everything that are allocated by all the runtime function calls.
 *
 * @retval void: None.
 *
 * @details Destroys and cleans up all the resources associated with the current
 * device in the current process, and closes all the opened devices.
 *
 * @note
 * - This function should be called at the end of any other runtime function calls.
 * - The number you called ::cnrtInit should be the same with the number you called
 *   this function in the program. So you need to call
 *   this function at the end of the program for each ::cnrtInit you called.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtInit function.
 *
 */
extern CNRT_DLL_API void cnrtDestroy(void);

/**
 * @brief Retrieves the device handle by a device ordinal.
 *
 * @param  pdev[out] A pointer to the device handle.
 * @param  ordinal[in] The device ordinal to retrieve. You can call the
 *                     ::cnrtGetDeviceCount function to get the total
 *                     number of devices in your system.
 *                     The device ID is in the range [0, ::cnrtGetDeviceCount() – 1].
 *                     The value ``-1`` represents a fake device.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because
 *         ``pdev`` is NULL or the value of ``ordinal`` is invalid.
 * @retval CNRT_RET_ERR_NODEV This function call failed because no MLU devices
 *                            were detected.
 *
 * @details Returns the device handle pointed by ``pdev`` with the given device ordinal
 * ``ordinal``. This function is used to set the device handle for the current
 * thread execution context with the ::cnrtSetCurrentDevice function.
 *
 * @note
 * - If the device used for the process is set in the ``CNRT_DEFAULT_DEVICE``
 *   environment variable, this function returns the handle of the device set in this
 *   environment variable. The ``ordinal`` parameter you set in this
 *   function will be ignored.
 * - This function should be called after calling the ::cnrtInit function.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
@verbatim
int main(void) {
  unsigned int dev_num = 0;
  cnrtDev_t dev;

  cnrtInit(0);
  // Get the total number of devices
  cnrtGetDeviceCount(&dev_num);
  if (dev_num < 1) {
    return -1;
  }

  int ordinal = dev_num - 1;
  // Get the device handle
  cnrtGetDeviceHandle(&dev, ordinal);
  // Set the current device
  cnrtSetCurrentDevice(dev);

  // ......

  // Get the current device handle
  cnrtGetCurrentDevice(&dev);

  // user program code
  // ......

  cnrtDestroy();
}
@endverbatim
 *
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetDeviceHandle(cnrtDev_t *pdev, int ordinal);

/**
 * @brief Sets the device handle for the current thread execution context.
 *
 * @param  dev[in] The device handle to be set.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because
 *         the ``dev`` parameter is 0 or an invalid handle.
 * @retval CNRT_RET_ERR_NODEV This function call failed because an error
 *         occurred when opening the device.
 *
 * @details Sets the device handle ``dev`` for the current thread execution context.
 * Any subsequent runtime function calls will use the device you set in this
 * function for execution.
 *
 * @note
 * - If the device you set is not opened, the device will be opened automatically
 *   after calling this function.
 * - If you want to use multi-devices to process tasks, you can call this function
 *   to set a different device for each thread separately.
 * - This function should be called after calling the ::cnrtInit function.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtGetDeviceHandle function.
 *
 */
extern CNRT_DLL_API cnrtRet_t cnrtSetCurrentDevice(cnrtDev_t dev);

/**
 * @brief Retrieves the device handle for the current thread execution context.
 *
 * @param  pdev[out] A pointer to the device handle.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because
 *         ``pdev`` is 0.
 * @retval CNRT_RET_ERR_NODEV This function call failed because an error
 *         occurred when opening the device, or the device cannot be found.
 *
 * @details Returns the current device handle pointed by ``pdev`` for the
 *          current thread execution context.
 *
 * @note
 * - If the device used for the process is set in the ``CNRT_DEFAULT_DEVICE``
 *   environment variable, this function returns the handle of the device set in this
 *   environment variable.
 * - If the device you set is not opened, the device will be opened automatically
 *   after calling this function.
 * - If the ::cnrtSetCurrentDevice function is not called, this function will return
 *   the handle of the default device, that is, the handle of device ordinal 0.
 * - This function should be called after calling the ::cnrtInit function.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtGetDeviceHandle function.
 *
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetCurrentDevice(cnrtDev_t *pdev);

/**
 * @brief Retrieves the total number of MLU devices in the system.
 *
 * @param  dev_num[out] Pointer to the total number of devices.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because
 *         ``dev_num`` is NULL.
 * @retval CNRT_RET_ERR_FUNC_CALL This function call failed because errors
 *         from driver functions are returned via CNRT. See the detailed
 *         error messages on the screen.
 *
 * @details Returns in ``dev_num`` the total number of MLU devices in
 * the system.
 *
 * @note
 * - This function should be called after calling the ::cnrtInit function.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtGetDeviceHandle function.
 *
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetDeviceCount(unsigned int *dev_num);

/**
 * @brief Retrieves the current device ordinal.
 *
 * @param  dev_ordinal[out] Pointer to the current device ordinal.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because
 *         ``dev_ordinal`` is NULL.
 *
 * @details Returns in ``dev_ordinal`` the current device ordinal.
 *
 * @note
 * - If current device is fake device, -1 is returned in ``dev_ordinal``.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 @verbatim
int main(void) {
  cnrtInit(0);
  unsigned int dev_num = 0;
  // Get total device number
  cnrtGetDeviceCount(&dev_num);
  if (dev_num < 1) {
    return -1;
  }

  int index = 0;
  while ((dev_num)--) {
    cnrtDev_t dev;
    cnrtGetDeviceHandle(&dev, index);
    cnrtSetCurrentDevice(dev);
    unsigned int dev_ordinal;
    cnrtGetCurrentOrdinal(&dev_ordinal);
    printf("device_%d, ordinal: %d\n", index++, dev_ordinal);
  }

  cnrtDestroy();
}
@endverbatim

 */
extern CNRT_DLL_API cnrtRet_t cnrtGetCurrentOrdinal(unsigned int *dev_ordinal);

/**
 * @brief Retrieves the device information by device ordinal.
 *
 * @param info[out] The information about the specified device.
 * @param device_ordinal[in] The device ordinal to retrieve. You can call the
 *                           ::cnrtGetDeviceCount function to get the total
 *                           number of devices in your system.
 *                           The device ID is in the range [0, ::cnrtGetDeviceCount() – 1].
 *                           The value ``-1`` represents a fake device.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because
 *         ``info`` is NULL, or the value of ``device_ordinal`` is smaller than 0.
 * @retval CNRT_RET_ERR_NODEV This function call failed because the device with the
 *         given device ordinal is not found, or retrieving the attribute from driver failed.
 * @retval CNRT_RET_ERR_FUNC_CALL This function call failed because errors
 *         from driver functions are returned via CNRT. See the detailed
 *         error messages on the screen.
 *
 * @details Returns in ``info`` the device information with the given device ordinal
 * ``device_ordinal``. The device information specified in the ::cnrtDeviceInfo_t struct,
 * includes device name, device core version, and the total number of device cores.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
@verbatim
int main(void) {
  unsigned int dev_num = 0;
  cnrtDev_t dev;

  cnrtInit(0);
  // Get the device number
  cnrtGetDeviceCount(&dev_num);
  if (dev_num < 1) {
    return -1;
  }

  int ordinal = 0;
  // Get the device handle
  cnrtGetDeviceHandle(&dev, ordinal);
  // Set the current device
  cnrtSetCurrentDevice(dev);

  cnrtDeviceInfo_t devInfo = {0};
  // Get the current device handle
  cnrtGetDeviceInfo(&devInfo, ordinal);

  // user program code
  // ......

  cnrtDestroy();
}
@endverbatim
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetDeviceInfo(cnrtDeviceInfo_t *info, int device_ordinal);

/**
 * @brief  Waits for the current device within the current process to complete all the
 *         preceding tasks.
 *
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_NODEV This function call failed because unable to get the current
 *         device.
 *
 * @details Blocks any further execution on the host until all the operations in the queues
 * within the current process on the current MLU device are completed. This function is used
 * for more than one queues within a process on the device.
 *
 *
 * @note
 * - This function should be called after calling the ::cnrtInit function.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
@verbatim
int main(void) {
  unsigned int dev_num = 0;
  cnrtDev_t dev;

  cnrtInit(0);
  // Get the device number
  cnrtGetDeviceCount(&dev_num);
  if (dev_num < 1) {
    return -1;
  }

  int ordinal = 0;
  // Get the device handle
  cnrtGetDeviceHandle(&dev, ordinal);
  // Set the current device
  cnrtSetCurrentDevice(dev);

  void *devPoint1 = NULL;
  void *devPoint2 = NULL;
  size_t size = 1024 * 1024;
  // Allocate the device memory
  cnrtMalloc(&devPoint1, size);
  cnrtMalloc(&devPoint2, size);
  void *hostPoint = malloc(size);
  if ((!hostPoint) || (!devPoint1) || (!devPoint2)) {
    return -1;
  }

  // Create two queue on the same device
  cnrtQueue_t queue[2];
  cnrtCreateQueue(&queue[0]);
  cnrtCreateQueue(&queue[1]);

  // Copy data in the two queue
  cnrtMemcpyAsync(devPoint1, hostPoint, size, queue[0], CNRT_MEM_TRANS_DIR_HOST2DEV);
  cnrtMemcpyAsync(devPoint2, hostPoint, size, queue[1], CNRT_MEM_TRANS_DIR_HOST2DEV);

  // Synchronize all the queues on the current device
  cnrtSyncDevice();

  cnrtDestroyQueue(queue[0]);
  cnrtDestroyQueue(queue[1]);
  cnrtFree(devPoint1);
  cnrtFree(devPoint2);
  cnrtFree(devPoint2);
  free(hostPoint);
  cnrtDestroy();
}
@endverbatim
 *
 */
extern CNRT_DLL_API cnrtRet_t cnrtSyncDevice(void);

/**
 * @brief Retrieves the device handle used for CNDRV API from thread execution context.
 *
 * @param  mlu_dev[out] Pointer to the device handle used for CNDRV API.
 * @param  cnrt_dev[in] Pointer to the CNRT device handle. You can get the current
 *         device handle by calling the ::cnrtGetCurrentDevice function.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because
 *         ``mlu_dev`` is NULL, or the value of ``cnrt_dev`` is 0.
 * @retval CNRT_RET_WARNING_FAKE_DEVICE This function call failed because
 *         a fake device is set in ``cnrt_dev``.
 * @retval CNRT_RET_ERR_FUNC_CALL This function call failed because errors
 *         from driver functions are returned via CNRT. See the detailed
 *         error messages on the screen.
 *
 * @details Returns in ``mlu_dev`` the device handle used for CNDRV API
 *          with the given CNRT device handle ``cnrt_dev``.
 *
 * @note
 * - To use CNDRV APIs, you need to call this function to get the device handle.
 * - This function should be called after calling the ::cnrtSetCurrentDevice
 *   or ::cnrtGetCurrentDevice function.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
@verbatim
int main(void) {
  unsigned int dev_num = 0;
  cnrtDev_t dev;

  cnrtInit(0);
  // Get the device number
  cnrtGetDeviceCount(&dev_num);
  if (dev_num < 1) {
    return -1;
  }

  int ordinal = 0;
  // Get the device handle
  cnrtGetDeviceHandle(&dev, ordinal);
  // Set the current device
  cnrtSetCurrentDevice(dev);

  // Get the device handle of CNDRV
  MLUdev_t mlu_dev;
  cnrtGetMLUdev(&mlu_dev, dev);

  // Call the CNDRV API to allocate device memory
  MLUaddr pAddr;
  mluMalloc(&pAddr, mlu_dev, 1024, 1024, -2);

  // ......

  mluFree(mlu_dev, pAddr);
  cnrtDestroy();
}
@endverbatim
 *
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetMLUdev(MLUdev_t *mlu_dev, cnrtDev_t cnrt_dev);

/**
 * @brief  Sets flags used for the current process executions on the current device.
 *
 * @param  flags[in] Options of how CPU interacts with the OS scheduler when
 *         waiting for the device execution result. Supported values are:
 *         - CNRT_QUEUE_SYNC_SPIN CPU actively spins when waiting for the device
 *           execution result. This option can have lower latency, but may decrease
 *           the performance of CPU threads if the works are executed in parallel
 *           with IPU. This value is used by default.
 *         - CNRT_QUEUE_SYNC_BLOCK CPU thread is blocked on a synchronization
 *           primitive when waiting for the device execution results.
 *         - CNRT_QUEUE_SYNC_YIELD CPU thread yields when waiting for the device
 *           execution results. This option can have higher latency, but can increase
 *           the performance of CPU threads if the works are executed in parallel
 *           with the device.

 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because the value of
 *         the ``flags`` parameter is bigger than ``1``.
 * @retval CNRT_RET_ERR_NODEV This function call failed because it is unable to open
 *         the device or the MLU devices are not found.
 *
 * @details After initializing the current device, specifies whether the
 * CPU thread yields or spins when waiting for the device execution result
 * in the ``flags``.
 * If no real device or fake device is set to the current thread, then
 * ::CNRT_RET_ERR_NODEV is returned.
 *
 * @note
 * - If the current device has been set and the device has already been
 *   initialized then this function will replace the option you set before.
 * - This function should be called after calling the ::cnrtInit function.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
@verbatim
int main(void) {
  unsigned int dev_num = 0;
  cnrtDev_t dev;

  cnrtInit(0);
  // Get the device number
  cnrtGetDeviceCount(&dev_num);
  if (dev_num < 1) {
    return -1;
  }

  int ordinal = 0;
  // Get the device handle
  cnrtGetDeviceHandle(&dev, ordinal);
  // Set the current device
  cnrtSetCurrentDevice(dev);

  unsigned int flags = CNRT_QUEUE_SYNC_SPIN;
  // Set the flag for the current process executions on the current device
  cnrtSetDeviceFlag(CNRT_QUEUE_SYNC_BLOCK);
  // Get the flag used for the current process executions on the current device.
  cnrtGetDeviceFlag(&flags);
  if (flags != CNRT_QUEUE_SYNC_BLOCK) {
    printf("set and get device flag error!\n");
  }

  cnrtDestroy();
}
@endverbatim
 *
 */
extern CNRT_DLL_API cnrtRet_t cnrtSetDeviceFlag(unsigned int flags);

/**
 * @brief  Retrieves the flag used for the current process executions on
 *         the current device.
 *
 * @param  flags[out] Pointer to the flag that specifies whether the
 *         CPU thread yields or spins when waiting for the device
 *         execution result. The flags are defined in ::cnrt_queue_sync_type.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because the value of
 *         the ``flags`` parameter is bigger than ``1``.
 * @retval CNRT_RET_ERR_NODEV This function call failed because it is unable to open
 *         the device or the MLU devices are not found.
 *
 * @details Returns in ``flags`` how CPU interacts with the OS scheduler when
 *          waiting for the device execution result.
 *
 *
 * @note
 * - This function should be called after calling the ::cnrtInit function.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtSetDeviceFlag function.
 *
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetDeviceFlag(unsigned int *flags);

/**
 * @brief Queries if a device is capable of directly accessing memories
 *        on another device.
 *
 * @param CanPeer[out] Pointer to the return value 1 or 0. If ``Dev`` is
 *        able to access memories on another MLU device ``PeerDev``,
 *        returns 1. Otherwise returns 0.
 * @param Dev[in] The ordinal of the device if can directly access
 *                memories on another device. You can call the
 *                ::cnrtGetDeviceCount function to get the total
 *                number of devices in your system.
 *                The device ID is in the range
 *                [0, ::cnrtGetDeviceCount() – 1].
 * @param PeerDev[in] The ordinal of the device on which memories can be
 *                    directly accessed ``Dev``. You can call the
 *                    ::cnrtGetDeviceCount function to get the total
 *                    number of devices in your system.
 *                    The device ID is in the range
 *                   [0, ::cnrtGetDeviceCount() – 1].
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because
 *         ``CanPeer`` is NULL.
 * @retval CNRT_RET_ERR_FUNC_CALL This function call failed because errors
 *         from driver functions are returned via CNRT. See the detailed
 *         error messages on the screen.
 * @retval CNRT_RET_ERR_ACCESS This function call failed because the two device
 *                             can not be accessed directly.
 *
 *
 * @details Returns in ``CanPeer`` if an MLU device ``Dev`` is able to access memories
 * on another MLU device ``PeerDev``. If ``Dev`` is able to access
 * memories on another MLU device ``PeerDev``, returns 1. Otherwise, returns 0.
 *
 *
 * @note
 * - This function should be called after calling the ::cnrtInit function.
 * - If the input parameter `Dev`` and ``PeerDev`` are the same value, this function
 *   will return CNRT_RET_SUCCESS and zero will be returned in ``CanPeer``.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
@verbatim
int main(void) {
  unsigned int dev_num = 0, canPeer = 0;
  cnrtDev_t dev_s, dev_d;

  cnrtInit(0);
  // Get the device number
  cnrtGetDeviceCount(&dev_num);
  if (dev_num < 2) {
    return -1;
  }

  // Get peer accessibility between device 0 and device 1
  cnrtGetPeerAccessibility(&canPeer, 0, 1);
  if (canPeer == 0) {
    printf("device 0 can not access memory from device 1 directly!\n");
  } else {
    printf("device 0 can access memory from device 1 directly!\n");
  }

  cnrtDestroy();
}
@endverbatim
 *
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetPeerAccessibility(unsigned int *CanPeer, int Dev, int PeerDev);

/**
 * @brief Retrieves the inter-process memory handle for an allocated host or
 *        device memory within the same process. This function is not supported
 *        on Edge platforms.
 *
 * @param handle[out] Pointer to the unique inter-process handle for the host
 *        or device memory to share.
 * @param memPtr[in] Base pointer to the allocated host or device memory.
 * @retval CNRT_RET_SUCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because
 *         ``handle`` or ``memPtr`` is NULL.
 * @retval CNRT_RET_ERR_NO_EXIST This function call failed because the allocated
 *         memory specified in ``*memPtr`` is not found.
 * @retval CNRT_RET_ERR_FUNC_CALL This function call failed because errors
 *         from driver functions are returned via CNRT. See the detailed
 *         error messages on the screen.
 *
 * @details Returns the inter-process memory handle ``handle`` with the given host
 * or device memory pointed by ``memPtr`` within the same process. The host or
 * device memory must be allocated with the ::cnrtMalloc or ::cnrtMallocHost function.
 * This function is used for sharing host or device memory pointer across processes.
 *
 *
 * @note
 * - If the allocated memory specified in ``memPtr`` is freed with the ::cnrtFree
 *   function before calling the ::cnrtAcquireMemHandle function,
 *   ::CNRT_RET_ERR_NO_EXIST is returned.
 * - This function should be called within the same process as the one that the host
 *   or device memory is allocated by the ::cnrtMalloc or ::cnrtMallocHost function.
 * - This function should be called after calling the ::cnrtInit function.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
@verbatim
int main(void) {
  unsigned int dev_num = 0, d_size = 1024 * 1024;
  cnrtDev_t dev;

  int fd[2];
  if (pipe(fd)) {
    return -1;
  }

  int p_index, ProcessCount = 2;
  for (int i = 1; i < ProcessCount; i++) {
    int pid = fork();
    if (pid == 0) {
      p_index = i;
    }
  }

  cnrtInit(0);
  // Get the device number
  cnrtGetDeviceCount(&dev_num);
  if (dev_num < 1) {
    return -1;
  }

  int ordinal = 0;
  // Get the device handle
  cnrtGetDeviceHandle(&dev, ordinal);
  // Set the current device
  cnrtSetCurrentDevice(dev);

  void *mem_ptr = NULL;
  cnrtIpcMemHandle ipc_handle;
  char ch = 'a';
  if (p_index == 0) {
    int status;
    char *h_results = (char *)calloc(1, d_size);
    cnrtMalloc(&mem_ptr, d_size);
    // Acquire an inter-process memory handle
    cnrtAcquireMemHandle(&ipc_handle, (void *)mem_ptr);

    close(fd[0]);
    write(fd[1], &ipc_handle, sizeof(ipc_handle));

    waitpid(0, &status, 0);

    // Copy data from the device memory and check it
    cnrtMemcpy(h_results, mem_ptr, d_size, CNRT_MEM_TRANS_DIR_DEV2HOST);
    for (int i = 0; i < d_size; i++) {
      if (h_results[i] != ch) {
        printf("ipc memory run error!");
        break;
      }
    }

    printf("check data ok!\n");
    if (h_results) free(h_results);
    if (mem_ptr) cnrtFree(mem_ptr);
  } else {
    void *d_addr = NULL;
    char *h_addr = (char *)malloc(d_size);
    memset(h_addr, ch, d_size);

    close(fd[1]);
    read(fd[0], &ipc_handle, sizeof(ipc_handle));

    // Map the inter-process memory handle exported from another process
    cnrtMapMemHandle(&mem_ptr, ipc_handle, 0);
    // Copy data to the device memroy
    cnrtMemcpy(mem_ptr, h_addr, d_size, CNRT_MEM_TRANS_DIR_HOST2DEV);
    cnrtUnMapMemHandle(mem_ptr);

    printf("set the value over in another process!\n");
    if (h_addr) free(h_addr);
  }

  cnrtDestroy();
}
@endverbatim
 *
 */
extern CNRT_DLL_API cnrtRet_t cnrtAcquireMemHandle(cnrtIpcMemHandle *handle, void *memPtr);

/**
 * @brief Maps an inter-process memory handle shared by another process into the memory
 *        address space of the host or the current device. This function is not supported
 *        on Edge platforms.
 *
 * @param memPtr[out] Pointer to the host or device memory pointer.
 * @param handle[in] The inter-process memory handle shared by another process.
 *                   You can call the ::cnrtAcquireMemHandle function to get the handle.
 * @param flag[in] Flags used in this operation. Reserved for further uses.
 *                 Currently only supports value 0.
 * @retval CNRT_RET_SUCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because
 *         ``memPtr`` is NULL.
 * @retval CNRT_RET_ERR_NODEV This function call failed because it is unable to
 *         get the current device.
 * @retval CNRT_RET_ERR_FUNC_CALL This function call failed because errors
 *         from driver functions are returned via CNRT. See the detailed
 *         error messages on the screen.
 * @retval CNRT_RET_ERR_NOMEM This function call failed because errors occurred when
 *         allocating host or dvice memory.
 *
 * @details Maps an inter-process memory handle ``handle`` shared by another process into
 * the memory address space of the current device or host. Returns the
 * host or device memory pointer pointed by ``memPtr`` used in the local process.
 * This function does not support different contexts of the current device.
 *
 *
 * @note
 * - The mapped memory resource ``memPtr`` should be unmapped only with the
 *   ::cnrtUnMapMemHandle function within the same process.
 * - The inter-process communication only supports on the same device. So the related
 *   functions ::cnrtMalloc, ::cnrtMapMemHandle, and ::cnrtUnMapMemHandle should be called
 *   on the same device.
 * - This function should be called after calling the ::cnrtInit function.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtAcquireMemHandle function.
 *
 */
extern CNRT_DLL_API cnrtRet_t cnrtMapMemHandle(void **memPtr, cnrtIpcMemHandle handle, int flag);

/**
 * @brief Unmaps the mapping between the memory address and the inter-process memory handle.
 *        This function is not supported on Edge platforms.
 *
 * @param memPtr[in] Pointer to the host or device memory.
 * @retval CNRT_RET_SUCESS This function has run successfully..
 * @retval CNRT_RET_ERR_INVALID This function call failed because
 *         ``memPtr`` is NULL.
 * @retval CNRT_RET_ERR_NODEV This function call failed because it is unable to find
 *         memory on the device.
 * @retval CNRT_RET_ERR_FUNC_CALL This function call failed because errors
 *         from driver functions are returned via CNRT. See the detailed
 *         error messages on the screen.
 *
 * @details Unmaps the mapping between the memory address space of the current device or host
 * pointed by ``memPtr`` and the inter-process memory handle shared by another process.
 * The mapping is created by the ::cnrtMapMemHandle function.
 *
 *
 * @note
 * - The memory resource ``memPtr`` should be mapped only with the ::cnrtMapMemHandle
 *   function.
 * - The ::cnrtMapMemHandle and ::cnrtMapMemHandle functions should be called within
 *   the same process.
 * - The inter-process communication only supports on the same device. So the related
 *   functions ::cnrtMalloc, ::cnrtMapMemHandle, and ::cnrtUnMapMemHandle should be called
 *   on the same device.
 * - You need to call the ::cnrtSetCurrentDevice function before calling this function.
 * - This function should be called after calling the ::cnrtInit function.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtAcquireMemHandle function.
 *
 */
extern CNRT_DLL_API cnrtRet_t cnrtUnMapMemHandle(void *memPtr);

/** @} */ /* END CNRT_DEVICE */

/******************************************************************************
 * Version and revision
 ******************************************************************************/

/**
 * @brief Retrieves the version of the current CNRT instance.
 *
 * @param ver[out] Pointer to the version of the current CNRT instance.
 *                 Return an unsigned int for version number.
 *
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because ``ver`` is NULL.
 *
 * @details Returns in ``ver`` the version of the current CNRT instance. \n
 *          The version is in the following format: \n
 *          major_version * 10000 + minor_version * 100 + patch_version \n
 *          Take version 4.8.0 for example, the return version number is 40800.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetVersion(unsigned int *ver);

/**
 * @brief Retrieves the major, minor, and patch of the current CNRT instance.
 *
 * @param major[out] Pointer to the major of version.
 * @param minor[out] Pointer to the minor of version.
 * @param patch[out] Pointer to the patch of version.
 *
 * @details Returns the major version in ``major``, minor version in ``minor``,
 *          and patch version in ``patch`` of the current CNRT instance.
 *
  * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
extern CNRT_DLL_API void cnrtGetLibVersion(int *major, int *minor, int *patch);

/******************************************************************************
 * Queue management
 ******************************************************************************/

/**
 * @brief Creates a queue.
 *
 * @param pQueue[out] Pointer to the newly created queue.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because input
 *         pointer ``pQueue`` is NULL.
 * @retval CNER_RET_ERR_NODEV This function call failed because no
 *         MLU devices were detected.
 * @retval CNRT_RET_WARNING_FAKE_DEVICE This function call failed because
 *         creating queue on fake devices are not supported.
 * @retval CNRT_RET_ERR_NOMEM This function call failed because not enough
 *         memory can be allocated.
 * @retval CNRT_RET_ERR_CNDRV_FUNC_CALL This function call failed because errors
 *         from driver functions are returned via CNRT. See the detailed error
 *         messages on the screen.
 *
 * @details Creates a queue on the current device, and returns a pointer
 * pQueue to the newly created queue. \n
 * You can define how the queues are synchronized with the ::cnrtSetDeviceFlag
 * function. By default CNRT_QUEUE_SYNC_BLOCK is used. You can call the
 * ::cnrtGetDeviceFlag function to query the current behavior.
 * - CNRT_QUEUE_SYNC_SPIN CPU actively spins when waiting for the device
 *   execution result. This option can have lower latency, but may decrease
 *   the performance of CPU threads if the works are executed in parallel
 *   with IPU. This value is used by default.
 * - CNRT_QUEUE_SYNC_BLOCK CPU thread is blocked on a synchronization
 *   primitive when waiting for the device execution result.
 * - CNRT_QUEUE_SYNC_YIELD CPU thread yields when waiting for the device
 *   execution results. This option can have higher latency, but can increase
 *   the performance of CPU threads if the works are executed in parallel
 *   with the device.
 *
 * @note
 * - You need to call the ::cnrtDestroyQueue function to release the queue resources;
 *   otherwise, the memory leaks may occurr.
 *
 * - To learn more about queue lifecycle management, please see
 *   "Cambricon Driver API Developer Guide".
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtSyncQueue function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtCreateQueue(cnrtQueue_t *pQueue);

/**
 * @brief Destroys a queue.
 *
 *
 * @param queue[in] The queue to be destroyed.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because ``queue`` is NULL.
 * @retval CNRT_RET_ERR_CNDRV_FUNC_CALL This function call failed because errors
 *         from driver functions are returned via CNRT. See the detailed error
 *         messages on the screen.
 * @retval CNRT_RET_ERR_QUEUE This function call failed because an error occurred when
 *         destroying the queue.
 *
 * @details Destroys a ``queue`` that is created by the ::cnrtCreateQueue funtion.
 * If the queue is still executing operations when the ::cnrtDestroyQueue function
 * is called, this function returns immediately, but the resources associated with
 * the queue is released automatically after all the operations in the queue are
 * completed.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtSyncQueue function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtDestroyQueue(cnrtQueue_t queue);

/**
 * @brief Waits for a notifier to complete before the specified queue to do any
 *        further executions.
 *
 *
 * @param notifier[in] The notifier to be waited for.
 * @param queue[in] The queue to wait.
 * @param flags Flags used in this operation. Reserved for further uses.
 *              It is recommended to set this parameter to 0.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because
 *         the input pointer ``notifier`` is NULL.
 * @retval CNRT_RET_ERR_CNDRV_FUNC_CALL This function call failed because errors
 *         from driver functions are returned via CNRT. See the detailed error
 *         messages on the screen.
 * @retval CNRT_RET_ERR_QUEUE This function call failed because an error occurred
 *         during obtaining the default queue handle.
 *
 * @details Blocks all future works in the ``queue`` until the ``notifier`` is completed.
 * The queue only waits for the completion of the most recent host call to
 * ::cnrtPlaceNotifier on the notifier. This function is used to synchronize the
 * queue efficiently on the device.
 *
 * @note
 * - If ``queue``is set to NULL, the default queue is used.
 * - This function is used for synchronization across queues. To synchronize in
 *   a single queue, use the ::cnrtWaitNotifier function.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtSyncQueue function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtQueueWaitNotifier(cnrtNotifier_t notifier,
                                                    cnrtQueue_t queue,
                                                    unsigned int flag);

/**
 * @brief Queries if a queue is completed.
 *
 * @param queue[in] The queue to query.
 * @retval CNRT_RET_SUCCESS This function has run successfully and all the operations
 *                          in the queue are completed.
 * @retval CNRT_RET_ERR_NOT_READY The operations in the queue are still executing.
 * @retval CNRT_RET_ERR_CNDRV_FUNC_CALL This function call failed because errors
 *         from driver functions are returned via CNRT. See the detailed error
 *         messages on the screen.
 * @retval CNRT_RET_ERR_QUEUE This function call failed because error occurred during
 *         obtaining the default queue handle.
 *
 * @details Returns the status of the ``queue``. If all the operations in the
 * queue are completed, ::CNRT_RET_SUCCESS is returned. If the operations in
 * the queue are still executing, ::CNRT_RET_ERR_NOT_READY is returned.
 *
 * @note
 * - If ``queue``is set to NULL, the default queue is used.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 @verbatim
 int main () {
  // This sample runs a simple offline model
  // The model name is specified by T_MEF_NAME
  // Offline model file (*.mef) should be prepared

  char m_file[128] = {0};
  cnrtModel_t model;
  cnrtFunction_t function;

  strcat(m_file, T_MEF_NAME);
  strcat(m_file, ".mef");

  // Initialize the CNRT environment
  cnrtInit(0);

  // Load a model from a model file
  cnrtLoadModel(&model, m_file);

  // Get function from the loaded model
  cnrtCreateFunction(&function);
  cnrtExtractFunction(&function, model, T_MEF_NAME);

  // Prepare data
  int inputNum, outputNum;
  int64_t *inputSizeS, *outputSizeS;
  cnrtGetInputDataSize(&inputSizeS, &inputNum,
                                      function);
  cnrtGetOutputDataSize(&outputSizeS, &outputNum,
                                      function);

  // Create a context
  cnrtRuntimeContext_t ctx;
  cnrtCreateRuntimeContext(&ctx, function, NULL);
  cnrtSetRuntimeContextDeviceId(ctx, 0);
  cnrtInitRuntimeContext(ctx, NULL);

  // Create a queue
  cnrtQueue_t queue;
  cnrtCreateQueue(&queue);

  void **inputCpuPtrS = (void **)malloc(sizeof(void *) * inputNum);
  for (int i = 0; i < inputNum; i++) {
    int64_t dataSize = inputSizeS[i];
    void *cpuPtr = (void *)malloc(dataSize);
    inputCpuPtrS[i] = (void *)cpuPtr;
  }

  void **inputMluPtrS = (void **)malloc(sizeof(void *) * inputNum);
  void **outputMluPtrS = (void **)malloc(sizeof(void *) * outputNum);

  for (int i = 0; i < inputNum; i++) {
    cnrtMalloc(&(inputMluPtrS[i]), inputSizeS[i]);
    cnrtMemcpy(inputMluPtrS[i], inputCpuPtrS[i], inputSizeS[i],
                                        CNRT_MEM_TRANS_DIR_HOST2DEV);
  }

  for (int i = 0; i < outputNum; i++) {
    cnrtMalloc(&(outputMluPtrS[i]), outputSizeS[i]);
  }

  void **param = (void **)malloc(sizeof(void *) * (inputNum + outputNum));
  for (int i = 0; i < inputNum; ++i) {
    param[i] = inputMluPtrS[i];
  }
  for (int j = 0; j < outputNum; ++j) {
    param[inputNum + j] = outputMluPtrS[j];
  }

  // Invoke the kernel from context
  cnrtInvokeRuntimeContext(ctx, param, queue, NULL);

  // Query the status of a queue and wait until all tasks in queue are completed
  cnrtQueryQueue(queue);
  cnrtSyncQueue(queue);
  cnrtQueryQueue(queue);

  // Destroy resources
  cnrtDestroyQueue(queue);

  cnrtDestroyRuntimeContext(ctx);
  cnrtDestroyFunction(function);
  cnrtUnloadModel(model);
  cnrtDestroy();

  return 0;
 }
 @endverbatim
 */

extern CNRT_DLL_API cnrtRet_t cnrtQueryQueue(cnrtQueue_t queue);

/**
 * @brief Waits for a queue operations to be completed.
 *
 * @param queue[in] The queue to be waited for.
 * @retval CNRT_RET_SUCCESS This function has run successfully and all the operations
 *                          in queue are completed.
 * @retval CNRT_RET_ERR_QUEUE This function call failed because error occurred during
 *         obtaining the default queue handle.
 * @retval CNRT_RET_ERR_CNDRV_FUNC_CALL This function call failed because errors
 *         from driver functions are returned via CNRT. See the detailed error
 *         messages on the screen.
 *
 * @details Blocks further executions in CPU thread until all the works
 *          in the queue on the current MLU device are completed.
 *
 * @note
 * - If ``queue`` is set to NULL, the default queue is used.
 * - This function may also return ::CNRT_RET_ERR_QUEUE from previous and asynchronous
 *   operations.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 @verbatim
 int main () {
   // Initialize device
   cnrtInit(0);

   cnrtDev_t dev;
   // Get device handle
   cnrtGetDeviceHandle(&dev, 0);

   cnrtGetCurrentDevice(&dev);

   // Set the device handle for current thread execution context
   cnrtSetCurrentDevice(dev);

   // Create a notifier
   cnrtNotifier_t notifier;
   cnrtCreateNotifier(&notifier);

   // Create a queue
   cnrtQueue_t queue_0;
   cnrtQueue_t queue_1;
   cnrtCreateQueue(&queue_0);
   cnrtCreateQueue(&queue_1);

   // Allocate memory on device and host
   size_t size = 0x100000;
   char *host_mem0 = NULL;
   char *host_mem1 = NULL;
   host_mem0 = (char *)malloc(size);
   host_mem1 = (char *)malloc(size);
   void *dev_mem0 = NULL;
   void *dev_mem1 = NULL;
   cnrtMalloc(&dev_mem0, size);
   cnrtMalloc(&dev_mem1, size);

   // Copy data asynchoronously in two queues
   cnrtMemcpyAsync(dev_mem0, host_mem0, size, queue_0, CNRT_MEM_TRANS_DIR_HOST2DEV);
   cnrtMemcpyAsync(host_mem1, dev_mem1, size, queue_1, CNRT_MEM_TRANS_DIR_DEV2HOST);
   // Put a notifier into a queue
   cnrtPlaceNotifier(notifier, queue_0);

   // Synchronize two queues
   cnrtQueueWaitNotifier(notifier, queue_1, 0);

   // Wait until all tasks are completed
   cnrtSyncQueue(queue_0);
   cnrtSyncQueue(queue_1);

   // Release resources
   cnrtDestroyQueue(queue_0);
   cnrtDestroyQueue(queue_1);
   cnrtDestroyNotifier(&notifier);
   cnrtDestroy();

   return 0;
 }
 @endverbatim
 */
extern CNRT_DLL_API cnrtRet_t cnrtSyncQueue(cnrtQueue_t queue);

/*********************************************************************************
 * Notifier. Supports both MLU220 and MLU270.
 *********************************************************************************/

/**
 * @brief Creates a notifier for the current device.
 *
 *
 * @param notifier[out] Pointer to the newly created notifier.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because
 *         the input pointer ``notifier`` is NULL.
 * @retval CNER_RET_ERR_NODEV This function call failed because no
 *         MLU devices were detected.
 * @retval CNRT_RET_WARNING_FAKE_DEVICE This function call failed because
 *         creating notifiers for fake devices are not supported.
 * @retval CNRT_RET_ERR_NOMEM This function call failed because not enough
 *         memory can be allocated.
 * @retval CNRT_RET_ERR_CNDRV_FUNC_CALL This function call failed because errors
 *         from driver functions are returned via CNRT. See the detailed error
 *         messages on the screen.
 *
 * @details Creates a notifier for the current device, and returns a pointer
 * ``notifier`` to the newly created notifier. For more information about notifier,
 * see "CNRT User Guide".
 *
 * @note
 * - You need to call the ::cnrtDestroyNotifier function to release the notifier memory;
 *   otherwise the memory leaks may occur.
 *
 * - To learn more about queue lifecycle management, please see
 *   "Cambricon Driver API Developer Guide".
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtNotifierDuration function.
 *
 *
 */
extern CNRT_DLL_API cnrtRet_t cnrtCreateNotifier(cnrtNotifier_t *notifier);

/**
 * @brief Destroys a notifier that was created by the ::cnrtCreateNotifier function.
 *
 * @param notifier[in] Pointer to notifier that was created by the
 *                     ::cnrtCreateNotifier function.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because
 *         the input pointer ``notifier`` or dereference of ``notifier`` is NULL.
 * @retval CNRT_RET_ERR_CNDRV_FUNC_CALL This function call failed because errors
 *         from driver functions are returned via CNRT. See the detailed error
 *         messages on the screen.
 *
 * @details Destroys a notifier pointed by ``notifier``. For more information about notifier,
 * see CNRT User Guide.
 *
 * @note
 * - If a notifier is placed into a queue but is not completed when the ::cnrtDestroyNotifier
 *   function is called, the ::cnrtDestroyNotifier function will return immediately, but
 *   the resources associated with this notifier will be released automatically only after
 *   the notifier is completed.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtNotifierDuration function.
 *
 *
 */
extern CNRT_DLL_API cnrtRet_t cnrtDestroyNotifier(cnrtNotifier_t *notifier);

/**
 * @brief Waits for a notifier to be completed.
 *
 *
 * @param notifier[in] The handle of the notifier to be waited for.
 * @retval CNRT_RET_SUCCESS This function has run successfully and the notifier
 *                          is completed.
 * @retval CNRT_RET_ERR_INVALID This function call failed because
 *         the input pointer ``notifier`` is NULL.
 * @retval CNRT_RET_ERR_CNDRV_FUNC_CALL This function call failed because errors
 *         from driver functions are returned via CNRT. See the detailed error
 *         messages on the screen.
 *
 *
 * @details Waits for a ``notifier`` in the queue to be completed before executing all
 *          future work in this queue. The ``notifier`` is the most recent one called by
 *          the ::cnrtPlaceNotifier function in the queue. Returns ::CNRT_RET_SUCCESS
 *          if the notifier is completed.
 *
 * @note
 * - If the ::cnrtPlaceNotifier function has not been called on the notifier before
 *   calling this function, CNRT_RET_SUCCESS is returned.
 * - This function is used for synchronization in a single queue. To synchronize across
 *   queues, use the ::cnrtQueueWaitNotifier function.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtNotifierDuration function.
 *
 */

extern CNRT_DLL_API cnrtRet_t cnrtWaitNotifier(cnrtNotifier_t notifier);

/**
 * @brief Queries the status of all the preceding works before a notifier in a queue.
 *
 *
 *
 * @param notifier[in] The handle of the notifier to query.
 * @retval CNRT_RET_SUCCESS This function has run successfully and all the preceding works
 *                          before the ``notifier`` in the queue are completed.
 * @retval CNRT_RET_ERR_BUSY The preceding works before the notifier in the queue are still
 *                           executing.
 * @retval CNRT_RET_ERR_INVALID This function call failed because
 *         the input pointer ``notifier`` or dereference of ``notifier`` is NULL.
 * @retval CNRT_RET_ERR_CNDRV_FUNC_CALL This function call failed because errors
 *         from driver functions are returned via CNRT. See the detailed error
 *         messages on the screen.
 *
 * @details Queries the status of all the preceding works before the ``notifier`` in a queue.
 * Returns ::CNRT_RET_SUCCESS if all the preceding works before the ``notifier`` in
 * the queue are completed. Returns CNRT_RET_ERR_BUSY if the preceding works before
 * the ``notifier`` are still executing.
 *
 * @note
 * - When querying the status of all the preceding works before the notifier, if you call the
 *   ::cnrtPlaceNotifier function again on the same notifier, the query will be based on the most
 *   recent call by the ::cnrtPlaceNotifier function. So all the existing status recorded
 *   for the previous call will be overwritten.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtNotifierDuration function.
 *
 */
extern CNRT_DLL_API cnrtRet_t cnrtQueryNotifier(cnrtNotifier_t notifier);

/**
 * @brief Places a notifier into a specified queue.
 *
 *
 * @param notifier[in] The handle of the notifier to be placed into the queue.
 *                     You can create the notifier by calling the
 *                     ::cnrtCreateNotifier function.
 * @param queue[in] The queue in which the notifier is placed. You can create
 *                  the queue by calling the ::cnrtCreateQueue function.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because
 *         the input pointer ``notifier`` is NULL.
 * @retval CNRT_RET_ERR_CNDRV_FUNC_CALL This function call failed because errors
 *         from driver functions are returned via CNRT. See the detailed error
 *         messages on the screen.
 *
 * @details Places a ``notifier`` into a specified ``queue``. The notifier can be
 * used to measure the execution time of all the works.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtNotifierDuration function.
 *
 */
extern CNRT_DLL_API cnrtRet_t cnrtPlaceNotifier(cnrtNotifier_t notifier, cnrtQueue_t queue);

/**
 * @brief Computes the time duration between starting and ending notifiers.
 *
 *
 * @param start[in] The handle of the starting notifier created by the
 *                  ::cnrtCreateNotifier function.
 * @param end[in] The handle of ending notifier created by the ::cnrtCreateNotifier
 *                function.
 * @param us[out] The time duration between the starting and ending notifier.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because
 *         the input pointer ``notifier`` or dereference of ``notifier`` is NULL.
 * @retval CNRT_RET_ERR_CNDRV_FUNC_CALL This function call failed because errors
 *         from driver functions are returned via CNRT. See the detailed error
 *         messages on the screen.
 *
 * @details Computes the time duration between the starting notifier ``start``
 * and the ending notifier ``end``. This function is used to measure the
 * device execution time of all the works between the starting notifier and
 * ending notifier. The measurement that can be used to help you improve the
 * performance.
 *
 * @note
 * - You need to call the ::cnrtPlaceNotifier function to place the starting and ending
 *   notifiers into the queue respectively before calling this function.
 *   Otherwise, ::CNRT_RET_ERR_CNDRV_FUNC_CALL is  returned.
 * - If the ::cnrtNotifierDuration function is called before the starting notifier
 *   or ending notifier is placed into the queue, ::CNRT_RET_ERR_CNDRV_FUNC_CALL is
 *   returned.
 *
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 @verbatim
 int main () {
   // Initialize device
   cnrtInit(0);
   cnrtDev_t dev;
   // Get device handle
   cnrtGetDeviceHandle(&dev, 0);
   // Set the device handle for current thread execution context
   cnrtSetCurrentDevice(dev);

   // Create notifier
   cnrtNotifier_t notifier_s;
   cnrtNotifier_t notifier_e;
   cnrtCreateNotifier(&notifier_s);
   cnrtCreateNotifier(&notifier_e);

   // Create queue
   cnrtQueue_t queue;
   cnrtCreateQueue(&queue);

   // Place notifier into a queue
   cnrtPlaceNotifier(notifier_s, queue);
   cnrtPlaceNotifier(notifier_e, queue);

   // Query notifier
   cnrtQueryNotifier(notifier_s);
   cnrtQueryNotifier(notifier_e);

   // Wait for notifier
   cnrtWaitNotifier(notifier_s);
   cnrtWaitNotifier(notifier_e);

   // Compute the duration
   float us;
   cnrtNotifierDuration(notifier_s, notifier_e, &us)
   printf("time cosuming between the two notifier is %f\n", us);

   cnrtDestroyNotifier(&notifier_s);
   cnrtDestroyNotifier(&notifier_e);

   cnrtDestroyQueue(queue);

   cnrtDestroy();
   return 0;
 }
 @endverbatim
 *
 */
extern CNRT_DLL_API cnrtRet_t cnrtNotifierDuration(cnrtNotifier_t start,
                                                   cnrtNotifier_t end,
                                                   float *us);

/*********************************************************************************
 * Execution control & BANG C Kernel
 *********************************************************************************/

/**
 * @brief Initializes a buffer that stores the kernel parameters.
 *
 *
 * @param params[in] Pointer to a kernel parameters buffer defined in
 *        ::cnrtKernelParamsBuffer_t enum.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because ``params``
 *         is NULL.
 * @retval CNRT_RET_ERR_NOMEM This function call failed because not enough memory
 *         can be allocated on the host.
 *
 * @details Initializes a kernel parameter buffer in ``params`` and allocates memory
 *          space for the buffer. The buffer is defined in
 *          ::cnrtKernelParamsBuffer_t enum for kernel launch.
 *
 * @note
 * - The ::cnrtDestroyKernelParamsBuffer function should be
 *   called to destroy the kernel parameter buffer and release the related resources.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtInvokeKernel_V2 or ::cnrtInvokeKernel_V3 function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetKernelParamsBuffer(cnrtKernelParamsBuffer_t *params);

/**
 * @brief Copies data from a kernel parameter buffer to another kernel parameter buffer.
 *
 * @param dst_params_buf[in] The kernel parameter buffer to be copied to. The buffer
 *         is defined in ::cnrtKernelParamsBuffer_t struct.
 * @param src_params_buf[in] The kernel parameters buffer to be copied from. The buffer
 *         is defined in ::cnrtKernelParamsBuffer_t struct.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because ``dst_params_buf``
 *         or ``src_params_buf`` is NULL.
 *
 * @details Copies data from the kernel parameter buffer ``src_params_buf`` to the kernel
 *          parameter buffer ``dst_params_buf``. The buffers are defined in
 *          ::cnrtKernelParamsBuffer_t struct.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
extern CNRT_DLL_API cnrtRet_t cnrtCopyKernelParamsBuffer(cnrtKernelParamsBuffer_t dst_params_buf,
                                                         cnrtKernelParamsBuffer_t src_params_buf);

/**
 * @brief Adds a parameter to a kernel parameter buffer.
 *
 * @param params[in] The kernel parameter buffer to be added into.
 * @param data[in] Pointer to the base address of the parameter to be added.
 * @param bytes[in] The size of the parameter in bytes.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because one of the following errors
 * occurred:
 *         - ``data`` or ``params`` is NULL.
 *         - The value of ``bytes`` is greater than 8.
 *         - The input kernel parameter buffer has reached the maximum size.
 *
 * @details Adds a parameter pointed by ``data`` in ``bytes`` bytes to
 * the kernel parameter buffer defined in ::cnrtKernelParamsBuffer_t struct ``params``.
 *
 * @note
 * - The kernel parameter buffer is 8-bytes aligned. If the size of the input parameter is less
 *   than 8 bytes, the parameter also takes 8 bytes in the buffer.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtInvokeKernel_V2 or ::cnrtInvokeKernel_V3 function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtKernelParamsBufferAddParam(cnrtKernelParamsBuffer_t params,
                                                             void *data,
                                                             size_t bytes);

/**
 * @brief Takes a placeholder for an input parameter in a kernel parameter buffer for
 *        further usage in the kernel launch.
 *
 *
 * @param params[in] The buffer that stores the kernel parameters to be added into.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because
 *         ``params`` is NULL or the input kernel parameters buffer has reached the
 *         maximum size.
 *
 * @details Adds an input parameter into the kernel parameter buffer ``params`` to take the
 *          placeholder for further usage in the kernel launch. The added parameter
 *          is initialized to value 0 in the buffer. When launching a kernel, the real
 *          parameter is stored in the placeholder you have taken via this function and replaces
 *          the initialized value 0 with a pointer pointed to the value of the real parameter.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtInvokeKernel_V2 or ::cnrtInvokeKernel_V3 function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtKernelParamsBufferMarkInput(cnrtKernelParamsBuffer_t params);

/**
 * @brief Takes a placeholder for an output parameter in a kernel parameter buffer for
 *        further usage in the kernel launch.
 *
 *
 * @param params[in] The buffer that stores the kernel parameters to be added into.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because
 *         ``params`` is NULL or the output kernel parameter buffer has reached the
 *         maximum size.
 *
 * @details Adds an output parameter into the kernel parameter buffer ``params`` to take the
 *          placeholder for further usage in the kernel launch. The added parameter
 *          is initialized to value 0 in the buffer. When launching a kernel, the real
 *          parameter is stored in the placeholder you have taken via this function and replaces
 *          the initialized value 0 with a pointer pointed to the value of the real parameter.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtInvokeKernel_V2 or ::cnrtInvokeKernel_V3 function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtKernelParamsBufferMarkOutput(cnrtKernelParamsBuffer_t params);

/**
 * @brief Takes a placeholder for a static parameter in a kernel parameter buffer for
 *        further usage in the kernel launch.
 *
 *
 * @param params[in] The buffer that stores the kernel parameters to be added into.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because
 *         ``params`` is NULL or the static kernel parameter buffer has reached the
 *         maximum size.
 *
 * @details Adds a static parameter into the kernel parameter buffer ``params`` to take the
 *          placeholder for further usage in the kernel launch. The added parameter
 *          is initialized to 0 in the buffer. When launching a kernel, the real
 *          parameter is stored in the placeholder you have taken via this function and replaces
 *          the initialized 0 with a pointer pointed to the value of the real parameter.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtInvokeKernel_V2 or ::cnrtInvokeKernel_V3 function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtKernelParamsBufferMarkStatic(cnrtKernelParamsBuffer_t params);

/**
 * @brief Destroys a kernel parameter buffer.
 *
 * @param params[in] The kernel parameter buffer to be destroyed. You can get the buffer via
 *                   the ::cnrtGetKernelParamsBuffer function.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because ``params`` is NULL.
 *
 *
 * @details Destroys a kernel parameter buffer defined in ::cnrtKernelParamsBuffer_t struct
 * ``params``.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtInvokeKernel_V2 or ::cnrtInvokeKernel_V3 function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtDestroyKernelParamsBuffer(cnrtKernelParamsBuffer_t params);

/**
 * @brief Launches a kernel to perform tasks on the MLU devices.
 *
 * @param function[in] Pointer to the MLU function. You can either define MLU functions
 *        using BANG C language or extract MLU functions from offline models.
 *        See "BANG C Developer Guide" and "CNRT Developer Guide" for detailed information.
 * @param dim[in] The grid dimensions defined in ::cnrtDim3_t struct.
 * @param params[in] The kernel parameter buffer used in the kernel launch.
 * @param func_type[in] The number of MLU cores used for parallel computing defined in
 *        ::cnrtFunctionType_t enum.
 * @param queue[in] The queue handle associated with the MLU function. You can create a
 *        queue by calling the ::cnrtCreateQueue function.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because
 *         ``function`` is NULL, or the value of ``dim``, ``params``, or ``func_type`` is invalid.
 * @retval CNRT_RET_WARNING_FAKE_DEVICE This function call failed because the fake device is used.
 * @retval CNRT_RET_ERR_NODEV This function call failed because no MLU devices are detected.
 * @retval CNRT_RET_ERR_KERNEL_VERSION_TOO_HIGH This function call failed because
 *         the CNRT library version need to be upgraded.
 * @retval CNRT_RET_ERR_NOMEM This function call failed because insufficient memory space
 *         on host or device can be allocated.
 * @retval CNRT_RET_ERR_CNDRV_FUNC_CALL This function call failed because errors
 *         from driver functions are returned via CNRT. See the detailed error
 *         messages on the screen.
 *
 * @details Launches a kernel ``function`` on ``dim`` with the given kernel parameters
 * stored in the buffer ``params``. You can specify the number of MLU
 * cores used for parallel computing in ::cnrtFunctionType_t enum ``func_type``. The ``dim``
 * is defined in ::cnrtDim3_t struct and ``params`` is defined in ::cnrtKernelParamsBuffer_t
 * struct.
 *
 *
 * @note
 * - This function does not support using fake device to launch kernels.
 * - This function is always performed by the device associated with a queue.
 *
 * @par Requirement
 * - None.
 *
 * @par Example
 @verbatim
 In the example below, kernel_add.mlu is written in BANG C language.
 See "BANG C Developer Guide" for detailed information.

 file < kernel_add.mlu >
 #include "bang.h"
 __mlu_global__ void kernelAdd(int *a, int *b) {
    *a = *a + *b;
 }


 file < test.c >
 #include "cnrt.h"
 int main () {
   //initialize the environment
   cnrtInit(0);
   cnrtDev_t dev;
   cnrtGetDeviceHandle(&dev, 0);
   cnrtSetCurrentDevice(dev);

   int a, b, out;
   uint32_t seed = 0x123;
   a = (int)rand_r(&seed) % 10000;
   b = (int)rand_r(&seed) % 10000;

   cnrtQueue_t queue;
   cnrtCreateQueue(&queue);

   int *k_a, *k_b;
   cnrtMalloc((void **)&k_a, sizeof(int));
   cnrtMalloc((void **)&k_b, sizeof(int));
   cnrtMemcpy(k_a, &a, sizeof(int), CNRT_MEM_TRANS_DIR_HOST2DEV);
   cnrtMemcpy(k_b, &b, sizeof(int), CNRT_MEM_TRANS_DIR_HOST2DEV);

   cnrtDim3_t dim;
   dim.x = 1;
   dim.y = 1;
   dim.z = 1;
   cnrtFunctionType_t func_type = CNRT_FUNC_TYPE_BLOCK;
   cnrtKernelParamsBuffer_t params;

   cnrtGetKernelParamsBuffer(&params);
   cnrtKernelParamsBufferAddParam(params, &k_a, sizeof(int *));
   cnrtKernelParamsBufferAddParam(params, &k_b, sizeof(int *));
   cnrtInvokeKernel_V2((void *)kernelAdd, dim, params, func_type, queue);
   cnrtSyncQueue(queue);

   cnrtMemcpy(&out, k_a, sizeof(int), CNRT_MEM_TRANS_DIR_DEV2HOST);
   cnrtFree(k_a);
   cnrtFree(k_b);
   cnrtDestroyKernelParamsBuffer(params);
   cnrtDestroyQueue(queue);

   printf("out:%d, a=%d, b=%d, a+b=%d\n", out, a, b, (a + b));

   cnrtDestroy();

   return 0;
 }
 @endverbatim
 *
 */
extern CNRT_DLL_API cnrtRet_t cnrtInvokeKernel_V2(const void *function,
                                                  cnrtDim3_t dim,
                                                  cnrtKernelParamsBuffer_t params,
                                                  cnrtFunctionType_t func_type,
                                                  cnrtQueue_t queue);

/**
 * @brief Creates a buffer to store MLU function related parameters used in
 *        the kernel launch.
 *
 * @param init_param[in] Pointer to the buffer that stores MLU function
 *                       related parameters.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because ``init_param`` is NULL.
 * @retval CNRT_RET_ERR_NOMEM This function call failed because insufficient host memory
 *         can be allocated.
 *
 *
 * @details Creates a buffer by allocating the memory to store the kernel
 *          parameters pointed by ``init_param``. The memory is initialized to 0.
 *          This buffer is used to store MLU function related parameters
 *          used in the kernel launch.
 *
 * @note
 * - After calling this function to create a buffer, you need to call the
 *   ::cnrtInitKernelMemory function to set MLU function related parameters.
 * - When the kernel launch is completed, you need to call the
 *   ::cnrtDestroyKernelInitParamAndMemory function to destroy the resources.
 *   Otherwise, the memory leak may be occurred.
 *
 * @par Requirement
 * - None.
 *
 * @par Example
 * - See example in ::cnrtInvokeKernel_V3 function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtCreateKernelInitParam(cnrtKernelInitParam_t *init_param);
/**
 * @brief Adds the MLU function related parameters into a buffer.
 *
 * @param function[in] Pointer to the MLU function. You can either define MLU functions
 *        using BANG C language or extract MLU functions from offline models.
 *        See "BANG C Developer Guide" and "CNRT Developer Guide" for detailed information.
 * @param init_param[in] The buffer to store MLU function related parameters
 *                       used in the kernel launch.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because
 *         ``function`` or ``init_param`` is NULL, or the value of ``init_param`` is invalid.
 * @retval CNRT_RET_ERR_NODEV This function call failed because no MLU devices are detected.
 * @retval CNRT_RET_ERR_UNSUPPORTED This function call failed because current platform
 *         dose not support this function.
 * @retval CNRT_RET_ERR_NOMEM This function call failed because insufficient host memory
 *         can be allocated.
 * @retval CNRT_RET_ERR_MEMCPY This function called failed because error occurred during
 *         copying data from host to MLU device.
 *
 * @details Extracts parameters from an MLU function pointed by ``function``, and adds
 *          the parameters to the buffer defined in ::cnrtKernelInitParam_t struct
 *          ``init_param``. The MLU function related parameters are used in the kernel launch.
 *
 * @note
 * - You need to call this function before launching kernels.
 *
 * @par Requirement
 * - None.
 *
 * @par Example
 * - See example in ::cnrtInvokeKernel_V3 function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtInitKernelMemory(const void *function,
                                                   cnrtKernelInitParam_t init_param);
/**
 * @brief Launches a kernel to perform tasks on MLU devices.
 *
 * @param function[in] Pointer to the MLU function. You can either define MLU functions
 *        using BANG C language or extract MLU functions from offline models.
 *        See "BANG C Developer Guide" and "CNRT Developer Guide" for detailed information.
 * @param init_param[in] The buffer that stores MLU function related parameters.
 * @param dim[in] The grid dimensions defined in ::cnrtDim3_t struct.
 * @param params[in] The kernel parameter buffer used in the kernel launch.
 * @param func_type[in] The number of MLU cores used for parallel computing defined in
 *                      ::cnrtFunctionType_t enum.
 * @param queue[in] The queue handle associated with the MLU function. You can create a
 *        queue by calling the ::cnrtCreateQueue function.
 * @param extra_param[in] Pointer to the parameters that specify the cluster affinity
 *                        information. You can set it to NULL if you do not want to
 *                        set cluster affinity. For best practices, you can define
 *                        DDR channel with the corresponding cluster in ::cnrtInvokeParam_t
 *                        struct. See "CNRT User Guide".
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function called failed because one of the following
 *         errors occurred:
 *         - ``function`` or ``init_param`` is NULL.
 *         - The MLU function set in the ::cnrtInitKernelMemory and cnrtInvokeKernel_V3
 *           functions is not the same one.
 *         - The value of ``dim``, ``param``, or ``func_type`` is invalid.
 * @retval CNRT_RET_ERR_INIT This function call failed because the ::cnrtInitKernelMemory
 *         function has not been called correctly.
 * @retval CNRT_RET_ERR_NODEV This function call failed because no MLU devices are detected.
 * @retval CNRT_RET_ERR_UNSUPPORTED This function call failed because the current platform
 *         dose not support this API.
 * @retval CNRT_RET_ERR_NOMEM This function called failed because insufficient memory space
 *         on host or device can be allocated.
 * @retval CNRT_RET_ERR_CNDRV_FUNC_CALL This function call failed because errors
 *         from driver functions are returned via CNRT. See the detailed error
 *         messages on the screen.
 *
 *
 * @details Launches a kernel function on "dim" with the given kernel parameters
 * stored in the buffer "params" and "init_param". You can specify the number of computation
 * MLU cores in cnrtFunctionType_t enum "func_type". The "params" is defined in
 * cnrtKernelParamsBuffer_t struct to store kernel parameters used in the kernel launch. The
 * "init_param" is defined in cnrtKernelInitParam_t struct to store MLU function related
 * kernel parameters. The "extra_param" points to the parameters about cluster affinity. \n
 * Compared with the ::cnrtInvokeKernel_V2 function, this function highly improves the usage
 * of MLU board card resources by setting the cluster affinity in the "extra_param" parameter.
 * Also, using the cnrtInvokeKernel_V3 function, the MLU function related parameters are
 * extracted before launching the kernel. While
 * using the ::cnrtInvokeKernel_V2 function, the MLU function related parameters are extracted when
 * launching the kernel. For more information about cluster affinity, see "CNRT Developer Guide".
 *
 * @note
 * - You need to call the ::cnrtCreateKernelInitParam and
 *   ::cnrtInitKernelMemory functions before calling this function.
 * - If you want to launch kernels via the ::cnrtInvokeKernel_V3 function for a single MLU
 *   function, you only need to call ::cnrtCreateKernelInitParam and ::cnrtInitKernelMemory
 *   functions once.
 * - The MLU function set in the ::cnrtInitKernelMemory and cnrtInvokeKernel_V3 functions
 *   should be the same one. Otherwise, CNRT_RET_ERR_INVALID is returned.
 * - This function does not support using fake device to launch kernels.
 * - This function is always performed by the device associated with a queue.
 *
 * @par Requirement
 * - None.
 *
 * @par Example
 @verbatim
 In the example below, kernel_add.mlu is written in BANG C language.
 See "BANG C Developer Guide" for detailed information.

 file < kernel_add.mlu >
 #include "bang.h"
 __mlu_global__ void kernelAdd(int *a, int *b) {
    *a = *a + *b;
 }


 file < test.c >
 #include "cnrt.h"
 int main () {
   //initialize the environment
   cnrtInit(0);
   cnrtDev_t dev;
   cnrtGetDeviceHandle(&dev, 0);
   cnrtSetCurrentDevice(dev);

   //prepare params
   int a, b, out;
   uint32_t seed = 0x321;
   a = (int)rand_r(&seed) % 10000;
   b = (int)rand_r(&seed) % 10000;

   //prepare queue
   cnrtQueue_t queue;
   cnrtCreateQueue(&queue);

   int *k_a, *k_b;
   cnrtMalloc((void **)&k_a, sizeof(int));
   cnrtMalloc((void **)&k_b, sizeof(int));
   cnrtMemcpy(k_a, &a, sizeof(int), CNRT_MEM_TRANS_DIR_HOST2DEV);
   cnrtMemcpy(k_b, &b, sizeof(int), CNRT_MEM_TRANS_DIR_HOST2DEV);

   cnrtKernelParamsBuffer_t params;
   cnrtGetKernelParamsBuffer(&params);
   cnrtKernelParamsBufferAddParam(params, &k_a, sizeof(int *));
   cnrtKernelParamsBufferAddParam(params, &k_b, sizeof(int *));

   //dim
   cnrtDim3_t dim;
   dim.x = 1;
   dim.y = 1;
   dim.z = 1;

   //func_type
   cnrtFunctionType_t func_type = CNRT_FUNC_TYPE_BLOCK;

   //init_param
   cnrtKernelInitParam_t init_param1;
   cnrtCreateKernelInitParam(&init_param1);
   cnrtInitKernelMemory((void *)kernelAdd, init_param1);

   //invoke_param
   unsigned int affinity1 = 0x00;
   cnrtInvokeParam_t invoke_param1;
   invoke_param1.invoke_param_type = CNRT_INVOKE_PARAM_TYPE_0;
   invoke_param1.cluster_affinity.affinity = &affinity1;

   printf("out:%d, a=%d, b=%d, a+b=%d\n", out, a, b, (a + b));

   //invoke_v3
   cnrtInvokeKernel_V3((void *)&kernel, init_param1, dim, params, func_type, queue,
                                  (void *)&invoke_param1);

   cnrtSyncQueue(queue);

   Get result
   cnrtMemcpy(&out, k_a, sizeof(int), CNRT_MEM_TRANS_DIR_DEV2HOST);

   //release resource
   cnrtFree(k_a);
   cnrtFree(k_b);
   cnrtDestroyKernelParamsBuffer(params);

   cnrtDestroyKernelInitParamAndMemory(init_param1);
   cnrtDestroyQueue(queue);
   cnrtDestroy();

   return 0;
 }
 @endverbatim
 *
 */

extern CNRT_DLL_API cnrtRet_t cnrtInvokeKernel_V3(const void *function,
                                                  cnrtKernelInitParam_t init_param,
                                                  cnrtDim3_t dim,
                                                  cnrtKernelParamsBuffer_t params,
                                                  cnrtFunctionType_t func_type,
                                                  cnrtQueue_t queue,
                                                  void *extra_param);
/**
 * @brief Destroys a buffer that stores MLU function related parameters used in
 *        the kernel launch.
 *
 * @param init_param[in] The buffer that stores MLU function related parameters.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because ``param`` is NULL.
 * @retval CNRT_RET_ERR_CNDRV_FUNC_CALL This function call failed because errors
 *         from driver functions are returned via CNRT. See the detailed error
 *         messages on the screen.
 * @retval CNRT_RET_ERR_FIND_DEV_ADDR This function call failed because it is
 *         unable to find the device memory address to be destroyed.
 *
 * @details Destroys the buffer ``param`` that stores MLU function related parameters for
 *        the kernel launch, and releases the related resources.
 *
 * @note
 * - None.
 *
 * @par Requirement
 * - None.
 *
 * @par Example
 * - See example in ::cnrtInvokeKernel_V3 function.
 */

extern CNRT_DLL_API cnrtRet_t cnrtDestroyKernelInitParamAndMemory(cnrtKernelInitParam_t param);

/*********************************************************************************
 * Model load and Function call
 *********************************************************************************/

/**
 * @brief Loads an offline model from a model file.
 *
 * @param fname[in] The name of the Cambricon offline model file to be loaded. You can specify
 *                  the absolute or relative path of the file where the file is located in this
 *                  parameter. For example, ``/home/abc/model/conv.cambricon`` or
 *                  ``./model/conv.cambricon``. For detailed information about how to generate a
 *                  Cambricon offline model file, see "CNML User Guide".
 * @param pmodel[out] Pointer to the loaded offline model file defined in ::cnrtModel_t struct.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_NOMEM This function call failed because not enough
 *         memory can be allocated.
 *
 * @details Loads an offline model from a Cambricon model file ``fname``, and returns in ``pmodel``
 *          the loaded offline model. This function allocates a block of memory to initialize
 *          the ::cnrtModel struct that holds the offline model information on the host.
 *
 * @note
 * - You need to release the resources with the ::cnrtUnloadModel function after executing the
 *   offline model.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtInvokeRuntimeContext function.

 */
extern CNRT_DLL_API cnrtRet_t cnrtLoadModel(cnrtModel_t *pmodel, const char *fname);

/**
 * @brief Loads an offline model from memory.
 *
 * @param pmodel[out] Pointer to the loaded offline model file defined in ::cnrtModel_t struct.
 * @param ptr[in] Pointer to the memory that stores the offline model to be loaded on the host.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID_POINTER This function call failed because ``ptr`` or ``pmodel``
 *         is NULL or invalid.
 * @retval CNRT_RET_ERR_NOMEM This function call failed because not enough
 *         memory can be allocated.
 *
 * @details Loads the offline model ``ptr`` from memory, and saves the extracted offline
 * model information defined in the ::cnrtModel_t struct in the host memory pointed by ``pmodel``.
 *
 * @note
 * - You need to call the ::cnrtUnloadModel function after executing the offline model to free
 *   the resources. The ::cnrtUnloadModel function should be called after the
 *   ::cnrtDestroyFunction function.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtGetModelSize function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtLoadModelFromMem(cnrtModel_t *pmodel, char *ptr);

/**
 * @brief Loads an offline model from memory.
 *
 * @param pmodel[out] Pointer to the loaded offline model file defined in ::cnrtModel_t struct.
 * @param ptr[in] Pointer to the memory that stores the offline model to be loaded on the host.
 * @param reuseUserMem[in] Whether the model pointer reuses the user memory.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID_POINTER This function call failed because ``ptr`` or ``pmodel``
 *         is NULL or invalid.
 * @retval CNRT_RET_ERR_NOMEM This function call failed because not enough
 *         memory can be allocated.
 *
 * @details Loads the offline model ``ptr`` from memory, and saves the extracted offline
 * model information defined in the ::cnrtModel_t struct in the host memory pointed by ``pmodel``.
 *
 * @note
 * - If the model reuses the user memory, the param pointer ptr[in] can be freed only if the
 *   model loaded from model ptr is not used to extract function and create runtimecontext.
 *   You should not free the ptr[in] before calling ::cnrtInitRuntimeContext.
 * - You need to call the ::cnrtUnloadModel function after executing the offline model to free
 *   the resources. The ::cnrtUnloadModel function should be called after the
 *   ::cnrtDestroyFunction function.
 * @par Requirements
 * - None.
 *
 * @par Example
 @verbatim
  ...
  char fname[100] = "";
  strcat(fname, (char *)name);
  FILE *fp = fopen(fname, "r");
  int model_size;
  cnrtGetModelSize(fname, &model_size));
  char *model_buffer = NULL;
  model_buffer = (char *)malloc(model_size);
  fread(model_buffer, model_size, 1, fp) != 1)
  fclose(fp);
  cnrtModel_t model;
  cnrtLoadModelFromMem_V2(&model, model_buffer, true);
  cnrtFunction_t function;
  cnrtCreateFunction(&function);
  cnrtExtractFunction(&function, model, func_name);
  ...
  cnrtInitRuntimeContext(ctx, NULL);
  free(model_buffer);
  ...
  cnrtInvokeRuntimeContext(ctx, param, queue, NULL);
  ...
 @endverbatim
 */
extern CNRT_DLL_API cnrtRet_t cnrtLoadModelFromMem_V2(cnrtModel_t *pmodel,
                                                      char *ptr,
                                                      bool reuseUserMem);

/**
 * @brief Frees all memory resources of an offline model on the host and MLU device.
 *
 * @param model[in] Pointer to the offline model file to be freed defined in ::cnrtModel_t struct.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID_POINTER This function call failed because ``model`` is NULL.
 *
 * @details Frees the memory that stores an offline model ``model``.
 *
 * @note
 * - This function should be called after CNRT Functions in the offline model are destroyed.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtInvokeRuntimeContext function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtUnloadModel(cnrtModel_t model);

/**
 * @brief Retrieves the size of an offline model file.
 *
 * @param fname[in] The file name of a Cambricon model. You can specify the absolute or relative
 *                  path of the file where the file is located in this parameter. For example,
 *                  ``/home/abc/model/conv.cambricon`` or ``./model/conv.cambricon``.
 * @param size[out] Pointer to the size of the offline model in bytes.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_SYS This function call failed because unable to open the offline model
 *         file.
 * @retval CNRT_RET_ERR_INVALID_POINTER This function call failed because ``fname`` or ``size``
 *         is NULL or invalid.
 * @retval CNRT_RET_ERR_INVALID This function call failed because ``fname`` cannot be read or
 *         ``fname`` is not a Cambricon offline model file..
 *
 * @details Returns in ``size`` the size of the give Cambricon offline model file ``fname``
 * in bytes. You can get the memory size to be allocated for the model file with this function.
 *
 * @note
 * - The file type and version of the offline model are checked in this function. The offline
 *   models with higher versions are not guaranteed to be parsed by the older versions of the
 *   CNRT library. You can retrieve the version of you offline model with
 *   ::cnrtGetModelLevelFromFile and check the latest version of the CNRT library that can
 *   be used.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 @verbatim
  ...
  char fname[100] = "";
  strcat(fname, (char *)name);
  FILE *fp = fopen(fname, "r");
  int model_size;
  cnrtGetModelSize(fname, &model_size));
  char *model_buffer = NULL;
  model_buffer = (char *)malloc(model_size);
  fread(model_buffer, model_size, 1, fp) != 1)
  fclose(fp);
  cnrtModel_t model;
  cnrtLoadModelFromMem(&model, model_buffer);
  int64_t model_size;
  cnrtGetModelMemUsed(model, &model_size);
  cnrtFunction_t function;
  cnrtCreateFunction(&function);
  cnrtExtractFunction(&function, model, func_name);
  int64_t function_size;
  cnrtGetFunctionMemUsed(function, &funciton_size);
  ...
 @endverbatim
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetModelSize(const char *fname, int *size);

/**
 * @brief Retrieves the core version of MLU device from an offline model file.
 *
 * @param model[in] Pointer to the offline model file to be retrieved.
 * @param coreVersion[out] Pointer to the core version of the offline model file.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID_POINTER This function call failed because ``model`` is invalid or
 *         ``coreVersion`` pointer NULL.
 *
 * @details  Returns in ``coreVersion`` the core version of MLU device from a given offline
 * model file ``model``.
 *
 * @note
 * - None
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 @verbatim
 ...
 cnrtModel_t model;
 cnrtLoadModelFromMem(&model, model_buffer);
 cnrtCoreVersion core_version;
 cnrtQueryCoreVersion(model, &core_version);
 ...
 @endverbatim
 */
extern CNRT_DLL_API cnrtRet_t cnrtQueryCoreVersion(cnrtModel_t model,
                                                   cnrtCoreVersion_t *coreVersion);

/**
 * @brief Retrieves the model parallelism of an offline model file.
 *
 * @param model[in] Pointer to the offline model file to be retrieved.
 * @param modelParallelism[out] Pointer to the model parallelism of the offline model file.
 *        An offline model file may contains multiple CNRT Functions. The model parallelism is the
 *        maximum core numbers that can be used in function for all CNRT functions in the
 *        offline model file.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID_POINTER This function call failed because ``model`` or
 *         ``modelParallelism`` is invalid.
 *
 * @details Returns in ``modelParallelism`` the maximum core numbers of the functions
 * used for computing in the offline model file.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 @verbatim
 ...
 cnrtModel_t model;
 cnrtLoadModelFromMem(&model, model_buffer);
 int model_parallelism = 0;
 cnrtQueryModelParallelism(model, &model_parallelism);
 ...
 @endverbatim
 */
extern CNRT_DLL_API cnrtRet_t cnrtQueryModelParallelism(cnrtModel_t model, int *modelParallelism);

/**
 * @brief Retrieves the maximum size of stack memory defined in an offline model file.
 *
 * @param model[in] Pointer to the offline model file to be retrieved.
 * @param size[out] Pointer to the stack size in MB defined in the offline model file. An offline
 *        model file may contains multiple CNRT Functions. This is the maximum size of the stack
 *        memory that can be used in functions for all CNRT Functions in the offline model file.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID_POINTER This function call failed because ``model`` is invalid.
 *
 * @details Returns in ``local_mem_size`` the maximum size (MB) of stack memory defined in the
 * offline model file ``model`` for computing. This function is deprecated. It is recommended
 * to use the ::cnrtQueryModelLocalMemSize function instead.
 *
 * @note
 * - None
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 @verbatim
 ...
 cnrtModel_t model;
 cnrtLoadModelFromMem(&model, model_buffer);
 uint64_t model_stack_size = 0;
 cnrtQueryModelStackSize(model, &model_stack_size);
 ...
 @endverbatim
 */
extern CNRT_DLL_API cnrtRet_t cnrtQueryModelStackSize(cnrtModel_t model, uint64_t *stack_size);

/**
 * @brief Retrieves the maximum size of local memory defined in an offline model file.
 *
 * @param model[in] Pointer to the offline model file to be retrieved.
 * @param local_mem_size[out] Pointer to the local memory in MB defined in the offline model file.
 *        An offline model file may contains multiple CNRT Functions. This is the maximum size of
 *        the private memory that can be used in functions for all CNRT Functions in the offline
 *        model file.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID_POINTER This function call failed because ``model`` is invalid.
 *
 * @details Returns in ``local_mem_size`` the maximum size (MB) of local memory defined in the
 * offline model file ``model`` for computing.
 *
 * @note
 * - None
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 @verbatim
 ...
 cnrtModel_t model;
 cnrtLoadModelFromMem(&model, model_buffer);
 uint64_t local_mem_size;
 cnrtQueryModelLocalMemSize(model,&local_mem_size);
 ...
 @endverbatim
 */
extern CNRT_DLL_API cnrtRet_t cnrtQueryModelLocalMemSize(cnrtModel_t model,
                                                         uint64_t *local_mem_size);

/**
 * @brief Retrieves the number of CNRT Functions in an offline model file.
 *
 * @param model[in] Pointer to the offline model file to be retrieved.
 * @param func_num[out] Pointer to the number of CNRT Functions in the offline model file.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID_POINTER This function call failed because ``model`` is NULL.
 *
 * @details Returns in ``func_num`` the total number of CNRT Functions in the offline model
 * file ``model``.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 @verbatim
 ...
 cnrtModel_t model;
 cnrtLoadModelFromMem(&model, model_buffer);
 int func_num = 0;
 cnrtGetFunctionNumber(model, &func_num);
 ...
 @endverbatim
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetFunctionNumber(cnrtModel_t model, int *func_num);

/**
 * @brief Retrieves the CNRT Function name defined in an offline model file.
 *
 * @param model[in] Pointer to the offline model file to be retrieved.
 * @param index[in] The index that specifies which CNRT Function in the offline model file to be
 *        retrieved. You can call the ::cnrtGetFunctionNumber function to get the total number
 *        of CNRT Functions. The value is in the range [0, ::cnrtGetFunctionNumber-1].
 * @param symbol[out] Pointer to the returned CNRT Function symbol.
 * @param name_size[out] The total number of characters of the returned CNRT Function symbol.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID_POINTER This function call failed because ``model`` pointer
 *         is NULL or the value of ``index`` is out of range [0, ::cnrtGetFunctionNumber-1].
 *
 * @details Returns in ``symbol`` and ``name_size`` the CNRT Function symbol and the total number of
 * characters of the CNRT Function symbol from the offline model file ``model``. CNRT Function
 * symbol is the name of a CNRT Function. You need to specify which CNRT Function symbol in the
 * offline model file to be retrieved in ``index``.
 *
 * @note
 * - You need to release the memory that stores the CNRT Function symbol with the free function.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 @verbatim
 ...
 cnrtModel_t model;
 cnrtLoadModelFromMem(&model, model_buffer);
 cnrtFunction_t function;
 cnrtCreateFunction(&function);
 char *func_name = NULL;
 int func_name_size = 0;
 cnrtGetFunctionSymbol(model, 0, &func_name, &func_name_size);
 cnrtExtractFunction(&function, model, func_name);
 ...
 free(func_name);
 cnrtDestroyFunction(function);
 cnrtUnloadModel(model);
 ...
 @endverbatim
 */
extern cnrtRet_t cnrtGetFunctionSymbol(cnrtModel_t model, int index, char **symbol, int *name_size);

/**
 * @brief Creates a CNRT Function.
 *
 * @param function[in] Pointer to a CNRT function extracted from an offline model file. This CNRT
 *        Function is defined in ::cnrtFunction_t.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_NOMEM This function call failed because not enough
 *         memory can be allocated.
 *
 * @details Creates a CNRT Function by allocating memory to store CNRT Function information, and
 * initialize the ::cnrtFunction_t struct that holds the information about the CNRT Function.
 *
 * @note
 * - You need to release the memory resources about the CNRT Function with the
 *   ::cnrtDestroyFunction function.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtInvokeRuntimeContext function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtCreateFunction(cnrtFunction_t *pfunction);

/**
 * @brief Retrieves a CNRT Function from an offline model file by a CNRT Function name.
 *
 * @param function[out] Pointer to a CNRT function extracted from an offline model file. This CNRT
 *        Function is defined in ::cnrtFunction_t.
 * @param model[in] Pointer to the offline model file to be extracted.
 * @param symbol[in] The name of the CNRT Function to be retrieved. You can retrieve the
 *        CNRT Function name with the ::cnrtGetFunctionSymbol function.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_NO_EXIST This function call failed because unable to find the CNRT
 *         Function with the given CNRT Function name.
 * @retval CNRT_RET_ERR_INVALID This function call failed because the offline model file is
 *         invalid. You need to regenerate the offline model file with the latest version of
 *         CNRT library.
 *
 * @details Returns in ``pfunction`` the CNRT Function from the offline model file ``model`` with
 * the given CNRT Function name ``symbol``. This function initializes the ::cnrtFunction_t
 * struct that holds the information about the CNRT Function to be retrieved.
 *
 * @note
 * - You need to create a CNRT Function with the ::cnrtCreateFunction function before calling
 *   this function.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtInvokeRuntimeContext function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtExtractFunction(cnrtFunction_t *pfunction,
                                                  cnrtModel_t model,
                                                  const char *symbol);

/**
 * @brief Destroys a CNRT Function and releases the related memory resources.
 *
 * @param function[in] Pointer to the CNRT function to be destroyed.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 *
 * @details Destroys the memory resources of the CNRT Function ``function``. This function should
 * be called before calling the ::cnrtUnloadModel function.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtInvokeRuntimeContext function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtDestroyFunction(cnrtFunction_t function);

/**
 * @brief Queries if the CNRT Function is in cache mode.
 *
 * @param function[in] Pointer to a CNRT function extracted from an offline model file. This CNRT
 *        Function is defined in ::cnrtFunction_t.
 * @param is_cache_mode[out] The pointer of a flag which shows if the CNRT Function is cache mode.
 * @param cache_num[out] The pointer of the number of cache in the CNRT Function.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID_POINTER This function call failed because the ``is_cache_mode`` or
 *         ``function`` is null.
 * @retval CNRT_RET_ERR_INVALID This function call failed because the ``function`` is invalid.
 *
 * @details Returns in ``is_cache_mode`` if the CNRT Function is in cache mode, and returns in
 *          ``cache_num`` the number of cache in CNRT Function. If ``cache_num`` is null, the
 *          number of cache won't be returned.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
extern CNRT_DLL_API cnrtRet_t cnrtQueryCacheMode(cnrtFunction_t function,
                                                 bool *is_cache_mode,
                                                 int *cache_num);

/**
 * @brief Queries all the batches of the CNRT function.
 *        Function will return the first input's first dimension if the kernel have more than one
 * input.
 *        The first dimension of the dimensions must be batch.
 *
 * @param function[in] Pointer to a CNRT function extracted from an offline model file. This CNRT
 *        Function is defined in ::cnrtFunction_t.
 * @param batches_array[out] The address of a one-dimensional array, which is allocated by user
 * based on "cache_num"
 *        get by interface cnrtQueryCacheMode if in cache mode.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID_POINTER This function call failed because the ``batches_array`` or
 *         ``function`` is NULL.
 * @retval CNRT_RET_ERR_INVALID This function call failed because the ``function`` is invalid.
 * @details Returns in ``batches_array`` of the CNRT Function no matter if the function is in cache
 * mode.
 *          Function will return the first input's batch if the kernel have more than one input.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
extern CNRT_DLL_API cnrtRet_t cnrtQueryBatches(cnrtFunction_t function, int *batches_array);

/**
 * @brief Retrieves the index of a specified input or output tensor in a CNRT Function.
 *
 * @param function[in] Pointer to a CNRT function extracted from an offline model file. This CNRT
 *        Function is defined in ::cnrtFunction_t.
 * @param name[in] The name of the input or output tensor to be retrieved in the CNRT Function.
 *        You can call the ::cnrtGetInputAndOutputNamesFromFunction function to get the name of
 *        the input and output tensors. The names of input and output tensors are set with the
 *        cnmlSetTensorName function.
 * @param index[out] The index for which input or output tensor to be retrieved in the CNRT
 *        Function, the range start from 0.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because ``function`` is invalid,
 *         ``name`` is NULL, or ``index`` is NULL.
 * @retval CNRT_RET_ERR_FAULT This function call failed because unable to find the matching
 *         ``name`` in the ``func``.
 * @retval CNRT_RET_ERR_FUNC_CALL This function call failed because the version of offline model
 *         is old that not support this function. You need to use current CNRT library and
 *         CNML library to regenerate the offline model.
 *
 * @details Returns in ``index`` the index of input or output in the given CNRT Function ``func``
 *          with the name of input or output tenor ``name``.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtGetInputAndOutputNamesFromFunction function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetParamIndexByName(cnrtFunction_t func,
                                                      const char *name,
                                                      int *index);

/**
 * @brief Retrieves the name of all input and output tensors in a CNRT Function.
 * @param func[in] Pointer to a CNRT Function extracted from an offline model file. This CNRT
 *        Function is defined in ::cnrtFunction_t.
 * @param input_names[out] Pointer to a 2-Dimensional array that stores all the inputs of a CNRT
 *        Function. The order of the inputs in the array is the input order when creating the
 *        fusion operation. You need to declare this parameter before calling this function and
 *        releases the related resources with the ::cnrtDestroyInputAndOutputNames function.
 *        For example, declare this parameter with ``char** input_names``, and then pass the
 *        parameter ``&input_names`` to this function. After calling this function,
 *        ``input_names[i]`` is the input name of the CNRT Function where ``i`` is the index of the
 *        input ranging from 0 to ``input_num``.
 * @param output_names[out] Pointer to a 2-Dimensional array that stores all the output of a CNRT
 *        Function. The order of the outputs in the array is the output order when creating the
 *        fusion operation. You need to declare this parameter before calling this function and
 *        releases the related resources with the ::cnrtDestroyInputAndOutputNames function. For
 *        example, declare this parameter with ``char** output_names``, and then pass the
 *        parameter ``&output_names`` to this function. After calling this function,
 *        ``output_names[i]`` is the output name of the CNRT Function where ``i`` is the index of
 the
 *        output ranging from 0 to ``output_num```.
 * @param input_num[out] Pointer to the number of inputs in the CNRT Function.
 * @param output_num[out] Pointer to the number of outputs in the CNRT Function.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID_POINTER This function call failed because the ``function``
 *         is invalid.
 * @retval CNRT_RET_ERR_UNSUPPORTED This function call failed because the version of the offline
 *         model is old that not support this function. You need to use current CNRT library and
 *         CNML library to regenerate the offline model.
 *
 * @details Returns in ``input_names`` and ``output_names`` the inputs and outputs in the CNRT
 *          Function ``func``. The total number of inputs and outputs in the CNRT Function are
 *          also returned in ``input_num`` and ``output_num``.
 *
 * @note
 * - You need to release the memory that stores all the inputs and outputs of a CNRT Function with
 * the ::cnrtDestroyInputAndOutputNames function.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 @verbatim
 main() {
  // Initializes device
  cnrtInit(0);
  int curr_device_id = 0;
  cnrtDev_t dev;
  cnrtGetDeviceHandle(&dev, curr_device_id);
  cnrtSetCurrentDevice(dev);

  // Load model and create CNRT Function
  cnrtLoadModel(&model, model_name);
  cnrtFunction_t function;
  cnrtCreateFunction(&function);
  cnrtExtractFunction(&function, model, func_name);

  cnrtQueue_t queue;
  cnrtCreateQueue(&queue);

  // Initialize runtime environment
  ...

  void **inputPtrs;
  void **outputPtrs;
  int input_num = 0;
  int output_num = 0;

  // Assuming model is hw mutable offline model, and function information is :
  // input_num = 1, output_num = 1, input_dim_num = 4, input_dim_shape = [a, b, c, d]

  cnrtGetInputAndOutputNamesFromFunction(function, &inputPtrs, &outputPtrs, &input_num,
      &output_num);

  cnrtParamDescArray_t input_param_desc, output_param_desc;
  cnrtCreateParamDescArray(&input_param_desc, input_num);
  cnrtCreateParamDescArray(&output_param_desc, output_num);

  for (int i = 0; i < input_num; i++) {
    int input_index = 0;
    cnrtGetParamIndexByName(function, inputPtrs[i], &input_index);
    cnrtSetShapeToParamDesc(input_param_desc[input_index], input_dim_shape, input_dim_num);
  }

  cnrtInferFunctionOutputShape(function, input_num, input_param_desc,
                               output_num, output_param_desc);

  // Invoke Runtime Context and get result
  ...

  for (int i = 0; i < input_num; i++) {
    cnrtDestroyParamDescArray(input_param_desc[i], input_num);
  }
  for (int i = 0; i < output_num; i++) {
    cnrtDestroyParamDescArray(output_param_desc[i], output_num);
  }

  cnrtDestroyInputAndOutputNames(inputPtrs, outputPtrs, input_num, output_num);
  cnrtDestroyQueue(queue);
  cnrtDestroyFunction(function);
  cnrtUnloadModel(model);
  cnrtDestroy();
 }
 @endverbatim
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetInputAndOutputNamesFromFunction(cnrtFunction_t func,
                                                                     char ***input_names,
                                                                     char ***output_names,
                                                                     int *input_num,
                                                                     int *output_num);
/**
 * @brief Frees the memory that stores the input and output tensor names used in the
 *        ::cnrtGetInputAndOutputNamesFromFunction function.
 * @param input_names[in] Pointer to a 2-Dimensional array that stores all the inputs of a CNRT
 *        Function.
 * @param output_names[in] Pointer to a 2-Dimensional array that stores all the outputs of a CNRT
 *        Function.
 * @param input_num[in] Pointer to the number of inputs in the CNRT Function.
 * @param output_num[in] Pointer to the number of outputs in the CNRT Function.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because ``input_num`` or
 *         ``output_num`` is less than or equal to 0.
 * @retval CNRT_RET_ERR_INVALID_POINTER This function call failed because the ``input_names``
 *         or ``output_names`` is NULL.
 *
 * @details Releases the memory that stores the input tensor names, output tensor names, the
 *          number of inputs and the number of outputs in ``input_names``, ``output_names``,
 *          ``input_num``, and ``output_num`` respectively that are used in the
 *          ::cnrtGetInputAndOutputNamesFromFunction function.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtGetInputAndOutputNamesFromFunction function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtDestroyInputAndOutputNames(char **input_names,
                                                             char **output_names,
                                                             int input_num,
                                                             int output_num);
/**
 * @brief Retrieves the number of dimensions of a specified input or output tensor in a CNRT
 *        Function.
 *
 * @param function[in] Pointer to a CNRT Function extracted from an offline model file. This CNRT
 *        Function is defined in ::cnrtFunction_t.
 * @param name[in] The name of the input or output tensor to be retrieved in the CNRT Function.
 *        You can call the ::cnrtGetInputAndOutputNamesFromFunction function to get the name of
 *        the input and output tensors.
 * @param dim_num[out] The number of dimensions of the input or output tensor to be retrieved
 *        in the CNRT Function.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because ``function``, ``name`` or
 *         ``dim_num`` is NULL.
 * @retval CNRT_RET_ERR_FUNC_CALL This function call failed because the version of the offline
 *         model is old that not support this function. You need to use current CNRT library
 *         and CNML library to regenerate offline model file or call the ::cnrtGetInputDataShape,
 *         ::cnrtGetInputDataSize, ::cnrtGetInputDataType, ::cnrtGetOutputDataShape,
 *         ::cnrtGetOutputDataSize, ::cnrtGetOutputDataType to get tensor information.
 * @retval CNRT_RET_ERR_FAULT This function call failed because unable to find the matching
 *         ``name`` in the ``func``.
 *
 * @details Returns in ``dim_num`` the number of dimensions of the input or output tensor with
 *          the given tensor name ``name`` in the CNRT Function ``func``.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtGetSupportedParamShapeByName function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetSupportedParamDimNumberByName(cnrtFunction_t func,
                                                                   const char *name,
                                                                   int *dim_num);

/**
 * @brief Retrieves the shape of a specified input or output tensor in a CNRT Function.
 *
 * @param function[in] Pointer to a CNRT Function extracted from an offline model file. This CNRT
 *        Function is defined in ::cnrtFunction_t.
 * @param name[in] The name of the input or output tensor to be retrieved in the CNRT Function.
 *        You can call the ::cnrtGetInputAndOutputNamesFromFunction function to get the name of the
 *        input and output tensors.
 * @param dim_shape[out] The shape of the input or output tensor to be retrieved in the CNRT
 *        Function.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because ``function``, ``name`` or
 *         ``dim_shape`` is NULL.
 * @retval CNRT_RET_ERR_FUNC_CALL This function call failed because the version of the offline
 *         model is old that not support this function. You need to use current CNRT library and
 *         CNML library to regenerate offline model file or call the ::cnrtGetInputDataShape,
 *         ::cnrtGetOutputDataShape to get tensor shape.
 * @retval CNRT_RET_ERR_FAULT This function call failed because unable to find the matching name
 *         in the function.
 *
 * @details Returns in ``dim_shape`` the shape of input or output tensor in the given CNRT Function
 *          ``func`` with the name of input or output tenor ``name``.
 *
 * @note
 * - The memory of ``dim_shape`` should malloc and free by user, and you can get dim number by
 *   calling ::cnrtGetSupportedParamDimNumberByName. This function is used for immutable shape
 *   of tensor.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 @verbatim
 main() {
  // Initialize runtime environment
  ...

  void **inputPtrs;
  void **outputPtrs;
  int input_num = 0;
  int output_num = 0;

  // Assuming the model is an immutable model file and this function is extracted from this model.
  cnrtGetInputAndOutputNamesFromFunction(function, &inputPtrs, &outputPtrs, &input_num,
      &output_num);

  for (int i = 0; i < input_num; i++) {
    int input_dim_num = 0;
    cnrtGetSupportedParamDimNumberByName(function, inputPtrs[i], input_dim_num);
    int *input_dim_shape = (int*)malloc(input_dim_num * sizeof(int));
    cnrtGetSupportedParamShapeByName(function, inputPtrs[i], input_dim_shape);
    free(input_dim_shape);
  }

  for (int i = 0; i < output_num; i++) {
    int output_dim_num = 0;
    cnrtGetSupportedParamDimNumberByName(function, outputPtrs[i], output_dim_num);
    int *output_dim_shape = (int*)malloc(output_dim_num * sizeof(int));
    cnrtGetSupportedParamShapeByName(function, outputPtrs[i], output_dim_shape);
    free(output_dim_shape);
  }

  // Invoke Runtime Context and get result
  ...
 }
 @endverbatim
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetSupportedParamShapeByName(cnrtFunction_t func,
                                                               const char *name,
                                                               int *dim_shape);

/**
 * @brief Retrieves the data type of a specified input or output tensor in a CNRT Function.
 *
 * @param function[in] Pointer to a CNRT function extracted from an offline model file. This CNRT
 *        Function is defined in ::cnrtFunction_t.
 * @param name[in] The name of the input or output tensor to be retrieved in the CNRT Function.
 *        You can call the ::cnrtGetInputAndOutputNamesFromFunction function to get the name of
 *        the input and output tensors.
 * @param dtype[out] The data type of the input or output tensor to be retrieved in the CNRT
 Function.
 *        The data type is defined in ::cnrtDataType_t.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because ``function``, ``name`` or
 *         ``dtype`` is NULL.
 * @retval CNRT_RET_ERR_FUNC_CALL This function call failed because the version of the offline
 *         model is old that not support this function. You need to use current CNRT library and
 *         CNML library to regenerate offline model file or call ::cnrtGetInputDataType,
 *         ::cnrtGetInputDataType to get tensor data type.
 * @retval CNRT_RET_ERR_FAULT This function call failed because unable to find the matching name
 *         in the function.
 *
 * @details Returns in ``dtype`` the shape of input or output tensor in the given CNRT Function
 *          ``func`` with the name of input or output tenor ``name``.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 @verbatim
 main() {
  // Initialize runtime environment
  ...
  void **inputPtrs;
  void **outputPtrs;
  int input_num = 0;
  int output_num = 0;

  cnrtGetInputAndOutputNamesFromFunction(function, &inputPtrs, &outputPtrs, &input_num,
      &output_num);

  // get input output datatype
  for (int i = 0; i < input_num; i++) {
    cnmlDataType input_datatype;
    cnrtGetSupportedParamDataTypeByName(function, inputPtrs[i], &input_datatype);
  }

  for (int i = 0; i < output_num; i++) {
    cnmlDataType output_datatype;
    cnrtGetSupportedParamDataTypeByName(function, outputPtrs[i], &output_datatype);
  }

  // Invoke Runtime Context and get result
  ...
 }
 @endverbatim
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetSupportedParamDataTypeByName(cnrtFunction_t func,
                                                                  const char *name,
                                                                  cnrtDataType_t *dtype);

/**
 * @brief Retrieves the dimension order of a specified input or output tensor in a CNRT Function.
 *
 * @param function[in] Pointer to a CNRT function extracted from an offline model file. This CNRT
 *        Function is defined in ::cnrtFunction_t.
 * @param name[in] The name of the input or output tensor to be retrieved in the CNRT Function.
 *        You can call the ::cnrtGetInputAndOutputNamesFromFunction function to get the name of
 *        the input and output tensors.
 * @param dorder[out] The dimension order of the input or output tensor to retrieve in the CNRT
 *        Function. The dimension order is defined in ::cnrtDimOrder_t.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function called failed because ``function``, ``name`` or
 *         ``dim_num`` is NULL.
 * @retval CNRT_RET_ERR_FUNC_CALL This function call failed because the version of the offline
 *         model is old that not support this function. You need to use CNRT library and CNML
 *         library to regenerate offline model file.
 * @retval CNRT_RET_ERR_FAULT This function called failed because no match name for ``name`` in
 *         ``function``.
 *
 * @details Returns in ``dorder`` the shape of input or output tensor in the given CNRT Function
 *          ``func`` with the name of input or output tensor ``name``.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtGetInputAndOutputNamesFromFunction function.
 @verbatim
 main() {
  // Initialize runtime environment
  ...
  void **inputPtrs;
  void **outputPtrs;
  int input_num = 0;
  int output_num = 0;

  cnrtGetInputAndOutputNamesFromFunction(function, &inputPtrs, &outputPtrs, &input_num,
      &output_num);

  // get input output data layout
  for (int i = 0; i < input_num; i++) {
    cnrtDimOrder_t input_dim_order;
    cnrtGetSupportedParamLayoutByName(function, inputPtrs[i], &input_dim_order)
  }

  for (int i = 0; i < output_num; i++) {
    cnrtDimOrder_t output_dim_order;
    cnrtGetSupportedParamLayoutByName(function, inputPtrs[i], &output_dim_order)
  }

  // Invoke Runtime Context and get result
  ...
 }
 @endverbatim
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetSupportedParamLayoutByName(cnrtFunction_t func,
                                                                const char *name,
                                                                cnrtDimOrder_t *dorder);

/**
 * @brief Creates a copy of CNRT Function.
 *
 * @param src[in] Pointer to the source memory address that stores the origin CNRT Function.
 * @param dst[out] Pointer to the destination memory address that stores the replicated CNRT
 *        Function.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_FUNC_CALL This function call failed because ``src`` or ``dst``
 *         function is invalid.
 * @retval CNRT_RET_ERR_SYS This function call failed because not enough memory can be allocated
 *         to store intermediate computing results.
 *
 * @details This function is deprecated. Creates a copy of CNRT Function pointed by the source
 *          address ``src``.
 *
 * @note
 * - If the CNRT Model is destroyed, the CNRT Function that is extracted from this model will
 *   be invalid. The replicated CNRT Function copyed from this CNRT Function will be invalid
 *   too.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
extern CNRT_DLL_API cnrtRet_t cnrtCopyFunction(cnrtFunction_t *dst, cnrtFunction_t src);

/*********************************************************************************
 * Memory management
 *********************************************************************************/

/**
 * @brief Allocates memory on the host.
 *
 * @param pPtr[out] Pointer to allocated memory on the host.
 * @param bytes[in] Requested memory size in bytes.
 * @param type[in]  The memory type of the memory to be allocated.
 *                  Supported value is CNRT_MEMTYPE_DEFAULT.
 *                  See ::cnrtMemType_t for details.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because the
 *         ``pPtr`` is NULL or ``type`` is invalid.
 * @retval CNRT_RET_ERR_NOMEM This function call failed because not enough
 *         memory can be allocated.
 *
 *
 * @details Allocates the size of bytes "bytes" of host memory with the memory
 * type specified in the cnrtMemType_t enum "type", and returns a pointer
 * "pPtr" to the allocated memory. This function supports allocating page-locked
 * memory type with IPC supports. \n
 * The memory ranges allocated with this function can be tracked by the driver
 * so that the function calls such as ::cnrtMemcpy are accelerated automatically. \n
 * So in comparison with pageable memory requested by the malloc function,
 * the page-locked memory has a better read and written performance. However,
 * large amount of page-locked memory obtained with this function may lead to
 * lower performance due to the reduce of available amount of memory for paging. So for
 * best practices, we recommend you use this function to allocate staging areas for data
 * exchange between host and the MLU device.
 *
 * @note
 * - You need to call the ::cnrtFreeHost function to release the allocated memory,
 *   otherwise the memory leaks may be occurred.
 * - If the size requested is 0, this function returns ::CNRT_RET_SUCCESS and the
 *   output pointer ``pPtr`` is set to NULL.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtMalloc function.
 *
 */
extern CNRT_DLL_API cnrtRet_t cnrtMallocHost(void **pPtr, size_t bytes, cnrtMemType_t type);

/**
 * @brief Frees the memory space on the host.
 *
 * @param ptr[in] Pointer to the host memory to be freed.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_FAULT This function call failed because it was
 *                            unable to find the corresponding memory address.
 *
 * @details Frees the host memory pointed by ``ptr``. This function is only used
 * to free the memory space that is allocated by the ::cnrtMallocHost
 * function.
 *
 * @note
 * - If the cnrtFreeHost function has already been called before, CNRT_RET_ERR_FAULT
 *   is returned.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtMalloc function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtFreeHost(void *ptr);

/**
 * @brief Allocates memory on the current MLU device.
 *
 * @param pPtr[out] Pointer to allocated memory on the MLU device.
 * @param bytes[in] Requested memory size in bytes.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because the
 *         ``pPtr`` is NULL.
 * @retval CNRT_RET_ERR_NODEV This function call failed because no
 *         MLU devices were detected.
 * @retval CNRT_RET_ERR_NOMEM This function call failed because it was
 *         unable to allocate memory. This may because there is not enough
 *         memory can be allocated.
 *
 * @details Allocates the size of ``bytes`` bytes of device memory on the current
 * MLU device, and returns a pointer ``pPtr`` to the allocated memory. The
 * allocated memory is not initialized.
 *
 * @note
 * - You need to call the ::cnrtSetCurrentDevice function to set the current device
 *   before allocating device memory.
 * - You need to call the cnrtFree function to release the allocated memory,
 *   otherwise the memory leaks may be occurred.
 * - If the size requested is 0, this function returns ::CNRT_RET_SUCCESS and the
 *   output pointer ``pPtr`` is set to NULL.
 * - To request for memory used for encoding or decoding videos and images on VPU or JPU,
 *   use the ::cnrtMallocFrameBuffer function.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 @verbatim
 int main () {
   // Initialize device
   cnrtInit(0);
   cnrtDev_t dev;
   // Get device handle
   cnrtGetDeviceHandle(&dev, 0);
   // Set the device handle for current thread execution context
   cnrtSetCurrentDevice(dev);

   // Specify the memory size
   size_t size = sizeof(size_t) * N;

   // Allocate page-locked memory. Optionally you can also use the
   // malloc to allocate pageable memory on the host. For more
   // information about the advantages of using the cnrtMallocHost
   // function, see cnrtMallocHost description.
   int *host_addr = NULL;
   cnrtMallocHost((void **)&host_addr, size, CNRT_MEMTYPE_DEFAULT);

   // Initialize host memory allocated before
   ...

   // Allocate device memory
   int *dev_addr = NULL;
   cnrtMalloc((void **)&dev_addr, size);

   // Copy data from host to device
   cnrtMemcpy(dev_addr, host_addr, size, CNRT_MEM_TRANS_DIR_HOST2DEV);
   // Copy data from device to host
   cnrtMemcpy(host_addr, dev_addr, size, CNRT_MEM_TRANS_DIR_DEV2HOST);

   // Free device memory
   cnrtFree(dev_addr);

   // Free page-locked memory
   cnrtFreeHost(host_addr);

   cnrtDestroy();
   return 0;
 }
 @endverbatim
 */
extern CNRT_DLL_API cnrtRet_t cnrtMalloc(void **pPtr, size_t bytes);

/**
 * @brief Allocates device memory on the current device.
 *
 * @param pPtr[out] Pointer to the starting address of the memory.
 * @param bytes[in] Requested memory size in bytes.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because the
 *                              ``pPtr`` is NULL.
 *
 * @details Allocates the size of ``bytes`` bytes of virtual memory on the
 * current device, and returns a pointer ``pPtr`` to the allocated memory.
 *
 * @note
 * - It is recommended to use ::cnrtMalloc function intead of this function.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
extern CNRT_DLL_API cnrtRet_t cnrtMallocPeerAble(void **pPtr, size_t bytes);

/**
 * @brief Allocates buffer memory on the current MLU device. This is only used
 *        for encoding or decoding videos and images on VPU or JPU with
 *        Cambricon CNCodec. The memory allocated by this function is a block
 *        of separate area from ::cnrtMalloc so that the performance of accessing
 *        VPU or JPU is better.
 *
 * @param pPtr[out] Pointer to allocated memory on the MLU device.
 * @param bytes[in] Requested memory size in bytes.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because ``pPtr``
 *                              is NULL.
 * @retval CNRT_RET_ERR_NODEV This function call failed because no
 *         MLU devices were detected.
 * @retval CNRT_RET_ERR_NOMEM This function call failed because it was
 *         unable to allocate memory. This is because there is not enough
 *         memory can be allocated.
 *
 * @details Allocates the size of ``bytes`` bytes of buffer memory on the
 * current MLU device, and returns a pointer ``pPtr`` to the allocated memory.
 * The allocated memory is not initialized.
 *
 * @note
 * - You need to call the ::cnrtSetCurrentDevice function to set the current device
 *   before allocating device memory.
 * - You need to call the ::cnrtFree function to release the allocated memory,
 *   otherwise the memory leaks may be occurred.
 * - If the size requested is 0, this function returns ::CNRT_RET_SUCCESS and the
 *   output pointer ``pPtr`` is set to NULL.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See "Cambricon CNCodec Developer Guide" for detailed information.
 *
 */
extern CNRT_DLL_API cnrtRet_t cnrtMallocFrameBuffer(void **pPtr, size_t bytes);

/**
 * @brief Allocates buffer memory on the current MLU device with information about frame
 *        and data parallelism. This is only used for encoding or decoding videos
 *        and images on VPU or JPU with Cambricon CNCodec.
 *        Because you no longer need to set data parallelism for MLU200 series or
 *        higher, we do not recommend that you use this function. You need to use
 *        the ::cnrtMallocFrameBuffer function instead for MLU200 series or higher.
 *
 *
 * @param pPtr[out] Pointer to allocated memory on the MLU device.
 * @param param[in] Pointer to the parameter context buffer that is created by
 *                  the cnrtAllocParam function.
 *                  You need to call the ::cnrtAddParam function to set the value of
 *                  this parameter. The value of this parameter should include the
 *                  following parameters:
 *                  - type: The memory type that is specified in the
 *                          ::cnrtMallocExType_t enum.
 *                  - data_parallelism: (int) The data parallelism used in encoding
 *                                       and decoding images and videos.
 *                  - frame_num: (int) The number of frames used in encoding and
 *                                decoding images and videos.
 *                  - frame_size: (int) The size of the frame used in encoding and
 *                                decoding images and videos.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because ``pPtr``
 *                              is NULL.
 * @retval CNRT_RET_ERR_NODEV This function call failed because no
 *         MLU devices were detected.
 * @retval CNRT_RET_ERR_NOMEM This function call failed because it was
 *         unable to allocate memory. This may because there is not enough
 *         memory can be allocated.
 *
 * @details This function is deprecated. It is recommended that you use the
 * ::cnrtMallocFrameBuffer function instead. \n
 * Allocates buffer memory on the current MLU device based on the information specified
 * in param, and returns a pointer pPtr to the allocated memory. \n
 * The allocated memory is not initialized. If an error occurred, ::CNRT_RET_ERR_INVALID,
 * ::CNRT_RET_ERR_NODEV or ::CNRT_RET_ERR_NOMEM is returned, and the output pointer
 * pPtr is set to NULL.
 *
 * @note
 * - You need to call the ::cnrtSetCurrentDevice function to set the current device
 *   before allocating device memory.
 * - You need to call the ::cnrtFree function to release the allocated memory,
 *   otherwise, the memory leaks may be occurred.
 * - If you use Cambricon CNCodec, you need to use this function to request
 *   for input and output buffer memory.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 *
 */
extern CNRT_DLL_API cnrtRet_t cnrtMallocBufferEx(void **pPtr, void *param);

/**
 * @brief Frees the device memory allocated before.
 *
 * @param ptr[in] Pointer to the device memory to be freed.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_FUNC_CALL This function call failed because errors
 *         from driver functions are returned via CNRT. See the detailed error
 *         messages on the screen.
 * @retval CNRT_RET_ERR_FIND_DEV_ADDR This function call failed because it was
 *                                unable to find the device memory address.
 *
 * @details Frees the MLU device memory pointed by ``ptr``. If ::cnrtFree
 * has already been called before, an error is returned.
 *
 *
 * @note
 * - None
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtMalloc function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtFree(void *ptr);

/**
 * @brief Frees a pointer array on the MLU device.
 *
 * @param ptr[in] Pointer to memory address of the array to be freed.
 * @param length[in] The length of the array to be freed.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_FIND_DEV_ADDR This function call failed because
 *                              it was unable to find the device memory address.
 *
 * @details Frees a pointer array pointed by "ptr" on the MLU device
 * with the given array size length. This function is used to
 * free multiple memory space at once. \n
 * This function is deprecated. It is recommended to use the ::cnrtFree function instead.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
extern CNRT_DLL_API cnrtRet_t cnrtFreeArray(void **ptr, int length);

/**
 * @brief Maps the memory address of an MLU device into the host memory space.
 *
 * @param host_ptr[out] Pointer to the host memory address to be mapped into.
 * @param dev_ptr[in] Pointer to the memory address of the MLU device
 *                    that is allocated by the ::cnrtMalloc function.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_FUNC_CALL This function call failed because errors
 *         from driver functions are returned via CNRT. See the detailed error
 *         messages on the screen.
 *
 * @details Maps memory address of the MLU device pointed by "dev_ptr" into the
 * host memory space, and returns the host memory address pointed by "host_ptr".
 * Then the device memory address can be accessed directly on the host. \n
 * This function is only used on Edge platform.
 *
 * @note
 * - You need to call the ::cnrtUnmap function to unmap the mapped host memory space.
 * - For the Edge platform, the host shares DDR with device, so the frequent data
 *   copy from MLU device to host may affect your computing
 *   performance. For best practices, you can map the memory address of the MLU device
 *   into the host to reduce the number of times on copying data from MLU device to the host.
 * - After mapping the memory address of the MLU device to the host with this function,
 *   you can call the ::cnrtCacheOperation function to synchronize data to make sure
 *   the consistency of data read and written.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 @verbatim
 int main () {
   // Initialize device
   cnrtInit(0);

   // Get device handle
   cnrtDev_t dev;
   cnrtGetDeviceHandle(&dev, 0);

   // Set the device handle for current thread execution context
   cnrtSetCurrentDevice(dev);

   size_t size = sizeof(size_t) * N;

   void *dev_src = NULL;
   void *dev_dest = NULL;
   cnrtMalloc(&dev_src, size);
   cnrtMalloc(&dev_dest, size);

   // Map device memory allocated before into host memory
   void *cpu_src = NULL;
   void *cpu_dest = NULL;
   cnrtMap(&cpu_src, dev_src);
   cnrtMap(&cpu_dest, dev_dest);

   // Flush cache to ensure data consistency
   memset(cpu_src, 'a', size);
   cnrtCacheOperation(cpu_src, CNRT_FLUSH_CACHE);
   memset(cpu_dest, 'b', size);
   cnrtCacheOperation(cpu_dest, CNRT_FLUSH_CACHE);

   void *dev_src_found_by_mapped = NULL;
   cnrtFindDevAddrByMappedAddr(cpu_src, &dev_src_found_by_mapped);
   if (dev_src_found_by_mapped != dev_src)
     printf("find device address by mapped host failed!\n");

   cnrtMemcpy(dev_src_found_by_mapped, dev_dest, size, CNRT_MEM_TRANS_DIR_DEV2DEV);
   if (memcmp(cpu_src, cpu_dest, size))
     printf("memcmp failed!\n");

   // Unmap
   cnrtUnmap(cpu_src);
   cnrtUnmap(cpu_dest);

   // Free device memory
   cnrtFree(dev_src);
   cnrtFree(dev_dest);

   cnrtDestroy();
   return 0;
 }
 @endverbatim
 */
extern CNRT_DLL_API cnrtRet_t cnrtMap(void **host_ptr, void *dev_ptr);

/**
 * @brief Maps a range of MLU device memory address into the address
 *        space of a host memory.
 *
 * @param host_ptr[out] Pointer to the host memory address to be mapped into.
 * @param dev_ptr[in] Pointer to the memory address of the MLU device
 *                    that is allocated by the ::cnrtMalloc function.
 * @param size[in] The size of the memory in bytes to be mapped into.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_MAP This function call failed because an error
 *                          occurred when mapping the device memory address
 *                          into the host memory address space.
 *
 * @details Maps a range of memory address on the MLU device into the address
 * space of host memory. The device memory is pointed by "dev_ptr" with
 * a given "size" bytes size of memory. The host memory address pointed by
 * "host_ptr" is returned. Then you can access this device memory address
 * directly on the host. \n
 * This function is only used on Edge platform.
 *
 * @note
 * - You need to call the ::cnrtUnmap function to unmap the mapped memory space on the host.
 * - For Edge devices, the host shares DDR with the device, so the frequent data
 *   copy from MLU device memory to host memory may affect your computing
 *   performance. For best practices, you can map the memory address of the MLU device
 *   into the host to reduce the number of times on copying data from MLU device memory
 *   to the host.
 * - After mapping the memory address of the MUL device to the host with this function,
 *   you can call the ::cnrtCacheOperationRange function to synchronize data on the host
 *   and MLU device to make sure the consistency of data read and written.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 @verbatim
 int main() {
   // Initialize device
   cnrtInit(0);

   // Get device handle
   cnrtDev_t dev;
   cnrtGetDeviceHandle(&dev, 0);

   // Set the device handle for current thread execution context
   cnrtSetCurrentDevice(dev);

   size_t count = sizeof(size_t) * N;
   size_t rangeCount = sizeof(size_t);

   void *dev_src = NULL;
   void *dev_dest = NULL;
   cnrtMalloc(&dev_src, count);
   cnrtMalloc(&dev_dest, count);

   void *cpu_src = NULL;
   void *cpu_dest = NULL;
   cnrtMapRange(&cpu_src, dev_src, count);
   cnrtMapRange(&cpu_dest, dev_dest, count);

   memset(cpu_src, 'a', count);
   cnrtCacheOperationRange(cpu_src, rangeCount, CNRT_FLUSH_CACHE);
   memset(cpu_dest, 'b', count);
   cnrtCacheOperation(cpu_dest, rangeCount, CNRT_FLUSH_CACHE);

   cnrtMemcpy(dev_src, dev_dest, rangeCount, CNRT_MEM_TRANS_DIR_DEV2DEV);
   if (memcmp(cpu_src, cpu_dest, rangeCount))
     printf("memcmp failed!\n");

   // Unmap address
   cnrtUnmap(cpu_src);
   cnrtUnmap(cpu_dest);

   // Free device memory
   cnrtFree(dev_src);
   cnrtFree(dev_dest);

   cnrtDestroy();
   return 0;
 }
 @endverbatim
 *
 */
extern CNRT_DLL_API cnrtRet_t cnrtMapRange(void **host_ptr, void *dev_ptr, size_t size);

/**
 * @brief Unmaps the mapping between host memory address and device memory address.
 *
 * @param host_ptr[in] Pointer to the host memory to be unmapped. The value of this parameter
 *                     should be obtained by calling the ::cnrtMap or ::cnrtMapRange function.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_UNMAP This function call failed because an error
 *                            occurred during unmapping.
 *
 * @details Unmaps the mapping between host memory address and device memory address
 * that is created using the ::cnrtMap or ::cnrtMapRange function. The host memory
 * space to be unmapped is pointed by "host_ptr". \n
 * This function is only used on Edge platform.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtMap function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtUnmap(void *host_ptr);

/**
 * @brief Retrieves the mapped memory address of the MLU device by the
 *        memory address of the host. This function is only used on Edge platform.
 *
 * @param dev_ptr[out] Pointer to the memory address of the MLU device.
 * @param mappped_host_ptr[in] The mapped address of host.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_FIND_DEV_ADDR This function call failed because
 *                                    it was unable to find the mapped memory
 *                                    address of the device.
 *
 * @details Returns a pointer ``dev_ptr`` to the mapped memory address of the MLU
 * device with the given memory address of the host pointed by ``mappped_host_ptr``.
 * The mapping of the memory address of the MLU device to the memory address
 * of the host address is created by the ::cnrtMap function.
 *
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtMap function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtFindDevAddrByMappedAddr(void *mappped_host_ptr, void **dev_ptr);

/**
 * @brief Retrieves the mapped memory address of the MLU device with offset by the
 *        memory address of the host. This function is only used on Edge platform.
 *
 * @param dev_ptr[out] Pointer to the memory address of the MLU device with offset.
 * @param mapped_host_ptr[in] The mapped address of host.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_FIND_DEV_ADDR This function call failed because
 *                                    it was unable to find the mapped memory
 *                                    address of the device.
 *
 * @details Returns a pointer ``dev_ptr`` to the mapped memory address of the MLU
 * device with the given memory address of the host pointed by ``mapped_host_ptr``.
 * The mapping of the memory address of the MLU device to the memory address
 * of the host address is created by the ::cnrtMap function.
 *
 *
 * @note
 * - This API will return the value of device address which contains the offset.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 @verbatim
 int main () {
   // Initialize device
   cnrtInit(0);

   // Get device handle
   cnrtDev_t dev;
   cnrtGetDeviceHandle(&dev, 0);

   // Set the device handle for current thread execution context
   cnrtSetCurrentDevice(dev);

   size_t size = sizeof(size_t) * N;

   void *dev_src = NULL;
   cnrtMalloc(&dev_src, size);

   // Map device memory allocated before into host memory
   void *cpu_src = NULL;
   void *cpu_dest = NULL;
   cnrtMap(&cpu_src, dev_src);

   // Flush cache to ensure data consistency
   memset(cpu_src, 'a', size);
   cnrtCacheOperation(cpu_src, CNRT_FLUSH_CACHE);

   int offset = 4;
   void *dev_src_found_by_mapped = NULL;
   cnrtFindDevAddrWithOffsetByMappedAddr(cpu_src + offset, &dev_src_found_by_mapped);
   if (dev_src_found_by_mapped != (dev_src + offset))
     printf("find device address by mapped host failed!\n");

   // Unmap
   cnrtUnmap(cpu_src);

   // Free device memory
   cnrtFree(dev_src);

   cnrtDestroy();
   return 0;
 }
 @endverbatim
 */
extern CNRT_DLL_API cnrtRet_t cnrtFindDevAddrWithOffsetByMappedAddr(void *mapped_host_ptr,
                                                                    void **dev_ptr);

/**
 * @brief Flushes or invalidates cache on the host. This function
 * is used with the ::cnrtMap function.
 *
 * @param host_ptr[in] Pointer to the host memory address to be
 *                     mapped into. The memory address should be mapped
 *                     with a device memory address by calling the
 *                     ::cnrtMap function.
 * @param opr[in] The operation type on cache. See ::cnrtCacheOps_t for details.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_CACHE This function call failed because an
 *         error occurred when flushing or invalidating cache.
 *
 * @details Flushes or invalidates the cache for the given memory address
 * pointed by "host_ptr". You can specify the operation on cache with
 * ::cnrtCacheOps_t enum "opr". Currently, only the flush operation is
 * supported. When memory contents have been modified and the dirty data
 * should be cleaned, you can call this function to flush and invalidate cache. \n
 * This function is only used on Edge platform.
 *
 * @note
 * - For Edge platform, the host shares DDR with the device, so the frequent data
 *   copy from MLU device memory to host memory may affect your computing
 *   performance. For best practices, you can map the memory address of the MLU device
 *   to the host to reduce the number of times on copying data from MLU device memory
 *   to the host.
 * - After mapping the memory address of the MUL device to the host with the ::cnrtMap
 *   function, you can call this function to synchronize data on host and MLU
 *   device to make sure the consistency of data read and written.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtMap function.
 *
 */
extern CNRT_DLL_API cnrtRet_t cnrtCacheOperation(void *host_ptr, cnrtCacheOps_t opr);

/**
 * @brief Flushes or invalidates a range of cache on the host. This function
 * is used with the ::cnrtMapRange function. This function is only used on Edge platform.
 *
 * @param host_ptr[in] Pointer to the host memory address.
 *                     The host memory address should be mapped with a device memory
 *                     address by calling the ::cnrtMapRange function.
 * @param size[in] The size of the host memory in bytes.
 * @param opr[in] Operation type on cache. See ::cnrtCacheOps_t for details.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_CACHE This function call failed because an
 *         error occurred when flushing or invalidating cache.
 *
 * @details Flushes or invalidates the cache for the memory address pointed by
 * host_ptr with the memory size specified in size. You can specify the
 * operation on cache with ::cnrtCacheOps_t enum ``opr``. Currently, only the flush
 * operation is supported. When memory contents have been updated or the stale
 * data should be cleaned, you can call this function to flush or invalidate cache.
 *
 *
 * @note
 * - For Edge platform, the host shares DDR with the device, so the frequent data
 *   copy from MLU device memory to host memory may affect computing
 *   performance. For best practices, you can map the memory address of the MLU device
 *   to host to reduce the number of times on copying data from MLU device memory to the host.
 * - After mapping the memory address of the MUL device to the host with the
 *   ::cnrtCacheOperationRange function, you can call this function to synchronize
 *   data on the host and MLU device to make sure the consistency of data read and written.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtMapRange function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtCacheOperationRange(void *host_ptr,
                                                      size_t size,
                                                      cnrtCacheOps_t opr);

/**
 * @brief Retrieves the device memory size and base address allocated by
 *        the ::cnrtMalloc or ::cnrtMallocFrameBuffer function.
 *
 * @param devBasePtr[out] Pointer to the base address of the device memory.
 * @param devPtr[in] Pointer to the address of the device memory
 *                   to retrieve size and base address.
 * @param bytes[out] The size of the allocated memory in bytes.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because ``devBasePtr``
 *                              or ``devPtr`` is NULL.
 * @retval CNRT_RET_ERR_FUNC_CALL This function call failed because errors
 *         from driver functions are returned via CNRT. See the detailed error
 *         messages on the screen.
 *
 * @details Returns the device memory size ``bytes`` in bytes and the pointer to the
 * base address of the device memory with the given device memory address
 * pointed by ``devPtr``. This function is used to retrieve the size and the
 * base address of the memory that are allocated by the ::cnrtMalloc or
 * ::cnrtMallocFrameBuffer function.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 @verbatim
 int main () {
   // Initialize device
   cnrtInit(0);

   // Get device handle
   cnrtDev_t dev;
   cnrtGetDeviceHandle(&dev, 0);

   // Set the device handle for current thread execution context
   cnrtSetCurrentDevice(dev);

   size_t size = sizeof(size_t) * N;
   size_t count = 0;
   void *dev_addr = NULL;
   void *base_addr = NULL;

   cnrtMalloc(&dev_addr, size);
   cnrtGetMemorySize(&base_addr, dev_addr, &count);

   if (count != size && dev_addr != base_addr)
     printf("cnrtGetMemorySize failed!\n");

   // Free device memory
   cnrtFree(dev_addr);

   cnrtDestroy();
   return 0;
 }
 @endverbatim
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetMemorySize(void **devBasePtr, void *devPtr, size_t *bytes);

/**
 * @brief Retrieves the device memory information by a specified memory channel.
 *
 * @param free[out] The total free memory size (MegaByte) of a specified memory channel
 *                  or all memory channels.
 * @param total[out] The total memory size (MegaByte) of a specified memory channel
  *                  or all memory channels.
 * @param channel[in] The memory channel ID. If you set the value of this
 *                    parameter to **-1** or **-2**, this function returns
 *                    the free memory size and total memory size of all the
 *                    memory channels.
 *                    see ::cnrtChannelType_t for details.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because ``free``
 *                              or ``total`` is NULL, or the value of channel ID
 *                              is invalid.
 * @retval CNRT_RET_ERR_NODEV This function call failed because no
 *                            MLU devices were detected.
 * @retval CNRT_RET_ERR_FUNC_CALL This function call failed because errors
 *         from driver functions are returned via CNRT. See the detailed error
 *         messages on the screen.
 *
 * @details Returns the total free memory size ``free`` and total memory
 * size ``total`` with the given channel ID ``channel`` of the current device.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 @verbatim
 int main () {
   // Initialize device
   cnrtInit(0);

   // Get device handle
   cnrtDev_t dev;
   cnrtGetDeviceHandle(&dev, 0);

   // Set the device handle for current thread execution context
   cnrtSetCurrentDevice(dev);

   size_t total = 0, free = 0;
   cnrtGetMemInfo(&free, &total, CNRT_CHANNEL_TYPE_NONE);
   cnrtGetMemInfo(&free, &total, CNRT_CHANNEL_TYPE_0);
   cnrtGetMemInfo(&free, &total, CNRT_CHANNEL_TYPE_1);
   cnrtGetMemInfo(&free, &total, CNRT_CHANNEL_TYPE_2);
   cnrtGetMemInfo(&free, &total, CNRT_CHANNEL_TYPE_3);

   cnrtDestroy();
   return 0;
 }
 @endverbatim
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetMemInfo(size_t *free,
                                             size_t *total,
                                             cnrtChannelType_t channel);

/**
 * @brief Copies memory from source address to destination address.
 *
 * @param dest[in] Pointer to the destination memory address.
 * @param src[in] Pointer to the source memory address.
 * @param bytes[in] The memory size in bytes to be copied.
 * @param dir[in] Copy direction.
 *                Supported values are as follows. See ::cnrtMemTransDir_t
 *                for details.
 *                - CNRT_MEM_TRANS_DIR_HOST2DEV
 *                - CNRT_MEM_TRANS_DIR_DEV2HOST
 *                - CNRT_MEM_TRANS_DIR_DEV2DEV
 *                - CNRT_MEM_TRANS_DIR_NODIR
 *                - CNRT_MEM_TRANS_DIR_HOST2HOST
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because ``dst``
 *         or ``src`` is NULL or ``dir`` is not a valid enum value in cnrtMemTransDir_t.
 * @retval CNRT_RET_ERR_NODEV This function call failed because no
 *         MLU devices were detected.
 * @retval CNRT_RET_ERR_FUNC_CALL This function call failed because errors
 *         from driver functions are returned via CNRT. See the detailed error
 *         messages on the screen.
 * @retval CNRT_RET_ERR_MEMCPY This function call failed because
 *         an error occurred during the data copy.
 *
 * @details Synchronously copies the size of ``bytes`` bytes of data from the location
 * pointed by the source address ``src`` to the memory address pointed by
 * ``dest`` with the copy direction specified in the ::cnrtMemTransDir_t enum
 * ``dir``. The supported direction of the data copy includes host
 * to device, device to host, host to host, from device to device,
 * and data copy without specifying the directions.
 *
 *
 * @note
 * - This function is used to copy data synchronously. To copy data asynchronously,
 *   call the ::cnrtMemcpyAsync function.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtMalloc function.
 *
 */
extern CNRT_DLL_API cnrtRet_t cnrtMemcpy(void *dest,
                                         void *src,
                                         size_t bytes,
                                         cnrtMemTransDir_t dir);

/**
 *
 * @brief Copies memory data from source address to destination address asynchronously.
 *
 * @param dest[in] Pointer to the destination memory address.
 * @param src[in] Pointer to the source memory address.
 * @param bytes[in] The memory size in bytes to be copied.
 * @param queue[in] The queue handle created by calling the ::cnrtCreateQueue function.
 * @param dir[in] The memory copy direction.
 *                Supported values are as follows. See ::cnrtMemTransDir_t for details.
 *                - CNRT_MEM_TRANS_DIR_HOST2DEV
 *                - CNRT_MEM_TRANS_DIR_DEV2HOST
 *                - CNRT_MEM_TRANS_DIR_DEV2DEV
 *                - CNRT_MEM_TRANS_DIR_NODIR
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because ``dest`` or
 *         ``src`` is NULL or ``dir`` is not a valid enum value in cnrtMemTransDir_t.
 * @retval CNRT_RET_ERR_NODEV This function call failed because no
 *         MLU devices were detected.
 * @retval CNRT_RET_ERR_FUNC_CALL This function call failed because errors
 *         from driver functions are returned via CNRT. See the detailed error
 *         messages on the screen.

 * @retval CNRT_RET_ERR_MEMCPY This function call failed because
 *         an error occurred during the data copy.
 *
 * @details Asynchronously copies the size of bytes ``bytes`` of data from the
 * location pointed by the source address ``src`` to the memory address pointed
 * by ``dest`` with the copy direction specified in the ::cnrtMemTransDir_t enum
 * ``dir``. The supported direction includes host to device, device to host,
 * from device to device, and data copy without specifying the directions.  \n
 * When copying large amount of data, it is recommended to use this function to copy
 * data asynchronously. For more information, see "CNRT User Guide".
 *
 *
 * @note
 * - This function is always performed by the device associated with a queue.
 * - This function is used to copy data asynchronously. To copy data synchronously,
 *   call the ::cnrtMemcpy function.
 * - Because the memory data is copied asynchronously, so the function call may
 *   return before the copy is completed.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 @verbatim
 int main () {
   // Initialize device
   cnrtInit(0);
   cnrtDev_t dev;

   // Get device handle
   cnrtGetDeviceHandle(&dev, 0);

   // Set the device handle for current thread execution context
   cnrtSetCurrentDevice(dev);

   // Prepare input and output
   size_t size = sizeof(size_t) * N;
   char *cpu_input = NULL;
   char *cpu_output = NULL;
   cnrtMallocHost((void **)&cpu_input, size, CNRT_MEMTYPE_DEFAULT);
   cnrtMallocHost((void **)&cpu_output, size, CNRT_MEMTYPE_DEFAULT);
   memset(cpu_input, 'a', size);

   void *dev_addr = NULL;
   cnrtMalloc(&dev_addr, size);

   // Create queue
   cnrtQueue_t queue = NULL;
   cnrtCreateQueue(&queue);

   // Memcpy Async
   cnrtMemcpyAsync(dev_addr, cpu_input, size, queue, CNRT_MEM_TRANS_DIR_HOST2DEV);
   cnrtMemcpyAsync(cpu_output, dev_addr, size, queue, CNRT_MEM_TRANS_DIR_DEV2HOST);
   cnrtSyncQueue(queue);

   if (memcmp(cpu_input, cpu_output, size))
     printf("memcmp async failed\n");

   // Destroy queue
   cnrtDestroyQueue(queue);

   // Free device memory
   cnrtFree(dev_addr);

   // Free host memory
   cnrtFreeHost((void *)cpu_input);
   cnrtFreeHost((void *)cpu_output);

   cnrtDestroy();
   return 0;
 }
 @endverbatim
 *
 */
extern CNRT_DLL_API cnrtRet_t
cnrtMemcpyAsync(void *dest, void *src, size_t bytes, cnrtQueue_t queue, cnrtMemTransDir_t dir);

/**
 * @brief Copies memory data from devices asynchronously.
 *
 * @param dst[in] Pointer to the destination memory address.
 * @param dstDevOrdinal[in] The destination device. You can call the cnrtGetDeviceCount function
 *                          to get the total number of devices in your system. The device ID
 *                          is in the range [0, cnrtGetDeviceCount() ‒ 1].
 * @param src[in] Pointer to the source memory address.
 * @param srcDevOrdinal[in] The source device. You can call the cnrtGetDeviceCount function
 *                          to get the total number of devices in your system. The device ID
 *                          is in the range [0, cnrtGetDeviceCount() ‒ 1].
 * @param bytes[in] The memory size in bytes to be copied.
 * @param queue[in] The queue handle created by calling the ::cnrtCreateQueue function.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because ``dst`` or ``src``
 *                              is NULL.
 * @retval CNRT_RET_ERR_FUNC_CALL This function call failed because errors
 *         from driver functions are returned via CNRT. See the detailed error
 *         messages on the screen.
 *
 * @details Asynchronously copies the size of bytes ``bytes`` of data from the location pointed
 * by the source address ``src`` of the ``srcDevOrdinal`` device to the memory block pointed by
 * ``dst`` of the dstDevOrdinal device. The two devices should be peerable. You can call the
 * ::cnrtGetPeerAccessibility function to check if the two devices are peerable.  \n
 * When copying large amount of data, it is recommended to use this function to copy data
 * asynchronously.
 *
 * @note
 * - This function is used to copy data asynchronously. To copy data synchronously,
 *   call the ::cnrtMemcpyPeer function.
 * - Cambricon MLU200 series supports peer-to-peer communications. If the two MLU devices
 *   are not peerable, it may be hardware problems.
 * - This function is always performed by the device associated with a queue.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 @verbatim
 int main() {
   // Initialize device
   cnrtInit(0);

   int dev_num = 0;
   // Get device count
   cnrtGetDeviceCount(&dev_num);

   // For p2p, device count must >= 2
   if (dev_num <= 1) {
     cnrtDestroy();
     exit(-1);
   }

   cnrtDev_t dev, peerdev;
   // To simplify sample code, suppose sdev is 0, ddev is 1
   int sdev = 0;
   int ddev = 1;

   int peer = 0;
   // Before do p2p, judge if the two devices can do p2p
   cnrtGetPeerAccessibility(&peer, sdev, ddev);
   if (peer != 1) {
     printf("the two devices can't do p2p\n");
     cnrtDestroy();
     exit(-1);
   }

   if (cnrtGetDeviceHandle(&dev, sdev) || cnrtGetDeviceHandle(&peerdev, ddev)) {
     printf("get sdev or ddev device handle failed!\n");
     cnrtDestroy();
     exit(-1);
   }

   size_t size = sizeof(size_t) * N;
   char *cpu_in = (char *)malloc(size);
   char *cpu_out = (char *)malloc(size);
   memset(cpu_in, 'a', size);

   // Before cnrtMalloc, must call cnrtSetCurrentDevice.
   cnrtSetCurrentDevice(peerdev);
   cnrtMalloc(&dinput, size);
   cnrtSetCurrentDevice(dev);
   cnrtMalloc(&sinput, size);

   cnrtMemset(sinput, 'a', size);

   cnrtMemcpyPeerAsync(dinput, ddev, sinput, sdev, size, queue);
   cnrtSyncQueue(queue);
   cnrtMemcpy(cpu_out, dinput, size, CNRT_MEM_TRANS_DIR_DEV2HOST)；

   if (memcmp(cpu_out, cpu_in, size))
     printf("p2p failed!\n");

   // Destroy queue
   cnrtDestroyQueue(queue);

   // Free device memory
   cnrtFree(dinput);
   cnrtFree(sinput);

   // Free host memory
   free(cpu_in);
   free(cpu_out);

   cnrtDestroy();
   return 0;
 }
 @endverbatim
 */
extern CNRT_DLL_API cnrtRet_t cnrtMemcpyPeerAsync(void *dst,
                                                  int dstDevOrdinal,
                                                  void *src,
                                                  int srcDevOrdinal,
                                                  size_t bytes,
                                                  cnrtQueue_t queue);

/**
 * @brief Fills device memory with a specified value.
 *
 * @param ptr[in] Pointer to the device memory address.
 * @param c[in] The value to be set in each byte. The value must
 *              be unsigned char. The value should
 *              be equal to or less than 255.
 * @param bytes[in] The number of bytes to be set.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because ``ptr``
 *                              is NULL.
 * @retval CNRT_RET_ERR_NODEV This function call failed because no
 *         MLU devices were detected.
 * @retval CNRT_RET_ERR_OUT_RANGE This function call failed because the value to
 *         be set is out of range.
 * @retval CNRT_RET_ERR_FUNC_CALL This function call failed because errors
 *         from driver functions are returned via CNRT. See the detailed error
 *         messages on the screen.
 *
 * @details Synchronously sets bytes "bytes" of the memory pointed
 * by "devPtr" with the constant value c. \n
 * This function is deprecated. It is recommended that you use the
 * ::cnrtMemsetD8 function instead.
 *
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
extern CNRT_DLL_API cnrtRet_t cnrtMemset(void *ptr, int c, size_t bytes);

/**
 * @brief Fills n bytes device memory with a specified value.
 *
 * @param ptr[in] Pointer to the device memory address.
 * @param uc[in] The value to be set in each byte. The value must
 *               be unsigned char. The value should
 *               be equal to or less than 255.
 * @param n[in] The number of bytes to be set.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because ``ptr`` is NULL.
 * @retval CNRT_RET_ERR_FIND_DEV_ADDR This function call failed because
 *         can't find dev address.
 * @retval CNRT_RET_ERR_FUNC_CALL This function call failed because errors
 *         from driver functions are returned via CNRT. See the detailed error
 *         messages on the screen.
 *
 * @details Synchronously sets ``n`` bytes of memory pointed by the memory
 * address ``ptr`` to the specified value ``uc``.
 *
 * @note
 * - This function is used to set memory synchronously. To set memory
 *   asynchronously, call the ::cnrtMemsetD8Async function.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 @verbatim
 int main () {
   size_t size = sizeof(uint8_t) * N;
   uint8_t *dev_addr = NULL;
   uint32_t seed = (unsigned)time(NULL);
   uint8_t value = rand_r(&seed) % 1024;
   uint8_t *host_addr = NULL;
   cnrtMallocHost((void **)&host_addr, size, CNRT_MEMTYPE_DEFAULT);

   // Initialize device
   cnrtInit(0);
   cnrtDev_t dev;

   // Get device handle
   cnrtGetDeviceHandle(&dev, 0);

   // Set the device handle for current thread execution context
   cnrtSetCurrentDevice(dev);

   cnrtMalloc((void **)&dev_addr, size);
   cnrtMemsetD8((void *)dev_addr, value, size);
   cnrtMemcpy((void *)host_addr, (void *)dev_addr, size, CNRT_MEM_TRANS_DIR_DEV2HOST);

   for (int i = 0; i < size; i++) {
     if (host_addr[i] != value) {
       printf("cnrtMemsetD8 failed!\n");
       break;
    }
   }

   // Free device memory
   cnrtFree(dev_addr);

   // Free host memory
   cnrtFreeHost(host_addr);

   cnrtDestroy();
   return 0;
 }
 @endverbatim
 */
extern CNRT_DLL_API cnrtRet_t cnrtMemsetD8(void *ptr, uint8_t uc, size_t n);

/**
 * @brief Fills n 32-bit bytes device memory with the specified value.
 *
 * @param ptr[in] Pointer to the device memory address.
 * @param ui[in] The value to be set in each byte.
 * @param n[in] The number of 32-bit to be set.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because ``ptr``
 *                              is NULL.
 * @retval CNRT_RET_ERR_FUNC_CALL This function call failed because errors
 *         from driver functions are returned via CNRT. See the detailed error
 *         messages on the screen.
 *
 * @details Synchronously sets ``n`` 32-bit bytes of memory pointed by the memory
 * address ``ptr`` to the specified value ``ui``.
 *
 * @note
 * - This function is used to set memory synchronously. To set memory
 *   asynchronously, call the ::cnrtMemsetD32Async function.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 @verbatim
 int main () {
   size_t size = sizeof(uint32_t) * N;
   void *dev_addr = NULL;
   uint32_t seed = (unsigned)time(NULL);
   uint32_t value = rand_r(&seed) % 1024;
   uint32_t *host_addr = NULL;
   cnrtMallocHost((void **)&host_addr, size, CNRT_MEMTYPE_DEFAULT);

   // Initialize device
   cnrtInit(0);
   cnrtDev_t dev;

   // Get device handle
   cnrtGetDeviceHandle(&dev, 0);

   // Set the device handle for current thread execution context
   cnrtSetCurrentDevice(dev);

   cnrtMalloc(&dev_addr, size);
   cnrtMemsetD32(dev_addr, value, size);
   cnrtMemcpy((void *)host_addr, dev_addr, size, CNRT_MEM_TRANS_DIR_DEV2HOST);

   for (int i = 0; i < size; i++) {
     if (host_addr[i] != value) {
       printf("cnrtMemsetD32 failed!\n");
       break;
     }
   }

   // Free device memory
   cnrtFree(dev_addr);

   // Free host memory
   cnrtFreeHost(host_addr);

   cnrtDestroy();
   return 0;
 }
 @endverbatim
 */
extern CNRT_DLL_API cnrtRet_t cnrtMemsetD32(void *ptr, uint32_t ui, size_t n);

/**
 * @brief Fills n bytes device memory with a specified value asynchronously.
 *
 * @param ptr[in] Pointer to the device memory address.
 * @param uc[in] The value to be set in each byte. The value must be unsigned char.
 * @param n[in] The number of bytes to be set.
 * @param queue[in] The queue handle created by calling the ::cnrtCreateQueue function.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because ``ptr`` is NULL.
 * @retval CNRT_RET_ERR_FAULT This function call failed because errors
 *         from driver functions are returned via CNRT. See the detailed error
 *         messages on the screen.
 *
 * @details Asynchronously sets ``n`` bytes of memory pointed by the memory
 * address ``ptr`` to the specified value ``uc``.
 *
 *
 * @note
 * - This function is performed by the device associated with a queue.
 * - This function is used to set memory asynchronously. To set memory
 *   synchronously, call the ::cnrtMemsetD8 function.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 @verbatim
 int main () {
   size_t size = sizeof(uint8_t) * N;
   uint32_t seed = (unsigned)time(NULL);
   uint8_t value = rand_r(&seed) % 1024;
   uint8_t *host_addr = NULL;
   cnrtMallocHost((void **)&host_addr, size, CNRT_MEMTYPE_DEFAULT);

   // Initialize device
   cnrtInit(0);
   cnrtDev_t dev;

   // Get device handle
   cnrtGetDeviceHandle(&dev, 0);

   // Set the device handle for current thread execution context
   cnrtSetCurrentDevice(dev);

   // Create queue
   cnrtQueue_t queue = NULL;
   cnrtCreateQueue(&queue);

   void *dev_addr = NULL;
   cnrtMalloc(&dev_addr, size);
   cnrtMemsetD8Async(dev_addr, value, size, queue);
   cnrtSyncQueue(queue);
   cnrtMemcpy((void *)host_addr, dev_addr, size, CNRT_MEM_TRANS_DIR_DEV2HOST);

   for (int i = 0; i < size; i++) {
     if (host_addr[i] != value) {
       printf("cnrtMemsetD8Async failed!\n");
       break;
     }
   }

   // Destroy queue
   cnrtDestroyQueue(queue);

   // Free device memory
   cnrtFree(dev_addr);

   // Free host memory
   cnrtFreeHost(host_addr);

   cnrtDestroy();
   return 0;
 }
 @endverbatim
 */
extern CNRT_DLL_API cnrtRet_t cnrtMemsetD8Async(void *ptr, uint8_t uc, size_t n, cnrtQueue_t queue);

/**
 * @brief Fills n 32-bit bytes device memory with the specified value asynchronously.
 *
 * @param ptr[in] Pointer to the device memory address.
 * @param ui[in] The value to be set in each byte.
 * @param n[in] The number of 32-bit to be set.
 * @param queue[in] The queue handle created by calling the ::cnrtCreateQueue function.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because ``ptr``
 *                              is NULL.
 * @retval CNRT_RET_ERR_FAULT This function call failed because the memory
 *         address is invalid.
 *
 * @details Asynchronously sets ``n`` 32-bit bytes of memory pointed by the memory
 * address ``ptr`` to the specified value ``ui``.
 *
 * @note
 * - This function is always performed by the device associated with a queue.
 * - This function is used to set memory asynchronously. To set memory
 *   synchronously, call the ::cnrtMemsetD32 function.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 @verbatim
 int main () {
   size_t size = sizeof(uint32_t) * N;
   void *dev_addr = NULL;
   uint32_t seed = (unsigned)time(NULL);
   uint32_t value = rand_r(&seed) % 1024;
   uint32_t *host_addr = NULL;
   cnrtMallocHost((void **)&host_addr, size, CNRT_MEMTYPE_DEFAULT);

   // Initialize device
   cnrtInit(0);
   cnrtDev_t dev;

   // Get device handle
   cnrtGetDeviceHandle(&dev, 0);

   // Set the device handle for current thread execution context
   cnrtSetCurrentDevice(dev);

   cnrtQueue_t queue = NULL;
   cnrtCreateQueue(&queue);

   cnrtMalloc(&dev_addr, size);
   cnrtMemsetD32Async(dev_addr, value, size, queue);
   cnrtSyncQueue(queue);
   cnrtMemcpy((void *)host_addr, dev_addr, size, CNRT_MEM_TRANS_DIR_DEV2HOST);

   for (int i = 0; i < size; i++) {
     if (host_addr[i] != value) {
       printf("cnrtMemsetD32Async failed!\n");
       break;
     }
   }

   // Destroy queue
   cnrtDestroyQueue(queue);

   // Free device memory
   cnrtFree(dev_addr);

   // Free host memory
   cnrtFreeHost(host_addr);

   cnrtDestroy();
   return 0;
 }
 @endverbatim
 */
extern CNRT_DLL_API cnrtRet_t cnrtMemsetD32Async(void *ptr,
                                                 uint32_t ui,
                                                 size_t n,
                                                 cnrtQueue_t queue);

/**
 * @brief Sets the maximum size of stack memory can be used for each IPU core on
 *        a MLU device.
 *
 * @param stacksize[in] The maximum size (MB) of stack memory can be used for
 *                      each IPU core. The value is in the range from 4 MB to 511 MB.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_NODEV This function call failed because no
 *         MLU devices were detected.
 * @retval CNRT_RET_ERR_FUNC_CALL This function call failed because errors
 *         from driver functions are returned via CNRT. See the detailed error
 *         messages on the screen.
 *
 * @details Sets the maximum size of the stack memory can be used to
 * "stacksize" in MB for each IPU core on a MLU device. \n
 * This function is deprecated. It is recommended to use the ::cnrtSetLocalMem
 * function instead.
 *
 * @note
 * - The maximum total size of the stack memory can be used on a MLU device
 *   is the total number of IPU cores of the MLU device multiplies ``stacksize``.
 *   Take MLU270 for example. MLU270 has 16 IPU cores in total. If you set the
 *   value of ``stacksize`` to 5 MB, the maximum total memory size of this
 *   MLU device is 80 MB.
 * - All the IPU cores in the MLU device are set to the ``stacksize`` MB size
 *   after this function been called. Setting the maximum size of a certain IPU
 *   core to a different value from other IPU cores' is not supported.
 *
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtSetLocalMem function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtSetStackMem(unsigned int stacksize);

/**
 * @brief Retrieves the maximum size of stack memory set for each
 *        IPU core on a MLU device.
 *
 * @param pStacksize[out] The maximum size (MB) of stack memory can be used
 *                        for each IPU core.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_NODEV This function call failed because no
 *         MLU devices were detected.
 * @retval CNRT_RET_ERR_FUNC_CALL This function call failed because errors
 *         from driver functions are returned via CNRT. See the detailed error
 *         messages on the screen.
 *
 * @details Returns the maximum size (MB) of the stack memory space "pStacksize"
 * that can be used for each IPU core on a MLU device.  \n
 * This function is deprecated. It is recommended to use the
 * ::cnrtGetLocalMem function instead.
 *
 * @note
 * - The maximum total size of the stack memory you can use on a MLU device is the total
 *   number of IPU cores of the MLU device multiplies ``stacksize``. Take MLU270
 *   for example. MLU270 has 16 IPU cores in total. If you set the value of
 *   ``stacksize`` to 5 MB, the maximum total memory size of this MLU device
 *   is 80 MB.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtSetLocalMem function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetStackMem(unsigned int *pStacksize);

/**
 * @brief Sets the maximum size of local memory can be used for each IPU core
 *        on a MLU device.
 *
 * @param local_mem_size[in] The maximum size (MB) of local memory can be used for
 *                           each IPU core. The value is in range from 4 MB to 511 MB.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_NODEV This function call failed because no
 *         MLU devices were detected.
 * @retval CNRT_RET_ERR_FUNC_CALL This function call failed because errors
 *         from driver functions are returned via CNRT. See the detailed error
 *         messages on the screen.
 *
 * @details Sets the maximum size of the local memory space can be used
 * for each IPU core on a MLU device. See "CNRT User Guide" for detailed information.
 *
 * @note
 * - Each IPU core has an exclusive local memory space which will be used to save
 *   information of each IPU core. The information can be instructions or temporary
 *   data for computing on each IPU core.
 * - The maximum total size of the local memory can be used on a MLU device is the
 *   number of IPU cores of the MLU device multiplies ``local_mem_size``. Take MLU270
 *   for example. MLU270 has 16 IPU cores. If you set the value of ``local_mem_size`` to 5 MB,
 *   the maximum total memory size of this MLU device is 80 MB.
 * - All the IPU cores in the MLU device are set to the ``local_mem_size`` MB
 *   size after this function been called. Setting the maximum size of an IPU core
 *   to a different value from other IPU cores' is not supported.
 * - If the actual usage exceeds the maximum size, the global memory is taken
 *   up for storage. If the actual usage is less than the maximum size, the
 *   global memory will not be taken up.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 @verbatim
 int main () {
   // Initialize device
   cnrtInit(0);

   // Get device handle
   cnrtDev_t dev;
   cnrtGetDeviceHandle(&dev, 0);

   // Set the device handle for current thread execution context
   cnrtSetCurrentDevice(dev);

   int expected_local_size = N; (4 <= N <= 511)
   unsigned int returned_local_size = 0;

   cnrtSetLocalMem(expected_local_size);
   cnrtGetLocalMem(&returned_local_size);

   if (expected_local_size != returned_local_size)
     printf("cnrtSetLocalMem failed!\n");

   cnrtDestroy();
   return 0;
 }
 @endverbatim
 */
extern CNRT_DLL_API cnrtRet_t cnrtSetLocalMem(unsigned int local_mem_size);

/**
 * @brief Retrieves the maximum size of local memory set for each IPU core on
 *        a MLU device.
 *
 * @param pLocalsize[out] The maximum size (MB) of local memory can be used for
 *                        each IPU core.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_NODEV This function call failed because no
 *         MLU devices were detected.
 * @retval CNRT_RET_ERR_FUNC_CALL This function call failed because errors
 *         from driver functions are returned via CNRT. See the detailed error
 *         messages on the screen.
 *
 * @details Returns the maximum size (MB) of the local memory space you set for
 * each IPU core on a MLU device.
 *
 * @note
 * - The maximum total size of the local memory you set on a MLU device is the total
 *   number of IPU cores of the MLU device multiplies ``local_mem_size``. Take MLU270
 *   for example. MLU270 has 16 IPU cores in total. If you set the value of
 *   ``local_mem_size`` to 5 MB, the maximum total memory size of this MLU device
 *   is 80 MB.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtSetLocalMem function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetLocalMem(unsigned int *pLocalsize);

/**
 * @brief Retrieves the memory size that is used for a CNRT Function.
 *
 * @param function[in] Pointer to a CNRT function extracted from an offline model file. This CNRT
 *        Function is defined in ::cnrtFunction_t.
 * @param pMemused[out] The memory size in bytes that the CNRT Function is required.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_FUNC_CALL This function call failed because the ``function`` is NULL.
 *
 * @details Returns in ``pMemused`` the size pf memory that is used to store the CNRT Function
 *          ``function``. The returned memory size includes all the device memory usage in
 *          executing the CNRT Function except the inputs and outputs of the CNRT Function.
 *          So you need a larger memory size for CNRT Function than ``pMemused``.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtGetModelSize function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetFunctionMemUsed(cnrtFunction_t function, int64_t *pMemused);

/**
 * @brief Retrieves the size of the device memory that is used in all CNRT Functions in an offline
 *        model file.
 *
 * @param model[in] Pointer to an offline model file defined in ::cnrtModel_t.
 * @param pMemused[out] The memory size in bytes that all CNRT Functions in an offline model
 *                      file is required.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because the ``model`` is NULL.
 *
 * @details Returns in `` pMemused `` the size of memory that is used to store all the CNRT
 *          Functions in an offline model file `` model `` are required. The returned memory
 *          size includes all the usages in
 *          executing all CNRT Functions except the inputs and outputs of the CNRT Functions.
 *          So you need a larger memory size for all the CNRT Functions than ``pMemused``.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtGetModelSize function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetModelMemUsed(cnrtModel_t model, int64_t *pMemused);

/*********************************************************************************
 * Channel control, only MLU270 support.
 *********************************************************************************/

/**
 * @brief Sets the memory channel to be used for operations on the current MLU device.
 *
 * @param cnrtChannelType_t[in] The ID of the memory channel to be used on the current
 *                   device. The value should be in the range [0, CNRT_CHANNEL_TYPE_3].
 *                   See ::cnrtChannelType_t for details.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because the value of the
 *                              ``channel`` paramenter is out of range.
 *
 * @details Sets the memory channel specified in the ::cnrtChannelType_t enum ``channel``
 *          to be used for memory and computation operations on the current MLU device.
 *          Once the memory channel is set, all memory allocation operations, such as
 *          ::cnrtMalloc and all function invokations are performed on this channel.
 *
 * @note
 * - You need to call the ::cnrtSetCurrentDevice function to set the current device
 *   before setting the memory channel.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtGetMemInfo function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtSetCurrentChannel(cnrtChannelType_t channel);

/**
 * @brief Retrieves the current memory channel used on the current CPU thread.
 *
 *
 * @param pChannel[out] Pointer to the ID of the memory channel used on the
 *                      current CPU thread.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because ``pChannel``
 *                              is NULL.
 *
 * @details Returns in ``pChannel`` the current memory channel used on the current
 *          CPU thread.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtGetMemInfo function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetCurrentChannel(cnrtChannelType_t *pChannel);

/*********************************************************************************
 * Parameter descriptor related API
 *********************************************************************************/

/**
 * @brief Creates a parameter descriptor for an input or output tensor.
 *
 * @param param_desc[in] Pointer to a parameter descriptor defined in ::cnrtParamDesc_t. The
 *        parameter descriptor describes the input or output information including shape, name,
 *        data type, and dimension order.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID_POINTER This function call failed because ``param_desc`` pointer is
 *         NULL or invalid.
 * @retval CNRT_RET_ERR_NOMEM This function call failed because not enough memory can be allocated.
 *
 * @details Allocates memory for a parameter descriptor "param_desc" that holds information about
 * an input or output tensor, and initializes the parameter descriptor including shape,
 * name, data type, and dimension order of the tensor. \n
 * The default values of the attributes in parameter descriptor are as follows: \n
 * - The number of dimension of the input or output: 0.
 * - The name of the input or output: NULL.
 * - The data type of the input or output: CNRT_FLOAT16.
 * - The dimension order of the input or output: CNRT_NHWC.
 *
 * @note
 * - This function is used to create parameter descriptors used in the
 *   ::cnrtInvokeRuntimeContext_V2 function.
 * - You need to destroy the parameter descriptor and release the memory resources with the
 *   ::cnrtDestroyParamDesc function. Otherwise, the memory leaks may be occurred.
 * - To create the parameter descriptor for a group of inputs or outputs, call the
 *   ::cnrtCreateParamDescArray function.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtGetInputAndOutputNamesFromFunction function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtCreateParamDesc(cnrtParamDesc_t *param_desc);

/**
 * @brief Destroys a parameter descriptor of an input or output tensor.
 *
 * @param param_desc[in] Pointer to a parameter descriptor defined in ::cnrtParamDesc_t. The
 *        parameter descriptor describes the input or output information including shape, name,
 *        data type, and dimension order.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 *
 * @details Destroys the parameter descriptor ``param_desc`` of an input or output tensor created
 * with the
 * ::cnrtCreateParamDesc function and releases the related memory resources.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtGetInputAndOutputNamesFromFunction function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtDestroyParamDesc(cnrtParamDesc_t param_desc);

/**
 * @brief Creates an array of parameter descriptor for an input or output tensor.
 *
 * @param param_descs[out] Pointer to an array that stores the parameter descriptors defined
 *        in ::cnrtParamDescArray_t. The parameter descriptor describes the input or output
 *        information including shape, name, data type, and dimension order.
 * @param param_num[in] The number of parameter descriptors in parameter descriptor array.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because ``param_num`` is 0 or a negative
 *         value.
 * @retval CNRT_RET_ERR_INVALID_POINTER This function call failed because ``param_descs`` is NULL or
 *         invalid.
 * @retval CNRT_RET_ERR_NOMEM This function call failed because not enough memory can be
 *         allocated.
 *
 * @details Allocates the memory for an array that stores a group of parameter descriptors for
 * input or output tensors, and initializes each parameter descriptor including shape, name,
 * data type, and dimension order. \n
 * The default values of the attributes in each parameter descriptor are as follows.
 * You can set the value of the attributes in each parameter descriptor with the
 * ::cnrtCreateParamDesc function. \n
 * - The number of dimension of input or output: 0.
 * - The name of input or output : NULL.
 * - The data type of input or output : CNRT_FLOAT16.
 * - The dimension order of input or output : CNRT_NHWC.
 *
 *
 * @note
 * - This function is used to create parameter descriptor array used in the
 *   ::cnrtInvokeRuntimeContext_V2 function.
 * - You need to call function ::cnrtDestroyParamDescArray to destroy parameter descriptor
 *   array. Otherwise the memory leaks may be occurred.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
extern CNRT_DLL_API cnrtRet_t cnrtCreateParamDescArray(cnrtParamDescArray_t *param_descs,
                                                       int param_num);

/**
 * @brief Destroys an array of parameter descriptor of an input or output tensor.
 *
 * @param param_descs[out] Pointer to an array that stores the parameter descriptors defined
 *        in ::cnrtParamDescArray_t. The parameter descriptor describes the input or output
 *        information including shape, name, data type, and dimension order.
 * @param param_num[in] The number of parameter descriptors in parameter descriptor array.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because the input parameter
 *         ``param_num`` is 0 or a negative value.
 *
 * @details Frees memory that stores an array of parameter descriptors ``param_descs`` created
 *          with the ::cnrtCreateParamDescArray function and releases the related memory resources.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
extern CNRT_DLL_API cnrtRet_t cnrtDestroyParamDescArray(cnrtParamDescArray_t param_descs,
                                                        int param_num);

/**
 * @brief Sets the shape of an input or output tensor into a tensor parameter descriptor.
 *
 * @param param_desc[in] Pointer to the parameter descriptor of the input or output tensor
 *        defined in ::cnrtParamDesc_t.
 * @param dim_num[in] The number of dimensions in the input or output tensor to be set.
 *        The value is 0 by default after creating the tensor
 *        parameter descriptor with the ::cnrtCreateParamDesc function.
 * @param dims[in] The number of elements for each dimension in the input or output tensor to be
 * set.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because the input parameter
 *         ``dim_num`` is 0 or a negative value.
 * @retval CNRT_RET_ERR_INVALID_POINTER This function call failed because ``param_desc`` or ``dims``
 * is NULL or invalid.
 *
 * @details Sets the shape information of an input or output tensor, including the number of
 * dimensions
 *          ``dim_num`` and the number of elements for each dimension ``dim_num`` into the parameter
 *          descriptor ``param_desc`` of the tensor.
 *
 * @note
 * - Before calling this function, you need to create a parameter descriptor for the input or
 *   output tensor with the ::cnrtCreateParamDesc function.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
extern CNRT_DLL_API cnrtRet_t cnrtSetShapeToParamDesc(cnrtParamDesc_t param_desc,
                                                      int *dims,
                                                      int dim_num);

/**
 * @brief Sets the name of the input or output tensor into a tensor parameter descriptor.
 *
 * @param param_desc[in] Pointer to the parameter descriptor of the input or output tensor
 *        defined in ::cnrtParamDesc_t.
 * @param name[in] Pointer to the name of the input or output tensor to be set.
 *                 The value is NULL by default after creating the tensor
 *                 parameter descriptor with the ::cnrtCreateParamDesc function.
 * @param name_size[in] The number of bytes of the name string.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because the input parameter
 *         ``name_size`` is 0 or a negative value.
 * @retval CNRT_RET_ERR_INVALID_POINTER This function call failed because ``param_desc`` or ``name``
 *         is NULL or invalid.
 *
 * @details Sets the name ``name`` of an input or output tensor into the parameter descriptor
 * ``param_desc``
 *          of the tensor with the number of bytes of the name string ``name_size``.
 *
 * @note
 * - Before calling this function, you need to create a parameter descriptor for the tensor
 *   with the ::cnrtCreateParamDesc function.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
extern CNRT_DLL_API cnrtRet_t cnrtSetNameToParamDesc(cnrtParamDesc_t param_desc,
                                                     char *name,
                                                     int name_size);

/**
 * @brief Sets the data type of an input or output tensor into a tensor parameter descriptor.
 *
 * @param param_desc[in] Pointer to the parameter descriptor of the input or output tensor
 *        defined in ::cnrtParamDesc_t.
 * @param dtype[in] Pointer to the data type of the input or output tensor to be set.
 *        The data type is defined in ::cnrtDataType_t.
 *        The value is ``CNRT_FLOAT16`` by default after creating the tensor
 *        parameter descriptor with the ::cnrtCreateParamDesc function.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID_POINTER This function call failed because ``param_desc`` or
 * ``dtype``
 *         is NULL or invalid.
 *
 * @details Sets the data type ``dtype`` of an input or output tensor into parameter descriptor
 *          ``param_desc`` of the tensor.
 *
 * @note
 * - Before calling this function, you need to create a parameter descriptor for the tensor to be
 * set
 *   with the ::cnrtCreateParamDesc function.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
extern CNRT_DLL_API cnrtRet_t cnrtSetDataTypeToParamDesc(cnrtParamDesc_t param_desc,
                                                         cnrtDataType_t dtype);

/**
 * @brief Retrieves the total number of elements of an input or output tensor from
 *        a tensor parameter descriptor.
 *
 * @param param_desc[in] Pointer to the parameter descriptor of the input or output tensor
 *        defined in ::cnrtParamDesc_t.
 * @param num[out] Pointer to the returned total number of elements in the input or output tensor.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID_POINTER This function call failed because ``param_desc`` or ``num``
 *         is NULL or invalid.
 * @retval CNRT_RET_ERR_FUNC_CALL This function call failed because the number of elements of one
 *         dimension in the input or output tensor is 0 or a negative value.
 *
 * @details Returns in ``num`` the total number of elements in an input or output tensor from
 *          the parameter descriptor ``param_desc`` of the tensor. The total number of elements is
 * the
 *          product of the number of elements of all dimensions.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetParamElementNum(cnrtParamDesc_t param_desc, size_t *num);

/**
 * @brief Retrieves the memory size that an input or output tensor required from a tensor
 *        parameter descriptor.
 *
 * @param param_desc[in] Pointer to the parameter descriptor of the input or output tensor
 *        defined in ::cnrtParamDesc_t.
 * @param size[out] Pointer to the returned memory size in bytes that the input or output
 *        tensor required.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID_POINTER This function call failed because ``param_desc`` or ``size``
 *         is NULL or invalid.
 * @retval CNRT_RET_ERR_FUNC_CALL This function call failed because the memory size of input or
 *         output from the parameter descriptor is 0 or a negative value.
 *
 * @details Returns in ``size`` the memory size in bytes that the input or output tensor required
 *          from the parameter descriptor ``param_desc`` of the tensor. The returned memory size
 *          is the product of the size of data type and the number of elements of all dimensions.
 *          You can get the memory size to be allocated for the tensor with this function.
 *
 * @note
 * - The value of any dimensions from the tensor parameter descriptor must be greater than
 *   or equal to 1.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetParamDescSize(cnrtParamDesc_t param_desc, int64_t *size);

/**
 * @brief Retrieves the shape of an input or output tensor from a tensor parameter descriptor.
 *
 * @param param_desc[in] Pointer to the parameter descriptor of the input or output tensor
 *        defined in ::cnrtParamDesc_t.
 * @param dim_num[out] The number of dimensions of the input or output tensor.
 * @param dims[out] The number of elements for each dimension of the input or output tensor.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID_POINTER This function call failed because ``param_desc``,
 * ``dim_num``, or ``dims`` is NULL or invalid.
 * @retval CNRT_RET_ERR_INVALID This function call failed because the shape is not set into
 *         the parameter descriptor ``param_desc``.
 *
 * @details Returns the number of dimensions in ``dims`` and the number of elements for each
 * dimension
 *          in ``dim_num`` of an input or output tensor from the parameter descriptor ``param_desc``
 *          of the tensor.
 *
 * @note
 * - You need to call ``free`` function to release memory used for ``dims``.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetShapeFromParamDesc(cnrtParamDesc_t param_desc,
                                                        int **dims,
                                                        int *dim_num);

/**
 * @brief Retrieves the name of an input or output tensor from a tensor parameter descriptor.
 *
 * @param param_desc[in] Pointer to the parameter descriptor of the input or output tensor
 *        defined in ::cnrtParamDesc_t.
 * @param name[out] The name of the input or output tensor to be retrieved.
 * @param name_size[out] The number of bytes of the name string.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID_POINTER This function call failed because ``param_desc``, ``name``
 * or ``name_size`` is NULL or invalid.
 * @retval CNRT_RET_ERR_INVALID This function call failed because the ``name`` is not set in
 * parameter descriptor ``param_desc``.
 *
 * @details Returns the name of the input or output tensor in ``name`` and the number of bytes
 *          of the name string in ``name_size`` from the parameter descriptor ``param_desc``
 *          of the tensor.
 *
 * @note
 * - You need to call ``free`` function to release memory used for ``name``.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetNameFromParamDesc(cnrtParamDesc_t param_desc,
                                                       char **name,
                                                       int *name_size);

/**
 * @brief Retrieves the data type of an input or output tensor from a tensor parameter descriptor.
 *
 * @param param_desc[in] Pointer to the parameter descriptor of the input or output tensor
 *        defined in ::cnrtParamDesc_t.
 * @param dtype[out] Pointer to the data type of the input or output tensor.
 *        The data type is defined in ::cnrtDataType_t.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID_POINTER This function call failed because ``param_desc`` or
 * ``dtype``
 * is NULL or invalid.
 *
 * @details Returns the data type of an input or output tensor in ``dtype`` from the parameter
 * descriptor
 *          ``param_desc`` of the tensor.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetDataTypeFromParamDesc(cnrtParamDesc_t param_desc,
                                                           cnrtDataType_t *dtype);

/**
 * @brief Retrieves dimension order of an input or output tensor from a tensor parameter descriptor.
 *
 * @param param_desc[in] Pointer to the parameter descriptor of the input or output tensor
 *        defined in ::cnrtParamDesc_t.
 * @param dim_order[out] Pointer to the dimension order of the input or output tensor
 *        defined in ::cnrtDimOrder_t.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID_POINTER This function call failed because ``param_desc`` or
 * ``dim_order`` is NULL or invalid.
 *
 * @details Returns in ``dim_order`` the dimension order of an input or output tensor from the
 * parameter
 *          descriptor ``param_desc`` of the tensor.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetDimOrderFromParamDesc(cnrtParamDesc_t param_desc,
                                                           cnrtDimOrder_t *dim_order);

/**
 * @brief Retrieves a tensor parameter descriptor from a parameter descriptor array by tensor name.
 *
 * @param param_descs[in] Pointer to an array that stores the tensor parameter descriptors defined
 * in
 *        ::cnrtParamDescArray_t.
 * @param param_desc[out] Pointer to the tensor parameter descriptor to be retrieved. The parameter
 *        descriptor is defined in ::cnrtParamDesc_t.
 * @param param_num[in] The number of parameter descriptors stored in the array.
 * @param name[in] Pointer to the name of input or output to be retrieved.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID_POINTER This function call failed because ``param_desc``,
 * ``param_descs`` or ``name`` is NULL or invalid.
 * @retval CNRT_RET_ERR_FAULT This function call failed because unable to find the matching
 *         ``name`` in the parameter descriptor array.
 *
 * @details Returns in ``param_desc`` a tensor parameter descriptor from a parameter descriptor
 * array
 *          ``param_descs`` with the given tensor name ``name`` and the number of parameter
 * descriptors
 *          in the array ``param_num``. The parameter descriptor describes the input or output
 * information
 *          including shape, name, data type, and dimension order.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
extern CNRT_DLL_API cnrtRet_t
cnrtGetParamDescFromParamDescArrayByName(cnrtParamDesc_t *param_desc,
                                         cnrtParamDescArray_t param_descs,
                                         int param_num,
                                         const char *name);

/**
 * @brief Retrieves the index of a tensor parameter descriptor in a parameter
 *        descriptor array by tensor name.
 *
 * @param param_descs[in] Pointer to an array that stores the tensor parameter descriptors defined
 * in
 *        ::cnrtParamDescArray_t.
 * @param index[out] The index of which the parameter descriptor is stored in the parameter
 *        descriptor array.
 * @param param_num[in] The number of parameter descriptors stored in the parameter descriptor
 *        array.
 * @param name[in] Pointer to the name of the input or output tensor to be retrieved.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID_POINTER This function call failed because ``param_desc``,
 * ``param_descs`` or ``name`` is NULL or invalid.
 * @retval CNRT_RET_ERR_FAULT This function call failed because unable to find the matching ``name``
 *         in the parameter descriptor array.
 *
 * @details Returns in ``index`` the index of a tensor parameter descriptor in the parameter
 * descriptor
 *          array ``param_descs`` with the given tensor name ``name`` and the number of parameter
 * descriptors
 *          ``param_num`` in the array.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetIndexFromParamDescArrayByName(int *index,
                                                                   cnrtParamDescArray_t param_descs,
                                                                   int param_num,
                                                                   const char *name);

/**
 * @brief Reshapes the input filter data.
 *
 *
 * @param dst[out] The destination address.
 * @param src[in] The source address.
 * @param n[in] The batch size.
 * @param h[in] The height.
 * @param w[in] The width.
 * @param c[in] The channel.
 * @param type[in] The data type of the source and destination data.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because errors occurred
 *                              when checking input parameter values.
 * @retval CNRT_RET_ERR_UNSUPPORTED The input value ``type`` is not supported
 *   at current platform.
 *
 * @details Reshapes the filter data from source memory address pointed
 *          by ``dst`` to destination memory address pointed by ``src`` with the
 *          source data shape src[NHWC] and data type ``type`` that is specified
 *          in ``cnrtDataType_t`` enum.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
extern CNRT_DLL_API cnrtRet_t
cnrtFilterReshape(void *dst, void *src, int n, int h, int w, int c, cnrtDataType_t type);

/**
 * @brief Reshapes the input filter data from NHWC to NCHW.
 *
 * @param dst[out] The destination address.
 * @param src[in] The source address.
 * @param n[in] The batch size.
 * @param h[in] The height.
 * @param w[in] The width.
 * @param c[in] The channel.
 * @param type[in] The data type of the source and destination data.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because errors occurred
 *                              when checking input parameter values.
 * @retval CNRT_RET_ERR_UNSUPPORTED The input value ``type`` is not supported
 *   at current platform.
 *
 * @details Reshapes the filter data from shape NCHW to NHWC. The source data
 *          is saved in the memory pointed by ``dst`` and the reshaped data is saved in the
 *          memory pointed by ``src``.
 *
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
extern CNRT_DLL_API cnrtRet_t
cnrtReshapeNCHWToNHWC(void *dst, void *src, int n, int h, int w, int c, cnrtDataType_t type);

/**
 * @brief Reshapes the input filter data from NHWC to NCHW.
 *
 * @param dst[out] The destination address.
 * @param src[in] The source address.
 * @param n[in] The batch size.
 * @param h[in] The height.
 * @param w[in] The width.
 * @param c[in] The channel.
 * @param type[in] The data type of the source and destination data.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because errors occurred
 *                              when checking input parameter values.
 * @retval CNRT_RET_ERR_UNSUPPORTED The input value ``type`` is not supported
 *   at current platform.
 *
 * @details Reshapes the filter data from shape NHWC to NCHW. The source data
 *          is saved in the memory pointed by ``dst`` and the reshaped data is saved in the
 *          memory pointed by ``src``.
 *
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
extern CNRT_DLL_API cnrtRet_t
cnrtReshapeNHWCToNCHW(void *dst, void *src, int n, int h, int w, int c, cnrtDataType_t type);

/**
 * @brief Retrieves the model version of a Cambricon offline model.
 *
 * @param fname[in] The file name of a Cambricon model to be retrieved. You can specify the absolute
 *        or relative path of the file in this parameter. For example,
 *        ``/home/abc/model/conv.cambricon`` or ``./model/conv.cambricon``.
 * @param model_level[out] The model version of the offline model in offline model file.
 *
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because it is unable to open or read the
 *         offline model file.
 * @retval CNRT_RET_ERR_INVALID_TYPE This function call failed because the model level from offline
 * file in not valid.
 *
 * @details Returns in "model_level" the model version of the given offline model file "fname". \n
 * The offline models with higher versions are not guaranteed to be parsed by the older versions
 * of the CNRT library. You can retrieve the version of you offline model with this function and
 * check the oldest version of the CNRT library that can be used. For detailed information, see
 * "CNRT User Guide".
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetModelLevelFromFile(const char *fname, int *model_level);

/****************************************************************************
 * Generic parameters handling
 ***************************************************************************/

/**
 * @brief Allocates a CNRT parameter context buffer.
 *
 * @param pParam[out] A pointer to the parameter context buffer.
 *
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because
 *                              the input parameter values are invalid.
 * @retval CNRT_RET_ERR_NOMEM This function call failed because errors occurred
 *                              when using the ``malloc`` function.
 *
 * @details Allocates a CNRT parameter context buffer that is used to store parameters
 * used in the ::cnrtMallocBufferEx function. The buffer address is returned
 * in ``pPram``.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 @verbatim
 int main () {
   // Initialize device
   cnrtInit(0);

   // Get device handle
   cnrtDev_t dev;
   cnrtGetDeviceHandle(&dev, 0);

   // Set the device handle for current thread execution context
   cnrtSetCurrentDevice(dev);

   void *param = NULL;
   cnrtAllocParam(&param);

   int type = CNRT_MALLOC_EX_PARALLEL_FRAMEBUFFER;
   int dp = 4;
   int frame_num = 0;
   int frame_size = 1024;

   cnrtAddParam(param, (char *)"type", sizeof(type), &type);
   cnrtAddParam(param, (char *)"data_parallelism", sizeof(dp), &dp);
   cnrtAddParam(param, (char *)"frame_num", sizeof(frame_num), &frame_num);
   cnrtAddParam(param, (char *)"frame_size", sizeof(frame_size), &frame_size);

   int out_param;
   cnrtGetParam(param, (char *)"type", &out_param, sizeof(out_param));

   void *dev_ptr = NULL;
   cnrtMallocBufferEx(&dev_ptr, param);

   if (dev_ptr)
     cnrtFree(dev_ptr);
   if (param)
     cnrtDestroyParam(param);

   cnrtDestroy();
   return 0;
 }
 @endverbatim
 */
extern CNRT_DLL_API cnrtRet_t cnrtAllocParam(void **pParam);

/**
 * @brief Destroys a CNRT parameter context buffer allocated with
 *        the ::cnrtAllocParam function.
 *
 * @param param[in] The parameter context buffer to be destroyed.
 *
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because
 *                              the input parameter values are invalid.
 *
 * @details Destroys a CNRT parameter context buffer pointed by ``param``.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtAllocParam function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtDestroyParam(void *param);

/**
 * @brief Adds a parameter to a parameter context buffer.
 *
 * @param param[in] The parameter to be added.
 * @param name[in] The name of the parameter to be added. The maximum
 * characters of the name length is 64.
 * @param len[in] The size in bytes of the parameter value to be added.
 * The maximum size is 1MB.
 * @param data[in] A pointer to the value of the parameter to be added.
 *
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because
 *                              the input parameter values are invalid.
 * @retval CNRT_RET_ERR_NOMEM This function call failed because errors occurred
 *                              when using the malloc function.
 *
 * @details Adds a parameter including name, size, and value
 * of the parameter into the parameter context buffer pointed by ``param``.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtAllocParam function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtAddParam(void *param, char *name, int len, void *data);

/**
 * @brief Retrieves a parameter from a parameter context buffer.
 *
 * @param param[in] The parameter context buffer to be retrieved from.
 * @param name[in] The name of the parameter to be retrieved.
 * @param out[out] Pointer to the value of the parameter to be retrieved.
 * @param outlen[in] The size in bytes of the parameter value.
 *
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_MEMCPY This function call failed because the size of the
 *         parameter value is larger than ``outlen``.
 * @retval CNRT_RET_ERR_NO_EXIST This function call failed because ``name`` parameter
 *         is not found in the parameter context buffer.
 *
 * @details Returns in ``out`` the value of the parameter from a parameter context
 * buffer ``param`` with the given paramenter name ``name`` and the size of the
 * parameter value ``outlen``.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetParam(void *param, char *name, void *out, int outlen);

/**
 * @brief Converts data from float or double data type to float16 data type.
 *
 * @param d[in] The data to be converted.
 * @param f16[out] Pointer to the converted data.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function called failed because input pointer
 *         ``f16`` is NULL.
 *
 * @details Converts the input data ``d`` in float or double data type to the float16
 *          data type, and returns in ``f16`` the converted data at specific
 *          position (*f16 = (f16)d).
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtConvertHalfToFloat function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtConvertDoubleToHalf(uint16_t *f16, double d);

/**
 * @brief Converts data from float data type to float16 data type.
 *
 * @param d[in] The data to be converted.
 * @param f16[out] Pointer to the converted data.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function called failed because input pointer ``f16`` is NULL.
 *
 * @details Converts the input data ``d`` in float data type to the float16
 *          data type, and returns in ``f16`` the converted data at specific
 *          position (*f16 = (f16)d).
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtConvertHalfToFloat function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtConvertFloatToHalf(uint16_t *f16, float d);

/**
 * @brief Converts data from float16 data type to double data type.
 *
 * @param f16[in] The data to be converted.
 * @param d[out] Pointer to the converted data.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function called failed because input pointer ``d`` is NULL.
 *
 * @details Converts the input data ``d`` in float16 data type to the double
 *          data type, and returns in ``f16`` the converted data at specific
 *          position (*d = (float or double)(f16)).
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtConvertHalfToFloat function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtConvertHalfToDouble(double *d, uint16_t f16);

/**
 * @brief Converts data from float16 data type to float data type.
 *
 *
 * @param f16[in] The data to be converted.
 * @param d[out] Pointer to the converted data.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function called failed because input pointer ``d`` is NULL.
 *
 * @details Converts the input data ``d`` in float16 data type to the float
 *          data type, and returns in ``f16`` the converted data at specific
 *          position (*d = (float or double)(f16)).
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 @verbatim
 int main () {

  float data0 = 1024.123;
  uint16_t half0 = 0;
  cnrtConvertFloatToHalf(&half0, data0);
  cnrtConvertHalfToFloat(&data0, half0);

  double data1 = 1024.123;
  uint16_t half1 = 0;
  cnrtConvertDoubleToHalf(&half1, data1);
  cnrtConvertHalfToDouble(&data1, half1);

  return 0;
 }
 @endverbatim
 *
 */

extern CNRT_DLL_API cnrtRet_t cnrtConvertHalfToFloat(float *d, uint16_t f16);

/**
 * @brief Retrieves the size of a data type defined with ::cnrtDataType_t.
 *
 * @param dt[in] The data type to be retrieved.
 * @retval The size of the data type in bytes.
 *
 * @details Returns the size of the data type ``dt``, which is defined with
 * the ::cnrtDataType_t enum.
 *
 * @note
 * - If the data type you set in ``dt`` is not supported, the value ``4`` is returned.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
extern CNRT_DLL_API int cnrtDataTypeSize(cnrtDataType_t dt);

/**
 * @brief Creates a CNRT Context based on a CNRT Function on the MLU device.
 *
 * @param pctx[out] Pointer to the created CNRT Context defined in ::cnrtRuntimeContext_t.
 * @param function[in] Pointer to an MLU function that is extracted from an offline model file via
 * the ::cnrtExtractFunction function.
 * @param extra[in] Reserved for future use. The value of this parameter should be NULL.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because the value of the
 * parameter ``function`` is invalid.
 * @retval CNRT_RET_ERR_NOMEM This function call failed because not enough memory can be allocated
 * on device.
 *
 * @details  Creates a CNRT Context with the given CNRT Function ``function`` on the MLU device, and
 * returns in ``pctx`` the created CNRT Context. The MLU device used in this function is set with
 * the ::cnrtSetRuntimeContextDeviceId function. The CNRT Context is used to run the CNRT Function
 * ``function`` with different configurations on multiple MLU devices.
 *
 * @note
 * - You need to call the ::cnrtDestroyRuntimeContext function to destroy the CNRT Context and
 *   release the related resources. Otherwise, the memory leaks may be occurred.
 * - Before calling this function, you need to call the ::cnrtLoadModel and ::cnrtExtractFunction
 *   functions to load the offline model and extract the CNRT Function.
 * - Multiple threads cannot be executed concurrently because the CNRT Context is not thread safe.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtInvokeRuntimeContext function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtCreateRuntimeContext(cnrtRuntimeContext_t *pctx,
                                                       cnrtFunction_t function,
                                                       void *extra);

/**
 * @brief Sets the DDR channel to be used for CNRT Context on the MLU device.
 *
 * @param pctx[in] Pointer to the CNRT Context that you want to set the DDR channel for. The CNRT
 * Context is defined in ::cnrtRuntimeContext_t.
 * @param channel[in] The memory channel ID. If you set the value of this parameter to **-1** or
 * **-2**, this function returns the free memory size and total memory size of all the memory
 * channels. See ::cnrtChannelType_t for details.
 * @param extra[in] Reserved for future use. The value of this parameter should be NULL.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INIT This CNRT Context has already been initialized before this function is
 * called.
 *
 * @details Sets the DDR channel of the current device used for executing CNRT Context specified in
 * "pctx". \n
 * This function is deprecated. It is recommended to use the ::cnrtSetCurrentChannel function
 * instead.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
extern CNRT_DLL_API cnrtRet_t cnrtSetRuntimeContextChannel(cnrtRuntimeContext_t pctx,
                                                           cnrtChannelType_t channel);

/**
 * @brief Sets the MLU device ordinal used for a CNRT Context on the MLU device.
 *
 * @param pctx[in] The CNRT Context that you want to set the MLU device for. The CNRT Context is
 * defined in ::cnrtRuntimeContext_t.
 * @param dev_ordinal[in] The device ordinal to be set for the CNRT Context. You can call the
 * ::cnrtGetDeviceCount function to get the total number of devices in your system.
 * The device ID is in the range [0, ::cnrtGetDeviceCount() – 1].
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INIT This CNRT Context has already been initialized before this function is
 * called.
 * @retval CNRT_RET_ERR_NODEV This function call failed because no MLU devices were detected.
 *
 * @details Sets the MLU device ``dev_ordinal`` as the current device for executing the CNRT Context
 * ``pctx``. You also need to call the ::cnrtInitRuntimeContext function to initialize computing
 * resources before executing the CNRT Context.
 *
 * @note
 * - Before calling this function, you need to call the ::cnrtCreateRuntimeContext function to
 *   create a CNRT Context object.
 * - The CNRT Context object can only be associated with a specified MLU device.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtInvokeRuntimeContext function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtSetRuntimeContextDeviceId(cnrtRuntimeContext_t pctx,
                                                            int dev_ordinal);

/**
 * @brief Initializes a CNRT Context on the MLU device.
 *
 * @param pctx[in] The CNRT Context to be initialized. The CNRT Context is defined in
 * ::cnrtRuntimeContext_t.
 * @param extra[in] Reserved for future use. The value of this parameter should be NULL.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INIT This CNRT Context has already been initialized before this function is
 * called.
 * @retval CNRT_RET_ERR_NODEV This function call failed because no MLU devices were detected.
 * @retval CNRT_RET_ERR_UNSUPPORTED This function call failed because the model version and the
 * device version are inconsistent. For example, the model file is generated by MLU270 and the
 * version of device is Edge.
 *
 * @details Initializes the memory, public and private computing
 * resources on the MLU device that are used for executing CNRT Context "pctx". This function is
 * used to initialize the CNRT Context for performing computing tasks. \n
 * To perform other operations, rather than computing, such as memory
 * allocation or memory data copy, call the ::cnrtSetCurrentContextDevice or
 * ::cnrtSetCurrentDevice function. Ensure that use the same device with the CNRT Context's.
 *
 * @note
 * - Before calling this function, you need to call the ::cnrtCreateRuntimeContext function to
 *   create a CNRT Context object and then call the
 *   ::cnrtSetRuntimeContextDeviceId function to specify the MLU device to be used for CNRT
 *   Context execution.
 * - After calling this function, you need to create a queue with the
 *   ::cnrtRuntimeContextCreateQueue or ::cnrtCreateQueue function. Ensure that the same device
 *   with the CNRT Context's before invoking CNRT Context with the
 *   ::cnrtInvokeRuntimeContext or ::cnrtInvokeRuntimeContext_V2 function.
 * - If the MLU device is not set for CNRT Context ``src_pctx``, the current MLU device is used if
 *   you have a single MLU device on the system for computing. You must set an MLU device for the
 *   CNRT Context if you have multiple MLU devices on the system for computing.
 * - Before doing any operations on CNRT Context, such as ::cnrtForkRuntimeContext and
 *   ::cnrtInvokeRuntimeContext, you need to call this function to initialize the CNRT Context
 *   first.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtInvokeRuntimeContext function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtInitRuntimeContext(cnrtRuntimeContext_t pctx, void *extra);

/**
 * @brief Creates and deploys a copy of the original CNRT Context on the MLU device.
 *
 * @param dst_pctx[out] Pointer to the replicated CNRT Context ``src_pctx``.
 * @param src_pctx[in] The original CNRT Context to be replicated.
 * @param extra[in] Reserved for future use. The value of this parameter should be NULL.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because the value of the parameter
 * ``src_pctx`` is invalid.
 *
 * @details Creates a copy of the original CNRT Context "src_pctx" and saves the replicated CNRT
 * Context in "dst_pctx".\n
 * This function also initializes the private space of the replicated Context that saves the
 * temporary data and other information.
 * You can call this function multiple times to create replicated multiple Contexts you need.
 * All the replicated Contexts share the public resources in the memory of the
 * original Context. Also, the replicated Contexts and the original Context can be passed to
 * different threads respectively for concurrent execution. For more information, see "CNRT User
 * Guide".
 *
 * @note
 * - This function is only used when executing with multiple queues or threads to avoid concurrent
 *   conflicts on accessing private spaces when invoking CNRT Context.
 * - Because the replicated CNRT Contexts and the origin CNRT Context are on the same MLU device,
 *   when multiple CNRT Contexts accessing their shared memory concurrently, it may cause
 *   performance decrease.
 * - If the MLU device is not set for CNRT Context ``src_pctx``, the current MLU device is used if
 *   you have a single MLU device on the system for computing. You must set an MLU device for the
 *   CNRT Context if you have multiple MLU devices on the system for computing.
 * - You need to free the memory used in ``dst_pctx`` with the ::cnrtDestroyRuntimeContext function.
 * - Before calling this function, you need to call the ::cnrtInitRuntimeContext function to
 *   initialize the original CNRT Context.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtInvokeRuntimeContext function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtForkRuntimeContext(cnrtRuntimeContext_t *dst_pctx,
                                                     cnrtRuntimeContext_t src_pctx,
                                                     void *extra);

/**
 * @brief Creates a queue for a CNRT Context on the MLU device.
 *
 * @param pctx[in] The CNRT Context to create the queue for. The CNRT Context is defined in
 * ::cnrtRuntimeContext_t.
 * @param queue[out] Pointer to the newly created queue.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INIT This CNRT Context has not been initialized while this function is
 * called.
 * @retval CNRT_RET_ERR_NODEV This function call failed because no MLU devices were detected.
 *
 * @details Creates a queue pointed by ``queue`` for a CNRT Context specified in ``pctx``.
 * This function is only used if you set the MLU device for CNRT Context with the
 * ::cnrtSetRuntimeContextDeviceId function. Otherwise, you should call the ::cnrtCreateQueue
 * function to create queues. Ensure that the same device is used with the CNRT Context.
 *
 * @note
 * - Before calling this function, you need to call the ::cnrtInitRuntimeContext function to
 *   initialize computing resources and bind the MLU device for executing the CNRT Context.
 * - You need to destroy the queue with the ::cnrtDestroyQueue function.
 * - After calling this function, you could call the ::cnrtInvokeRuntimeContext or
 *   ::cnrtInvokeRuntimeContext_V2 function to launch CNRT Context.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtInvokeRuntimeContext function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtRuntimeContextCreateQueue(cnrtRuntimeContext_t pctx,
                                                            cnrtQueue_t *queue);

/**
 * @brief Creates a notifier for a CNRT Context on the MLU device.
 *
 * @param pctx[in] The CNRT Context to create a notifier for. The CNRT Context is defined in
 * ::cnrtRuntimeContext_t.
 * @param pnotifier[out] Pointer to the created notifier for the CNRT Context.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INIT This CNRT Context has not been initialized while this function is
 * called.
 * @retval CNRT_RET_ERR_INVALID_POINTER This function call failed because the value of the pointer
 * ``pnotifier`` is invalid.
 * @retval CNRT_RET_ERR_NODEV This function call failed because no MLU devices were detected.
 *
 * @details Creates a notifier for a CNRT Context "pctx", and returns a pointer
 * "pnotifier" to the newly created notifier. \n
 * This function is used to compute the time duration for a CNRT Context. You can create
 * a starting and ending notifier before and after invoking CNRT Context respectively with
 * this function, and then call the ::cnrtNotifierDuration function to compute the execution
 * time of the CNRT Context.
 *
 * @note
 * - Before calling this function, you need to initialize computing resources with the
 *   ::cnrtInitRuntimeContext function.
 * - You need to free the memory used in ``pnotifier`` with the ::cnrtDestroyNotifier function.
 * - If the MLU device is not set for CNRT Context ``src_pctx``, the current MLU device is used if
 *   you have a single MLU device on the system for computing. You must set an MLU device for the
 *   CNRT Context if you have multiple MLU devices on the system for computing.
 * - This function is only used if you set the MLU device for CNRT Context with the
 *   ::cnrtSetRuntimeContextDeviceId function. Otherwise, you need to call the ::cnrtCreateNotifier
 *   function to create notifiers. For more information about notifier, see "CNRT User Guide".
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtInvokeRuntimeContext function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtRuntimeContextCreateNotifier(cnrtRuntimeContext_t pctx,
                                                               cnrtNotifier_t *pnotifier);

/**
 * @brief Allocates memory for an array used for a CNRT Context on the MLU device.
 *
 * @param pctx[in] The CNRT Context to allocate memory for. The CNRT Context is defined in
 * ::cnrtRuntimeContext_t.
 * @param ptr[out] Pointer to allocated memory for the array on the MLU device.
 * @param bytesArray[in] Requested memory size in bytes.
 * @param length[in] The length of the array to be allocated.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INIT This CNRT Context has not been initialized while this function is
 * called.
 * @retval CNRT_RET_ERR_INVALID This function call failed because the value of the function
 * parameters ``bytesArray`` or ``length`` is invalid.
 * @retval CNRT_RET_ERR_NOMEM This function call failed because not enough memory can be allocated
 * on device.
 *
 * @details Allocates the size of ``bytesArray`` bytes of device memory for an array with the given
 * array size ``length`` on the MLU device of the CNRT Context, and returns in a pointer
 * ``pPtr`` to the allocated memory. Ensure that use the same MLU device with the CNRT Context.
 * The array is used to store the input and output when executing the CNRT Context ``pctx``.
 *
 * @note
 * - Before calling this function, you need to call the ::cnrtInitRuntimeContext function to
 *   initialize computing resources.
 * - You need to free the memory used in ``ptr`` with the ::cnrtFree function.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
extern CNRT_DLL_API cnrtRet_t cnrtRuntimeContextMallocBySizeArray(cnrtRuntimeContext_t pctx,
                                                                  void ***ptr,
                                                                  size_t *bytesArray,
                                                                  int length);

/**
 * @brief Frees the device memory allocated for a CNRT Context on the MLU device.
 *
 * @param pctx[in] The CNRT Context to free the memory for. The CNRT Context is defined in
 * ::cnrtRuntimeContext_t.
 * @param ptr[in] Pointer to the device memory to be freed.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INIT This CNRT Context has not been initialized while this function is
 * called.
 *
 * @details Frees the MLU device memory pointed by ``ptr`` for CNRT Context ``ptr``. This function
 * is deprecated. It is recommended to use the ::cnrtFree function instead.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
extern CNRT_DLL_API cnrtRet_t cnrtRuntimeContextFree(cnrtRuntimeContext_t pctx, void *ptr);

/**
 * @brief Frees a pointer array for a CNRT Context on the MLU device.
 *
 * @param pctx[in] The CNRT Context to free the memory for. The CNRT Context is defined in
 * ::cnrtRuntimeContext_t.
 * @param ptr[in] Pointer to the device memory to be freed.
 * @param length[in] The length of the array to be freed.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INIT This CNRT Context has not been initialized while this function is
 * called.
 *
 * @details Frees a pointer array pointed by ``ptr`` for CNRT Context ``pctx`` on the MLU device
 * with the given array size ``length``. This function is deprecated. It is recommended to use the
 * ::cnrtFree function instead.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
extern CNRT_DLL_API cnrtRet_t cnrtRuntimeContextFreeArray(cnrtRuntimeContext_t pctx,
                                                          void **ptr,
                                                          int length);

/**
 * @brief Destroys a CNRT Context that was created via the ::cnrtCreateRuntimeContext function.
 *
 * @param pctx[in] The CNRT Context to be destroyed. The CNRT Context is defined in
 * ::cnrtRuntimeContext_t.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because the value of the function
 * parameters ``pctx`` is invalid.
 *
 * @details Releases the resources of the CNRT Context ``pctx`` created with the
 * ::cnrtCreateRuntimeContext function.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtInvokeRuntimeContext function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtDestroyRuntimeContext(cnrtRuntimeContext_t pctx);

/**
 * @brief Launches a CNRT Context for computing on the MLU device.
 *
 * @param pctx[in] The CNRT Context to be launched. The CNRT Context is defined in
 * ::cnrtRuntimeContext_t.
 * @param params[in] Pointer to the memory that stores input and output.  You need to call the
 * ::cnrtMalloc or ::cnrtRuntimeContextMallocBySizeArray function to allocate memory for the input
 * and output before calling this function.
 * @param queue[in] The queue handle associated with the MLU function. You can create a
 * queue by calling the ::cnrtCreateQueue or ::cnrtRuntimeContextCreateQueue function.
 * @param extra[in] Pointer to the parameters that specify the cluster affinity information. You can
 * set it to NULL if you do not want to set cluster affinity. For best practices, you can define
 * cluster with the corresponding DDR channel on MLU hardware platforms with multiple clusters
 * in ::cnrtInvokeParam_t struct. See "CNRT User Guide".
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INIT This CNRT Context has not been initialized by ::cnrtInitRuntimeContext
 * while this function is called.
 * @retval CNRT_RET_ERR_INVALID This function call failed because the value of the function
 * parameters ``params`` is invalid.
 * @retval CNRT_RET_ERR_NODEV This function call failed because no MLU devices were detected.
 * @retval CNRT_RET_ERR_NOMEM This function call failed because not enough memory can be allocated
 * on device.
 *
 * @details Launches a CNRT Context "pctx" for computing. The "extra" points to the parameters about
 * cluster affinity. \n
 * This function is only used for immutable shape of tensors. To launch CNRT Context with mutable
 * shape of tensors, call the ::cnrtInvokeRuntimeContext_V2 function.
 *
 * @note
 * - You must set an MLU device for the CNRT Context if you have multiple MLU devices on the system
 *   for computing.
 * - Before calling this function, you need to call the ::cnrtInitRuntimeContext function to
 *   initialize computing resources or call the ::cnrtForkRuntimeContext function to use the
 *   initialized computing resources.
 * - This function is always performed by the device associated with a queue. If ``queue`` is set to
 *   NULL, the default queue is used.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 @verbatim
 int main() {
   // Initializes device
   // Assuming there are multiple MLU devices
   cnrtInit(0);
   int curr_device_id = 0, ctx_device_id = 1;
   cnrtDev_t dev;
   cnrtGetDeviceHandle(&dev, curr_device_id);
   cnrtSetCurrentDevice(dev);

   // Loads model and extracts function
   // Assuming the model is an immutable model file
   cnrtModel_t model;
   cnrtFunction_t function;
   cnrtLoadModel(&model, fname);
   cnrtCreateFunction(&function);
   cnrtExtractFunction(&function, model, symbol);

   // Creates CNRT Context and sets device configuration
   cnrtRuntimeContext_t ctx, ctx_copy;
   cnrtCreateRuntimeContext(&ctx, function, NULL);
   cnrtSetRuntimeContextDeviceId(ctx, ctx_device_id);

   // Initailizes and forks CNRT Context
   cnrtInitRuntimeContext(ctx, NULL);
   cnrtForkRuntimeContext(&ctx_copy, ctx, NULL);

   // Creates queue and notifier from CNRT Context as the following two methods
   // which you can choose one to achieve
   cnrtQueue_t queue;
   cnrtNotifier_t notifier_start = NULL, notifier_end = NULL;
   // (1) By CNRT Context functions
   cnrtRuntimeContextCreateQueue(ctx, &queue);
   cnrtRuntimeContextCreateNotifier(ctx, &notifier_start);
   cnrtRuntimeContextCreateNotifier(ctx, &notifier_end);
   // (2) By normal functions, make sure the current device is same with the CNRT Context
   // If the current device is different from the CNRT Context, call cnrtSetCurrentDevice or
   // cnrtSetCurrentContextDevice
   // (2.1) By cnrtSetCurrentDevice
   cnrtGetDeviceHandle(&dev, ctx_device_id);
   cnrtSetCurrentDevice(dev);
   // (2.2) By cnrtSetCurrentContextDevice
   cnrtSetCurrentContextDevice(ctx);

   cnrtCreateQueue(&queue);
   cnrtCreateNotifier(&notifier_start);
   cnrtCreateNotifier(&notifier_end);

   // Gets input and output data size of the function
   int input_num, output_num;
   int64_t *input_size_array, *output_size_array;
   cnrtGetInputDataSize(&input_size_array, &input_num, function);
   cnrtGetOutputDataSize(&output_size_array, &output_num, function);
   // input_cpu_ptr_array is initialized by users
   void **input_mlu_ptr_array = (void **)malloc(sizeof(void *) * input_num);
   void **output_mlu_ptr_array = (void **)malloc(sizeof(void *) * output_num);
   for (int i = 0; i < input_num; i++) {
     cnrtMalloc(&(input_mlu_ptr_array[i]), input_size_array[i]);
     cnrtMemcpy(input_mlu_ptr_array[i], input_cpu_ptr_array[i], input_size_array[i],
                CNRT_MEM_TRANS_DIR_HOST2DEV);
   }
   for (int i = 0; i < output_num; i++) {
     cnrtMalloc(&(output_mlu_ptr_array[i]), output_size_array[i]);
   }

   // Prepares parameters for cnrtInvokeRuntimeContext
   // input_mlu_ptr_array and output_mlu_ptr_array are device addresses
   // of all input and output tensor
   void **param = (void **)malloc(sizeof(void *) * (input_num + output_num));
   for (int i = 0; i < input_num; ++i) {
     param[i] = input_mlu_ptr_array[i];
   }
   for (int j = 0; j < output_num; ++j) {
     param[input_num + j] = output_mlu_ptr_array[j];
   }

   // Invokes CNRT Context and gets time of invoking time as the following two methods
   // If you need to set cluster affinity, see sample cnrtInvokeRuntimeContext_V2
   float *ptv = (float *)malloc(sizeof(float));
   cnrtPlaceNotifier(notifier_start, queue);
   // (1) By cnrtInvokeRuntimeContext
   cnrtInvokeRuntimeContext(ctx, param, queue, NULL);
   // (2) By cnrtInvokeRuntimeContext_V2 which the param_descs is NULL
   cnrtInvokeRuntimeContext_V2(ctx, NULL, param, queue, NULL);
   cnrtPlaceNotifier(notifier_end, queue);
   cnrtSyncQueue(queue);
   cnrtNotifierDuration(notifier_start, notifier_end, ptv);
   print("Invoking CNRT Context time is: %f\n", *ptv);

   // Free input and output device memory by cnrtFree or cnrtFreeArray function
   // (1) By cnrtFree
   for (int i = 0; i < input_num; i++) {
     cnrtFree(input_mlu_ptr_array[i]);
   }
   for (int i = 0; i < output_num; i++) {
     cnrtFree(output_mlu_ptr_array[i]);
   }
   // (2) By cnrtFreeArray
   cnrtFreeArray(input_mlu_ptr_array, input_num);
   cnrtFreeArray(output_mlu_ptr_array, output_num);

   // Frees resources
   cnrtDestroyQueue(queue);
   cnrtDestroyNotifier(&notifier_start);
   cnrtDestroyNotifier(&notifier_end);
   cnrtDestroyRuntimeContext(ctx_copy);
   cnrtDestroyRuntimeContext(ctx);
   cnrtDestroyFunction(function);
   cnrtUnloadModel(model);
   cnrtDestroy();
   free(input_size_array);
   free(output_size_array);
   free(param);
   free(ptv);
 }
 @endverbatim
 */
extern CNRT_DLL_API cnrtRet_t cnrtInvokeRuntimeContext(cnrtRuntimeContext_t pctx,
                                                       void **params,
                                                       cnrtQueue_t queue,
                                                       void *extra);

/**
 * @brief Launches a CNRT Context for computing on the MLU device.
 *
 * @param pctx[in] The CNRT Context to be launched. The CNRT Context is defined in
 * ::cnrtRuntimeContext_t.
 * @param param_descs[in] The array of parameter descriptors that hold the information about input
 * shapes. The descriptors are created with the ::cnrtCreateParamDescArray or ::cnrtCreateParamDesc
 * function. If the shape of the input is immutable, set this parameter to NULL.
 * @param param_buffers[in] An array pointer to the input and output addresses on the MLU device.
 * @param queue[in] The queue handle associated with the MLU function. You can create a
 * queue by calling the ::cnrtCreateQueue or ::cnrtRuntimeContextCreateQueue function.
 * @param extra[in]  Pointer to the parameters that specify the cluster affinity information. You
 * can set it to NULL if you do not want to set cluster affinity. For best practices, you can define
 * cluster with the corresponding DDR channel on MLU hardware platforms with multiple clusters
 * in ::cnrtInvokeParam_t struct. See "CNRT User Guide".
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INIT This CNRT Context has not been initialized while this function is
 * called.
 * @retval CNRT_RET_ERR_INVALID This function call failed because the value of the function
 * parameters ``param_buffers`` is invalid or the variable dims of ``param_descs`` is invalid.
 * @retval CNRT_RET_ERR_NODEV This function call failed because no MLU devices were detected.
 * @retval CNRT_RET_ERR_INVALID_DATADESC This function call failed because the value of the function
 * parameters ``param_descs`` is invalid.
 * @retval CNRT_RET_ERR_NOMEM This function call failed because not enough memory can be allocated
 * on device.
 *
 * @details Launches a CNRT Context "pctx" for computing. The "param_descs" is defined in
 * cnrtParamDesc_t struct to store the shape of input. The "extra" points to the parameters about
 * cluster affinity. \n
 * This function is used for both immutable and mutable shape of tensors. To launch CNRT Context
 * with immutable shape of tensors, set param_descs to NULL or call the ::cnrtInvokeRuntimeContext
 * function.
 *
 * @note
 * - If the MLU device is not set for CNRT Context ``src_pctx``, the current MLU device is used if
 *   you have a single MLU device on the system for computing. You must set an MLU device for the
 *   CNRT Context if you have multiple MLU devices on the system for computing.
 * - Before calling this function, you need to call the ::cnrtInitRuntimeContext function to
 *   initialize computing resources.
 * - If ``queue`` is set to NULL, the default queue is used.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * @verbatim
 int main() {
   // Initializes device
   cnrtInit(0);
   cnrtDev_t dev;
   int device_id = 0;
   cnrtGetDeviceHandle(&dev, device_id);
   cnrtSetCurrentDevice(dev);

   // Loads model and extracts function
   // Assuming the model is a mutable model file
   cnrtModel_t model;
   cnrtFunction_t function;
   cnrtLoadModel(&model, fname);
   cnrtCreateFunction(&function);
   cnrtExtractFunction(&function, model, symbol);

   // Creates CNRT Context and sets device configuration
   cnrtRuntimeContext_t ctx;
   cnrtCreateRuntimeContext(&ctx, function, NULL);
   cnrtSetRuntimeContextDeviceId(ctx, device_id);

   // Initailizes CNRT Context
   cnrtInitRuntimeContext(ctx, NULL);

   // Creates a queue to invoke CNRT Context
   // If the device of CNRT Context is different from current device, call
   // cnrtSetCurrentContextDevice or cnrtSetCurrentDevice to modify current device to the device
   // assocaited with the CNRT Context. See sample cnrtInvokeRuntimeContext
   cnrtQueue_t queue;
   cnrtCreateQueue(&queue);

   // Creates parameters and invokes CNRT Context
   // Create param_descs and param_buffers to invoke CNRT Context
   // shape_num is the number of invoking mutable functions
   // input_num is the number of input tensors for each variable dims
   // output_num is the number of output tensors for each variable dims
   // input_shape_x is the input shape of input tensor
   // input_dim_num is the number of input tensor dimensions
   // shape_num, input_num, output_num, input_shape_x and input_dim_num are defined by users
   // Assuming shape_num = 3, input_num = 1, output_num = 1, input_dim_num = 4
   int *input_shape_array[shape_num][input_num] = {{input_shape_1},
                                                   {input_shape_2},
                                                   {input_shape_3}};
   cnrtParamDescArray_t input_param_desc_array[shape_num] = {NULL};
   cnrtParamDescArray_t output_param_desc_array[shape_num] = {NULL};
   for (int shape_index = 0; shape_index < shape_num; shape_index++) {
     cnrtParamDescArray_t input_param_descs, output_param_descs;
     cnrtCreateParamDescArray(&input_param_descs, input_num);
     cnrtCreateParamDescArray(&output_param_descs, output_num);
     for (int i = 0; i < input_num; i++) {
       cnrtSetShapeToParamDesc(input_param_descs[i], input_shape_array[shape_index][i],
                               input_dim_num[i]);
     }
     cnrtInferFunctionOutputShape(function, input_num, input_param_descs,
                                  output_num, output_param_descs);
     input_param_desc_array[shape_index] = input_param_descs;
     output_param_desc_array[shape_index] = output_param_descs;
   }
   int64_t **input_size_array = (int64_t **)malloc(sizeof(int64_t) * shape_num);
   for (int i = 0; i < shape_num; i++) {
     input_size_array[i] = (int64_t *)malloc(sizeof(int64_t) * input_num);
     for (int j = 0; j < input_num; j++) {
       cnrtGetParamDescSize(input_param_desc_array[i][j], &input_size_array[i][j]);
     }
   }
   int64_t **output_size_array = (int64_t **)malloc(sizeof(int64_t) * shape_num);
   for (int i = 0; i < shape_num; i++) {
     output_size_array[i] = (int64_t *)malloc(sizeof(int64_t) * output_num);
     for (int j = 0; j < output_num; j++) {
       cnrtGetParamDescSize(output_param_desc_array[i][j], &output_size_array[i][j]);
     }
   }
   // input_cpu_ptr_array is initialized by users
   for (int invoke_index = 0; invoke_index < shape_num; invoke_index++) {
     void **input_mlu_ptr_array = (void **)malloc(sizeof(void *) * input_num);
     void **output_mlu_ptr_array = (void **)malloc(sizeof(void *) * output_num);
     for (int i = 0; i < input_num; i++) {
       cnrtMalloc(&(input_mlu_ptr_array[i]), input_size_array[invoke_index][i]);
       cnrtMemcpy(input_mlu_ptr_array[i], input_cpu_ptr_array[invoke_index][i],
           input_size_array[invoke_index][i], CNRT_MEM_TRANS_DIR_HOST2DEV);
     }
     for (int i = 0; i < output_num; i++) {
       cnrtMalloc(&(output_mlu_ptr_array[i]), output_size_array[invoke_index][i]);
     }
     cnrtParamDesc_t *param_descs =
         (cnrtParamDesc_t *)malloc(sizeof(cnrtParamDesc_t) * (input_num + output_num));
     void **param = (void **)malloc(sizeof(void *) * (input_num + output_num));
     for (int i = 0; i < input_num; ++i) {
       param[i] = input_mlu_ptr_array[i];
       param_descs[i] = input_param_desc_array[invoke_index][i];
     }
     for (int j = 0; j < output_num; ++j) {
       param[input_num + j] = output_mlu_ptr_array[j];
       param_descs[input_num + j] = output_param_desc_array[invoke_index][j];
     }

     // Sets cluster affinity to execute kernel function. If you do not need to set affinity, the
     // parameter extra could be set to NULL.
     u32_t affinity = 0x01;
     cnrtInvokeParam_t invoke_param;
     invoke_param.cluster_affinity.affinity = &affinity;
     invoke_param.invoke_param_type = CNRT_INVOKE_PARAM_TYPE_0;
     // Invokes CNRT Context
     cnrtInvokeRuntimeContext_V2(ctx, param_descs, param, queue, (void *)&invoke_param);
     cnrtSyncQueue(queue);

     // Gets output after executing kernel function
     // output_cpu_ptr_array is initialized by users
     for (int i = 0; i < output_num; i++) {
       cnrtMemcpy(output_cpu_ptr_array[invoke_index][i], output_mlu_ptr_array[i],
           output_size_array[invoke_index][i], CNRT_MEM_TRANS_DIR_DEV2HOST);
     }

     cnrtFreeArray(input_mlu_ptr_array, input_num);
     cnrtFreeArray(output_mlu_ptr_array, output_num);
     free(param_descs);
     if (NULL != param)
       free(param);
   }

   // Frees resources
   for (int i = 0; i < shape_num; i++) {
     cnrtDestroyParamDescArray(input_param_desc_array[i], input_num);
     cnrtDestroyParamDescArray(output_param_desc_array[i], output_num);
   }
   cnrtDestroyQueue(queue);
   cnrtDestroyRuntimeContext(ctx);
   cnrtDestroyFunction(function);
   cnrtUnloadModel(model);
   cnrtDestroy();
   free(input_size_array);
   free(output_size_array);
 }
 * @endverbatim
 */
extern CNRT_DLL_API cnrtRet_t cnrtInvokeRuntimeContext_V2(cnrtRuntimeContext_t pctx,
                                                          cnrtParamDesc_t *param_descs,
                                                          void **param_buffers,
                                                          cnrtQueue_t queue,
                                                          void *extra);

/**
 * @brief Launches a CNRT Context for computing on the MLU device.
 *
 * @param pctx[in] The CNRT Context to be launched. The CNRT Context is defined in
 * ::cnrtRuntimeContext_t.
 * @param batch_sizes[in] The array of input batch sizes. If there is more than one input, you need
 * to set all input batch sizes accordingly. When not in cache mode, this parameter could be set as
 * NULL.
 * @param param_buffers[in] An array pointer to the input and output addresses on the MLU device.
 * @param queue[in] The queue handle associated with the MLU function. You can create a
 * queue by calling the ::cnrtCreateQueue or ::cnrtRuntimeContextCreateQueue function.
 * @param extra[in]  Pointer to the parameters that specify the cluster affinity information. You
 * can set it to NULL if you do not want to set cluster affinity. For best practices, you can define
 * cluster with the corresponding DDR channel on MLU hardware platforms with multiple clusters
 * in ::cnrtInvokeParam_t struct. See "CNRT User Guide".
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INIT This CNRT Context has not been initialized while this function is
 * called.
 * @retval CNRT_RET_ERR_INVALID This function call failed because the value of the function
 * parameters ``param_buffers`` is invalid or the variable dims of ``param_descs`` is invalid.
 * @retval CNRT_RET_ERR_NODEV This function call failed because no MLU devices were detected.
 * @retval CNRT_RET_ERR_INVALID_DATADESC This function call failed because the value of the function
 * parameters ``param_descs`` is invalid.
 * @retval CNRT_RET_ERR_NOMEM This function call failed because not enough memory can be allocated
 * on device.
 *
 * @details Launches a CNRT Context "pctx" for computing. The "batch_sizes" is an array of the batch
 * sizes of inputs. The "extra" points to the parameters about cluster affinity. \n
 * This function is used for both cache and non-cache mode. To launch CNRT Context with cache mode,
 * make sure batch_sizes set correctly or set to NULL, or call the ::cnrtInvokeRuntimeContext
 * function.
 *
 * @note
 * - If the MLU device is not set for CNRT Context ``src_pctx``, the current MLU device is used if
 *   you have a single MLU device on the system for computing. You must set an MLU device for the
 *   CNRT Context if you have multiple MLU devices on the system for computing.
 * - Before calling this function, you need to call the ::cnrtInitRuntimeContext function to
 *   initialize computing resources.
 * - If ``queue`` is set to NULL, the default queue is used.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * @verbatim
 int main() {
   // Initializes device
   cnrtInit(0);
   cnrtDev_t dev;
   int device_id = 0;
   cnrtGetDeviceHandle(&dev, device_id);
   cnrtSetCurrentDevice(dev);

   // Loads model and extracts function
   // Assuming the model is a mutable model file
   cnrtModel_t model;
   cnrtFunction_t function;
   cnrtLoadModel(&model, fname);
   cnrtCreateFunction(&function);
   cnrtExtractFunction(&function, model, symbol);

   // Creates CNRT Context and sets device configuration
   cnrtRuntimeContext_t ctx;
   cnrtCreateRuntimeContext(&ctx, function, NULL);
   cnrtSetRuntimeContextDeviceId(ctx, device_id);

   // Initailizes CNRT Context
   cnrtInitRuntimeContext(ctx, NULL);

   // Creates a queue to invoke CNRT Context
   // If the device of CNRT Context is different from current device, call
   // cnrtSetCurrentContextDevice or cnrtSetCurrentDevice to modify current device to the device
   // assocaited with the CNRT Context. See sample cnrtInvokeRuntimeContext
   cnrtQueue_t queue;
   cnrtCreateQueue(&queue);

   // The batch of cache model: {4}, {1}
   int shape_num = 4;
   int input_num = 0;
   int output_num = 0;
   // Run batch : 1, 3, 4, 5
   // Cache model: 1, 3, 4 should be success, others should be failed.
   // Non-cache model: 4, null should be success, others should be failed.
   int batch_sizes[4][1] = {{1}, {3}, {4}, {5}};
   int64_t *input_size;
   int64_t *output_size;
   cnrtGetInputDataSize(&input_size, &input_num, function);
   cnrtGetOutputDataSize(&output_size, &output_num, function);

   for (int invoke_index = 0; invoke_index < shape_num; invoke_index++) {
     void **input_mlu_ptr_array = (void **)malloc(sizeof(void *) * input_num);
     void **output_mlu_ptr_array = (void **)malloc(sizeof(void *) * output_num);
     void **input_cpu_ptr_array = (void **)malloc(sizeof(void *) * input_num);
     void **output_cpu_ptr_array = (void **)malloc(sizeof(void *) * output_num);
     cnrtDataType_t *dtype;
     cnrtGetInputDataType(&dtype, &input_num, function);
     for (int i = 0; i < input_num; i++) {
       cnrtMalloc(&(input_mlu_ptr_array[i]), input_size[i]);
       input_cpu_ptr_array[i] = (void *)malloc(input_size[i]);

       int dim_num = 0;
       int *shape;
       int elem_num = 1;
       cnrtGetInputDataShape(&shape, &dim_num, i, function);

       for (int j = 0; j < dim_num; j++) {
         elem_num *= shape[j];
       }
       float *fp32_input = (float *)calloc(1, sizeof(float) * elem_num);
       for (int j = 0; j < elem_num; j++) {
         fp32_input[j] = 0.5;
       }
       cnrtCastDataType(fp32_input, CNRT_FLOAT32, input_cpu_ptr_array[i], dtype[i], elem_num, NULL);

       cnrtMemcpy(input_mlu_ptr_array[i], input_cpu_ptr_array[invoke_index][i],
           input_size_array[invoke_index][i], CNRT_MEM_TRANS_DIR_HOST2DEV);
       free(shape);
     }
     for (int i = 0; i < output_num; i++) {
       cnrtMalloc(&(output_mlu_ptr_array[i]), output_size_array[invoke_index][i]);
       output_cpu_ptr_array[i] = (void *)malloc(output_size[i]);
     }
     void **param = (void **)malloc(sizeof(void *) * (input_num + output_num));
     for (int i = 0; i < input_num; ++i) {
       param[i] = input_mlu_ptr_array[i];
     }
     for (int j = 0; j < output_num; ++j) {
       param[input_num + j] = output_mlu_ptr_array[j];
     }

     // Sets cluster affinity to execute kernel function. If you do not need to set affinity, the
     // parameter extra could be set to NULL.
     u32_t affinity = 0x01;
     cnrtInvokeParam_t invoke_param;
     invoke_param.cluster_affinity.affinity = &affinity;
     invoke_param.invoke_param_type = CNRT_INVOKE_PARAM_TYPE_0;
     // Invokes CNRT Context
     cnrtInvokeRuntimeContext_V3(ctx, batch_sizes, param, queue, (void *)&invoke_param);
     cnrtSyncQueue(queue);

     // Gets output after executing kernel function
     // output_cpu_ptr_array is initialized by users
     for (int i = 0; i < output_num; i++) {
       cnrtMemcpy(output_cpu_ptr_array[invoke_index][i], output_mlu_ptr_array[i],
           output_size_array[invoke_index][i], CNRT_MEM_TRANS_DIR_DEV2HOST);
     }

     cnrtFreeArray(input_mlu_ptr_array, input_num);
     cnrtFreeArray(output_mlu_ptr_array, output_num);
     free(param_descs);
     if (NULL != param)
       free(param);
   }

   // Frees resources
   for (int i = 0; i < shape_num; i++) {
     cnrtDestroyParamDescArray(input_param_desc_array[i], input_num);
     cnrtDestroyParamDescArray(output_param_desc_array[i], output_num);
   }
   cnrtDestroyQueue(queue);
   cnrtDestroyRuntimeContext(ctx);
   cnrtDestroyFunction(function);
   cnrtUnloadModel(model);
   cnrtDestroy();
 }
 * @endverbatim
 */
extern CNRT_DLL_API cnrtRet_t cnrtInvokeRuntimeContext_V3(cnrtRuntimeContext_t pctx,
                                                          int *batch_sizes,
                                                          void **param_buffers,
                                                          cnrtQueue_t queue,
                                                          void *extra);

/**
 * @brief Retrieves information about a CNRT Context on the MLU device.
 *
 * @param pctx[in] The CNRT Context to retrieve. The CNRT Context is defined in
 * ::cnrtRuntimeContext_t.
 * @param key[in] The information of CNRT Context you want to retrieve,
 * including CNRT Function, MLU device ID, the core number used for compiling. This parameter is
 * defined in ::cnrtRuntimeContextInfo enum, and supported values are CNRT_RT_CTX_FUNCTION,
 * CNRT_RT_CTX_DEV_ORDINAL, and CNRT_RT_CTX_MODEL_PARALLEL. If the MLU device is not set with the
 * ::cnrtSetCurrentDevice or ::cnrtSetRuntimeContextDeviceId function, **-2** is returned to this
 * parameter.
 * @param out[out] The returned information about the CNRT Context.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because the value of the function
 * parameters ``key`` is invalid.
 * @retval CNRT_RET_ERR_INVALID_POINTER This function call failed because the value of the pointer
 * ``out`` is invalid.
 *
 * @details Returns in ``out`` a pointer to the information about CNRT Context ``pctx`` on MLU
 * device. You can get the information about CNRT Function, MLU device ID for launching the CNRT
 * Context, or the core number for compiling by setting the ``key`` parameter.
 *
 * @note
 * - Before calling this function, you need to call the ::cnrtCreateRuntimeContext function to
 *   create CNRT Context, and set the MLU device to be used with the
 *   ::cnrtSetRuntimeContextDeviceId function.
 * - If the MLU device is not specified with the ::cnrtSetCurrentDevice or
 *   ::cnrtSetRuntimeContextDeviceId function, the current device is used if a single MLU device is
 *   used. You must set an MLU device if multiple MLU devices you want to used for computing.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetRuntimeContextInfo(cnrtRuntimeContext_t pctx,
                                                        cnrtRuntimeContextInfo_t key,
                                                        void **out);

/**
 * @brief Sets the device handle for a CNRT Context.
 *
 * @param pctx[in] The CNRT Context to set the device handle for. The CNRT Context is defined in
 * ::cnrtRuntimeContext_t.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because the value of the function
 * parameters ``pctx`` is invalid or the ``pctx`` has not been created.
 * @retval CNRT_RET_ERR_NODEV This function call failed because no MLU devices were detected.
 *
 * @details Sets the device handle for the CNRT Context "pctx". Any subsequent runtime function
 * calls
 * will use the device set in this function for execution. \n
 * This function is used to set the device handle for a CNRT Context. If the operations are
 * performed without using CNRT Contexts, such as memory allocation or memory data copy, call the
 * ::cnrtSetCurrentDevice function to set the device handle.
 *
 * @note
 * - Before calling this function, you need to call the ::cnrtCreateRuntimeContext function to
 *   create CNRT Context, and then call the ::cnrtSetRuntimeContextDeviceId function to set the MLU
 *   device used for executing the CNRT Context.
 * - If you want to perform computing operations, call the ::cnrtInitRuntimeContext function to set
 *   the device handle for a CNRT Context. If you want to perform other operations, rather than
 *   computing, call the cnrtSetCurrentContextDevice function. To share CNRT Context computing
 *   resources with other CNRT Contexts, you can either call the ::cnrtSetRuntimeContextDeviceId and
 *   ::cnrtSetCurrentContextDevice functions, or ::cnrtInitRuntimeContext function. If you call the
 *   ::cnrtInitRuntimeContext function, you do not need to call cnrtSetRuntimeContextDeviceId
 *   function.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
extern CNRT_DLL_API cnrtRet_t cnrtSetCurrentContextDevice(cnrtRuntimeContext_t pctx);

/**
 * @brief Retrieves the CPU affinity for the current thread of an MLU device.
 *
 *
 *
 * @param affinity[out] The returned CPU affinity information specified in
 *                      ::cnrtDeviceAffinity_t struct.
 * @param dev_ordinal[in] The device ordinal to retrieve. You can call the
 *                     ::cnrtGetDeviceCount function to get the total
 *                     number of devices in your system.
 *                     The device ID is in the range [0, ::cnrtGetDeviceCount() – 1].
 *                     The value ``-1`` represents a fake device.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because ``affinity`` is NULL.
 * @retval CNRT_RET_ERR_NO_EXIST This function call failed because it is unable to find CNDev
 *         functions, such as failed to find the CNDev library.
 * @retval CNRT_RET_ERR_CNDEV_FUNC_CALL This function call failed because errors
 *         from CNDev functions are returned via CNRT, such as the value of ``dev_ordinal``
 *         is out of range.
 *
 * @details Returns in ``affinity`` the information about CPU affinity bitmask for the current
 * thread of the device ``dev_ordinal``.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtTopologyGetCpuRelatedDevices function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetDeviceAffinity(cnrtDeviceAffinity_t *affinity,
                                                    int dev_ordinal);

/**
 * @brief Removes the CPU affinity for the current thread of an MLU device.
 *
 *
 * @param dev_ordinal[in] The device ordinal to be cleared. You can call the
 *                     ::cnrtGetDeviceCount function to get the total
 *                     number of devices in your system.
 *                     The device ID is in the range [0, ::cnrtGetDeviceCount() – 1].
 *                     The value ``-1`` represents a fake device.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_NO_EXIST This function call failed because it is unable to find CNDev
 *         functions, such as failed to find the CNDev library.
 * @retval CNRT_RET_ERR_CNDEV_FUNC_CALL This function call failed because errors
 *         from CNDev functions are returned via CNRT, such as the value of ``dev_ordinal``
 *         is out of range.
 *
 * @details Removes the CPU affinity for the current thread of the device ``dev_ordinal``,
 * and sets all CPU bit masks to 1. Then the thread is run on a random CPU core.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtTopologyGetCpuRelatedDevices function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtClearCurrentThreadAffinity(int dev_ordinal);

/**
 * @brief Sets CPU affinity for the current thread of an MLU device.
 *
 * @param dev_ordinal[in] The device ordinal to be set. You can call the
 *                     ::cnrtGetDeviceCount function to get the total
 *                     number of devices in your system.
 *                     The device ID is in the range [0, ::cnrtGetDeviceCount() – 1].
 *                     The value ``-1`` represents a fake device.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_NO_EXIST This function call failed because it is unable to find CNDev
 *         functions, such as failed to find the CNDev library.
 * @retval CNRT_RET_ERR_CNDEV_FUNC_CALL This function call failed because errors
 *         from CNDev functions are returned via CNRT, such as the value of ``dev_ordinal``
 *         is out of range.
 *
 * @details Sets the CPU affinity for the current thread of the device ``dev_ordinal``
 * so that the thread can be run on the CPU core directly.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtTopologyGetCpuRelatedDevices function.
 */
extern CNRT_DLL_API cnrtRet_t cnrtSetCurrentThreadAffinity(int dev_ordinal);

/**
 * @brief Retrieves the topology relationship between two MLU devices.
 *
 *
 *
 * @param relationship[out] The topology relationship of two MLU devices specified in
 *                          ::cnrtTopologyRelationshipEnum_t enum.
 * @param dev_ordinal1[in] The device ordinal to retrieve. You can call the
 *                     ::cnrtGetDeviceCount function to get the total
 *                     number of devices in your system.
 *                     The device ID is in the range [0, ::cnrtGetDeviceCount() – 1].
 *                     The value ``-1`` represents a fake device.
 * @param dev_ordinal2[in] The device ordinal to retrieve. You can call the
 *                     ::cnrtGetDeviceCount function to get the total
 *                     number of devices in your system.
 *                     The device ID is in the range [0, ::cnrtGetDeviceCount() – 1].
 *                     The value ``-1`` represents a fake device.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because ``relationship`` is NULL.
 * @retval CNRT_RET_ERR_NO_EXIST This function call failed because it is unable to find CNDev
 *         functions, such as failed to find the CNDev library.
 * @retval CNRT_RET_ERR_CNDEV_FUNC_CALL This function call failed because errors
 *         from CNDev functions are returned via CNRT, such as the value of ``dev_ordinal``
 *         is out of range.
 *
 * @details Returns in ``relationship`` the topology relationship between two MLU devices specified
 * with ``dev_ordinal1`` and ``dev_ordinal2``.
 * The relationship is specified in ::cnrtTopologyRelationshipEnum_t enum. For detailed
 * information about topology relationship, see "CNDev Developer Guide".
 *
 * @note
 * - This function does not support on Windows.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtTopologyGetCpuRelatedDevices function.
 */
extern CNRT_DLL_API cnrtRet_t
cnrtTopologyGetRelationship(cnrtTopologyRelationshipEnum_t *relationship,
                            int dev_ordinal1,
                            int dev_ordinal2);

/**
 * @brief Retrieves MLU devices that are in the topology relationship with
 *        a specified MLU device.
 *
 * @param relationship[in] The topology relationship specified in ::cnrtTopologyRelationshipEnum_t
 *                         enum.
 * @param count[in,out] Specify the number of devices to retrieve as input. Return the total number
 *                      of devices that are in the topology relationship with a specified MLU
 *                      device as output.
 * @param ordinalArray[out] The ordinal of devices that are in topology relationship
 *                          with a specified MLU device.
 * @param dev_ordinal[in] The device ordinal to retrieve. You can call the
 *                     ::cnrtGetDeviceCount function to get the total
 *                     number of devices in your system.
 *                     The device ID is in the range [0, ::cnrtGetDeviceCount() – 1].
 *                     The value ``-1`` represents a fake device.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because ``count`` or
 *                              ``ordinal_array`` is NULL.
 * @retval CNRT_RET_ERR_NO_EXIST This function call failed because it is unable to find CNDev
 *         functions, such as failed to find the CNDev library.
 * @retval CNRT_RET_ERR_CNDEV_FUNC_CALL This function call failed because errors
 *         from CNDev functions are returned via CNRT, or the ``count`` you set is less than
 *         the total number of devices that are in the topology relationship with
 *         a specified MLU device.
 *
 * @details Returns the ordinal of devices in ``ordinal_array`` and the total number of devices
 *          in ``count`` that have the topology relationship ``relationship`` with the device
 *          ``dev_ordinal``. You can specify the number of devices to retrieve
 *          in ``count``. The topology relationship is defined in ::cnrtTopologyRelationshipEnum_t
 *          enum. For detailed information about topology relationship, see "CNDev Developer Guide".
 *
 *
 * @note
 * - If the value of ``count`` is set to be less than the total number of devices that are in the
 *   topology relationship with a specified MLU device, ::CNRT_RET_ERR_CNDEV_FUNC_CALL is returned
 *   to remind you that the total number of devices is greater than ``count`` you set, and the total
 *   number of devices is returned in ``count``. The ``ordinalArray``
 *   only returns ``count`` number of device ordinals, and the returned device ordinals
 *   is based on oridinal in ASC.
 * - If the value of ``count`` is set to be greater than the total number of devices that are in the
 *   topology relationship with a specified MLU device, the total number of devices is returned in
 * ``count``.
 *   The ``ordinalArray`` returns the device ordinals for all devices that are in the
 *   topology relationship with a specified MLU device.
 * - This function does not support on Windows.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtTopologyGetCpuRelatedDevices function.
 */
extern CNRT_DLL_API cnrtRet_t
cnrtTopologyGetNearestDevices(cnrtTopologyRelationshipEnum_t relationship,
                              uint64_t *count,
                              uint64_t *ordinal_array,
                              int dev_ordinal);

/**
 * @brief Retrieves MLU devices that have the affinity with a specified CPU.
 *
 *
 * @param cpuid[in] The CPU ID that has affinity with MLU devices. You can get the total number
 *                  of CPUs on your system by calling the ``get_nprocs_conf`` function on Linux.
 * @param count[in,out] Specify the number of devices to retrieve as input. Return the total number
 *                      of devices that have the affinity with a specified CPU as output.
 * @param ordinalArray[out] The ordinal of the devices that have the affinity
 *                          with a specified CPU.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because ``affinity`` is NULL.
 * @retval CNRT_RET_ERR_NOMEM This function call failed because not enough
 *         memory can be allocated.
 * @retval CNRT_RET_ERR_NO_EXIST This function call failed because it is unable to find CNDev
 *         functions, such as failed to find the CNDev library.
 * @retval CNRT_RET_ERR_CNDEV_FUNC_CALL This function call failed because errors
 *         from CNDev functions are returned via CNRT, or the ``count`` you set is less than
 *         the total number of devices that have the affinity with the specified CPU.
 *
 *
 * @details Returns the ordinal of devices in ``ordinal_array`` and the total number of devices
 * in ``count`` that have the affinity with a specified CPU ``cpuid`` and number of devices
 * ``count``to retrieve. You can allocate memory space for ``ordinal_array`` based on ``count``.
 *
 * @note
 * - If the value of ``count`` is set to be less than the total number of devices that have
 *   the affinity with the specified CPU, ::CNRT_RET_ERR_CNDEV_FUNC_CALL is returned to remind
 *   you that the total number of devices is greater than ``count`` you set, and the total
 *   number of devices is returned in ``count``. The ``ordinalArray``
 *   only returns ``count`` number of device ordinals, and the returned device ordinals
 *   is based on oridinal in ASC.
 * - If the value of ``count`` is set to be greater than the total number of devices that have
 *   the affinity with the specified CPU, the total number of device is returned in ``count``.
 *   The ``ordinalArray`` returns the device ordinals for all devices that have the affinity with
 *   the CPU.
 * - This function does not support on Windows.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 @verbatim
 int main () {
   // Initialize device and set cuurent device
   cnrtInit(0);
   cnrtDev_t dev;
   cnrtGetDeviceHandle(&dev, 0);
   cnrtSetCurrentDevice(dev);

   // Set current thread affinity for device ordinal 0
   cnrtSetCurrentThreadAffinity(0);

   // clear current thread affinity binding for device ordinal 0
   cnrtClearCurrentThreadAffinity(0);

   // Get device(ordinal 0) affinity
   cnrtDeviceAffinity_t deviceAffinity;
   cnrtGetDeviceAffinity(&deviceAffinity, 0);

   // Get relationship
   int dev_count;
   cnrtGetDeviceCount(&dev_count);
   cnrtTopologyRelationshipEnum_t relationship;
   if (dev_count > 1) {
     cnrtTopologyGetRelationship(&relationship, 0, 1);
   }

   //Get nearest device
   uint64_t count = 10;
   uint64_t *ordinalArray = NULL;
   ordinalArray = (uint64_t *)calloc(1, sizeof(uint64_t) * count);
   if (!ordinalArray) {
     //error happened
     return 0;
   }
   cnrtTopologyGetNearestDevices(CNRT_TOPO_SELF, &count, ordinalArray, 0);

   // Get related devices
   cnrtTopologyGetCpuRelatedDevices(0, &count, ordinalArray);

   free(ordinalArray);

   cnrtDestroy();
   return 0;
 }
 @endverbatim .
 */
extern CNRT_DLL_API cnrtRet_t cnrtTopologyGetCpuRelatedDevices(int cpuid,
                                                               uint64_t *count,
                                                               uint64_t *ordinal_array);
/**
 * @brief Copies memory data from two devices.
 *
 * @param dst[in] Pointer to the destination memory address.
 * @param dstDevOrdinal[in] The destination device. You can call the ::cnrtGetDeviceCount function
 *                          to get the total number of devices in your system. The device ID
 *                          is in the range [0, ::cnrtGetDeviceCount() ‒ 1].
 * @param src[in] Pointer to the source memory address.
 * @param srcDevOrdinal[in] The source device. You can call the ::cnrtGetDeviceCount function
 *                          to get the total number of devices in your system. The device ID
 *                          is in the range [0, ::cnrtGetDeviceCount() ‒ 1].
 * @param bytes[in] The memory size in bytes to be copied.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because the
 *         value of this function parameters is invalid.
 * @retval CNRT_RET_ERR_FUNC_CALL This function call failed because
 *         an error occurred from driver function.
 *
 * @details Synchronously copies the size of ``bytes`` bytes of data from the location pointed
 * by the source address ``src`` of the ``srcDevOrdinal`` device to the memory address pointed
 * by ``dst`` of the ``dstDevOrdinal`` device. The two devices should be peerable. You need tocall
 * the ::cnrtGetPeerAccessibility function to check if the two devices are peerable. If an error
 * occurred, ::CNRT_RET_ERR_INVALID or ::CNRT_RET_ERR_FUNC_CALL is returned.
 *
 * @note
 * - This function is used to copy data synchronously. To copy data asynchronously,
 *   call the ::cnrtMemcpyPeerAsync function.
 * - Cambricon devices support peer-to-peer communications. If the two MLU devices
 *   are not peerable, it may be hardware problems.
 *
 * **Requirements**
 *
 * None.
 *
 * **Example**
 *
 @verbatim
 int main() {
   // Initialize device
   cnrtInit(0);

   int dev_num = 0;
   // Get device count
   cnrtGetDeviceCount(&dev_num);

   // For p2p, device count must >= 2
   if (dev_num <= 1) {
     cnrtDestroy();
     exit(-1);
   }

   cnrtDev_t dev, peerdev;
   // To simplify sample code, suppose sdev is 0, ddev is 1
   int sdev = 0;
   int ddev = 1;

   int peer = 0;
   // Before do p2p, judge if the two devices can do p2p
   cnrtGetPeerAccessibility(&peer, sdev, ddev);
   if (peer != 1) {
     printf("the two devices can't do p2p!\n");
     cnrtDestroy();
     exit(-1);
   }

   if (cnrtGetDeviceHandle(&dev, sdev) || cnrtGetDeviceHandle(&peerdev, ddev)) {
     printf("get sdev or ddev device hanle failed!\n");
     cnrtDestroy();
     exit(-1);
   }

   size_t size = sizeof(size_t) * N;
   char *cpu_in = (char *)malloc(size);
   char *cpu_out = (char *)malloc(size);
   memset(cpu_in, 'a', size);

   // Before cnrtMalloc, must call cnrtSetCurrentDevice.
   cnrtSetCurrentDevice(peerdev);
   cnrtMalloc(&dinput, size);
   cnrtSetCurrentDevice(dev);
   cnrtMalloc(&sinput, size);

   cnrtMemset(sinput, 'a', size);

   cnrtMemcpyPeer(dinput, ddev, sinput, sdev, size);
   cnrtMemcpy(cpu_out, dinput, size, CNRT_MEM_TRANS_DIR_DEV2HOST)；

   if (memcmp(cpu_out, cpu_in, size))
     printf("p2p failed!\n");

   // Free device memory
   cnrtFree(dinput);
   cnrtFree(sinput);

   // Free host memory
   free(cpu_in);
   free(cpu_out);

   cnrtDestroy();
   return 0;
 }
 @endverbatim
 */
extern CNRT_DLL_API cnrtRet_t
cnrtMemcpyPeer(void *dst, int dstDevOrdinal, void *src, int srcDevOrdinal, size_t bytes);

/**
 * @brief Creates and sets the quantization parameters used for casting data types.
 *
 * @param param[out] Pointer to the quantization parameters defined in ::cnrtQuantizedParam_t.
 * @param pos[in] The position factor used in quantization.
 * @param scale[in] The scale factor used in quantization.
 * @param offset[in] The offset factor used in quantization.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INIT This function call failed because ``param`` is NULL or not enough
 * memory can be allocated.
 *
 * @details Creates the quantization parameters pos, scale, and offset, and sets the
 * values to these parameters. Returns in param the quantization parameters used for casting
 * data types. For more information about quantization, see "CNRT User Guide".
 *
 * @note
 * - After this function is called, the output pointer ``param`` is used in the ::cnrtCastDataType
 * or ::cnrtTransOrderAndCast function to cast data type.
 * - The ::cnrtDestroyQuantizedParam function should be called to release the memory space when
 * ``param`` is no longer needed.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtCastDataType function.
 *
 */
extern CNRT_DLL_API cnrtRet_t cnrtCreateQuantizedParam(cnrtQuantizedParam_t *param,
                                                       int pos,
                                                       float scale,
                                                       int offset);

/**
 * @brief Creates and sets the quantization parameters used for casting data types. This function
 * is used to quantize data by channel.
 *
 * @param param[out] Pointer to the quantization parameters defined in ::cnrtQuantizedParam_t.
 * @param poses[in] The position factor used in quantization.
 * @param scales[in] The scale factor used in quantization.
 * @param offsets[in] The offset factor used in quantization.
 * @param dimNum[in] The number of dimensions of the filter data to be quantized.
 * @param dimValues[in] The number of elements for each dimension of the filter data to quantized.
 * @param channelDim[in] The dimension index of the channel in the filter data layout.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INIT This function call failed because ``param``, ``poses``, ``scales``,
 * ``offsets`` or ``dimValues`` is NULL, or not enough memory can be allocated.
 *
 * @details Creates quantization parameters "poses", "scales" and "offsets", and sets the
 * values to these parameters. Returns in "param" the quantization parameters defined in
 * cnrtQuantizedParam_t used for casting data type. \n
 * This function is used to quantize filter data
 * by channel. The data is divided into groups based on the number of elements of the channel
 * dimension and quantized filter data for each group. The ::cnrtCreateQuantizedParam function
 * quantizes data without division, and usually for input or output data. \n
 * Compared with the
 * ::cnrtCreateQuantizedParam function, this function has a higher precision quantization. For more
 * information about quantization, see "CNRT User Guide".
 *
 * @note
 * - After this function is called, the output pointer ``param`` is used in the ::cnrtCastDataType
 * or ::cnrtTransOrderAndCast function to cast data type.
 * - The ::cnrtDestroyQuantizedParam function should be called to release the memory space when
 * ``param`` is no longer needed.
 * - This function is to set parameters for quantization by channel, so the number of ``scales``,
 * ``poses``, ``offsets`` should be the same as the number of channel elements.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 @verbatim
 int main() {
   // Initialize quantization parameter
   cnrtQuantizedParam_t param = NULL;
   // Set parameters
   #define channel_num 3
   int poses[channel_num] = {1, 2, 3};
   float scales[channel_num] = {1.0, 1.5, 1.2};
   float offsets[channel_num] = {0.0, 0.5, -0.5};
   // Dimensions of data which is to be quantized
   int dim_num = 4;
   // Define the number of elements of the channel dimension
   int channel_dim = 0;
   // Set the value of each dimension of data
   int dim_values[4] = {channel_num, 1, 16, 16};

   cnrtCreateQuantizedParamByChannel(&param, poses, scales, offsets, dim_num, dim_values,
 channel_dim);

   // Prepare for input and output
   int num = channel_num * 16 * 16;
   // Allocate memory for input and output
   void *input = (void *)malloc(sizeof(float) * num);
   void *output = (void *)malloc(sizeof(int16_t) * num);
   // Set value for input

   cnrtCastDataType(input, CNRT_FLOAT32, output, CNRT_INT16, num, param);

   // Release releated resoureces
   free(input);
   free(output);
   cnrtDestroyQuantizedParam(param);
   return 0;
 }
 @endverbatim
 *
 */
extern CNRT_DLL_API cnrtRet_t cnrtCreateQuantizedParamByChannel(cnrtQuantizedParam_t *param,
                                                                int *poses,
                                                                float *scales,
                                                                float *offsets,
                                                                int dimNum,
                                                                int *dimValues,
                                                                int channelDim);

/**
 * @brief Destroys the quantization parameters and releases the memory resources.
 *
 * @param param[in] Pointer to the quantization parameters defined in ::cnrtQuantizedParam_t.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because the input pointer ``param`` is
 * NULL.
 *
 * @details Destroys the quantization parameters ``param`` and releases the parameter resources.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - See example in ::cnrtCastDataType function.
 *
 */
extern CNRT_DLL_API cnrtRet_t cnrtDestroyQuantizedParam(cnrtQuantizedParam_t param);

/**
 * @brief Converts the data into another data type.
 *
 * @param src_addr[in] Pointer to the input data to be converted.
 * @param src_data_type[in] The data type of the input data to be converted. The data type is
 * defined in ::cnrtDataType_t.
 * @param dst_addr[out] Pointer to the converted output data.
 * @param dst_data_type[in] The data type of the data to be converted into. The data type is defined
 * in ::cnrtDataType_t.
 * @param data_num[in] The number of data to be converted.
 * @param param[in] Pointer to the quantization parameters defined in ::cnrtQuantizedParam_t. You
 * can create and set quantization parameters via the ::cnrtCreateQuantizedParam or
 * ::cnrtCreateQuantizedParamByChannel function. If you do not want to quantize data, set this
 * parameter to NULL.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_UNSUPPORTED This function call failed because the data types set in
 * ``src_data_type`` and ``dst_data_type`` are not in the supported combination. See ``Note``
 section
 * below for details.
 *
 * @details Converts data pointed by ``src_addr`` in ``src_data_type`` data type into the
 * ``dst_data_type`` data type with data quantization if the quantization parameter ``param`` is
 * not set to NULL. Returns the converted data in ``dst_addr``.
 *
 * @note
 * - If the input pointer ``param`` is set to NULL, the data will not be quantized.
 *
 *   The supported combinations of the data types are as follows. The data types are shown in
 *   the following order:
 *
 *    ``src_data_type``-``dst_data_type``
 *
 *   Supported combinations are:
 *
 *    float32-float16, float32-uint8, int64-float16, float16-float32, float16-uint8,
 *    uint8-float32, uint8-float16, float32-float32
 *
 * - If the input pointer ``param`` is not set to NULL, the data will be quantized.
 *
 *   The supported combinations of tensor data types are as follows. The data types are shown in the
 *   following order:
 *
 *    ``src_data_type``-``dst_data_type``
 *
 *   Supported combinations are:
 *
 *    float32-float16, float32-int16, float32-int8, float32-int32, int32-float32,
 *    float16-int16, int16-float32, int8-float32, float32-float32
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 @verbatim
 int main() {
   // Initialize quantized param, and set position = 1, scale = 1.2, offset = 0
   cnrtQuantizedParam_t param = NULL;
   cnrtCreateQuantizedParam(&param, 1, 1.2, 0);

   // Prepare for input and output
   int num = 500;
   // Allocate memory for input and output
   void *input = (void *)malloc(sizeof(float) * num);
   void *output = (void *)malloc(sizeof(int16_t) * num);
   //Set value to input

   cnrtCastDataType(input, CNRT_FLOAT32, output, CNRT_INT16, num, param);

   // Release related resources
   free(input);
   free(output);
   cnrtDestroyQuantizedParam(param);
 @endverbatim
 *
 */
extern CNRT_DLL_API cnrtRet_t cnrtCastDataType(void *src_addr,
                                               cnrtDataType_t src_data_type,
                                               void *dst_addr,
                                               cnrtDataType_t dst_data_type,
                                               int data_num,
                                               cnrtQuantizedParam_t param);

/**
 * @brief Adds strides to the input data.
 *
 * @param src_addr[in] Pointer to the memory address of the input data to be strided.
 * @param data_type[in] Pointer to the data type of the input and output data defined in
 * ::cnrtDataType_t.
 * @param dst_addr[out] Pointer to the memory address of the strided output data.
 * @param dimNum[in] The number of dimensions of the input data to be strided.
 * @param dimValues[in] The number of elements for each dimension of the input data.
 * @param dimStride[in] An array that stores the number of elements to be added for each dimension
 * of input data.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID_POINTER This function call failed because ``src_addr``,
 * ``dst_addr``,
 * ``dimValues`` or ``dimStride`` is NULL.
 * @retval CNRT_RET_ERR_INVALID This function call failed because ``dimNum`` is a negative value.
 * @retval CNRT_RET_ERR_SYS This function call failed because not enough memory can be allocated to
 * store intermidiate computing results.
 *
 * @details Adds strides to the input pointed by "src_addr" with the stride of the sliding window
 * dimStride for each dimension, and returns the strided output data in "dst_addr". Zero is padded
 * at the end of each dimension. You need to specify the data type, the number of dimensions,
 * and the number of elements for each dimension of the input in "data_type", "dimNum", and
 * "dimValues"
 * accordingly. \n
 * The resulting output is computed with the following formula: \n
 *  output_shape = dimNum + dimStride \n
 * For each dimension, the number of elements of the output equals the number of elements of input
 * set in dimNum plus the number of elements to be strided set in dimStride.
 *
 * @note
 * - The number of dimensions set in ``dimValues`` and ``dimStride`` should be the same as
 * ``dimNum``.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
extern CNRT_DLL_API cnrtRet_t cnrtAddDataStride(void *src_addr,
                                                cnrtDataType_t data_type,
                                                void *dst_addr,
                                                int dimNum,
                                                int *dimValues,
                                                int *dimStride);

/**
 * @brief Reorders N-dimensional data to the specified dimension order.
 *
 * @param src_addr[in] Pointer to the memory address of the input data to be reordered.
 * @param data_type[in] Pointer to the data type of the input and output data defined in
 * ::cnrtDataType_t.
 * Supported data_types are CNRT_FLOAT32, CNRT_FLOAT16, CNRT_INT32, CNRT_INT16, CNRT_INT8,
 * CNRT_UINT16, and CNRT_UINT8.
 * @param dst_addr[out] Pointer to the memory address of the reordered output data.
 * @param dimNum[in] The number of dimensions of the input and output data.
 * @param dimValues[in] The number of elements for each dimension of the input data.
 * @param dimOrder[in] The order of dimensions to be reordered to.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_SYS This function call failed because not enough memory can be allocated to
 * store intermidiate computing results.
 * @retval CNRT_RET_ERR_UNSUPPORTED This function call failed because the value of ``data_type`` is
 * invalid.
 * @retval CNRT_RET_ERR_INVALID This function call failed because the number of dimensions set in
 * ``dimOrder`` is not the same as the number of dimensions set in ``dimNum``.
 *
 * @details Reorders the dimensions of input data pointed by ``src_addr`` to the dimension order
 * ``dimOrder``, and returns the reordered output data in ``dst_addr``. You need to specify the
 * total number of dimensions, the data type, and the number of elements for each dimension of the
 * input data in ``dimNum``, ``data_type``, and ``dimValues`` accordingly.
 *
 * @note
 * - The number of dimensions set in ``dimOrder`` should be the same as ``dimNum``.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 *
 */
extern CNRT_DLL_API cnrtRet_t cnrtTransDataOrder(void *src_addr,
                                                 cnrtDataType_t data_type,
                                                 void *dst_addr,
                                                 int dimNum,
                                                 int dimValues[],
                                                 int dimOrder[]);

/**
 * @brief Reorders the dimensions and converts the data type for the N-dimensional data.
 *
 * @param src_addr[in] Pointer to the memory address of the input data to be reordered and
 * converted.
 * @param src_type[in] Pointer to the data type of the input data to be converted. The data type is
 * defined in ::cnrtDataType_t.
 * @param dst_addr[out] Pointer to the memory address of the output data that is reordered the
 * dimensions and converted the data type.
 * @param dst_type[in] The data type of the input data to be converted into. The data type is
 * defined in ::cnrtDataType_t.
 * @param param[in] Pointer to the quantization parameters defined in ::cnrtQuantizedParam_t. You
 * can create and set quantization parameters via the ::cnrtCreateQuantizedParam or
 * ::cnrtCreateQuantizedParamByChannel function. If you do not want to quantize data, set this
 * parameter to NULL.
 *
 * @param dimNum[in] The number of dimensions of the input data to be reordered.
 * @param dimValues[in] The number of elements for each dimension of the input data.
 * @param dimOrder[in] The order of dimensions to be reordered to.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_UNSUPPORTED This function call failed because the data types set in
 * ``src_data_type`` and ``dst_data_type`` are not in the supported combination. See ``Note``
 * section below for details.
 * @retval CNRT_RET_ERR_INVALID This function call failed because the number of dimensions set
 * in ``dimOrder`` is not the same as the number of dimensions set in ``dimNum``.
 * @retval CNRT_RET_ERR_SYS This function call failed because not enough memory can be allocated to
 * store intermediate computing results.
 *
 * @details Reorders the dimensions of input data to the dimension order "dimOrder", and converts
 * input data in data type src_type into the data type dst_type. If
 * the quantization parameter param is not set to NULL, data quantization is also performed.
 * Returns in dst_addr the data with the
 * reordered dimensions and converted data type. The input data is pointed by src_addr. You need
 * to specify the total number of dimensions and the number of elements for each dimension of input
 * data in "dimNum" and "dimValues" accordingly. \n
 * This function is used to reorder dimensions and convert data types on a single call. To reorder
 * dimensions and convert data types separately, call the ::cnrtTransDataOrder and
 * ::cnrtCastDataType functions respectively.
 *
 *
 * @note
 * - The number of dimensions set in ``dimOrder`` should be the same as the number of dimensions set
 * in ``dimNum``.
 * - If the input pointer param is set to NULL, the data will not be quantized.
 *
 *   The supported combinations of the data types are as follows. The data types are shown in
 *   the following order:
 *
 *    ``src_data_type``-``dst_data_type``
 *
 *   Supported combinations are:
 *
 *   float32-float16, float32-uint8, int64-float16, float16-float32, float16-uint8, uint8-float32,
 *   uint8-float16, float32-float32
 *
 * - If the input pointer param is not set to NULL, the data will be quantized.
 *
 *   The supported combinations of the data types are as follows. The data types are shown in
 *   the following order:
 *
 *    ``src_data_type``-``dst_data_type``
 *
 *   Supported combinations are:
 *
 *   float32-float16, float32-int16, float32-int8, float32-int32, int32-float32, float16-int16,
 *   int16-float32, int8-float32, float32-float32
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 *
 */
extern CNRT_DLL_API cnrtRet_t cnrtTransOrderAndCast(void *src_addr,
                                                    cnrtDataType_t src_type,
                                                    void *dst_addr,
                                                    cnrtDataType_t dst_type,
                                                    cnrtQuantizedParam_t param,
                                                    int dimNum,
                                                    int dimValues[],
                                                    int dimOrder[]);
/**
 * @brief Reorders the elements of the channel dimension of the input data.
 *
 * @param src_addr[in] Pointer to the memory address of the input data to be reordered.
 * @param dst_addr[out] Pointer to the memory address of the reordered output data.
 * @param data_type[in] Pointer to the data type of the input and output data defined in
 * ::cnrtDataType_t.
 * Supported data types are CNRT_FLOAT32, CNRT_FLOAT16, CNRT_INT32, CNRT_INT16, CNRT_INT8,
 * CNRT_UINT16, and CNRT_UINT8.
 * @param dim_order[in] The dimension order of input and output data defined in ::cnrtDimOrder_t.
 * The CNRT_TNC and CNRT_NTC values in ::cnrtDimOrder_t are not supported in this parameter.
 * @param dimValues[in] The number of elements for each dimension of the input data. The value you
 * set in this parameter should correspond to the dimension order you set in the dim_order
 * parameter.
 * @param colorOrder[in] The order of elements in the channel dimension to be reordered to.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID This function call failed because one of the following errors
 * occurred:
 * - src_addr, dst_addr, dimValues, colorOrder is NULL.
 * - dimValues is set to a negative number.
 * - The total size of the data is too large. The total size of data should be less than INT_MAX.
 * @retval CNRT_RET_ERR_UNSUPPORTED This function call failed because one of the following errors
 * occurred:
 * - Invalid value of the dim_order parameter.
 * - Invalid number of elements of the channel dimension.
 * - Unsupported data type set in the data_type parameter.
 * @retval CNRT_RET_ERR_NOMEM This function call failed because not enough memory can be allocated.
 *
 * @details Reorders the elements of the channel dimension of the input data pointed by ``src_addr``
 * to the element order ``colorOrder``. For example, change the element order from ARGB to RGBA, or
 * from RGB to BGR. Returns the reordered data in ``dst_addr``. You need to specify the number of
 * elements for each dimension, the data type, and the dimension order for the data in
 * ``dimValues``,
 * ``data_type``, and ``dim_order`` accordingly.
 *
 * @note
 * - The number of elements of the channel dimension can only be 3 or 4.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 *
 */
extern CNRT_DLL_API cnrtRet_t cnrtTransDataColorOrder(void *src_addr,
                                                      void *dst_addr,
                                                      cnrtDataType_t data_type,
                                                      cnrtDimOrder_t dim_order,
                                                      int dimValues[],
                                                      int colorOrder[]);

/**
 * @brief Retrieves the size of input data from a CNRT Function.
 *
 * @param sizeArray[out] Pointer to an array that consists of the size of each input data.
 * @param num[out] The number of input data to be retrieved in a CNRT Function.
 * @param function[in] Pointer to a CNRT Function that holds the input data to be retrieved. The
 * CNRT Function is extracted from an offline model file via the ::cnrtExtractFunction function.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID_POINTER This function call failed because ``function`` is NULL.
 * @retval CNRT_RET_ERR_INVALID This function call failed because the shape of the input data should
 * be immutable.
 * @retval CNRT_RET_ERR_NO_EXIST This function call failed because the input pointer ``function`` is
 * not initialized via the ::cnrtExtractFunction function.
 *
 * @details Returns the size of each input data in ``sizeArray`` and the number of input data in
 * ``num`` based on the given CNRT Function ``function``. You can get the memory size to be
 * allocated for the input data via this function.
 *
 * @note
 * - This function is only used for immutable shape of tensors. To retrieve the size of input data
 * for mutable shape of tensors, call the ::cnrtInferFunctionOutputShape function.
 * - The ``num`` and ``sizeArray`` will become invalid if the CNRT Function ``function`` is
 * released.
 * - You do not need to free the memory for ``sizeArray`` parameter after usage.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 *
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetInputDataSize(int64_t **sizeArray,
                                                   int *num,
                                                   cnrtFunction_t function);

/**
 * @brief Retrieves the size of output data from a CNRT Function.
 *
 * @param sizeArray[out] Pointer to an array that consists of the size of each output data.
 * @param num[out] The number of output data to be retrieved in a CNRT Function.
 * @param function[in] Pointer to a CNRT Function that holds the output data to be retrieved. The
 * CNRT Function is extracted from an offline model file via the ::cnrtExtractFunction function.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID_POINTER This function call failed because ``function`` is NULL.
 * @retval CNRT_RET_ERR_INVALID This function call failed because the shape of the output data
 * should be immutable.
 * @retval CNRT_RET_ERR_NO_EXIST This function call failed because the input pointer ``function`` is
 * not initialized via the ::cnrtExtractFunction function.
 *
 * @details Returns the size of each output data in ``sizeArray`` and the number of output data in
 * ``num`` based on the given CNRT Function ``function``. You can get the memory size to be
 * allocated for the input data via this function.
 *
 * @note
 * - This function is only used for immutable shape of tensors. To retrieve the size of input data
 * for mutable shape of tensors, call the ::cnrtInferFunctionOutputShape function.
 * - The ``num`` and ``sizeArray`` will become invalid if the CNRT Function ``function`` is
 * released.
 * - You do not need to free the memory for ``sizeArray`` parameter after usage.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 *
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetOutputDataSize(int64_t **sizeArray,
                                                    int *num,
                                                    cnrtFunction_t function);

/**
 * @brief Retrieves the data type of input data from a CNRT Function.
 *
 * @param dtype[out] Pointer to an array that consists of the data type of each input data.
 * @param num[out] The number of input data to be retrieved in a CNRT Function.
 * @param function[in] Pointer to a CNRT Function that holds the input data to be retrieved. The
 * CNRT Function is extracted from an offline model file via the ::cnrtExtractFunction function.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID_POINTER This function call failed because ``function`` is NULL.
 * @retval CNRT_RET_ERR_INVALID This function call failed because the shape of the output data
 * should be immutable.
 * @retval CNRT_RET_ERR_NO_EXIST This function call failed because the input pointer ``function`` is
 * not initialized via the ::cnrtExtractFunction function.
 *
 * @details Returns the data type of a set of input data in ``dtype`` and the number of input data
 * to be retrieved in ``num`` based on the given CNRT Function ``function``.
 *
 * @note
 * - This function is only used for immutable shape of tensors. To retrieve the size of input data
 * for mutable shape of tensors, call the ::cnrtInferFunctionOutputShape or
 * ::cnrtGetSupportedParamDataTypeByName function.
 *
 * - The ``num`` and ``dtype`` will become invalid if the CNRT Function ``function`` is released
 * after calling this function.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 *
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetInputDataType(cnrtDataType_t **dtype,
                                                   int *num,
                                                   cnrtFunction_t function);

/**
 * @brief Retrieves the data type of output data from a CNRT Function.
 *
 * @param dtype[out] Pointer to an array that consists of the data type of each output data.
 * @param num[out] The number of output data to be retrieved in a CNRT Function.
 * @param function[in] Pointer to a CNRT Function that holds the output data to be retrieved. The
 * CNRT Function is extracted from an offline model file via the ::cnrtExtractFunction function.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID_POINTER This function call failed because ``function`` is NULL.
 * @retval CNRT_RET_ERR_INVALID This function call failed because the shape of the output data
 * should be immutable.
 * @retval CNRT_RET_ERR_NO_EXIST This function call failed because the input pointer ``function`` is
 * not initialized via the ::cnrtExtractFunction function.
 *
 * @details Returns the data type of a set of output data in ``dtype`` and the number of output data
 * to be retrieved in ``num`` based on the given CNRT Function ``function``.
 *
 * @note
 * - This function is only used for immutable shape of tensors. To retrieve the size of output data
 * for mutable shape of tensors, call the ::cnrtInferFunctionOutputShape or
 * ::cnrtGetSupportedParamDataTypeByName function.
 *
 * - The ``num`` and ``dtype`` will become invalid if the CNRT Function ``function`` is released
 * after calling this function.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 *
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetOutputDataType(cnrtDataType_t **dtype,
                                                    int *num,
                                                    cnrtFunction_t function);

/**
 * @brief Retrieves the shape of the specified input data from a CNRT Function.
 *
 * @param dimValues[out] Pointer to an array that consists of the shape of the specified input data.
 * @param dimNum[out] The number of dimensions of the input data to be retrieved in a CNRT Function.
 * @param index[in] The index that specifies which input data in the CNRT Function to be retrieved.
 * You can call the ::cnrtGetInputDataSize function to get the total number of input data in CNRT
 * Function. The value is in the range [0, input_num-1].
 * @param function[in] Pointer to a CNRT Function that holds the input data to be retrieved. The
 * CNRT Function is extracted from an offline model file via the ::cnrtExtractFunction function.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID_POINTER This function call failed because ``function``,
 * ``dimValues``, or ``dimNum`` is NULL.
 * @retval CNRT_RET_ERR_INVALID This function call failed because the shape of the output data
 * should be immutable.
 * @retval CNRT_RET_ERR_NO_EXIST This function call failed because the input pointer ``function`` is
 * not initialized via the ::cnrtExtractFunction function or the index exceeds the total number of
 * input data in the CNRT Function.
 *
 * @details Returns the shape and the number of dimensions of the specified input data in
 * ``dimValues`` and ``num`` based on the given CNRT Function function. You need to specify the
 * input data to be retrieved in the CNRT Function in ``index``. To retrieve all the input data in
 * the CNRT Function, you need to call this function for each input data.
 *
 * @note
 * - This function is only used for immutable shape of tensors. To retrieve the size of input data
 * for mutable shape of tensors, call the ::cnrtGetSupportedParamShapeByName function.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 *
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetInputDataShape(int **dimValues,
                                                    int *dimNum,
                                                    int index,
                                                    cnrtFunction_t function);

/**
 * @brief Retrieves the shape of the specified output data from a CNRT Function.
 *
 * @param dimValues[out] Pointer to an array that consists of the shape of the specified output
 * data.
 * @param dimNum[out] The number of dimensions of the output data to be retrieved in a CNRT
 * Function.
 * @param index[in] The index that specifies which output data in the CNRT Function to be retrieved.
 * You can call the ::cnrtGetInputDataSize function to get the total number of output data in CNRT
 * Function. The value is in the range [0, output_num-1].
 * @param function[in] Pointer to a CNRT Function that holds the output data to be retrieved. The
 * CNRT Function is extracted from an offline model file via the ::cnrtExtractFunction function.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_INVALID_POINTER This function call failed because ``function``,
 * ``dimValues``, or ``dimNum`` is NULL.
 * @retval CNRT_RET_ERR_INVALID This function call failed because the shape of the output data
 * should be immutable.
 * @retval CNRT_RET_ERR_NO_EXIST This function call failed because the input pointer ``function`` is
 * not initialized via the ::cnrtExtractFunction function or the index exceeds the total number of
 * output data in the CNRT Function.
 *
 * @details Returns the shape and the number of dimensions of the specified output data in
 * ``dimValues`` and ``num`` based on the given CNRT Function function. You need to specify the
 * output data to be retrieved in the CNRT Function in ``index``. To retrieve all the output data in
 * the CNRT Function, you need to call this function for each output data.
 *
 * @note
 * - This function is only used for immutable shape of tensors. To retrieve the size of output data
 * for mutable shape of tensors, call the ::cnrtGetSupportedParamShapeByName function.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 *
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetOutputDataShape(int **dimValues,
                                                     int *dimNum,
                                                     int index,
                                                     cnrtFunction_t function);
/**
 * @brief Retrieves the shape, size, and data type of output data from a CNRT Function with the
 * specified shape of input data.
 *
 * @param func[in] Pointer to a CNRT Function that holds the input and output data. The CNRT
 * Function is extracted from an offline model file via the ::cnrtExtractFunction function.
 * @param input_num[in] The number of input data in a CNRT Function.
 * @param input_params[in,out] Pointer to the description of the input data including shape, size,
 * and data type in a CNRT Function. Only the shape of the input data is used as the input to get
 * the shape, size, and data type of the output data from the CNRT Function. The size and data type
 * of the input data in the CNRT Function are also returned to this parameter.
 * @param output_num[in] The number of output data in a CNRT Function. The value of this parameter
 * should be the same as the number of input data in the CNRT Function. Otherwise,
 * ::CNRT_RET_ERR_INVALID is returned.
 * @param output_params[out] Pointer to the returned description of the output data including shape,
 * size, and data type.
 * @retval CNRT_RET_SUCCESS This function has run successfully.
 * @retval CNRT_RET_ERR_FUNC_CALL This function call failed because the CNRT Function func is
 * extracted from an old version of offline model that is not supported in this CNRT version, or the
 * shape of the input data is immutable.
 * @retval CNRT_RET_ERR_INVALID This function call failed because one of the following errors
 * occurred:
 * - ``func``, ``input_params``, or ``output_params`` is NULL.
 * - The number of input data set in ``input_num`` mismatched the number of data in CNRT Function
 * ``func``.
 * - The number of output data set in ``output_num`` mismatched the number of data in CNRT Function
 * ``func``.
 *
 * @details Retrieves the shape, size, and data type of output data in ``output_params`` from a CNRT
 * Function ``func`` with the given shape ``input_params`` of the input data. The size and data type
 * of the input data in ``func`` are also returned in ``input_params``. The total number of input
 * data and output data stored in ``func`` should match the values set in ``input_num`` and
 * ``output_num``. Otherwise, ::CNRT_RET_ERR_INVALID is returned. You can get the memory size to be
 * allocated for the input and output data via this function.
 *
 * @note
 * - This function is used for the shape of the input data is mutable. If the shape of the input
 * data is immutable, call the ::cnrtGetInputDataShape, ::cnrtGetInputDataSize,
 * ::cnrtGetInputDataType, ::cnrtGetOutputDataShape, ::cnrtGetOutputDataSize, or
 * ::cnrtGetOutputDataType functions to get data information.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 *
 */
extern CNRT_DLL_API cnrtRet_t cnrtInferFunctionOutputShape(cnrtFunction_t func,
                                                           int input_num,
                                                           cnrtParamDescArray_t input_params,
                                                           int output_num,
                                                           cnrtParamDescArray_t output_params);

#define cnrtDestoryParam cnrtDestroyParam

#if defined(__cplusplus)
}
#endif /*__cplusplus*/
#endif /*__CNRT_H*/
