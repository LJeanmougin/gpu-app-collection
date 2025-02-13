#ifdef __cplusplus
extern "C" {
#endif

//========================================================================================================================================================================================================200
//	INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	COMMON
//======================================================================================================================================================150

#include "../common.h"									// (in the main program folder)	needed to recognized input parameters

//======================================================================================================================================================150
//	UTILITIES
//======================================================================================================================================================150

#include "../util/cuda/cuda.h"							// (in library path specified to compiler)	needed by for device functions
#include "../util/timer/timer.h"						// (in library path specified to compiler)	needed by timer

//======================================================================================================================================================150
//	KERNEL
//======================================================================================================================================================150

#include "./kernel_gpu_cuda_2.cu"						// (in the current directory)	GPU kernel, cannot include with header file because of complications with passing of constant memory variables

//======================================================================================================================================================150
//	HEADER
//======================================================================================================================================================150

#include "./kernel_gpu_cuda_wrapper_2.h"				// (in the current directory)

//========================================================================================================================================================================================================200
//	FUNCTION
//========================================================================================================================================================================================================200

void 
kernel_gpu_cuda_wrapper_2(	knode *knodes,
							long knodes_elem,
							long knodes_mem,

							int order,
							long maxheight,
							int count,

							long *currKnode,
							long *offset,
							long *lastKnode,
							long *offset_2,
							int *start,
							int *end,
							int *recstart,
							int *reclength)
{

	//======================================================================================================================================================150
	//	CPU VARIABLES
	//======================================================================================================================================================150

	// timer
	long long time0;
	long long time1;
	long long time2;
	long long time3;
	long long time4;
	long long time5;
	long long time6;

	time0 = get_time();

	//======================================================================================================================================================150
	//	GPU SETUP
	//======================================================================================================================================================150

	//====================================================================================================100
	//	INITIAL DRIVER OVERHEAD
	//====================================================================================================100

	cudaThreadSynchronize();

	//====================================================================================================100
	//	EXECUTION PARAMETERS
	//====================================================================================================100

	int numBlocks;
	numBlocks = count;
	int threadsPerBlock;
	threadsPerBlock = order < 1024 ? order : 1024;

	printf("# of blocks = %d, # of threads/block = %d (ensure that device can handle)\n", numBlocks, threadsPerBlock);

	time1 = get_time();

	//======================================================================================================================================================150
	//	GPU MEMORY				MALLOC
	//======================================================================================================================================================150

	//====================================================================================================100
	//	DEVICE IN
	//====================================================================================================100

	//==================================================50
	//	knodesD
	//==================================================50

	knode *knodesD;
	cudaMalloc((void**)&knodesD, knodes_mem);
	checkCUDAError("cudaMalloc  recordsD");

	//==================================================50
	//	currKnodeD
	//==================================================50

	long *currKnodeD;
	cudaMalloc((void**)&currKnodeD, count*sizeof(long));
	checkCUDAError("cudaMalloc  currKnodeD");

	//==================================================50
	//	offsetD
	//==================================================50

	long *offsetD;
	cudaMalloc((void**)&offsetD, count*sizeof(long));
	checkCUDAError("cudaMalloc  offsetD");

	//==================================================50
	//	lastKnodeD
	//==================================================50

	long *lastKnodeD;
	cudaMalloc((void**)&lastKnodeD, count*sizeof(long));
	checkCUDAError("cudaMalloc  lastKnodeD");

	//==================================================50
	//	offset_2D
	//==================================================50

	long *offset_2D;
	cudaMalloc((void**)&offset_2D, count*sizeof(long));
	checkCUDAError("cudaMalloc  offset_2D");

	//==================================================50
	//	startD
	//==================================================50

	int *startD;
	cudaMalloc((void**)&startD, count*sizeof(int));
	checkCUDAError("cudaMalloc startD");

	//==================================================50
	//	endD
	//==================================================50

	int *endD;
	cudaMalloc((void**)&endD, count*sizeof(int));
	checkCUDAError("cudaMalloc endD");

	//====================================================================================================100
	//	DEVICE IN/OUT
	//====================================================================================================100

	//==================================================50
	//	ansDStart
	//==================================================50

	int *ansDStart;
	cudaMalloc((void**)&ansDStart, count*sizeof(int));
	checkCUDAError("cudaMalloc ansDStart");

	//==================================================50
	//	ansDLength
	//==================================================50

	int *ansDLength;
	cudaMalloc((void**)&ansDLength, count*sizeof(int));
	checkCUDAError("cudaMalloc ansDLength");

	time2 = get_time();

	//======================================================================================================================================================150
	//	GPU MEMORY			COPY
	//======================================================================================================================================================150

	//====================================================================================================100
	//	DEVICE IN
	//====================================================================================================100

	//==================================================50
	//	knodesD
	//==================================================50

	cudaMemcpy(knodesD, knodes, knodes_mem, cudaMemcpyHostToDevice);
	checkCUDAError("cudaMalloc cudaMemcpy memD");

	//==================================================50
	//	currKnodeD
	//==================================================50

	cudaMemcpy(currKnodeD, currKnode, count*sizeof(long), cudaMemcpyHostToDevice);
	checkCUDAError("cudaMalloc cudaMemcpy currKnodeD");

	//==================================================50
	//	offsetD
	//==================================================50

	cudaMemcpy(offsetD, offset, count*sizeof(long), cudaMemcpyHostToDevice);
	checkCUDAError("cudaMalloc cudaMemcpy offsetD");

	//==================================================50
	//	lastKnodeD
	//==================================================50

	cudaMemcpy(lastKnodeD, lastKnode, count*sizeof(long), cudaMemcpyHostToDevice);
	checkCUDAError("cudaMalloc cudaMemcpy lastKnodeD");

	//==================================================50
	//	offset_2D
	//==================================================50

	cudaMemcpy(offset_2D, offset_2, count*sizeof(long), cudaMemcpyHostToDevice);
	checkCUDAError("cudaMalloc cudaMemcpy offset_2D");

	//==================================================50
	//	startD
	//==================================================50

	cudaMemcpy(startD, start, count*sizeof(int), cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpy startD");

	//==================================================50
	//	endD
	//==================================================50

	cudaMemcpy(endD, end, count*sizeof(int), cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpy endD");

	//====================================================================================================100
	//	DEVICE IN/OUT
	//====================================================================================================100

	//==================================================50
	//	ansDStart
	//==================================================50

	cudaMemcpy(ansDStart, recstart, count*sizeof(int), cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpy ansDStart");

	//==================================================50
	//	ansDLength
	//==================================================50

	cudaMemcpy(ansDLength, reclength, count*sizeof(int), cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpy ansDLength");

	time3 = get_time();

	//======================================================================================================================================================150
	//	KERNEL
	//======================================================================================================================================================150

	// [GPU] findRangeK kernel
	// findRangeK<<<numBlocks, threadsPerBlock>>>(	maxheight,
	findRangeK<<<1, threadsPerBlock>>>(	maxheight,
												knodesD,
												knodes_elem,

												currKnodeD,
												offsetD,
												lastKnodeD,
												offset_2D,
												startD,
												endD,
												ansDStart,
												ansDLength);
	cudaThreadSynchronize();
	checkCUDAError("findRangeK");

	time4 = get_time();

	//======================================================================================================================================================150
	//	GPU MEMORY			COPY (CONTD.)
	//======================================================================================================================================================150

	//====================================================================================================100
	//	DEVICE IN/OUT
	//====================================================================================================100

	//==================================================50
	//	ansDStart
	//==================================================50

	cudaMemcpy(recstart, ansDStart, count*sizeof(int), cudaMemcpyDeviceToHost);
	checkCUDAError("cudaMemcpy ansDStart");

	//==================================================50
	//	ansDLength
	//==================================================50

	cudaMemcpy(reclength, ansDLength, count*sizeof(int), cudaMemcpyDeviceToHost);
	checkCUDAError("cudaMemcpy ansDLength");

	time5 = get_time();

	//======================================================================================================================================================150
	//	GPU MEMORY DEALLOCATION
	//======================================================================================================================================================150

	cudaFree(knodesD);

	cudaFree(currKnodeD);
	cudaFree(offsetD);
	cudaFree(lastKnodeD);
	cudaFree(offset_2D);
	cudaFree(startD);
	cudaFree(endD);
	cudaFree(ansDStart);
	cudaFree(ansDLength);

	time6 = get_time();

	//======================================================================================================================================================150
	//	DISPLAY TIMING
	//======================================================================================================================================================150

	printf("Time spent in different stages of GPU_CUDA KERNEL:\n");

	printf("%15.12f s, %15.12f % : GPU: SET DEVICE / DRIVER INIT\n",	(float) (time1-time0) / 1000000, (float) (time1-time0) / (float) (time6-time0) * 100);
	printf("%15.12f s, %15.12f % : GPU MEM: ALO\n", 					(float) (time2-time1) / 1000000, (float) (time2-time1) / (float) (time6-time0) * 100);
	printf("%15.12f s, %15.12f % : GPU MEM: COPY IN\n",					(float) (time3-time2) / 1000000, (float) (time3-time2) / (float) (time6-time0) * 100);

	printf("%15.12f s, %15.12f % : GPU: KERNEL\n",						(float) (time4-time3) / 1000000, (float) (time4-time3) / (float) (time6-time0) * 100);

	printf("%15.12f s, %15.12f % : GPU MEM: COPY OUT\n",				(float) (time5-time4) / 1000000, (float) (time5-time4) / (float) (time6-time0) * 100);
	printf("%15.12f s, %15.12f % : GPU MEM: FRE\n", 					(float) (time6-time5) / 1000000, (float) (time6-time5) / (float) (time6-time0) * 100);

	printf("Total time:\n");
	printf("%.12f s\n", 												(float) (time6-time0) / 1000000);

}

//========================================================================================================================================================================================================200
//	END
//========================================================================================================================================================================================================200

#ifdef __cplusplus
}
#endif
