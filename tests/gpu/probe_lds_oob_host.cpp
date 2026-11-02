// Minimal host for probe_lds_oob: one 64-lane block, prints the four dwords per lane.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do { hipError_t e_=(x); if(e_!=hipSuccess){std::fprintf(stderr,"%s failed: %s\n",#x,hipGetErrorString(e_)); return 1;} } while(0)
int main(int argc,char**argv){
    if(argc<2){std::fprintf(stderr,"usage: %s kernel.co\n",argv[0]);return 2;}
    void* dout=nullptr; CK(hipMalloc(&dout,64*16)); CK(hipMemset(dout,0xA5,64*16));
    hipModule_t mod; hipFunction_t fn; CK(hipModuleLoad(&mod,argv[1])); CK(hipModuleGetFunction(&fn,mod,"probe_lds_oob"));
    void* p=dout; void* args[]={&p};
    CK(hipModuleLaunchKernel(fn,1,1,1,64,1,1,0,nullptr,args,nullptr)); CK(hipDeviceSynchronize());
    std::vector<unsigned> out(64*4); CK(hipMemcpy(out.data(),dout,64*16,hipMemcpyDeviceToHost));
    int bad=0;
    for(int t=0;t<64;t++){
        unsigned valid=out[4*t], r256=out[4*t+1], r1024=out[4*t+2], r4096=out[4*t+3];
        if(valid!=0x11111111u||r4096!=0x22222222u){ if(bad<8) std::printf("lane %d valid=%08x r256=%08x r1024=%08x r4096=%08x\n",t,valid,r256,r1024,r4096); bad++; }
    }
    std::printf("LDS_OOB lanes=%d bad=%d first: valid=%08x r256=%08x r1024=%08x r4096=%08x\n",64,bad,out[0],out[1],out[2],out[3]);
    std::printf("LDS_OOB_OUTCOME %s\n", bad? "UNEXPECTED":"NO_FAULT_SENTINEL_READBACK");
    return bad?1:0;
}
