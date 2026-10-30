"""Compile and exercise the actual patched NCCL host topology; no GPU required."""
import os
from pathlib import Path
import subprocess
import tempfile

source = Path(os.environ["NCCL_SOURCE_DIR"])
connect = (source / "src/graph/connect.cc").read_text()
host = connect[connect.index("static ncclResult_t getIndexes("):
               connect.index("static ncclResult_t connectCollNet(")]
trees = (source / "src/graph/trees.cc").read_text().replace('#include "nccl.h"', "")
transport = (source / "src/transport/generic.cc").read_text()
transport = transport[transport.index("ncclResult_t ncclTransportTreeConnect("):
                      transport.index("ncclResult_t ncclTransportPatConnect(")]
prefix = r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
enum ncclResult_t { ncclSuccess = 0, ncclInternalError = 3 };
#define NCCL_MAX_TREE_ARITY 3
#define NCCL_ALGO_TREE 0
#define INFO(...) ((void)0)
#define WARN(...) ((void)0)
#define NCCLCHECK(x) do { auto e=(x); if(e!=ncclSuccess) return e; } while(0)
#define NCCLCHECKGOTO(x,r,l) do { r=(x); if(r!=ncclSuccess) goto l; } while(0)
struct ncclTree { int up=-1; int down[3]={-1,-1,-1}; int depth=0; };
struct ncclChannel { ncclTree tree; };
struct Graph {};
struct ncclComm { int nChannels, nNodes, node, nRanks, rank; ncclChannel* channels; Graph graphs[1]; };
static bool enabled=true;
int64_t ncclParamSwitchlessRingOnly() { return enabled; }
int log2i(int n) { return 31-__builtin_clz(n); }
static int calls=0, baseChannels=0;
static int* actualNext=nullptr;
ncclResult_t ncclTransportP2pConnect(ncclComm* comm, int c, int nr, int* recv, int ns, int* send, int) {
  calls++;
  const int* next=actualNext+(c%baseChannels)*4;
  auto check=[&](int peer) {
    if(peer<0) return;
    assert(peer<4 && peer!=comm->rank);
    assert(next[comm->rank]==peer || next[peer]==comm->rank);
  };
  for(int i=0;i<nr;i++) check(recv[i]);
  for(int i=0;i<ns;i++) check(send[i]);
  return ncclSuccess;
}
ncclResult_t ncclTransportP2pSetup(ncclComm*, Graph*, int) { return ncclSuccess; }
"""
main = r"""
int main() {
  int topologies=0, rankCases=0;
  std::array<int,4> permutation={0,1,2,3};
  do {
    for(int channels=1;channels<=4;channels++) {
      int next[16], indexes[16], patterns[4]={};
      ncclTree results[4][8];
      for(int c=0;c<channels;c++) {
        for(int i=0;i<4;i++) {
          next[c*4+permutation[i]]=permutation[(i+(c%2 ? 3:1))%4];
          indexes[c*4+i]=i;
        }
      }
      for(int rank=0;rank<4;rank++) {
        ncclChannel data[8];
        ncclComm comm{channels,4,rank,4,rank,data,{}};
        enabled=true;
        assert(connectTrees(&comm,indexes,indexes,indexes,patterns,next)==ncclSuccess);
        for(int c=0;c<2*channels;c++) results[rank][c]=data[c].tree;
        baseChannels=channels; actualNext=next; calls=0;
        comm.nChannels=2*channels;
        assert(ncclTransportTreeConnect(&comm)==ncclSuccess);
        assert(calls==4*channels);
        rankCases++;
      }
      for(int c=0;c<2*channels;c++) {
        int roots=0, edges=0, root=-1;
        for(int rank=0;rank<4;rank++) {
          const auto& t=results[rank][c];
          assert(t.depth==2 && t.down[2]==-1);
          if(t.up<0) { roots++; root=rank; }
          else {
            const auto& parent=results[t.up][c];
            assert(parent.down[0]==rank || parent.down[1]==rank);
            edges++;
          }
          int degree=t.up>=0;
          for(int child:t.down) if(child>=0) {
            assert(results[child][c].up==rank);
            degree++;
          }
          assert(degree<=2);
        }
        assert(roots==1 && edges==3);
        for(int rank=0;rank<4;rank++) {
          int visited=0, node=rank, depth=0;
          while(node>=0) {
            assert(!(visited&(1<<node))); visited|=1<<node;
            if(results[node][c].up>=0) depth++;
            else assert(node==root);
            node=results[node][c].up;
          }
          assert(depth<=2);
        }
      }
      for(int c=0;c<channels;c++) for(int rank=0;rank<4;rank++) {
        const auto& a=results[rank][c]; const auto& b=results[rank][c+channels];
        assert(!(a.down[0]>=0 && b.down[0]>=0));
      }
      // The disabled feature must still use the actual upstream double tree.
      enabled=false;
      for(int rank=0;rank<4;rank++) {
        ncclChannel data[8]; ncclComm comm{channels,4,rank,4,rank,data,{}};
        assert(connectTrees(&comm,indexes,indexes,indexes,patterns,next)==ncclSuccess);
        int u0,d00,d01,pc0,u1,d10,d11,pc1;
        ncclGetDtree(4,rank,&u0,&d00,&d01,&pc0,&u1,&d10,&d11,&pc1);
        for(int t=0;t<2;t++) for(int c=0;c<channels;c++) {
          const auto& a=data[c+t*channels].tree;
          assert(a.up==(t?u1:u0));
          int x=t?d10:d00, y=t?d11:d01;
          assert((a.down[0]==x && a.down[1]==y) ||
                 (x<0 && a.down[0]==y && a.down[1]<0));
        }
      }
      topologies++;
    }
  } while(std::next_permutation(permutation.begin(),permutation.end()));
  enabled=true;
  for(int bad: {-1,0,4}) {
    int next[4]={bad,2,3,0}, indexes[4]={0,1,2,3}, patterns[4]={};
    ncclChannel data[2]; ncclComm comm{1,4,0,4,0,data,{}};
    assert(connectTrees(&comm,indexes,indexes,indexes,patterns,next)==ncclInternalError);
  }
  {
    int next[4]={1,2,3,1}, indexes[4]={0,1,2,3}, patterns[4]={};
    ncclChannel data[2]; ncclComm comm{1,4,0,4,0,data,{}};
    assert(connectTrees(&comm,indexes,indexes,indexes,patterns,next)==ncclInternalError);
  }
  // Other communicator sizes retain the existing switchless transport skip.
  {
    ncclChannel data[2]; ncclComm comm{1,2,0,2,0,data,{}};
    calls=0; assert(ncclTransportTreeConnect(&comm)==ncclSuccess); assert(calls==0);
  }
  std::printf("{\"topologies\":%d,\"rank_cases\":%d,\"invalid_rings\":4,\"status\":\"CPU_PASS\"}\n",topologies,rankCases);
}
"""
with tempfile.TemporaryDirectory(prefix="nccl-ring-tree-cpu-") as tmp:
    cpp=Path(tmp)/"test.cc"; binary=Path(tmp)/"test"
    cpp.write_text(prefix+trees+host+transport+main)
    subprocess.run(["g++","-std=c++17","-O2",str(cpp),"-o",str(binary)],check=True)
    subprocess.run([str(binary)],check=True)
