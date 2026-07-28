/*
 * feddrl_msg.h — ns-3 시나리오와 Python actor 사이의 셰어드 메모리 wire
 * format. ns3-ai 의 ``Ns3AiMsgInterfaceImpl<EnvStruct, ActStruct>`` 템플릿
 * 으로 사용되는 두 struct (Cpp2Py = Env, Py2Cpp = Act) 를 정의한다.
 *
 * 본 헤더는 C++ 시나리오(``feddrl_scenario.cc``) 와 pybind11 모듈
 * (``feddrl_py.cc``) 양쪽에서 동일하게 include 되어, struct layout 의
 * byte-level 일치를 보장한다. (ctypes 기반의 sim/ns3/bridge.py 도 동일
 * field 순서를 사용한다.)
 *
 * 토폴로지 상수 (16-AP dense 2D OBSS): N_AP=16, N_STA_PER_AP=5, N_LINKS=3.
 */

#ifndef FEDDRL_MSG_H
#define FEDDRL_MSG_H

#include <cstdint>

namespace feddrl
{

constexpr uint32_t kNumAp = 16;
constexpr uint32_t kStaPerAp = 5;
constexpr uint32_t kNumLinks = 3;
constexpr uint32_t kNumSta = kNumAp * kStaPerAp;

// C++ -> Python: ns-3 측이 매 매크로 슬롯에 측정한 obs.
struct EnvMsg
{
    uint32_t queueLen[kNumSta];
    uint32_t holUs[kNumSta];
    uint8_t cbr[kNumAp][kNumLinks];
    uint32_t served[kNumAp];
    uint32_t violation99[kNumAp];
    uint32_t violation999[kNumAp];
    // per-slot MAC-queue drops (aged-out at MaxDelay=deadline999 + queue-full).
    // eq:uhr 의 "decided = served or aged out" 분모를 per-slot 에서 성립시켜
    // dual(Z) 신호의 survivor-bias 를 제거한다: v += dropped, d = served+dropped.
    uint32_t dropped[kNumAp];
    // per-STA-per-link 채널 품질 [0,1] (ns-3 물리 채널이 소유·측정; Python 은
    // 해석식 재유도 없이 이 값을 그대로 학습에 사용). 정적 large-scale fading.
    float csi[kNumSta][kNumLinks];
    uint64_t nowUs;
};

// Python -> C++: 학습된 actor 의 action.
struct ActMsg
{
    int8_t selectedLink[kNumAp][kStaPerAp];  // -1 = inactive
    uint8_t mapMode;  // 0=none, 1=Co-TDMA, 2=Co-OFDMA
    uint8_t reserved[8];
};

}  // namespace feddrl

#endif  // FEDDRL_MSG_H
