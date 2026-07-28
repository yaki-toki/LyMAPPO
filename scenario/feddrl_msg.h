/*
 * feddrl_msg.h — shared-memory wire format between the ns-3 scenario and the
 * Python actor. Defines the two structs (Cpp2Py = Env, Py2Cpp = Act) that are
 * used as ns3-ai's ``Ns3AiMsgInterfaceImpl<EnvStruct, ActStruct>`` template
 * arguments.
 *
 * This header is included identically by the C++ scenario
 * (``feddrl_scenario.cc``) and the pybind11 module (``feddrl_py.cc``), which
 * guarantees byte-level agreement of the struct layout. (The ctypes-based
 * sim/ns3/bridge.py uses the same field order.)
 *
 * Topology constants (16-AP dense 2D OBSS): N_AP=16, N_STA_PER_AP=5, N_LINKS=3.
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

// C++ -> Python: the obs measured by the ns-3 side on each macro slot.
struct EnvMsg
{
    uint32_t queueLen[kNumSta];
    uint32_t holUs[kNumSta];
    uint8_t cbr[kNumAp][kNumLinks];
    uint32_t served[kNumAp];
    uint32_t violation99[kNumAp];
    uint32_t violation999[kNumAp];
    // per-slot MAC-queue drops (aged-out at MaxDelay=deadline999 + queue-full).
    // Makes eq:uhr's "decided = served or aged out" denominator hold per-slot,
    // which removes the survivor bias of the dual(Z) signal: v += dropped,
    // d = served+dropped.
    uint32_t dropped[kNumAp];
    // per-STA-per-link channel quality [0,1] (owned and measured by the ns-3
    // physical channel; Python uses this value for learning as-is, with no
    // analytical re-derivation). Static large-scale fading.
    float csi[kNumSta][kNumLinks];
    uint64_t nowUs;
};

// Python -> C++: the action of the learned actor.
struct ActMsg
{
    int8_t selectedLink[kNumAp][kStaPerAp];  // -1 = inactive
    uint8_t mapMode;  // 0=none, 1=Co-TDMA, 2=Co-OFDMA
    uint8_t reserved[8];
};

}  // namespace feddrl

#endif  // FEDDRL_MSG_H
