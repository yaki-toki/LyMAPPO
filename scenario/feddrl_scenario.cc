/*
 * feddrl_scenario.cc — ns3-ai 통합 4 AP × 5 STA × 3 link 시나리오.
 *
 * Phase-3 fidelity 확장:
 *   - 링크당 독립 PHY/SSID/IP 서브넷 (4ap20sta.cc 와 동일 구조).
 *   - STA 마다 3 개 UDP client 인스턴스 (링크별 송신 큐). ActMsg.selectedLink
 *     이 매 매크로 슬롯에서 활성 client 를 토글 (다른 링크 client 는 즉시
 *     stop, 선택된 링크 client 는 stopTime 을 다음 슬롯 끝까지 연장).
 *   - EnvMsg.cbr[ap][link] 를 PHY busy 상태 trace 로 측정 (PhyState).
 *   - EnvMsg.queueLen[sta] / holUs[sta] 를 STA 의 WifiMacQueue 깊이에서 측정.
 *
 * Python(driver) 가 셰어드 메모리 creator 이고, C++(본 시나리오) 는 attacher
 * 이다 (a-plus-b 의 pattern 과 동일).
 *
 * 빌드:
 *   sim/ns3/build.sh 가 본 디렉토리(feddrl_ai/) 를 ~/ns-3-dev/contrib/ai/
 *   examples/feddrl/ 로 복사하고, contrib/ai 빌드 시 pybind11 모듈
 *   ``ns3ai_feddrl_py`` 와 함께 컴파일된다.
 *
 * 실행 (Python 측):
 *   cd ~/ns-3-dev/contrib/ai/examples/feddrl
 *   python3 feddrl.py --seed=0 --arrivalPps=5500
 */

#include "feddrl_msg.h"

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/propagation-module.h"
#include "ns3/seq-ts-header.h"
#include "ns3/udp-socket-factory.h"
#include "ns3/wifi-module.h"

#include <ns3/ai-module.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <deque>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

using namespace ns3;
using namespace feddrl;

NS_LOG_COMPONENT_DEFINE("FedDrlAiScenario");

namespace {

Time kMacroSlot = MilliSeconds(5);  // RL decision + measurement window (settable)

// UHR deadline pair (settable, density-scaled): a decided packet counts as a
// p99 (resp. p99.9) violation if its delay exceeds these. The canonical 4-AP
// pair is (5ms, 10ms); denser scales relax it (e.g. 16-AP -> (10ms, 20ms))
// because the structural tail floor grows with co-channel AP density.
double g_l99Sec = 5e-3;
double g_l999Sec = 10e-3;
Time kEpisodeDuration = MilliSeconds(180);  // measurement window (settable)
// ns-3 STAs need association time (passive scan, ~100-150ms) that the surrogate
// has no analogue for; a bare 180ms episode drops ~75% of packets sent before
// association completes. Run a WARMUP so all STAs associate BEFORE the window;
// traffic + KPI counting happen only during [kWarmup, kSimEnd).
const Time kWarmup = MilliSeconds(200);
Time kSimEnd = kWarmup + kEpisodeDuration;  // 380ms (recomputed in main)
// P1: stop sending 10ms before sim end so every sent packet reaches a terminal
// state (delivered or dropped) and is counted in the decided denominator.
Time kSendStop = kSimEnd - MilliSeconds(10);

static std::vector<double> g_delaysSec;
// Debug: cumulative per-AP delivered UHR count over the measurement window, to
// test whether per-BSS EDCA actually reallocates airtime (compare the SAME AP
// across a spread vs a uniform run to control for the per-AP load asymmetry).
static std::array<uint64_t, kNumAp> g_apRxTotal{};
// Per-AP KPI over the measurement window (핵심주장 검증용): 네트워크 P1 공식의
// per-AP 버전 p99_a = (late_a + lost_a)/decided_a, lost_a = decided_a - rx_a.
// zq 의 positive-result 지표(worst-AP p999, feasible AP 수)는 이걸로 계산한다.
static std::array<uint64_t, kNumAp> g_apTxTotal{};      // decided (sends) per AP
static std::array<uint64_t, kNumAp> g_apLate99Total{};  // delivered late > L99
static std::array<uint64_t, kNumAp> g_apLate999Total{}; // delivered late > L999
// Debug: AP-side AC_BE EDCAF + per-AP received BACKGROUND count. Background is a
// clean backlogged OnOff on AC_BE with UNIFORM per-band load, so any per-AP
// delivery difference under an AC_BE spread is a pure EDCA effect -> isolates
// whether EDCA reallocates airtime for a standard generator (vs the g_staBuf UHR).
static std::array<std::vector<Ptr<QosTxop>>, kNumAp> g_bgTxop;
static std::array<uint64_t, kNumAp> g_bgRxTotal{};

// Steady-state measurement (settable settleMs, default 0 = disabled, fully
// backward compatible). Traffic + slots still start at kWarmup; KPI counting is
// reset at g_measStart = kWarmup + kSettle so the post-association queue-buildup
// transient is excluded (standard transient removal). At settleMs=0 the reset is
// not scheduled and the baselines stay 0 -> the metric is byte-identical to the
// legacy path.
Time kSettle = MilliSeconds(0);
Time g_measStart = kWarmup;
static uint64_t g_txBase = 0;   // FlowMonitor tx/rx snapshot at g_measStart, so
static uint64_t g_rxBase = 0;   // the end-of-run KPI counts only [g_measStart,end).

// PHY busy time accumulator (per AP × link). Macro-slot delta 로 CBR 산출.
static std::array<std::array<double, kNumLinks>, kNumAp> g_busySecAcc{};
static std::array<std::array<double, kNumLinks>, kNumAp> g_busySecPrev{};
// per-AP 슬롯 내 delivered + deadline 위반 카운트. served 와 violation 을 동일
// OnPacketRx 콜백·동일 apIdx 로 세어 violation ⊆ served (p99 ≤ 1) 를 보장한다.
static std::array<uint32_t, kNumAp> g_slotServed{};
static std::array<uint32_t, kNumAp> g_slotViol99{};
static std::array<uint32_t, kNumAp> g_slotViol999{};
// per-slot MAC-queue drops (Expired = aged-out at MaxDelay=deadline999, +
// DropBeforeEnqueue = queue-full). eq:uhr 의 decided(=served or aged out) 를
// per-slot 에서 성립시켜 dual 신호의 survivor-bias 를 제거한다 (E1 수정).
static std::array<uint32_t, kNumAp> g_slotDropped{};
// 액션 → 트래픽 인과 커플링: SendTick 이 매 송신마다 참조.
static std::array<int8_t, kNumSta> g_selLink;    // 선택 링크 (-1=inactive)
static std::array<bool, kNumAp> g_apActive;      // Co-TDMA: 이번 슬롯 AP 활성
static std::array<uint32_t, kNumSta> g_seq{};    // per-STA seq
static uint32_t g_pktBytes = 1488;
static Time g_sendInterval = MilliSeconds(1);
// Heterogeneous per-AP load (settable loadSpread, default 0 = homogeneous). Each
// STA accrues its AP's multiplier m[ap] each tick and enqueues one arrival when
// the accumulator crosses 1 (deterministic thinning, no burst). m[ap] = 1 -
// loadSpread*(ap/(kNumAp-1)) so AP0 is hottest (full arrivalPps) and AP15 the
// lightest ((1-loadSpread)*arrivalPps) -> heterogeneous per-AP constraint
// pressure (Z), the condition under which constraint-weighted federation (zq)
// could beat independent learning. At loadSpread=0 every m=1 -> one arrival per
// STA per tick -> byte-identical to the homogeneous CBR path.
static std::array<double, kNumAp> g_loadMult;   // per-AP arrival-rate multiplier
static std::array<double, kNumSta> g_sendAcc{}; // per-STA fractional-rate accumulator
// 관측용 per-STA per-link MAC 큐 핸들 (uhrAc on 이면 AC_VI, off 면 AC_BE —
// 송신 AC 와 일치). 디바이스 설치 후 수집; g_staObsTxop[s][l] = 링크 l 의 txop.
static std::array<std::vector<Ptr<QosTxop>>, kNumSta> g_staObsTxop;

// F8 (E3 수정): 공유 상위버퍼 + drain-on-demand — 논문 eq:state 의 "shared
// buffer, HoL routed at SERVICE time" 구현. 도착은 SeqTs(도착시각) 스탬프 후
// 공유버퍼 대기; 선택 링크의 MAC 큐 깊이 < g_drainTarget 일 때만 그 링크로
// 송신 → 라우팅이 drain(service) 시점의 정책을 따르고, 대기 패킷은 정책 변경
// 시 재라우팅된다(MLO 다이버시티). target 은 A-MPDU 집성 효율을 위해 얕지
// 않게(기본 8). 구 g_staBuf drip(arrival-rate pacing → MAC 병목 우회 아티팩트)
// 과 달리 demand-driven 이라 MAC 큐가 진짜 병목으로 유지된다. 비활성/게이트
// AP 의 도착은 드롭이 아니라 대기(P3). g_drainTarget=0 → 공유버퍼 우회
// (direct-send 레거시, A/B 회귀용).
static std::array<std::deque<Ptr<Packet>>, kNumSta> g_shBuf;
static uint32_t g_drainTarget = 8;   // per-link MAC 큐 목표 깊이 (CLI)
Time kDrainTick = MilliSeconds(1);   // 주기 drain (arrival-시점 drain 보완)
// 공유버퍼 aged-out (도착 후 상한 초과) — KPI decided/lost 및 dual 신호에 포함.
static std::array<uint64_t, kNumAp> g_apShDropTotal{};

// --- AI-AC (learning-controlled dedicated UHR access category) --------------
// UHR traffic is placed on a dedicated access category (AC_VI, socket priority
// 5) whose EDCA contention parameters are set PER-BSS at runtime — the learned
// control surface. A saturating best-effort background (AC_BE, priority 0)
// makes the medium contention-limited, so per-BSS UHR-AC airtime priority
// becomes the decisive lever that round-robin (link selection only) cannot
// express. All of this is inert unless --uhr-ac / --bg-pps are set, so with the
// flags off the scenario is byte-identical to the single-class baseline.
// Aggressiveness level -> (CWmin, AIFSN). Level 0 = neutral (== AC_BE params,
// no priority over background); rising levels claw more airtime.
const uint32_t kUhrLevels = 4;
const uint32_t kUhrCwMin[kUhrLevels] = {15, 7, 3, 1};
const uint8_t kUhrAifsn[kUhrLevels] = {3, 2, 2, 1};
static bool g_uhrAcEnabled = false;
static double g_uhrEdcaSpread = 0.0;  // static heuristic strength (validation)
static int g_uhrLevelUniform = -1;    // >=0 overrides ALL APs to this level (test)
static bool g_bgEdca = false;         // also apply per-BSS level to AC_BE (test)
// Background (AC_BE) UDP flows use destination ports >= kBgPort; UHR uses the
// 9000-range. The KPI + settle-baseline filter on this so best-effort
// background never enters the UHR violation-rate statistics.
const uint16_t kBgPort = 20000;
// AP-side AC_VI EDCAF handles per AP (one single-link device per link = 3 per
// AP). Setting these updates the AP's advertised EdcaParameterSet, which the
// BSS's STAs adopt from beacons -> reallocates that BSS's UHR airtime share.
static std::array<std::vector<Ptr<QosTxop>>, kNumAp> g_uhrTxop;
static std::array<uint8_t, kNumAp> g_uhrLevel{};  // current per-AP level
// Debug: STA-side AC_VI EDCAF for AP0-STA0 and AP15-STA0, to observe whether the
// STA actually ADOPTS the AP's advertised UHR-AC params at measurement time.
static std::array<Ptr<QosTxop>, 2> g_dbgStaVi{};

// per-STA-per-link 채널 품질 [0,1] (높을수록 좋음) — F6 (E2 수정): 유한상태
// Markov 채널. 상태 ∈ {0.9, 0.5, 0.3} (nearest-neighbor random walk, 평균 체류
// chanDwellMs). A5 의 "finite-state Markov channel of bounded mixing" 가정을
// 실제로 성립시키고, 학습이 추적할 시간 동역학을 만든다 (구: 완전 정적 →
// 고정 RSSI 맵이 최적 재현 = 학습이 이길 구조 없음). 초기 상태 = 구 정적
// 패턴 (l+s)%3 이라 chanDwellMs=0(전이 없음)이면 byte-identical 레거시.
// shadowing 은 그 STA 로 향하는 전 AP 경로에 대칭 적용(기존 규칙 유지),
// 매 슬롯 env->csi 로 obs 에 실려 Python 이 그대로 학습에 쓴다.
// 상태 레벨: 스텝 ~4dB (extra loss 2/6/10dB). 구 {0.9,0.5,0.3}(스텝 8dB)은
// 전이 직후 IdealWifiManager 가 고정 고MCS 로 전 프레임 실패(피드백 부재→재시도
// 소진→3%+ 손실, L60 전 AP p999>0.05 실증) — rate control 이 따라갈 수 있는
// 폭으로 완화 (동역학·추적가능성은 유지).
const double kCsiLevels[3] = {0.9, 0.7, 0.5};
static std::array<std::array<uint8_t, kNumLinks>, kNumSta> g_chanState;
static double g_chanDwellMs = 200.0;  // 평균 상태 체류(ms); 0 = 정적(레거시)
static bool g_hetBands = true;        // F7: 이질 밴드(2.4/5/6GHz); 0=레거시
static uint32_t g_link2Width = 80;    // link2(6GHz) 폭 MHz: 80(현행) 또는 40
Time kChanTick = MilliSeconds(20);    // 전이 검사 주기
static Ptr<UniformRandomVariable> g_chanRng;
// 채널 갱신 핸들 (setup 에서 채움): per-link Matrix loss + 노드 mobility.
static std::array<Ptr<MatrixPropagationLossModel>, kNumLinks> g_chanLoss{};
static std::array<Ptr<MobilityModel>, kNumAp> g_apMob{};
static std::array<Ptr<MobilityModel>, kNumSta> g_staMob{};

void
InitChanStates()
{
    for (uint32_t s = 0; s < kNumSta; ++s)
    {
        for (uint32_t l = 0; l < kNumLinks; ++l)
        {
            g_chanState[s][l] = static_cast<uint8_t>((l + s) % kNumLinks);
        }
    }
}

double
CsiOf(uint32_t s, uint32_t l)
{
    return kCsiLevels[g_chanState[s][l]];
}

// Markov 전이 + 물리 반영: 전이 확률 p = tick/dwell 로 이웃 상태 이동, 바뀐
// (s,l) 은 그 STA 로 향하는 전 AP 경로의 Matrix loss 를 갱신 (대칭 규칙 유지).
void
ChanTick()
{
    const double p = kChanTick.GetMilliSeconds() / g_chanDwellMs;
    for (uint32_t s = 0; s < kNumSta; ++s)
    {
        for (uint32_t l = 0; l < kNumLinks; ++l)
        {
            if (g_chanRng->GetValue() >= p)
            {
                continue;
            }
            uint8_t st = g_chanState[s][l];
            if (st == 0) { st = 1; }
            else if (st == 2) { st = 1; }
            else { st = (g_chanRng->GetValue() < 0.5) ? 0 : 2; }
            if (st == g_chanState[s][l])
            {
                continue;
            }
            g_chanState[s][l] = st;
            const double extra = (1.0 - kCsiLevels[st]) * 20.0;
            if (g_chanLoss[l] && g_staMob[s])
            {
                for (uint32_t a = 0; a < kNumAp; ++a)
                {
                    if (g_apMob[a])
                    {
                        g_chanLoss[l]->SetLoss(g_apMob[a], g_staMob[s],
                                               extra, true);
                    }
                }
            }
        }
    }
    Simulator::Schedule(kChanTick, &ChanTick);
}

// per-AP allowed link set K_i (canonical asymmetric OBSS, tiled by i%4):
//   AP%4==0,1 -> {0,1} (2.4+5 GHz),  ==2 -> {1,2} (5+6 GHz),  ==3 -> {2} (6 GHz)
// APs share a link (band) only if it is in BOTH sets; with all APs within
// mutual carrier-sense range this makes co-channel contention (OBSS) follow the
// band-sharing graph -> o=(2,2,3,1) at 4 AP, growing with N. Enforced at service
// time so an AP only ever transmits on its allowed bands.
bool
LinkAllowed(uint32_t ap, int link)
{
    static const int sets[4][3] = {
        {1, 1, 0}, {1, 1, 0}, {0, 1, 1}, {0, 0, 1}};
    if (link < 0 || link >= static_cast<int>(kNumLinks))
    {
        return false;
    }
    return sets[ap % 4][link] != 0;
}

// First allowed link for an AP (fallback when a chosen link is out of K_i).
int
FirstAllowedLink(uint32_t ap)
{
    for (int l = 0; l < static_cast<int>(kNumLinks); ++l)
    {
        if (LinkAllowed(ap, l))
        {
            return l;
        }
    }
    return 0;
}

// STA 마다 링크별 UDP client. 선택된 링크의 client 만 활성, 나머지는 stop.
struct PerStaApps
{
    Ptr<Node> node;
    std::array<Ptr<Socket>, kNumLinks> sockPerLink;
};

struct AppHandles
{
    ApplicationContainer servers;
    std::vector<PerStaApps> stas;  // size = kNumSta
};

void
OnPacketRx(uint32_t apIdx,
           Ptr<const Packet> packet,
           const Address& /*src*/,
           const Address& /*dst*/)
{
    SeqTsHeader hdr;
    Ptr<Packet> copy = packet->Copy();
    if (copy->PeekHeader(hdr) > 0)
    {
        const Time delay = Simulator::Now() - hdr.GetTs();
        const double d = delay.GetSeconds();
        g_delaysSec.push_back(d);
        if (apIdx < kNumAp)
        {
            ++g_slotServed[apIdx];  // 이 슬롯에 delivered (= decided) 패킷.
            ++g_apRxTotal[apIdx];   // cumulative (windowed) for airtime debug.
            if (d > g_l99Sec) { ++g_slotViol99[apIdx]; ++g_apLate99Total[apIdx]; }
            if (d > g_l999Sec) { ++g_slotViol999[apIdx]; ++g_apLate999Total[apIdx]; }
        }
    }
}

// 진단 (FEDDRL_DBG_DROP=1): per-링크 × per-사유 MPDU 드롭 카운터 — 6GHz/80MHz
// 링크의 잔여 손실이 재시도 소진인지 수명 만료인지 귀속하기 위한 계측.
// 사유: 0=FAILED_ENQUEUE, 1=EXPIRED_LIFETIME, 2=REACHED_RETRY_LIMIT, 3=QOS_OLD.
static std::array<std::array<uint64_t, 4>, kNumLinks> g_dbgDropByLink{};

void
OnDbgMacDrop(uint32_t linkIdx, WifiMacDropReason reason,
             Ptr<const WifiMpdu> /*mpdu*/)
{
    if (linkIdx < kNumLinks && static_cast<uint8_t>(reason) < 4)
    {
        ++g_dbgDropByLink[linkIdx][static_cast<uint8_t>(reason)];
    }
}

// MAC-queue drop (aged-out at MaxDelay=deadline999, or queue-full). A dropped
// UHR packet missed BOTH deadlines but never reaches OnPacketRx — count it
// per-slot so the dual signal is not survivor-biased (E1). 주의: bgPps>0 이면서
// uhrAc off 인 진단 조합에선 배경도 같은 AC_BE 큐라 드롭이 섞임 — 학습/평가
// 프로토콜(bg=0 또는 uhrAc on)에선 UHR 전용.
void
OnMacDrop(uint32_t apIdx, Ptr<const WifiMpdu> /*mpdu*/)
{
    if (apIdx < kNumAp)
    {
        ++g_slotDropped[apIdx];
    }
}

// Per-AP background delivery counter (AC_BE OnOff), for the EDCA-isolation test.
void
OnBgRx(uint32_t apIdx, Ptr<const Packet> /*packet*/, const Address& /*from*/)
{
    if (apIdx < kNumAp)
    {
        ++g_bgRxTotal[apIdx];
    }
}

// Steady-state transient removal: at g_measStart snapshot the FlowMonitor tx/rx
// baseline and clear the delivered-delay buffer, so the end-of-run KPI reflects
// only [g_measStart, kSimEnd). Scheduled only when settleMs>0; keeps the exact
// decided-denominator metric (formula unchanged), just windowed past the
// queue-buildup transient. A packet straddling g_measStart is a sub-ms boundary
// effect (lost is floored at 0) and is negligible over an 800ms+ settle.
void
ResetMeasurement(Ptr<FlowMonitor> monitor, Ptr<Ipv4FlowClassifier> classifier)
{
    uint64_t tx = 0;
    uint64_t rx = 0;
    for (auto& kv : monitor->GetFlowStats())
    {
        if (classifier)
        {
            Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(kv.first);
            if (t.destinationPort >= kBgPort)
            {
                continue;  // exclude AC_BE background — same filter as end KPI
            }
        }
        tx += kv.second.txPackets;
        rx += kv.second.rxPackets;
    }
    g_txBase = tx;
    g_rxBase = rx;
    g_delaysSec.clear();
    g_apRxTotal.fill(0);  // window the per-AP airtime debug to [g_measStart,end).
    g_bgRxTotal.fill(0);
    g_apTxTotal.fill(0);      // window the per-AP KPI to [g_measStart, end).
    g_apLate99Total.fill(0);
    g_apLate999Total.fill(0);
    g_apShDropTotal.fill(0);  // F8: shared-buffer aged-out, same window.
    if (std::getenv("FEDDRL_DBG_EDCA"))
    {
        // Adopted STA-side AC_VI CWmin at measurement time: if the STA truly
        // adopted the AP's advertised UHR-AC, AP0-STA0 should reflect its BSS's
        // aggressive level while AP15-STA0 stays neutral.
        std::cerr << "[EDCA_STA] AP0-STA0 viMinCw="
                  << (g_dbgStaVi[0] ? (int)g_dbgStaVi[0]->GetMinCw(0) : -1)
                  << " AP15-STA0 viMinCw="
                  << (g_dbgStaVi[1] ? (int)g_dbgStaVi[1]->GetMinCw(0) : -1)
                  << std::endl;
    }
}

// PhyState callback: PHY 가 RX/TX/BUSY 상태로 머문 누적 시간을 측정.
// 시그니처: (Time start, Time duration, WifiPhyState state).
void
OnPhyStateChange(uint32_t apIdx, uint32_t linkIdx,
                 Time /*start*/, Time duration, WifiPhyState state)
{
    if (state == WifiPhyState::CCA_BUSY ||
        state == WifiPhyState::RX ||
        state == WifiPhyState::TX)
    {
        g_busySecAcc[apIdx][linkIdx] += duration.GetSeconds();
    }
}

// F8: 공유버퍼 drain — (1) aging: 도착 후 상한(5×deadline999, ≥100ms) 초과
// HoL 은 aged-out drop (dual 신호 + KPI decided 포함), (2) 선택 링크의 MAC 큐
// 깊이 < g_drainTarget 인 동안 HoL 을 그 링크 소켓으로 송신 (service-time
// routing). 게이트/비활성이면 대기 (P3).
void
DrainSta(AppHandles* apps, uint32_t s)
{
    const uint32_t ap = s / kStaPerAp;
    const double ageBound = std::max(5.0 * g_l999Sec, 0.1);
    while (!g_shBuf[s].empty())
    {
        SeqTsHeader hdr;
        g_shBuf[s].front()->PeekHeader(hdr);
        if ((Simulator::Now() - hdr.GetTs()).GetSeconds() <= ageBound)
        {
            break;
        }
        g_shBuf[s].pop_front();
        ++g_slotDropped[ap];
        ++g_apShDropTotal[ap];
    }
    if (g_shBuf[s].empty() || !g_apActive[ap])
    {
        return;
    }
    int link = g_selLink[s];
    if (link >= 0 && !LinkAllowed(ap, link))
    {
        link = FirstAllowedLink(ap);
    }
    if (link < 0 || static_cast<uint32_t>(link) >= kNumLinks)
    {
        return;  // inactive: 대기 (드롭 아님 — aging 이 상한을 강제)
    }
    Ptr<Socket> sock = apps->stas[s].sockPerLink[link];
    Ptr<QosTxop> txop =
        (g_staObsTxop[s].size() > static_cast<size_t>(link))
            ? g_staObsTxop[s][link] : nullptr;
    Ptr<WifiMacQueue> q = txop ? txop->GetWifiMacQueue() : nullptr;
    if (!sock || !q)
    {
        return;
    }
    while (!g_shBuf[s].empty() && q->GetNPackets() < g_drainTarget)
    {
        sock->Send(g_shBuf[s].front());
        g_shBuf[s].pop_front();
        ++g_apTxTotal[ap];  // per-AP decided-at-send (KPI 분모의 send 성분)
    }
}

// 주기 drain: MAC 큐가 비워질 때 공유버퍼가 후속 공급되도록 1ms 마다 전 STA
// drain (arrival-시점 drain 이 놓치는 "큐가 나중에 빈" 경우 보완).
void
DrainTick(AppHandles* apps)
{
    for (uint32_t s = 0; s < kNumSta; ++s)
    {
        DrainSta(apps, s);
    }
    if (Simulator::Now() < kSimEnd)
    {
        Simulator::Schedule(kDrainTick, &DrainTick, apps);
    }
}

// 액션이 선택한 링크로만 STA 트래픽을 송신 (인과적 action→KPI 커플링).
// UdpClient 자동 송신은 슬롯별 링크 전환을 신뢰성 있게 못 하므로 수동 소켓.
// F8: 도착 = SeqTs 스탬프 + 공유버퍼 push + 즉시 drain 시도. g_drainTarget=0
// 이면 공유버퍼 우회(direct-send 레거시: 즉시 송신, 비활성 도착 드롭) — A/B.
void
SendTick(AppHandles* apps)
{
    if (Simulator::Now() >= kSendStop)
    {
        return;
    }
    for (uint32_t s = 0; s < kNumSta; ++s)
    {
        g_sendAcc[s] += g_loadMult[s / kStaPerAp];
        if (g_sendAcc[s] < 1.0)
        {
            continue;
        }
        g_sendAcc[s] -= 1.0;

        const uint32_t ap = s / kStaPerAp;
        SeqTsHeader seqTs;
        seqTs.SetSeq(g_seq[s]++);
        Ptr<Packet> p = Create<Packet>(g_pktBytes);
        p->AddHeader(seqTs);

        if (g_drainTarget == 0)
        {
            // direct-send 레거시 (회귀 A/B): 도착 즉시 선택 링크로 송신.
            int link = g_selLink[s];
            if (link >= 0 && !LinkAllowed(ap, link))
            {
                link = FirstAllowedLink(ap);
            }
            if (!g_apActive[ap] || link < 0 ||
                static_cast<uint32_t>(link) >= kNumLinks)
            {
                continue;
            }
            Ptr<Socket> sock = apps->stas[s].sockPerLink[link];
            if (!sock)
            {
                continue;
            }
            sock->Send(p);
            ++g_apTxTotal[ap];
            continue;
        }
        g_shBuf[s].push_back(p);
        DrainSta(apps, s);
    }
    Simulator::Schedule(g_sendInterval, &SendTick, apps);
}

// Apply a per-BSS UHR-AC aggressiveness level: rewrite the AC_VI EDCA
// (CWmin/MaxCw/AIFSN) of every STA link-device of AP `ap`. This is the real
// MAC operation; the level is chosen by the Python controller (or the static
// heuristic). No-op unless the dedicated UHR-AC is enabled.
void
ApplyUhrEdca(uint32_t ap, uint8_t level)
{
    if (!g_uhrAcEnabled || ap >= kNumAp)
    {
        return;
    }
    if (level >= kUhrLevels)
    {
        level = static_cast<uint8_t>(kUhrLevels - 1);
    }
    g_uhrLevel[ap] = level;
    for (const Ptr<QosTxop>& txop : g_uhrTxop[ap])
    {
        if (!txop)
        {
            continue;
        }
        // g_uhrTxop holds each AP's single-link AC_VI EDCAF (linkId 0). Setting
        // the AP-side CW/AIFSN makes the next beacon's EdcaParameterSet advertise
        // it (ApWifiMac::GetEdcaParameterSet reads GetMinCw(linkId) live), which
        // this BSS's STAs adopt (StaWifiMac::SetEdcaParameters) -> per-BSS uplink
        // airtime control. STA-side SetMinCw would be overwritten by that adoption.
        txop->SetMinCw(kUhrCwMin[level], 0);
        txop->SetMaxCw(1023, 0);
        txop->SetAifsn(kUhrAifsn[level], 0);
    }
}

// 매크로 슬롯 끝마다 Python ↔ C++ 핸드셰이크 + 다음 슬롯 액션 적용.
void
SlotTick(Ns3AiMsgInterfaceImpl<EnvMsg, ActMsg>* msgIf,
         AppHandles* apps,
         Ptr<FlowMonitor> monitor)
{
    if (Simulator::Now() >= kSimEnd)
    {
        return;
    }

    // ---- (1) C++ -> Python: EnvMsg ----
    msgIf->CppSendBegin();
    EnvMsg* env = msgIf->GetCpp2PyStruct();
    env->nowUs = static_cast<uint64_t>(Simulator::Now().GetMicroSeconds());

    // 1a. per-AP delivered + deadline-violation counts THIS slot, all from the
    // same OnPacketRx callback / apIdx so violation ⊆ served (p99 ≤ 1). (구:
    // FlowMonitor flowId→AP 매핑이 어긋나 served 를 과소·오배정 → p99>1.)
    for (uint32_t a = 0; a < kNumAp; ++a)
    {
        env->served[a] = g_slotServed[a];
        env->violation99[a] = g_slotViol99[a];
        env->violation999[a] = g_slotViol999[a];
        env->dropped[a] = g_slotDropped[a];
        g_slotServed[a] = 0;
        g_slotViol99[a] = 0;
        g_slotViol999[a] = 0;
        g_slotDropped[a] = 0;
    }

    // 1b. 실측 backlog + HoL age (F8): STA 의 총 백로그 = 공유버퍼 + per-link
    //     MAC 큐 합. HoL age = 공유버퍼 front 의 도착시각(SeqTs) vs MAC 큐
    //     head 들의 enqueue 시각(WifiMpdu::GetTimestamp) 중 최고령.
    for (uint32_t s = 0; s < kNumSta; ++s)
    {
        uint32_t nPkts = static_cast<uint32_t>(g_shBuf[s].size());
        Time oldest = Time::Max();
        if (!g_shBuf[s].empty())
        {
            SeqTsHeader hdr;
            g_shBuf[s].front()->PeekHeader(hdr);
            oldest = hdr.GetTs();
        }
        for (const Ptr<QosTxop>& txop : g_staObsTxop[s])
        {
            Ptr<WifiMacQueue> q = txop ? txop->GetWifiMacQueue() : nullptr;
            if (!q)
            {
                continue;
            }
            nPkts += q->GetNPackets();
            Ptr<const WifiMpdu> head = q->Peek();
            if (head && head->GetTimestamp() < oldest)
            {
                oldest = head->GetTimestamp();
            }
        }
        env->queueLen[s] = nPkts;
        env->holUs[s] = (nPkts > 0 && oldest < Time::Max())
            ? static_cast<uint32_t>(
                  (Simulator::Now() - oldest).GetMicroSeconds())
            : 0;
    }
    // 관측 검증용 (FEDDRL_DBG_QLEN=1): 주기적으로 backlog 합/최대 HoL 출력 —
    // direct-send 로 관측이 실제 MAC 큐를 읽는지(전부 0 이 아닌지) 확인.
    if (std::getenv("FEDDRL_DBG_QLEN"))
    {
        static uint32_t dbgSlot = 0;
        if (++dbgSlot % 50 == 0)
        {
            uint64_t qSum = 0;
            uint32_t holMax = 0;
            for (uint32_t s = 0; s < kNumSta; ++s)
            {
                qSum += env->queueLen[s];
                holMax = std::max(holMax, env->holUs[s]);
            }
            uint64_t dSum = 0;
            for (uint32_t a = 0; a < kNumAp; ++a)
            {
                dSum += env->dropped[a];
            }
            std::cerr << "[QLEN_DBG] slot=" << dbgSlot << " qSum=" << qSum
                      << " holMaxUs=" << holMax << " dropSlot=" << dSum
                      << " csi0=" << CsiOf(0, 0) << "/" << CsiOf(0, 1)
                      << "/" << CsiOf(0, 2) << std::endl;
        }
    }

    // 1c. CBR per AP × link (PHY busy 누적값의 슬롯 delta / kMacroSlot).
    const double slotSec = kMacroSlot.GetSeconds();
    for (uint32_t a = 0; a < kNumAp; ++a)
    {
        for (uint32_t l = 0; l < kNumLinks; ++l)
        {
            const double busyDelta =
                g_busySecAcc[a][l] - g_busySecPrev[a][l];
            g_busySecPrev[a][l] = g_busySecAcc[a][l];
            const double ratio = std::min(1.0,
                                          std::max(0.0, busyDelta / slotSec));
            env->cbr[a][l] = static_cast<uint8_t>(ratio * 255.0);
        }
    }

    // 1d. per-STA-per-link 채널 품질 (정적 large-scale fading). ns-3 물리 채널이
    //     실제로 적용하는 shadowing 파라미터(CsiOf)를 그대로 obs 로 전달 →
    //     Python 은 해석식 재유도 없이 이 값을 학습 입력으로 사용. 정적 채널에서
    //     완전 CSI 추정기가 보고할 large-scale 채널 품질과 동일하다.
    for (uint32_t s = 0; s < kNumSta; ++s)
    {
        for (uint32_t l = 0; l < kNumLinks; ++l)
        {
            env->csi[s][l] = static_cast<float>(CsiOf(s, l));
        }
    }
    msgIf->CppSendEnd();

    // ---- (2) Python -> C++: ActMsg ----
    msgIf->CppRecvBegin();
    const ActMsg* act = msgIf->GetPy2CppStruct();
    // STR/NSTR behavioral approximation:
    //   mapMode == 0 (no coord): 모든 AP 가 동시 활성 (선택된 링크 만).
    //   mapMode == 1 (Co-TDMA):  현재 슬롯에 한 AP 만 활성 (round-robin),
    //                           나머지 AP 는 현재 슬롯 동안 모든 client stop.
    //   mapMode == 2 (Co-OFDMA): 모든 AP 동시 활성 (현재 모델에서는
    //                           mapMode==0 와 동일 동작 — full PHY-level
    //                           STR/NSTR 은 별도 ns-3 patch 가 필요,
    //                           REIMPL_PLAN.md 항목 3 참고).
    const uint8_t mode = act->mapMode;
    const uint32_t coTdmaSlot = static_cast<uint32_t>(
        Simulator::Now().GetMicroSeconds() /
        kMacroSlot.GetMicroSeconds()) % kNumAp;
    for (uint32_t a = 0; a < kNumAp; ++a)
    {
        // mode==1 (Co-TDMA): 현재 슬롯에 한 AP 만 활성. 다음 SendTick 부터
        // 반영 (SlotTick 이 SendTick 보다 성기므로 슬롯 내내 유지).
        g_apActive[a] = (mode != 1) || (a == coTdmaSlot);
        for (uint32_t s = 0; s < kStaPerAp; ++s)
        {
            const uint32_t staIdx = a * kStaPerAp + s;
            if (staIdx < kNumSta)
            {
                g_selLink[staIdx] = act->selectedLink[a][s];
            }
        }
    }
    msgIf->CppRecvEnd();

    Simulator::Schedule(kMacroSlot, &SlotTick, msgIf, apps, monitor);
}

}  // namespace

int
main(int argc, char* argv[])
{
    uint32_t seed = 0;
    double arrivalPps = 5500.0;
    bool verbose = false;
    std::string baselineTag = "ns3_feddrl_ai";
    double episodeMs = 180.0;
    double macroSlotMs = 5.0;
    double deadline99Ms = 5.0;
    double deadline999Ms = 10.0;
    double settleMs = 0.0;
    double loadSpread = 0.0;
    bool uhrAc = false;
    double bgPps = 0.0;
    double chanDwellMs = 200.0;  // F6 Markov 채널 체류; 0=정적(레거시)
    bool hetBands = true;        // F7 이질 밴드; 0=레거시
    uint32_t drainTarget = 8;    // F8 per-link MAC 큐 목표 깊이; 0=direct-send
    uint32_t link2Width = 80;    // link2(6GHz) 폭 MHz
    double bgLinkSkew = 0.0;    // per-link background asymmetry: link l weight =
                                // 1 + skew*(2l/(K-1)-1) (sum preserved). 0 = symmetric.
    double uhrEdcaSpread = 0.0;
    int uhrLevel = -1;
    uint32_t rtsCts = 4692000;  // RTS/CTS threshold (bytes); huge = off (default)
    bool uhrOnoff = false;      // diagnostic: drive UHR via backlogged OnOff
    bool bgEdca = false;        // diagnostic: apply per-BSS level to AC_BE too
    std::string segSuffix = "";

    CommandLine cmd(__FILE__);
    cmd.AddValue("seed", "RNG seed", seed);
    cmd.AddValue("arrivalPps", "Per-STA arrival rate (pkts/s); every STA "
                 "enqueues one packet each 1/arrivalPps s, so network load = "
                 "kNumSta * arrivalPps. Drive high to induce saturation.",
                 arrivalPps);
    cmd.AddValue("baselineTag", "CSV baseline column tag", baselineTag);
    cmd.AddValue("episodeDurationMs",
                 "Measurement-window length in ms after warmup. Large values "
                 "(e.g. 100000) give one long continuous run for RL training; "
                 "default 180 = single eval episode.",
                 episodeMs);
    cmd.AddValue("macroSlotMs",
                 "RL decision + KPI measurement window in ms (served/violation "
                 "are per-slot deltas). Longer slots integrate more packets per "
                 "decision -> lower-variance reward/Z; default 5.",
                 macroSlotMs);
    cmd.AddValue("deadline99Ms",
                 "p99 UHR deadline in ms; delay above this = violation99. "
                 "Density-scaled: 5 at 4-AP, larger (e.g. 10) at denser scales "
                 "whose structural tail floor precludes 5ms. Default 5.",
                 deadline99Ms);
    cmd.AddValue("deadline999Ms",
                 "p99.9 UHR deadline in ms; delay above this = violation999. "
                 "Default 10 (4-AP); e.g. 20 at 16-AP.",
                 deadline999Ms);
    cmd.AddValue("segSuffix",
                 "Unique suffix for the ns3-ai shared-memory object names "
                 "(segment/msgs/lock). Empty = library defaults. Set a distinct "
                 "value per concurrent run so parallel training jobs do not "
                 "collide on the same shared memory. Must match the Python side.",
                 segSuffix);
    cmd.AddValue("settleMs",
                 "Queue-settle warmup in ms AFTER the association warmup. Traffic "
                 "starts at kWarmup(200ms) with empty queues; KPI counting is "
                 "reset at kWarmup+settleMs so the queue-buildup transient is "
                 "excluded -> steady-state UHR. Default 0 = no settle (legacy, "
                 "byte-identical metric).",
                 settleMs);
    cmd.AddValue("loadSpread",
                 "Per-AP load heterogeneity in [0,1]. 0 = homogeneous (all APs "
                 "at arrivalPps). >0 assigns m[ap]=1-loadSpread*(ap/(N-1)) so AP0 "
                 "stays at arrivalPps and AP15 drops to (1-loadSpread)*arrivalPps "
                 "-> heterogeneous per-AP constraint pressure (Z). Same value in "
                 "train and eval.",
                 loadSpread);
    cmd.AddValue("uhrAc",
                 "Enable the dedicated learning-controlled UHR access category: "
                 "route UHR flows to AC_VI (priority 5) whose per-BSS EDCA "
                 "params are set at runtime. Off (default) = legacy single-class "
                 "AC_BE, byte-identical.",
                 uhrAc);
    cmd.AddValue("bgPps",
                 "Per-STA best-effort background load (pkts/s) on AC_BE, sent to "
                 "the STA's AP on its first allowed link. Saturates the medium so "
                 "the regime becomes contention-limited. 0 (default) = no "
                 "background.",
                 bgPps);
    cmd.AddValue("chanDwellMs",
                 "F6: Markov 채널 평균 상태 체류(ms). 각 (STA,link) CSI 가 "
                 "{0.9,0.5,0.3} nearest-neighbor walk 로 전이 (A5 Markov 가정 "
                 "성립 + 학습이 추적할 동역학). 0 = 정적 채널(레거시 byte-"
                 "identical).",
                 chanDwellMs);
    cmd.AddValue("hetBands",
                 "F7: 링크별 이질 밴드 — link0=2.4GHz/20MHz, link1=5GHz/40MHz, "
                 "link2=6GHz/{80|40}MHz. 0 = 레거시(전 링크 기본 채널).",
                 hetBands);
    cmd.AddValue("link2Width",
                 "link2(6GHz) 채널폭 MHz: 80(기본) 또는 40. 80MHz 는 노이즈 "
                 "+6dB 로 Markov churn 하 재시도 폭풍(6GHz AP 손실 바닥) — "
                 "40MHz 가 완화 후보.",
                 link2Width);
    cmd.AddValue("drainTarget",
                 "F8: 공유버퍼 drain-on-demand 의 per-link MAC 큐 목표 깊이. "
                 "도착은 공유버퍼 대기 후 선택 링크 MAC 큐가 이보다 얕을 때만 "
                 "송신(service-time routing, 논문 shared-buffer MLO). 0 = "
                 "공유버퍼 우회(direct-send 레거시, A/B 회귀용).",
                 drainTarget);
    cmd.AddValue("bgLinkSkew",
                 "Per-link background asymmetry in [0,1]: link l gets background "
                 "rate * (1 + skew*(2l/(K-1)-1)), sum preserved. skew=1 clears "
                 "link0 and doubles the top link, so a clear link exists for UHR "
                 "steering to exploit. 0 (default) = symmetric background.",
                 bgLinkSkew);
    cmd.AddValue("uhrEdcaSpread",
                 "Static UHR-AC heuristic strength in [0,1] (validation only): "
                 "sets each BSS's UHR-AC level proportional to its load rank "
                 "(AP0 hottest -> most aggressive). 0 = all neutral (level 0). "
                 "Overridden by the RL action when present.",
                 uhrEdcaSpread);
    cmd.AddValue("uhrLevel",
                 "Uniform UHR-AC aggressiveness level override (0..3): sets ALL "
                 "BSSs to this level, ignoring uhrEdcaSpread. -1 (default) = use "
                 "the spread heuristic. Diagnostic for whether within-BSS EDCA "
                 "moves the UHR tail at all.",
                 uhrLevel);
    cmd.AddValue("rtsCts",
                 "RTS/CTS threshold in bytes: data frames >= this use RTS/CTS. "
                 "0 = RTS/CTS on every data frame (protects against hidden-terminal "
                 "OBSS collisions). Default 4692000 = effectively off (legacy).",
                 rtsCts);
    cmd.AddValue("uhrOnoff",
                 "Diagnostic: drive UHR with a backlogged OnOff generator on AC_VI "
                 "(first allowed link) instead of the g_staBuf manual path, to "
                 "isolate whether the traffic architecture masks EDCA. Off = normal.",
                 uhrOnoff);
    cmd.AddValue("bgEdca",
                 "Diagnostic: apply the per-BSS UHR level to AC_BE too, so per-AP "
                 "background (clean backlogged OnOff, uniform load) delivery reveals "
                 "whether EDCA reallocates airtime for a standard generator.",
                 bgEdca);
    cmd.AddValue("verbose", "Enable NS_LOG_INFO", verbose);
    cmd.Parse(argc, argv);

    // RTS/CTS must be set before device creation. Global here; a per-BSS RL
    // lever would set it per remote-station-manager instead.
    Config::SetDefault("ns3::WifiRemoteStationManager::RtsCtsThreshold",
                       UintegerValue(rtsCts));

    g_uhrAcEnabled = uhrAc;
    g_uhrEdcaSpread = uhrEdcaSpread;
    g_uhrLevelUniform = uhrLevel;
    g_bgEdca = bgEdca;
    g_chanDwellMs = chanDwellMs;
    g_hetBands = hetBands;
    g_drainTarget = drainTarget;
    g_link2Width = link2Width;

    // Runtime-configurable episode length + macro-slot: recompute derived
    // times BEFORE any Simulator::Schedule / Stop below reads them.
    kMacroSlot = MilliSeconds(macroSlotMs);
    g_l99Sec = deadline99Ms * 1e-3;
    g_l999Sec = deadline999Ms * 1e-3;
    kEpisodeDuration = MilliSeconds(episodeMs);
    kSettle = MilliSeconds(settleMs);
    g_measStart = kWarmup + kSettle;
    kSimEnd = g_measStart + kEpisodeDuration;
    kSendStop = kSimEnd - MilliSeconds(10);

    // Zero-sum per-AP load redistribution: preserve the MEAN load (= arrivalPps,
    // so total co-channel contention stays at the operating point) while spreading
    // per-AP rate. AP0 hottest = (1+loadSpread)*arrivalPps, AP15 coldest =
    // (1-loadSpread)*arrivalPps, mean = arrivalPps. The base tick is refined by
    // maxMult=1+loadSpread (g_sendInterval below) so g_loadMult is a thinning
    // fraction in (0,1] (hottest AP=1). Hot APs become genuinely overloaded (high
    // Z), cold APs slack (Z~0) -> heterogeneous binding constraint pressure without
    // relieving the system. loadSpread=0 -> maxMult=1, all g_loadMult=1 ->
    // homogeneous CBR (byte-identical).
    {
        const double maxMult = 1.0 + loadSpread;
        for (uint32_t a = 0; a < kNumAp; ++a)
        {
            const double frac = (kNumAp > 1)
                ? static_cast<double>(a) / (kNumAp - 1) : 0.0;
            const double m = 1.0 + loadSpread * (1.0 - 2.0 * frac);  // [1-s, 1+s]
            g_loadMult[a] = m / maxMult;                             // (0,1], hot=1
        }
    }

    RngSeedManager::SetSeed(seed + 1);
    RngSeedManager::SetRun(seed + 1);

    if (verbose)
    {
        LogComponentEnable("FedDrlAiScenario", LOG_LEVEL_INFO);
    }

    auto interface = Ns3AiMsgInterface::Get();
    interface->SetIsMemoryCreator(false);
    interface->SetUseVector(false);
    interface->SetHandleFinish(true);
    // Per-run unique shared-memory names so concurrent training jobs don't
    // collide. Must match the Python Experiment(...) names exactly. Empty
    // suffix keeps the library defaults (single-run backward compatibility).
    if (!segSuffix.empty())
    {
        interface->SetNames("ns3ai_seg_" + segSuffix,
                            "ns3ai_c2p_" + segSuffix,
                            "ns3ai_p2c_" + segSuffix,
                            "ns3ai_lock_" + segSuffix);
    }
    Ns3AiMsgInterfaceImpl<EnvMsg, ActMsg>* msgInterface =
        interface->GetInterface<EnvMsg, ActMsg>();

    // -------- 토폴로지 --------
    NodeContainer apNodes;
    apNodes.Create(kNumAp);
    NodeContainer staNodes;
    staNodes.Create(kNumSta);

    // Dense 2D deployment: all APs sit in a compact grid so every AP is within
    // mutual carrier-sense range. Combined with the per-AP link set K_i
    // (LinkAllowed), co-channel contention (OBSS) is then governed purely by
    // band sharing -> heterogeneous o=(2,2,3,1) at 4 AP that DENSIFIES as N_AP
    // grows (the canonical scenario, scaled). STAs sit on a ring inside the BSS.
    const double kApPitch = 8.0;       // m between adjacent APs (<< CS range)
    const uint32_t kGridCols = 4;      // AP grid width
    const double kStaRing = 3.0;       // m, STA close to its own AP (good SINR;
                                       // association is forced by per-AP SSID)
    const double kPi = 3.14159265358979323846;
    MobilityHelper mobility;
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    Ptr<ListPositionAllocator> apPos = CreateObject<ListPositionAllocator>();
    for (uint32_t i = 0; i < kNumAp; ++i)
    {
        apPos->Add(Vector(kApPitch * (i % kGridCols),
                          kApPitch * (i / kGridCols), 3.0));
    }
    mobility.SetPositionAllocator(apPos);
    mobility.Install(apNodes);

    Ptr<ListPositionAllocator> staPos = CreateObject<ListPositionAllocator>();
    for (uint32_t i = 0; i < kNumSta; ++i)
    {
        const uint32_t apIdx = i / kStaPerAp;
        const uint32_t local = i % kStaPerAp;
        const double ax = kApPitch * (apIdx % kGridCols);
        const double ay = kApPitch * (apIdx / kGridCols);
        const double ang = 2.0 * kPi * local / kStaPerAp;
        staPos->Add(Vector(ax + kStaRing * std::cos(ang),
                           ay + kStaRing * std::sin(ang), 1.0));
    }
    mobility.SetPositionAllocator(staPos);
    mobility.Install(staNodes);

    // 링크당 독립 PHY/MAC 인스턴스.
    std::array<NetDeviceContainer, kNumLinks> apDevsPerLink;
    std::array<NetDeviceContainer, kNumLinks> staDevsPerLink;

    // 링크별 채널 = LogDistance(OBSS 간섭 유지) + MatrixPropagationLoss(per-STA
    // -per-link CSI 오프셋). 좋은 링크(높은 CSI)는 추가 손실 0, 나쁜 링크는 최대
    // +20dB → STA 마다 링크 선택이 KPI 에 실제로 반영됨.
    YansWifiPhyHelper phyHelpers[kNumLinks];
    InitChanStates();  // F6: Markov 채널 초기 상태 = 구 정적 패턴 (l+s)%3
    for (uint32_t l = 0; l < kNumLinks; ++l)
    {
        Ptr<LogDistancePropagationLossModel> logd =
            CreateObject<LogDistancePropagationLossModel>();
        Ptr<MatrixPropagationLossModel> off =
            CreateObject<MatrixPropagationLossModel>();
        off->SetDefaultLoss(0.0);
        // 대칭 채널: STA s 의 링크 l CSI 로 유도된 shadowing 을 그 STA 로 향하는
        // 모든 AP 경로(desired + interferer)에 동일 적용. → SINR 의 signal/
        // interference 비는 기하(LogDistance)가 지배(가까운 own-AP 승), 절대
        // RSSI 는 CSI 가 지배(낮은 CSI = 큰 shadowing = noise 대비 열세). 구:
        // own-AP 경로만 penalize → 비대칭 → dense co-channel SINR 인위적 붕괴.
        for (uint32_t s = 0; s < kNumSta; ++s)
        {
            Ptr<MobilityModel> staMob =
                staNodes.Get(s)->GetObject<MobilityModel>();
            g_staMob[s] = staMob;  // F6: ChanTick 이 loss 갱신에 사용
            const double extra = (1.0 - CsiOf(s, l)) * 20.0;
            for (uint32_t tx = 0; tx < kNumAp; ++tx)
            {
                Ptr<MobilityModel> txMob =
                    apNodes.Get(tx)->GetObject<MobilityModel>();
                g_apMob[tx] = txMob;
                off->SetLoss(txMob, staMob, extra, true);
            }
        }
        g_chanLoss[l] = off;  // F6: Markov 전이 시 SetLoss 대상
        logd->SetNext(off);
        Ptr<YansWifiChannel> channel = CreateObject<YansWifiChannel>();
        channel->SetPropagationLossModel(logd);
        channel->SetPropagationDelayModel(
            CreateObject<ConstantSpeedPropagationDelayModel>());
        phyHelpers[l].SetChannel(channel);
        // F7 (E5 수정): 이질 밴드/폭 — link0=2.4GHz/20MHz, link1=5GHz/40MHz,
        // link2=6GHz/80MHz (PHY 용량 ~1:2:4). 링크선택이 "어느 파이프가
        // 굵은가"라는 실제 구조를 갖게 한다 (구: 3링크 동일 기본 채널 →
        // 선택의 구조적 가치 미미). hetBands=0 이면 레거시(기본 채널) 유지.
        if (g_hetBands)
        {
            // link2 폭은 CLI(link2Width): 80MHz 는 노이즈 +6dB 로 SNR 마진이
            // 얇아 Markov 4dB 스텝이 MCS 실패 임계를 넘는 재시도 폭풍 유발
            // (DROP_DBG: 정적 13 vs churn 180 드롭 실증) — 40MHz(+3dB)는
            // 마진 2 배로 churn 을 흡수하면서 밴드 이질성(2.4/5/6GHz)은 유지.
            static const char* kChan01[2] = {
                "{1, 20, BAND_2_4GHZ, 0}", "{38, 40, BAND_5GHZ, 0}"};
            const char* link2 = (g_link2Width >= 80)
                ? "{7, 80, BAND_6GHZ, 0}" : "{3, 40, BAND_6GHZ, 0}";
            phyHelpers[l].Set(
                "ChannelSettings",
                StringValue(l < 2 ? kChan01[l] : link2));
        }
    }

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211ax);
    wifi.SetRemoteStationManager("ns3::IdealWifiManager");

    WifiMacHelper mac;
    // Per-AP SSID: each STA associates ONLY to its own AP regardless of geometry
    // or the per-link CSI penalty. This turns the co-channel APs into a proper
    // multi-BSS OBSS (same channel, different BSS, contending via CSMA) and
    // avoids mis-association in the dense layout. Device order stays AP 0..N-1
    // and STA 0..M-1 so IP / flow / trace indexing is unchanged.
    for (uint32_t l = 0; l < kNumLinks; ++l)
    {
        for (uint32_t ap = 0; ap < kNumAp; ++ap)
        {
            Ssid ssid(("feddrl-ap-" + std::to_string(ap)).c_str());
            mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid));
            apDevsPerLink[l].Add(
                wifi.Install(phyHelpers[l], mac, apNodes.Get(ap)));

            mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid),
                        "ActiveProbing", BooleanValue(false));
            for (uint32_t k = 0; k < kStaPerAp; ++k)
            {
                staDevsPerLink[l].Add(wifi.Install(
                    phyHelpers[l], mac, staNodes.Get(ap * kStaPerAp + k)));
            }
        }
    }

    InternetStackHelper internet;
    internet.Install(apNodes);
    internet.Install(staNodes);

    std::array<Ipv4InterfaceContainer, kNumLinks> apIfs;
    std::array<Ipv4InterfaceContainer, kNumLinks> staIfs;
    for (uint32_t l = 0; l < kNumLinks; ++l)
    {
        Ipv4AddressHelper ipv4;
        std::ostringstream net;
        net << "10." << (1 + l) << ".0.0";
        ipv4.SetBase(net.str().c_str(), "255.255.0.0");
        apIfs[l] = ipv4.Assign(apDevsPerLink[l]);
        staIfs[l] = ipv4.Assign(staDevsPerLink[l]);
    }

    // PHY busy state trace — link 별 CBR 측정용.
    for (uint32_t l = 0; l < kNumLinks; ++l)
    {
        for (uint32_t a = 0; a < kNumAp; ++a)
        {
            Ptr<WifiNetDevice> wnd =
                DynamicCast<WifiNetDevice>(apDevsPerLink[l].Get(a));
            if (!wnd)
            {
                continue;
            }
            Ptr<WifiPhy> phy = wnd->GetPhy();
            if (!phy)
            {
                continue;
            }
            Ptr<WifiPhyStateHelper> stateHelper = phy->GetState();
            if (!stateHelper)
            {
                continue;
            }
            stateHelper->TraceConnectWithoutContext(
                "State",
                MakeBoundCallback(&OnPhyStateChange, a, l));
        }
    }

    // Collect each BSS's AP-side AC_VI EDCAF (one single-link device per link).
    // Control is applied at the AP because 802.11 STAs adopt the AP's advertised
    // EdcaParameterSet from beacons (a STA-side setting would be overwritten).
    // apDevsPerLink[l].Get(a) is AP a on link l (install loop order).
    if (g_uhrAcEnabled)
    {
        for (uint32_t l = 0; l < kNumLinks; ++l)
        {
            for (uint32_t a = 0; a < kNumAp; ++a)
            {
                Ptr<WifiNetDevice> wnd =
                    DynamicCast<WifiNetDevice>(apDevsPerLink[l].Get(a));
                if (!wnd)
                {
                    continue;
                }
                Ptr<WifiMac> wmac = wnd->GetMac();
                if (!wmac)
                {
                    continue;
                }
                Ptr<QosTxop> txop = wmac->GetQosTxop(AC_VI);
                if (txop)
                {
                    g_uhrTxop[a].push_back(txop);
                }
                Ptr<QosTxop> beTxop = wmac->GetQosTxop(AC_BE);
                if (beTxop)
                {
                    g_bgTxop[a].push_back(beTxop);  // for the EDCA-isolation test
                }
            }
        }
        // Static heuristic (validation lever): AP0 is hottest, so map load rank
        // (1 - a/(N-1)) to aggressiveness. uhrEdcaSpread=0 -> all neutral.
        for (uint32_t a = 0; a < kNumAp; ++a)
        {
            uint8_t lvl = 0;
            if (g_uhrLevelUniform >= 0)
            {
                lvl = static_cast<uint8_t>(
                    std::min(g_uhrLevelUniform, static_cast<int>(kUhrLevels) - 1));
            }
            else if (g_uhrEdcaSpread > 0.0)
            {
                const double hotFrac = (kNumAp > 1)
                    ? (1.0 - static_cast<double>(a) / (kNumAp - 1)) : 1.0;
                lvl = static_cast<uint8_t>(std::lround(
                    hotFrac * g_uhrEdcaSpread * (kUhrLevels - 1)));
            }
            ApplyUhrEdca(a, lvl);
            // EDCA-isolation test: also drive AC_BE (the clean OnOff background)
            // with the same per-BSS level, so per-AP background delivery reveals
            // whether EDCA reallocates airtime for a standard generator.
            if (g_bgEdca)
            {
                for (const Ptr<QosTxop>& t : g_bgTxop[a])
                {
                    if (!t) { continue; }
                    t->SetMinCw(kUhrCwMin[lvl], 0);
                    t->SetMaxCw(1023, 0);
                    t->SetAifsn(kUhrAifsn[lvl], 0);
                }
            }
        }
        // Debug handles to STA-side AC_VI (AP0-STA0=idx0, AP15-STA0=idx75).
        for (int j = 0; j < 2; ++j)
        {
            const uint32_t sidx = (j == 0) ? 0u : (kNumSta - kStaPerAp);
            Ptr<WifiNetDevice> sd =
                DynamicCast<WifiNetDevice>(staDevsPerLink[0].Get(sidx));
            if (sd && sd->GetMac())
            {
                g_dbgStaVi[j] = sd->GetMac()->GetQosTxop(AC_VI);
            }
        }
        if (std::getenv("FEDDRL_DBG_EDCA"))
        {
            const uint32_t nHi = g_uhrTxop[0].size();
            const uint32_t nLo = g_uhrTxop[kNumAp - 1].size();
            std::cerr << "[EDCA_DBG] AP0 txop=" << nHi << " level="
                      << +g_uhrLevel[0] << " minCw="
                      << (nHi ? g_uhrTxop[0][0]->GetMinCw(0) : 9999)
                      << " | AP" << (kNumAp - 1) << " txop=" << nLo << " level="
                      << +g_uhrLevel[kNumAp - 1] << " minCw="
                      << (nLo ? g_uhrTxop[kNumAp - 1][0]->GetMinCw(0) : 9999)
                      << std::endl;
        }
    }

    // Observation handles: per-STA per-link MAC queue of the AC that UHR is
    // actually sent on (uhrAc on -> socket priority 5 -> AC_VI; off -> AC_BE).
    // SlotTick 1b reads GetNPackets()/Peek() from these for queueLen/holUs.
    {
        const AcIndex obsAc = g_uhrAcEnabled ? AC_VI : AC_BE;
        for (uint32_t s = 0; s < kNumSta; ++s)
        {
            for (uint32_t l = 0; l < kNumLinks; ++l)
            {
                Ptr<WifiNetDevice> sd =
                    DynamicCast<WifiNetDevice>(staDevsPerLink[l].Get(s));
                if (!sd || !sd->GetMac())
                {
                    continue;
                }
                if (std::getenv("FEDDRL_DBG_DROP"))
                {
                    sd->GetMac()->TraceConnectWithoutContext(
                        "DroppedMpdu",
                        MakeBoundCallback(&OnDbgMacDrop, l));
                }
                Ptr<QosTxop> txop = sd->GetMac()->GetQosTxop(obsAc);
                if (txop)
                {
                    g_staObsTxop[s].push_back(txop);
                    Ptr<WifiMacQueue> q = txop->GetWifiMacQueue();
                    if (q)
                    {
                        // Stale-packet discard bound. NOT deadline999 자체:
                        // lifetime 이 혼잡 지연 스케일(수십 ms)과 겹치면 만료
                        // 직전 dequeue 된 프레임이 전송/재전송 도중 수명 초과로
                        // 버려져 airtime 만 소모(thrashing) — L140 스모크에서
                        // AP0 rx 20% 붕괴로 실증. 5×deadline999(≥100ms) 는
                        // staleness 를 유계로 막되 작업지연과 분리된다. 회계는
                        // 무편향: 초과-지연 전달은 OnPacketRx 가 late 로,
                        // >bound 좌초는 Expired 가 drop 으로 잡는다.
                        q->SetMaxDelay(
                            Seconds(std::max(5.0 * g_l999Sec, 0.1)));
                        const uint32_t apIdx = s / kStaPerAp;
                        q->TraceConnectWithoutContext(
                            "Expired", MakeBoundCallback(&OnMacDrop, apIdx));
                        q->TraceConnectWithoutContext(
                            "DropBeforeEnqueue",
                            MakeBoundCallback(&OnMacDrop, apIdx));
                    }
                }
            }
        }
    }

    // -------- UDP 트래픽: STA 마다 링크별 client + AP 별 link 별 server ----
    const uint16_t kAppPort = 9000;
    const uint32_t kPktSizeBytes = 1500;
    // 보정: arrivalPps 를 STA당 부하로 해석 (surrogate wlan_env 와 의미 일치).
    const double perStaPps = arrivalPps;
    const double sendInterval = 1.0 / perStaPps;

    AppHandles apps;
    apps.stas.resize(kNumSta);

    for (uint32_t i = 0; i < kNumSta; ++i)
    {
        const uint32_t apIdx = i / kStaPerAp;
        apps.stas[i].node = staNodes.Get(i);
        for (uint32_t l = 0; l < kNumLinks; ++l)
        {
            const Ipv4Address dst = apIfs[l].GetAddress(apIdx);

            UdpServerHelper server(kAppPort + i * kNumLinks + l);
            ApplicationContainer srv = server.Install(apNodes.Get(apIdx));
            apps.servers.Add(srv);

            // 액션이 링크를 선택하므로 자동 송신 UdpClient 대신 수동 소켓.
            Ptr<Socket> sock = Socket::CreateSocket(
                staNodes.Get(i), UdpSocketFactory::GetTypeId());
            sock->Bind();
            sock->Connect(InetSocketAddress(
                dst, kAppPort + i * kNumLinks + l));
            // UHR flows go on the dedicated AC (AC_VI). Socket priority 5 ->
            // TID 5 -> AC_VI (QosUtilsMapTidToAc). When disabled the socket
            // keeps default priority 0 -> AC_BE (legacy single-class behavior).
            if (g_uhrAcEnabled)
            {
                sock->SetPriority(5);
            }
            apps.stas[i].sockPerLink[l] = sock;
        }
    }
    apps.servers.Start(Seconds(0.0));
    apps.servers.Stop(kSimEnd);

    // 액션→트래픽 커플링용 전역 초기화.
    g_pktBytes = kPktSizeBytes - 12;  // SeqTsHeader(12B) 여유.
    // Refine the tick by maxMult so the hottest AP (g_loadMult=1) sends every tick
    // at (1+loadSpread)*arrivalPps; cold APs thin down. loadSpread=0 -> unchanged.
    g_sendInterval = Seconds(sendInterval / (1.0 + loadSpread));
    g_selLink.fill(0);       // 초기: 모든 STA link 0.
    g_apActive.fill(true);

    g_delaysSec.reserve(static_cast<size_t>(arrivalPps *
                                            kEpisodeDuration.GetSeconds() *
                                            1.2));
    for (uint32_t i = 0; i < apps.servers.GetN(); ++i)
    {
        Ptr<UdpServer> srv = DynamicCast<UdpServer>(apps.servers.Get(i));
        if (srv)
        {
            // servers 는 STA×link 순서로 Add 되므로 (idx = sta*kNumLinks+link),
            // AP = idx / (kStaPerAp*kNumLinks).
            const uint32_t apIdx = i / (kStaPerAp * kNumLinks);
            srv->TraceConnectWithoutContext(
                "RxWithAddresses",
                MakeBoundCallback(&OnPacketRx, apIdx));
        }
    }

    // -------- Best-effort background (AC_BE) to make the medium contention-
    // limited. bgPps is the PER-BAND background rate: each STA sends a constant-
    // rate UDP stream to its AP on EVERY link the BSS is allowed to use, so UHR
    // faces background contention whichever link the policy selects. Default
    // socket priority 0 -> AC_BE. Separate ports (kBgPort+) and dedicated
    // PacketSinks keep it OUT of the UHR KPI (filtered below by destination
    // port). Inert when bgPps<=0 -> byte-identical legacy behavior.
    if (bgPps > 0.0)
    {
        for (uint32_t i = 0; i < kNumSta; ++i)
        {
            const uint32_t apIdx = i / kStaPerAp;
            for (uint32_t l = 0; l < kNumLinks; ++l)
            {
                if (!LinkAllowed(apIdx, l))
                {
                    continue;  // background only on bands this BSS uses
                }
                // Per-link asymmetry: weight = 1 + skew*(2l/(K-1)-1), sum preserved
                // across links so total offered background is skew-independent. A
                // heterogeneous background is what makes congestion-aware link
                // steering strictly beat blind round-robin (a clear link exists).
                double bgMult = 1.0;
                if (kNumLinks > 1)
                {
                    bgMult = 1.0 + bgLinkSkew *
                        (2.0 * static_cast<double>(l) / (kNumLinks - 1) - 1.0);
                }
                const uint64_t bgBps = static_cast<uint64_t>(
                    bgPps * bgMult * kPktSizeBytes * 8.0);
                if (bgBps == 0)
                {
                    continue;  // fully-cleared link (skew=1, l=0): no background
                }
                const Ipv4Address dst = apIfs[l].GetAddress(apIdx);
                const uint16_t port =
                    kBgPort + static_cast<uint16_t>(i * kNumLinks + l);

                PacketSinkHelper sink(
                    "ns3::UdpSocketFactory",
                    InetSocketAddress(Ipv4Address::GetAny(), port));
                ApplicationContainer sinkApp = sink.Install(apNodes.Get(apIdx));
                sinkApp.Start(Seconds(0.0));
                sinkApp.Stop(kSimEnd);
                sinkApp.Get(0)->TraceConnectWithoutContext(
                    "Rx", MakeBoundCallback(&OnBgRx, apIdx));

                OnOffHelper onoff("ns3::UdpSocketFactory",
                                 InetSocketAddress(dst, port));
                onoff.SetConstantRate(DataRate(bgBps), kPktSizeBytes);
                // Default Tos=0 -> priority 0 -> AC_BE (best-effort background).
                ApplicationContainer bgApp = onoff.Install(staNodes.Get(i));
                bgApp.Start(kWarmup);
                bgApp.Stop(kSendStop);
            }
        }
    }

    FlowMonitorHelper flowmon;
    Ptr<FlowMonitor> monitor = flowmon.InstallAll();
    // Classifier for filtering AC_BE background out of the UHR KPI; shared by
    // the settle-baseline reset and the end-of-run aggregation.
    Ptr<Ipv4FlowClassifier> classifier =
        DynamicCast<Ipv4FlowClassifier>(flowmon.GetClassifier());

    // Traffic + Python handshake start AFTER the warmup so association is done.
    Simulator::Schedule(kWarmup, &SendTick, &apps);
    Simulator::Schedule(kWarmup, &SlotTick, msgInterface, &apps, monitor);
    // F6: Markov 채널 전이 시작 (association 은 초기 정적 상태에서 완료).
    if (g_chanDwellMs > 0.0)
    {
        g_chanRng = CreateObject<UniformRandomVariable>();
        Simulator::Schedule(kWarmup, &ChanTick);
    }
    // F8: 주기 drain — MAC 큐가 빌 때 공유버퍼가 후속 공급되도록.
    if (g_drainTarget > 0)
    {
        Simulator::Schedule(kWarmup, &DrainTick, &apps);
    }
    // Steady-state: reset KPI accumulators after the queue-settle period so the
    // final KPI excludes the post-association queue-buildup transient. No-op
    // (unscheduled) when settleMs=0 -> legacy metric preserved exactly.
    if (kSettle > MilliSeconds(0))
    {
        Simulator::Schedule(g_measStart, &ResetMeasurement, monitor,
                            classifier);
    }
    Simulator::Stop(kSimEnd);
    Simulator::Run();

    // -------- KPI 집계 --------
    monitor->CheckForLostPackets();
    auto stats = monitor->GetFlowStats();
    // Exclude AC_BE background flows (destinationPort >= kBgPort) so the UHR KPI
    // denominator (tx = decided + lost) counts UHR traffic only. When bgPps<=0
    // there are no such flows and this changes nothing.
    uint64_t totalRxPkts = 0;
    uint64_t totalTxPkts = 0;
    for (auto& kv : stats)
    {
        if (classifier)
        {
            Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(kv.first);
            if (t.destinationPort >= kBgPort)
            {
                continue;  // best-effort background — not a UHR flow
            }
        }
        totalTxPkts += kv.second.txPackets;
        totalRxPkts += kv.second.rxPackets;
    }
    // Steady-state windowing: subtract the pre-g_measStart settle baseline so
    // tx/rx (denominator + lost) count only [g_measStart, kSimEnd). Bases are 0
    // when settleMs=0, leaving the legacy totals unchanged. Floor at 0 to absorb
    // the sub-ms straddle at the reset boundary.
    totalTxPkts = (totalTxPkts >= g_txBase) ? totalTxPkts - g_txBase : 0;
    totalRxPkts = (totalRxPkts >= g_rxBase) ? totalRxPkts - g_rxBase : 0;

    // P1: decided-denominator violation rate (surrogate-consistent, anti-
    // survivor-bias). A packet violates if delivered LATE *or* never delivered
    // (dropped/stranded). Denominator = all decided packets ~= txPackets (the
    // 10ms send-drain guarantees late sends reach a terminal state before Stop).
    const double kL99 = g_l99Sec;
    const double kL999 = g_l999Sec;
    uint64_t deliveredLate99 = 0;
    uint64_t deliveredLate999 = 0;
    for (double d : g_delaysSec)
    {
        if (d > kL99) { ++deliveredLate99; }
        if (d > kL999) { ++deliveredLate999; }
    }
    const uint64_t lostPkts =
        (totalTxPkts >= totalRxPkts) ? totalTxPkts - totalRxPkts : 0;
    // F8: 공유버퍼 aged-out(+종료 잔류)은 FlowMonitor 에 안 잡히는 미전달-결정
    // — 분자(두 deadline 모두 위반)와 분모(decided)에 더한다. 잔류분은 여기서
    // 한 번만 가산 (아래 [KPI_AP] 도 같은 g_apShDropTotal 을 읽어 정합).
    for (uint32_t s = 0; s < kNumSta; ++s)
    {
        g_apShDropTotal[s / kStaPerAp] += g_shBuf[s].size();
    }
    uint64_t shDropSum = 0;
    for (uint32_t a = 0; a < kNumAp; ++a)
    {
        shDropSum += g_apShDropTotal[a];
    }
    const uint64_t v99 = deliveredLate99 + lostPkts + shDropSum;
    const uint64_t v999 = deliveredLate999 + lostPkts + shDropSum;
    const uint64_t decided = totalTxPkts + shDropSum;  // sent(→delivered|dropped) + shBuf aged-out
    const double p99Rate = (decided > 0)
        ? static_cast<double>(v99) / decided : 0.0;
    const double p999Rate = (decided > 0)
        ? static_cast<double>(v999) / decided : 0.0;

    // DIAGNOSTIC: where do packets go? tx/rx/lost + STA-device association.
    uint32_t assocDevs = 0;
    for (uint32_t s = 0; s < kNumSta; ++s)
    {
        Ptr<Node> node = staNodes.Get(s);
        for (uint32_t d = 0; d < node->GetNDevices(); ++d)
        {
            Ptr<WifiNetDevice> wnd =
                DynamicCast<WifiNetDevice>(node->GetDevice(d));
            if (!wnd) { continue; }
            Ptr<StaWifiMac> smac = DynamicCast<StaWifiMac>(wnd->GetMac());
            if (smac && smac->IsAssociated()) { ++assocDevs; }
        }
    }
    if (std::getenv("FEDDRL_DBG_DROP"))
    {
        for (uint32_t l = 0; l < kNumLinks; ++l)
        {
            std::cerr << "[DROP_DBG] link" << l
                      << " enq=" << g_dbgDropByLink[l][0]
                      << " expired=" << g_dbgDropByLink[l][1]
                      << " retry=" << g_dbgDropByLink[l][2]
                      << " qosOld=" << g_dbgDropByLink[l][3] << std::endl;
        }
    }
    std::cout << "[DIAG] tx=" << totalTxPkts << " rx=" << totalRxPkts
              << " lost=" << lostPkts << " shDrop=" << shDropSum
              << " delivered=" << g_delaysSec.size()
              << " assocDevs=" << assocDevs << "/" << (kNumSta * kNumLinks)
              << std::endl;
    // Per-AP KPI (network P1 공식의 per-AP 버전): p99_a = (late_a+lost_a)/decided_a.
    // zq 핵심주장 지표 (worst-AP p999, feasible AP 수)의 원천 — always printed.
    // F8: decided = sends + shared-buffer aged-out (잔류분은 위 P1 블록에서
    // 이미 g_apShDropTotal 에 가산됨 — 이중 가산 금지).
    for (uint32_t a = 0; a < kNumAp; ++a)
    {
        const uint64_t dec = g_apTxTotal[a] + g_apShDropTotal[a];
        const uint64_t rx = g_apRxTotal[a];
        const uint64_t lost = (dec >= rx) ? dec - rx : 0;
        const double ap99 = (dec > 0)
            ? static_cast<double>(g_apLate99Total[a] + lost) / dec : 0.0;
        const double ap999 = (dec > 0)
            ? static_cast<double>(g_apLate999Total[a] + lost) / dec : 0.0;
        std::cout << "[KPI_AP] " << a << ",decided=" << dec << ",rx=" << rx
                  << ",lost=" << lost
                  << ",p99=" << std::fixed << std::setprecision(5) << ap99
                  << ",p999=" << ap999 << std::endl;
    }
    if (std::getenv("FEDDRL_DBG_EDCA"))
    {
        // Per-AP delivered UHR over the window. Compare the SAME AP index across
        // a spread vs uniform run: if AP0's rx rises when it is made aggressive,
        // per-BSS EDCA reallocates airtime; if identical, EDCA is inert.
        std::cerr << "[EDCA_APRX]";
        for (uint32_t a = 0; a < kNumAp; ++a)
        {
            std::cerr << " " << a << ":" << g_apRxTotal[a];
        }
        std::cerr << std::endl;
        // Per-AP background (AC_BE OnOff) delivery: uniform load, so differences
        // under --bgEdca spread isolate whether EDCA reallocates airtime for a
        // clean standard generator.
        std::cerr << "[EDCA_BGRX]";
        for (uint32_t a = 0; a < kNumAp; ++a)
        {
            std::cerr << " " << a << ":" << g_bgRxTotal[a];
        }
        std::cerr << std::endl;
    }

    Simulator::Destroy();
    const double simTimeMs = kEpisodeDuration.GetMilliSeconds();

    std::time_t now = std::time(nullptr);
    char isoBuf[32];
    std::strftime(isoBuf, sizeof(isoBuf), "%Y-%m-%dT%H:%M:%S",
                  std::localtime(&now));

    std::cout
        << "[KPI] "
        << seed << ","
        << baselineTag << ","
        << arrivalPps << ","
        << "20000" << ","
        << "0.0" << ","
        << std::fixed << std::setprecision(4) << p99Rate << ","
        << p999Rate << ","
        << totalRxPkts << ","
        << "0.0" << ","
        << "0.0" << ","
        << "0.0" << ","
        << simTimeMs << ","
        << isoBuf
        << std::endl;

    NS_LOG_INFO("KPI: rx=" << totalRxPkts << "/tx=" << totalTxPkts
                           << " p99=" << p99Rate
                           << " p99.9=" << p999Rate);
    return 0;
}
