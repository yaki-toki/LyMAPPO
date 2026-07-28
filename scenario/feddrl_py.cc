/*
 * feddrl_py.cc — pybind11 binding for the feddrl ns3-ai shared-memory wire
 * format. Mirrors a-plus-b/use-msg-stru/apb_py.cc with our EnvMsg / ActMsg
 * structs.
 *
 * Built as ``ns3ai_feddrl_py`` shared library; loaded by feddrl.py as the
 * ``py_binding`` argument to ``ns3ai_utils.Experiment``.
 */

#include "feddrl_msg.h"

#include <ns3/ai-module.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using feddrl::EnvMsg;
using feddrl::ActMsg;
using feddrl::kNumAp;
using feddrl::kStaPerAp;
using feddrl::kNumLinks;
using feddrl::kNumSta;

PYBIND11_MODULE(ns3ai_feddrl_py, m)
{
    m.attr("N_AP") = py::int_(kNumAp);
    m.attr("N_STA_PER_AP") = py::int_(kStaPerAp);
    m.attr("N_LINKS") = py::int_(kNumLinks);
    m.attr("N_STA") = py::int_(kNumSta);

    // EnvMsg: C++ -> Python observation.
    // 배열 필드는 raw memoryview 로 노출 (pybind11 가 fixed-size C 배열을
    // numpy buffer 로 자동 변환하지 못하므로 lambda accessor 사용).
    py::class_<EnvMsg>(m, "PyEnvMsg")
        .def(py::init<>())
        .def_readwrite("nowUs", &EnvMsg::nowUs)
        .def(
            "queueLen",
            [](EnvMsg& self) {
                return py::memoryview::from_buffer(
                    self.queueLen,
                    sizeof(uint32_t),
                    "I",
                    {static_cast<py::ssize_t>(kNumSta)},
                    {static_cast<py::ssize_t>(sizeof(uint32_t))});
            })
        .def(
            "holUs",
            [](EnvMsg& self) {
                return py::memoryview::from_buffer(
                    self.holUs,
                    sizeof(uint32_t),
                    "I",
                    {static_cast<py::ssize_t>(kNumSta)},
                    {static_cast<py::ssize_t>(sizeof(uint32_t))});
            })
        .def(
            "cbr",
            [](EnvMsg& self) {
                return py::memoryview::from_buffer(
                    &self.cbr[0][0],
                    sizeof(uint8_t),
                    "B",
                    {static_cast<py::ssize_t>(kNumAp),
                     static_cast<py::ssize_t>(kNumLinks)},
                    {static_cast<py::ssize_t>(kNumLinks * sizeof(uint8_t)),
                     static_cast<py::ssize_t>(sizeof(uint8_t))});
            })
        .def(
            "served",
            [](EnvMsg& self) {
                return py::memoryview::from_buffer(
                    self.served,
                    sizeof(uint32_t),
                    "I",
                    {static_cast<py::ssize_t>(kNumAp)},
                    {static_cast<py::ssize_t>(sizeof(uint32_t))});
            })
        .def(
            "violation99",
            [](EnvMsg& self) {
                return py::memoryview::from_buffer(
                    self.violation99,
                    sizeof(uint32_t),
                    "I",
                    {static_cast<py::ssize_t>(kNumAp)},
                    {static_cast<py::ssize_t>(sizeof(uint32_t))});
            })
        .def(
            "violation999",
            [](EnvMsg& self) {
                return py::memoryview::from_buffer(
                    self.violation999,
                    sizeof(uint32_t),
                    "I",
                    {static_cast<py::ssize_t>(kNumAp)},
                    {static_cast<py::ssize_t>(sizeof(uint32_t))});
            })
        .def(
            "dropped",
            [](EnvMsg& self) {
                return py::memoryview::from_buffer(
                    self.dropped,
                    sizeof(uint32_t),
                    "I",
                    {static_cast<py::ssize_t>(kNumAp)},
                    {static_cast<py::ssize_t>(sizeof(uint32_t))});
            })
        .def(
            "csi",
            [](EnvMsg& self) {
                return py::memoryview::from_buffer(
                    &self.csi[0][0],
                    sizeof(float),
                    "f",
                    {static_cast<py::ssize_t>(kNumSta),
                     static_cast<py::ssize_t>(kNumLinks)},
                    {static_cast<py::ssize_t>(kNumLinks * sizeof(float)),
                     static_cast<py::ssize_t>(sizeof(float))});
            });

    // ActMsg: Python -> C++ action.
    py::class_<ActMsg>(m, "PyActMsg")
        .def(py::init<>())
        .def_readwrite("mapMode", &ActMsg::mapMode)
        .def(
            "selectedLink",
            [](ActMsg& self) {
                return py::memoryview::from_buffer(
                    &self.selectedLink[0][0],
                    sizeof(int8_t),
                    "b",
                    {static_cast<py::ssize_t>(kNumAp),
                     static_cast<py::ssize_t>(kStaPerAp)},
                    {static_cast<py::ssize_t>(kStaPerAp * sizeof(int8_t)),
                     static_cast<py::ssize_t>(sizeof(int8_t))});
            })
        .def(
            "set_selected_link",
            [](ActMsg& self, uint32_t ap, uint32_t sta, int8_t link) {
                if (ap < kNumAp && sta < kStaPerAp)
                {
                    self.selectedLink[ap][sta] = link;
                }
            },
            py::arg("ap"),
            py::arg("sta"),
            py::arg("link"));

    py::class_<ns3::Ns3AiMsgInterfaceImpl<EnvMsg, ActMsg>>(
        m,
        "Ns3AiMsgInterfaceImpl")
        .def(py::init<bool,
                      bool,
                      bool,
                      uint32_t,
                      const char*,
                      const char*,
                      const char*,
                      const char*>())
        .def("PyRecvBegin",
             &ns3::Ns3AiMsgInterfaceImpl<EnvMsg, ActMsg>::PyRecvBegin)
        .def("PyRecvEnd",
             &ns3::Ns3AiMsgInterfaceImpl<EnvMsg, ActMsg>::PyRecvEnd)
        .def("PySendBegin",
             &ns3::Ns3AiMsgInterfaceImpl<EnvMsg, ActMsg>::PySendBegin)
        .def("PySendEnd",
             &ns3::Ns3AiMsgInterfaceImpl<EnvMsg, ActMsg>::PySendEnd)
        .def("PyGetFinished",
             &ns3::Ns3AiMsgInterfaceImpl<EnvMsg, ActMsg>::PyGetFinished)
        .def("GetCpp2PyStruct",
             &ns3::Ns3AiMsgInterfaceImpl<EnvMsg, ActMsg>::GetCpp2PyStruct,
             py::return_value_policy::reference)
        .def("GetPy2CppStruct",
             &ns3::Ns3AiMsgInterfaceImpl<EnvMsg, ActMsg>::GetPy2CppStruct,
             py::return_value_policy::reference);
}
