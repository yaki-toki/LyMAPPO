# CMakeLists.txt — ns3-ai contrib example for the FedDRL scenario.
#
# This file is copied to ~/ns-3-dev/contrib/ai/examples/feddrl/ and is evaluated
# when the ns-3 top-level cmake add_subdirectory()s it through
# contrib/ai/examples/CMakeLists.txt (same pattern as a-plus-b).

build_lib_example(
        NAME ns3ai_feddrl
        SOURCE_FILES feddrl_scenario.cc
        LIBRARIES_TO_LINK ${libai} ${libwifi} ${libapplications} ${libinternet}
                          ${libmobility} ${libflow-monitor}
)

pybind11_add_module(ns3ai_feddrl_py feddrl_py.cc)
set_target_properties(ns3ai_feddrl_py PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Build the pybind11 binding as well, before the C++ scenario build starts.
add_dependencies(ns3ai_feddrl ns3ai_feddrl_py)
