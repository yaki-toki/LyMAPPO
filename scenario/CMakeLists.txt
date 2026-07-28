# CMakeLists.txt — ns3-ai contrib example for the FedDRL scenario.
#
# 본 파일은 ~/ns-3-dev/contrib/ai/examples/feddrl/ 로 복사된 후 ns-3 의
# top-level cmake 가 contrib/ai/examples/CMakeLists.txt 를 통해 add_subdirectory
# 할 때 평가된다 (a-plus-b 와 동일 패턴).

build_lib_example(
        NAME ns3ai_feddrl
        SOURCE_FILES feddrl_scenario.cc
        LIBRARIES_TO_LINK ${libai} ${libwifi} ${libapplications} ${libinternet}
                          ${libmobility} ${libflow-monitor}
)

pybind11_add_module(ns3ai_feddrl_py feddrl_py.cc)
set_target_properties(ns3ai_feddrl_py PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# C++ 시나리오 빌드가 시작되기 전에 pybind11 binding 도 함께 빌드.
add_dependencies(ns3ai_feddrl ns3ai_feddrl_py)
