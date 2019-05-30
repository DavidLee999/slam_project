add_library( slam SHARED
		config.cpp
		orb.cpp
		Frame.cpp)

include_directories( "/usr/local/Cellar/eigen/3.3.7/include/eigen3" )

target_link_libraries( slam ${THIRD_PARTY_LIBS} )
