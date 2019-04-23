add_library( slam SHARED
		config.cpp
		RGBDOdometry.cpp )

target_link_libraries( slam ${THIRD_PARTY_LIBS} )
