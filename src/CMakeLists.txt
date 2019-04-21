add_library( slam SHARED 
	config.cpp	)

target_link_libraries( slam ${THIRD_PARTY_LIBS} )
