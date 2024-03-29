cmake_minimum_required(VERSION 3.24)

#Name your project here
project(raytrace LANGUAGES CXX)

configure_file(spheres.txt 
				spheres.txt COPYONLY)

configure_file(lights.txt 
				lights.txt COPYONLY)

#create an executable
add_executable(raytrace
				Prog_AlanDevkota.cpp
)

target_link_libraries(raytrace
)