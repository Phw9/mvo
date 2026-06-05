cmake_minimum_required(VERSION 3.5...4.3)
project(DBoW2 LANGUAGES CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(HDRS_DBOW2
  DBoW2/BowVector.h
  DBoW2/FORB.h 
  DBoW2/FClass.h       
  DBoW2/FeatureVector.h
  DBoW2/ScoringObject.h   
  DBoW2/TemplatedVocabulary.h
  DBoW2/QueryResults.h)
  
set(SRCS_DBOW2
  DBoW2/BowVector.cpp
  DBoW2/FORB.cpp      
  DBoW2/FeatureVector.cpp
  DBoW2/ScoringObject.cpp
  DBoW2/QueryResults.cpp)

set(HDRS_DUTILS
  DUtils/Random.h
  DUtils/Timestamp.h)
set(SRCS_DUTILS
  DUtils/Random.cpp
  DUtils/Timestamp.cpp)

find_package(OpenCV REQUIRED)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

add_library(DBoW2 STATIC ${SRCS_DBOW2} ${SRCS_DUTILS})
target_include_directories(DBoW2 PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${OpenCV_INCLUDE_DIRS})
target_link_libraries(DBoW2 PUBLIC ${OpenCV_LIBS})
target_compile_features(DBoW2 PUBLIC cxx_std_11)

if(MSVC)
  target_compile_options(DBoW2 PRIVATE /W3)
else()
  target_compile_options(DBoW2 PRIVATE -Wall -Wextra -O3)
endif()

