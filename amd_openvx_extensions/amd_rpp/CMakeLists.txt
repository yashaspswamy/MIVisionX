# Copyright (c) 2019 - 2020 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

cmake_minimum_required(VERSION 3.0)
project(vx_rpp)

set(CMAKE_CXX_STANDARD 11)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../amd_openvx/cmake)

find_package(AMDRPP    REQUIRED)
find_package(OpenCL    QUIET)

include_directories(../../amd_openvx/openvx/include/
                    ${AMDRPP_INCLUDE_DIRS}
                    include
                   )

list(APPEND SOURCES
        source/AbsoluteDifferencebatchPD.cpp
        source/AccumulatebatchPD.cpp
        source/AccumulateSquaredbatchPD.cpp
        source/AccumulateWeightedbatchPD.cpp
        source/AddbatchPD.cpp
        source/BilateralFilterbatchPD.cpp
        source/BitwiseANDbatchPD.cpp
        source/BitwiseNOTbatchPD.cpp
        source/BlendbatchPD.cpp
        source/BlurbatchPD.cpp
        source/BoxFilterbatchPD.cpp
        source/BrightnessbatchPD.cpp
        source/CannyEdgeDetector.cpp
        source/ChannelCombinebatchPD.cpp
        source/ChannelExtractbatchPD.cpp
        source/ColorTemperaturebatchPD.cpp
        source/ColorTwistbatchPD.cpp
        source/ContrastbatchPD.cpp
        source/ControlFlowbatchPD.cpp 
        source/copy.cpp
        source/CropMirrorNormalizePD.cpp
        source/CropPD.cpp
        source/CustomConvolutionbatchPD.cpp
        source/DataObjectCopybatchPD.cpp
        source/DilatebatchPD.cpp
        source/ErodebatchPD.cpp
        source/ExclusiveORbatchPD.cpp
        source/ExposurebatchPD.cpp
        source/FastCornerDetector.cpp  
        source/FisheyebatchPD.cpp
        source/FlipbatchPD.cpp
        source/FogbatchPD.cpp
        source/GammaCorrectionbatchPD.cpp
        source/GaussianFilterbatchPD.cpp 
        source/GaussianImagePyramidbatchPD.cpp
        source/HarrisCornerDetector.cpp   
        source/Histogram.cpp
        source/HistogramBalancebatchPD.cpp
        source/HistogramEqualizebatchPD.cpp
        source/HuebatchPD.cpp
        source/InclusiveORbatchPD.cpp
        source/JitterbatchPD.cpp
        source/LaplacianImagePyramid.cpp
        source/LensCorrectionbatchPD.cpp
        source/LocalBinaryPatternbatchPD.cpp
        source/LookUpTablebatchPD.cpp 
        source/MagnitudebatchPD.cpp
        source/MaxbatchPD.cpp
        source/MeanStddev.cpp
        source/MedianFilterbatchPD.cpp
        source/MinbatchPD.cpp
        source/MinMaxLoc.cpp
        source/Multiply.cpp
        source/MultiplybatchPD.cpp
        source/MultiplybatchPDROID.cpp
        source/MultiplybatchPS.cpp
        source/Noise.cpp
        source/NoisebatchPD.cpp
        source/NoisebatchPDROID.cpp
        source/NoisebatchPS.cpp
        source/NonLinearFilter.cpp   
        source/NonLinearFilterbatchPD.cpp
        source/NonLinearFilterbatchPDROID.cpp
        source/NonLinearFilterbatchPS.cpp   
        source/NonMaxSupression.cpp
        source/NonMaxSupressionbatchPD.cpp
        source/NonMaxSupressionbatchPDROID.cpp
        source/NonMaxSupressionbatchPS.cpp   
        source/nop.cpp
        source/Occlusion.cpp
        source/OcclusionbatchPD.cpp
        source/OcclusionbatchPDROID.cpp
        source/OcclusionbatchPS.cpp   
        source/Phase.cpp
        source/PhasebatchPD.cpp
        source/PhasebatchPDROID.cpp
        source/PhasebatchPS.cpp
        source/Pixelate.cpp
        source/PixelatebatchPD.cpp
        source/PixelatebatchPDROID.cpp
        source/PixelatebatchPS.cpp
        source/Rain.cpp
        source/RainbatchPD.cpp
        source/RainbatchPDROID.cpp
        source/RainbatchPS.cpp
        source/RandomCropLetterBox.cpp
        source/RandomCropLetterBoxbatchPD.cpp
        source/RandomCropLetterBoxbatchPDROID.cpp
        source/RandomCropLetterBoxbatchPS.cpp   
        source/RandomShadow.cpp
        source/RandomShadowbatchPD.cpp
        source/RandomShadowbatchPDROID.cpp
        source/RandomShadowbatchPS.cpp   
        source/Remap.cpp
        source/Resize.cpp
        source/ResizebatchPD.cpp
        source/ResizebatchPDROID.cpp
        source/ResizebatchPS.cpp   
        source/ResizeCrop.cpp
        source/ResizeCropbatchPD.cpp
        source/ResizeCropbatchPDROID.cpp
        source/ResizeCropbatchPS.cpp   
        source/ResizeCropMirrorPD.cpp
        source/Rotate.cpp   
        source/RotatebatchPD.cpp
        source/RotatebatchPDROID.cpp
        source/RotatebatchPS.cpp   
        source/Saturation.cpp
        source/SaturationbatchPD.cpp
        source/SaturationbatchPDROID.cpp
        source/SaturationbatchPS.cpp   
        source/Scale.cpp    
        source/ScalebatchPD.cpp
        source/ScalebatchPDROID.cpp
        source/ScalebatchPS.cpp   
        source/Snow.cpp
        source/SnowbatchPD.cpp
        source/SnowbatchPDROID.cpp
        source/SnowbatchPS.cpp
        source/Sobel.cpp
        source/SobelbatchPD.cpp
        source/SobelbatchPDROID.cpp   
        source/SobelbatchPS.cpp   
        source/Subtract.cpp
        source/SubtractbatchPD.cpp
        source/SubtractbatchPDROID.cpp
        source/SubtractbatchPS.cpp
        source/TensorAdd.cpp
        source/TensorLookup.cpp   
        source/TensorMatrixMultiply.cpp   
        source/TensorMultiply.cpp
        source/TensorSubtract.cpp   
        source/Thresholding.cpp
        source/ThresholdingbatchPD.cpp
        source/ThresholdingbatchPDROID.cpp
        source/ThresholdingbatchPS.cpp
        source/Vignette.cpp
        source/VignettebatchPD.cpp
        source/VignettebatchPDROID.cpp
        source/VignettebatchPS.cpp   
        source/WarpAffine.cpp
        source/WarpAffinebatchPD.cpp
        source/WarpAffinebatchPDROID.cpp
        source/WarpAffinebatchPS.cpp
        source/WarpPerspective.cpp
        source/WarpPerspectivebatchPD.cpp
        source/WarpPerspectivebatchPDROID.cpp
        source/WarpPerspectivebatchPS.cpp
        source/kernel_rpp.cpp
        source/internal_publishKernels.cpp
    )

link_directories(${AMDRPP_LIBRARIES_DIR})

if(OpenCL_FOUND)
	message("-- ${Green}AMD VX RPP -- BUILDING WITH OPENCL BACKEND${ColourReset}")
	set(ENABLE_OPENCL 1)
	add_definitions(-DENABLE_OPENCL=${ENABLE_OPENCL} -DRPP_BACKEND_OPENCL=1)
	include_directories(${OpenCL_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS}/Headers)
	add_library(vx_rpp SHARED ${SOURCES})
	target_link_libraries(vx_rpp amd_rpp boost_filesystem boost_system ${OpenCL_LIBRARIES})
else(OpenCL_FOUND)
	message("-- ${Red}AMD VX RPP -- OPENCL NOT FOUND${ColourReset}")
	set(ENABLE_OPENCL 0)
	add_definitions(-DENABLE_OPENCL=${ENABLE_OPENCL})
	add_library(vx_rpp SHARED ${SOURCES})
	target_link_libraries(vx_rpp amd_rpp boost_filesystem boost_system)
endif(OpenCL_FOUND)

install(TARGETS vx_rpp DESTINATION lib)
install(FILES include/vx_ext_rpp.h DESTINATION include)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	target_link_libraries(OpenVX)
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
else()
	target_link_libraries( openvx)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdata-sections")
endif()
