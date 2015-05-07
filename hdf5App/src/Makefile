TOP=../..
include $(TOP)/configure/CONFIG
#----------------------------------------
#  ADD MACRO DEFINITIONS AFTER THIS LINE

LIBRARY_IOC_Linux = hdf5Driver
LIBRARY_IOC_Darwin = hdf5Driver

LIB_SRCS += hdf5Driver.cpp

DBD += hdf5DriverSupport.dbd

LIB_LIBS_Linux += hdf5_hl 
hdf5_hl_DIR = $(HDF5_LIB)
USR_INCLUDES += $(HDF5_INCLUDE)

include $(ADCORE)/ADApp/commonLibraryMakefile

#=============================

include $(TOP)/configure/RULES
#----------------------------------------
#  ADD RULES AFTER THIS LINE

