< envPaths
errlogInit(20000)

dbLoadDatabase("$(TOP)/dbd/hdf5DriverApp.dbd")
hdf5DriverApp_registerRecordDeviceDriver(pdbbase)

#epicsEnvSet("PREFIX", "13HDF1:")
epicsEnvSet("PREFIX", "Eiger1M:")
epicsEnvSet("PORT",   "HDF")
epicsEnvSet("QSIZE",  "20")
epicsEnvSet("NCHANS", "2048")
epicsEnvSet("XSIZE",  "1030")
epicsEnvSet("YSIZE",  "1065")
epicsEnvSet("CBUFFS", "500")
epicsEnvSet("EPICS_DB_INCLUDE_PATH", "$(ADCORE)/db:$(ADHDF5)/db")
epicsEnvSet("EPICS_CA_MAX_ARRAY_BYTES", "5000000")

hdf5DriverConfig("$(PORT)", 0, 0)
#dbLoadRecords("$(ADCORE)/db/ADBase.template", "P=$(PREFIX),R=cam1:,PORT=$(PORT),ADDR=0,TIMEOUT=1")
dbLoadRecords("ADBase.template", "P=$(PREFIX),R=cam1:,PORT=$(PORT),ADDR=0,TIMEOUT=1")
#dbLoadRecords("$(ADHDF5)/db/hdf5.template","P=$(PREFIX),R=cam1:,PORT=$(PORT),ADDR=0,TIMEOUT=1,SERVER_PORT=server")
dbLoadRecords("hdf5.template","P=$(PREFIX),R=cam1:,PORT=$(PORT),ADDR=0,TIMEOUT=1,SERVER_PORT=server")

NDPvaConfigure("Pva1", 3, 0, "$(PORT)", 0, "$(PREFIX)pva1:Image")
dbLoadRecords("NDPva.template","P=$(PREFIX),R=pva1:,PORT=Pva1,ADDR=0,TIMEOUT=1,NDARRAY_PORT=$(PORT),NDARRAY_ADDR=0")

#NDStatsConfigure("Stats", 3, 0, "$(PORT)", 0)
#dbLoadRecords("NDStats.template", "P=$(PREFIX),R=stats:,PORT=Stats,ADDR=0,TIMEOUT=1")

# Create a standard arrays plugin 1030x1065
NDStdArraysConfigure("Image1", 5, 0, "$(PORT)", 0, 0)
dbLoadRecords("$(ADCORE)/db/NDStdArrays.template", "P=$(PREFIX),R=image1:,PORT=Image1,ADDR=0,TIMEOUT=1,TYPE=Int32,FTVL=LONG,NELEMENTS=1096950, NDARRAY_PORT=$(PORT)")

# Load all other plugins using commonPlugins.cmd
< $(ADCORE)/iocBoot/commonPlugins.cmd
set_requestfile_path("$(ADHDF5)/hdf5App/Db")

#asynSetTraceMask("$(PORT)",0,255)
#asynSetTraceMask("$(PORT)",0,3)


iocInit()

# save things every thirty seconds
# create_monitor_set("auto_settings.req", 30,"P=$(PREFIX)")

dbpf $(PREFIX)cam1:HDF5FilePath "/home/bmartins/eiger_series/x1_insu_0p05d_0p05s_0p01beam_master.h5"
