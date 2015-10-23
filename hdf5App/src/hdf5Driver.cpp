/* hdf5Driver.cpp
 *
 * This is a driver for HDF5 files.
 *
 * Authors: Bruno Martins
 *          Brookhaven National Laboratory
 *
 * Created: May 6, 2015
 */

#include <epicsExport.h>
#include <epicsThread.h>
#include <iocsh.h>
#include <asynOctetSyncIO.h>
#include <algorithm>
#include <epicsString.h>

#include <hdf5.h>
#include <hdf5_hl.h>

#include "ADDriver.h"

#if defined(_WIN32)              // Windows
  #include <direct.h>
  #define MKDIR(a,b) _mkdir(a)
  #define DELIM '\\'
#elif defined(vxWorks)           // VxWorks
  #include <sys/stat.h>
  #define MKDIR(a,b) mkdir(a)
  #define DELIM '/'
#else                            // Linux
  #include <sys/stat.h>
  #include <sys/types.h>
  #define MKDIR(a,b) mkdir(a,b)
  #define DELIM '/'
#endif

#define FAIL_IF(cond,statement,msg)\
    do{if(cond){\
        asynPrint(pasynUserSelf, ASYN_TRACE_ERROR, "%s:%s %s\n",\
                driverName, functionName, msg);\
        status = asynError;\
        statement;\
    }}while(0)

#define FAIL_IF_ARGS(cond,statement,fmt,...)\
    do{if(cond){\
        asynPrint(pasynUserSelf, ASYN_TRACE_ERROR, "%s:%s"fmt"\n",\
                driverName, functionName, __VA_ARGS__);\
        status = asynError;\
        statement;\
    }}while(0)

#define HDF5FilePathString      "HDF5_FILE_PATH"
#define HDF5FileExistsString    "HDF5_FILE_EXISTS"
#define HDF5DatasetsCountString "HDF5_DATASETS_COUNT"
#define HDF5FirstFrameString    "HDF5_FIRST_FRAME"
#define HDF5LastFrameString     "HDF5_LAST_FRAME"
#define HDF5TotalFramesString   "HDF5_TOTAL_FRAMES"
#define HDF5CurrentFrameString  "HDF5_CURRENT_FRAME"
#define HDF5AutoLoadFrameString "HDF5_AUTO_LOAD"
#define HDF5LoopString          "HDF5_LOOP"
#define HDF5InMemoryString      "HDF5_IN_MEMORY"

static const char *driverName = "hdf5Driver";

// Driver for HDF5 files
class hdf5Driver : public ADDriver
{
public:
    hdf5Driver(const char *portName, int maxBuffers, size_t maxMemory,
            int priority, int stackSize);

    /* These are the methods that we override from ADDriver */
    virtual asynStatus writeInt32(asynUser *pasynUser, epicsInt32 value);
    virtual asynStatus writeOctet(asynUser *pasynUser, const char *value,
                                    size_t nChars, size_t *nActual);
    void report(FILE *fp, int details);

    /* This should be private but is called from C so must be public */
    void hdf5Task();

protected:
    int HDF5FilePath;
    #define FIRST_HDF5_PARAM HDF5FilePath
    int HDF5FileExists;
    int HDF5DatasetsCount;
    int HDF5FirstFrame;
    int HDF5LastFrame;
    int HDF5TotalFrames;
    int HDF5CurrentFrame;
    int HDF5AutoLoadFrame;
    int HDF5Loop;
    int HDF5InMemory;
    #define LAST_HDF5_PARAM HDF5InMemory

private:
    typedef struct dSetInfo
    {
        hid_t id;
        int imageNrLow;
        int imageNrHigh;
        size_t width, height, nFrames;
        NDDataType_t type;
    }dSetInfo_t;

    epicsEventId mStartEventId;
    epicsEventId mStopEventId;
    size_t       mDatasetsCount;
    dSetInfo_t   *mpDatasets;

    /* These are the methods that are new to this class */
    bool checkFilePath      (const char *path);
    char *getFolder         (const char *path, size_t *folderLen);
    asynStatus loadFile     (const char *path);
    asynStatus openFile     (const char *path, hid_t *fId);
    asynStatus openDataset  (hid_t gId, dSetInfo_t *dInfo, const char *dName,
            const char *folder);
    struct dSetInfo *getDatasetByFrame (int frame);
    asynStatus getFrameInfo (int frame, size_t *pDims, NDDataType_t *pType);
    asynStatus getFrameData (int frame, void *pData);
    asynStatus parseType    (hid_t id, NDDataType_t *pNDType);
};

#define NUM_HDF5_PARAMS ((int)(&LAST_HDF5_PARAM - &FIRST_HDF5_PARAM + 1))

static void hdf5TaskC (void *pDrvPvt)
{
    hdf5Driver *pPvt = (hdf5Driver *)pDrvPvt;
    pPvt->hdf5Task();
}

/** Constructor for HDF5 driver; most parameters are simply passed to
  * ADDriver::ADDriver.
  * After calling the base class constructor this method creates a thread to
  * collect the detector data, and sets reasonable default values for the
  * parameters defined in this class, asynNDArrayDriver, and ADDriver.
  * \param[in] portName The name of the asyn port driver to be created.
  * \param[in] maxBuffers The maximum number of NDArray buffers that the
  *            NDArrayPool for this driver is allowed to allocate. Set this to
  *            -1 to allow an unlimited number of buffers.
  * \param[in] maxMemory The maximum amount of memory that the NDArrayPool for
  *            this driver is allowed to allocate. Set this to -1 to allow an
  *            unlimited amount of memory.
  * \param[in] priority The thread priority for the asyn port driver thread if
  *            ASYN_CANBLOCK is set in asynFlags.
  * \param[in] stackSize The stack size for the asyn port driver thread if
  *            ASYN_CANBLOCK is set in asynFlags.
  */
hdf5Driver::hdf5Driver (const char *portName, int maxBuffers, size_t maxMemory,
        int priority, int stackSize)
    : ADDriver(portName, 1, NUM_HDF5_PARAMS, maxBuffers, maxMemory,
            0, 0,             /* No interfaces beyond ADDriver.cpp */
            ASYN_CANBLOCK,    /* ASYN_CANBLOCK=1, ASYN_MULTIDEVICE=0 */
            1,                /* autoConnect=1 */
            priority, stackSize),
      mStartEventId(epicsEventCreate(epicsEventEmpty)),
      mStopEventId(epicsEventCreate(epicsEventEmpty)),
      mDatasetsCount(0), mpDatasets(NULL)
{
    int status = asynSuccess;
    const char *functionName = "hdf5Driver";

    if (!mStartEventId)
    {
        asynPrint(pasynUserSelf, ASYN_TRACE_ERROR,
                "%s:%s epicsEventCreate failure for start event\n",
                driverName, functionName);
        return;
    }

    if (!mStopEventId)
    {
        asynPrint(pasynUserSelf, ASYN_TRACE_ERROR,
                "%s:%s epicsEventCreate failure for stop event\n",
                driverName, functionName);
        return;
    }

    createParam(HDF5FilePathString,       asynParamOctet, &HDF5FilePath);
    createParam(HDF5FileExistsString,     asynParamInt32, &HDF5FileExists);
    createParam(HDF5DatasetsCountString,  asynParamInt32, &HDF5DatasetsCount);
    createParam(HDF5FirstFrameString,     asynParamInt32, &HDF5FirstFrame);
    createParam(HDF5LastFrameString,      asynParamInt32, &HDF5LastFrame);
    createParam(HDF5TotalFramesString,    asynParamInt32, &HDF5TotalFrames);
    createParam(HDF5CurrentFrameString,   asynParamInt32, &HDF5CurrentFrame);
    createParam(HDF5AutoLoadFrameString,  asynParamInt32, &HDF5AutoLoadFrame);
    createParam(HDF5LoopString,           asynParamInt32, &HDF5Loop);
    createParam(HDF5InMemoryString,       asynParamInt32, &HDF5InMemory);

    status = asynSuccess;

    status |= setStringParam (HDF5FilePath,         "");
    status |= setIntegerParam(HDF5FileExists,       0);
    status |= setIntegerParam(HDF5FirstFrame,       0);
    status |= setIntegerParam(HDF5LastFrame,        0);
    status |= setIntegerParam(HDF5TotalFrames,      0);
    status |= setIntegerParam(HDF5CurrentFrame,     0);
    status |= setIntegerParam(HDF5AutoLoadFrame,    0);
    status |= setIntegerParam(HDF5Loop,             0);
    status |= setIntegerParam(HDF5InMemory,         0);
    status |= setDoubleParam (ADAcquirePeriod,      0.02);

    callParamCallbacks();

    if (status)
    {
        printf("%s: unable to set driver parameters\n", functionName);
        return;
    }

    /* Create the thread that updates the images */
    status = (epicsThreadCreate("hdf5DrvTask", epicsThreadPriorityMedium,
            epicsThreadGetStackSize(epicsThreadStackMedium),
            (EPICSTHREADFUNC)hdf5TaskC, this) == NULL);

    if (status)
    {
        asynPrint(pasynUserSelf, ASYN_TRACE_ERROR,
                "%s:%s epicsThreadCreate failure for task\n",
                driverName, functionName);
        return;
    }
}

/** Called when asyn clients call pasynInt32->write().
  * This function performs actions for some parameters, including ADAcquire,
  * ADTriggerMode, etc.
  * For all parameters it sets the value in the parameter library and calls any
  * registered callbacks..
  * \param[in] pasynUser pasynUser structure that encodes the reason and
  *            address.
  * \param[in] value Value to write. */
asynStatus hdf5Driver::writeInt32 (asynUser *pasynUser, epicsInt32 value)
{
    int function = pasynUser->reason;
    asynStatus status = asynSuccess;
    const char *functionName = "writeInt32";
    int adstatus, acquiring, imageMode;

    getIntegerParam(ADAcquire,   &acquiring);
    getIntegerParam(ADStatus,    &adstatus);
    getIntegerParam(ADImageMode, &imageMode);

    if (function == ADAcquire)
    {
        if (value && !acquiring)
        {
            setStringParam(ADStatusMessage, "Acquiring data");
            setIntegerParam(ADStatus, ADStatusAcquire);
            callParamCallbacks();
            epicsEventSignal(this->mStartEventId);
        }
        else if (!value && acquiring)
        {
            setStringParam(ADStatusMessage, "Acquisition stopped");
            if(imageMode == ADImageContinuous)
                setIntegerParam(ADStatus, ADStatusIdle);
            else
                setIntegerParam(ADStatus, ADStatusAborted);
            callParamCallbacks();
            epicsEventSignal(this->mStopEventId);
        }
    }
    else if(function == HDF5CurrentFrame)
    {
        int first, last;
        getIntegerParam(HDF5FirstFrame, &first);
        getIntegerParam(HDF5LastFrame, &last);

        if(value < first)
            value = first;
        else if(value > last)
            value = last;
    }
    else if(function < FIRST_HDF5_PARAM)
        status = ADDriver::writeInt32(pasynUser, value);

    if(status)
    {
        asynPrint(pasynUser, ASYN_TRACE_ERROR,
              "%s:%s: error, status=%d function=%d, value=%d\n",
              driverName, functionName, status, function, value);
        return status;
    }

    status = setIntegerParam(function, value);
    callParamCallbacks();

    if (status)
        asynPrint(pasynUser, ASYN_TRACE_ERROR,
              "%s:%s: error, status=%d function=%d, value=%d\n",
              driverName, functionName, status, function, value);
    else
        asynPrint(pasynUser, ASYN_TRACEIO_DRIVER,
              "%s:%s: function=%d, value=%d\n",
              driverName, functionName, function, value);

    return status;
}

/** Called when asyn clients call pasynOctet->write().
  * For all parameters it sets the value in the parameter library and calls any
  * registered callbacks.
  * \param[in] pasynUser pasynUser structure that encodes the reason and
  *            address.
  * \param[in] value Address of the string to write.
  * \param[in] nChars Number of characters to write.
  * \param[out] nActual Number of characters actually written. */
asynStatus hdf5Driver::writeOctet (asynUser *pasynUser, const char *value,
                                    size_t nChars, size_t *nActual)
{
    int function = pasynUser->reason;
    asynStatus status = asynSuccess;
    const char *functionName = "writeOctet";

    // Not interested in base class parameters
    if(function < FIRST_HDF5_PARAM)
        return ADDriver::writeOctet(pasynUser, value, nChars, nActual);

    if(function == HDF5FilePath)
    {
        bool exists = checkFilePath(value);
        setIntegerParam(HDF5FileExists, exists);
        setStringParam(function, value);
        if(exists)
        {
            int acquiring;
            getIntegerParam(ADStatus, &acquiring);
            if(acquiring != ADStatusAcquire)
                status = loadFile(value);
            else
            {
                asynPrint(pasynUser, ASYN_TRACE_ERROR,
                        "%s:%s: can't change file path while acquiring",
                        driverName, functionName);
                status = asynError;
            }
        }
    }

    callParamCallbacks();

    if (status)
        asynPrint(pasynUser, ASYN_TRACE_ERROR,
                "%s:%s: status=%d, function=%d, value=%s",
                driverName, functionName, status, function, value);
    else
        asynPrint(pasynUser, ASYN_TRACEIO_DRIVER,
                "%s:%s: function=%d, value=%s\n",
                driverName, functionName, function, value);

    *nActual = nChars;
    return status;
}

/** Report status of the driver.
  * Prints details about the driver if details>0.
  * It then calls the ADDriver::report() method.
  * \param[in] fp File pointed passed by caller where the output is written to.
  * \param[in] details If >0 then driver details are printed.
  */
void hdf5Driver::report (FILE *fp, int details)
{
    fprintf(fp, "Eiger detector %s\n", this->portName);
    if (details > 0) {
        int nx, ny, dataType;
        getIntegerParam(ADSizeX, &nx);
        getIntegerParam(ADSizeY, &ny);
        getIntegerParam(NDDataType, &dataType);
        fprintf(fp, "  NX, NY:            %d  %d\n", nx, ny);
        fprintf(fp, "  Data type:         %d\n", dataType);
    }
    /* Invoke the base class method */
    ADDriver::report(fp, details);
}

/** This thread controls acquisition, reads image files to get the image data,
  * and does the callbacks to send it to higher layers */
void hdf5Driver::hdf5Task (void)
{
    const char *functionName = "hdf5Task";
    int status = asynSuccess;
    epicsTimeStamp startTime, endTime;
    int imageMode, currentFrame, colorMode;
    double acquirePeriod, elapsedTime, delay;

    this->lock();

    for(;;)
    {
        int acquire;
        getIntegerParam(ADAcquire, &acquire);

        if (!acquire)
        {
            this->unlock(); // Wait for semaphore unlocked

            asynPrint(this->pasynUserSelf, ASYN_TRACE_FLOW,
                    "%s:%s: waiting for acquire to start\n",
                    driverName, functionName);

            status = epicsEventWait(this->mStartEventId);

            this->lock();

            acquire = 1;
            setStringParam(ADStatusMessage, "Acquiring data");
            setIntegerParam(ADNumImagesCounter, 0);
        }

        // Are there datasets loaded?
        if(!mDatasetsCount)
        {
            setStringParam(ADStatusMessage, "No datasets loaded");
            goto error;
        }

        // Get acquisition parameters
        epicsTimeGetCurrent(&startTime);
        getIntegerParam(ADImageMode, &imageMode);
        getDoubleParam(ADAcquirePeriod, &acquirePeriod);
        getIntegerParam(HDF5CurrentFrame, &currentFrame);
        setIntegerParam(ADStatus, ADStatusAcquire);
        callParamCallbacks();

        // Get information to allocate NDArray
        size_t dims[2];
        NDDataType_t dataType;
        if(getFrameInfo(currentFrame, dims, &dataType))
        {
            setStringParam(ADStatusMessage, "Failed to get frame info");
            goto error;
        }

        // Allocate NDArray
        NDArray *pImage;
        if(!(pImage = pNDArrayPool->alloc(2, dims, dataType, 0, NULL)))
        {
            setStringParam(ADStatusMessage, "Failed to allocate frame");
            goto error;
        }

        // Copy data into NDArray
        if(getFrameData(currentFrame, pImage->pData))
        {
            setStringParam(ADStatusMessage, "Failed to read frame data");
            goto error;
        }

        // Set ColorMode
        colorMode = NDColorModeMono;
        pImage->pAttributeList->add("ColorMode", "Color mode", NDAttrInt32,
                &colorMode);

        // Call plugins callbacks
        int arrayCallbacks;
        getIntegerParam(NDArrayCallbacks, &arrayCallbacks);
        if (arrayCallbacks)
        {
          this->unlock();
          asynPrint(this->pasynUserSelf, ASYN_TRACE_FLOW,
                    "%s:%s: calling imageData callback\n",
                    driverName, functionName);
          doCallbacksGenericPointer(pImage, NDArrayData, 0);
          this->lock();
        }
        pImage->release();

        // Get the current parameters
        int lastFrame, imageCounter, numImages, numImagesCounter;
        getIntegerParam(HDF5LastFrame,      &lastFrame);
        getIntegerParam(NDArrayCounter,     &imageCounter);
        getIntegerParam(ADNumImages,        &numImages);
        getIntegerParam(ADNumImagesCounter, &numImagesCounter);

        setIntegerParam(NDArrayCounter,     ++imageCounter);
        setIntegerParam(ADNumImagesCounter, ++numImagesCounter);
        setIntegerParam(HDF5CurrentFrame,   ++currentFrame);

        // Put the frame number and time stamp into the buffer
        pImage->uniqueId = imageCounter;
        pImage->timeStamp = startTime.secPastEpoch + startTime.nsec / 1.e9;
        updateTimeStamp(&pImage->epicsTS);

        // Prepare loop if necessary
        int loop;
        getIntegerParam(HDF5Loop, &loop);

        if (loop && currentFrame > lastFrame)
        {
            getIntegerParam(HDF5FirstFrame,   &currentFrame);
            setIntegerParam(HDF5CurrentFrame, currentFrame);
        }

        // See if acquisition is done
        if (imageMode == ADImageSingle || currentFrame > lastFrame ||
            (imageMode == ADImageMultiple && numImagesCounter >= numImages))
        {
          // First do callback on ADStatus
          setStringParam(ADStatusMessage, "Waiting for acquisition");
          setIntegerParam(ADStatus, ADStatusIdle);

          acquire = 0;
          setIntegerParam(ADAcquire, acquire);

          asynPrint(this->pasynUserSelf, ASYN_TRACE_FLOW,
                  "%s:%s: acquisition completed\n",
                  driverName, functionName);
        }

        callParamCallbacks();

        // Delay next acquisition and check if received STOP signal
        if(acquire)
        {
            epicsTimeGetCurrent(&endTime);
            elapsedTime = epicsTimeDiffInSeconds(&endTime, &startTime);
            delay = acquirePeriod - elapsedTime;
            asynPrint(this->pasynUserSelf, ASYN_TRACE_FLOW,
                      "%s:%s: delay=%f\n",
                      driverName, functionName, delay);
            if(delay > 0.0)
            {
                // Set the status to waiting to indicate we are in the delay
                setIntegerParam(ADStatus, ADStatusWaiting);
                callParamCallbacks();
                this->unlock();
                status = epicsEventWaitWithTimeout(mStopEventId, delay);
                this->lock();

                if (status == epicsEventWaitOK)
                {
                    acquire = 0;
                    if (imageMode == ADImageContinuous)
                        setIntegerParam(ADStatus, ADStatusIdle);
                    else
                        setIntegerParam(ADStatus, ADStatusAborted);

                  callParamCallbacks();
                }
            }
        }
        continue;

error:
        setIntegerParam(ADAcquire, 0);
        setIntegerParam(ADStatus, ADStatusError);
        callParamCallbacks();
        continue;
    }
}

bool hdf5Driver::checkFilePath (const char *path)
{
    struct stat buff;
    return !stat(path, &buff) && (S_IFREG & buff.st_mode);
}

char *hdf5Driver::getFolder (const char *path, size_t *folderLen)
{
    char *folder = epicsStrDup(path);
    char *delim = strrchr(folder, DELIM);

    if(!delim)
        folder[0] = '\0';
    else
        *(delim + 1) = '\0';

    if(folderLen)
        *folderLen = strlen(folder);

    return folder;
}

asynStatus hdf5Driver::openFile (const char *path, hid_t *fId)
{
    asynStatus status = asynSuccess;
    const char *functionName = "openFile";
    FILE *fHandle;
    size_t size, left;
    char *buf;
    int inMemory;

    getIntegerParam(HDF5InMemory, &inMemory);

    // Simple case: let the HDF5 library open the file
    if(!inMemory)
    {
        *fId = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
        FAIL_IF_ARGS(*fId < 0, , "unable to open file %s", path);
        goto end;
    }

    // Portably open file
    fHandle = fopen(path, "rb");
    FAIL_IF_ARGS(!fHandle, goto end, "unable to open file %s", path);

    // Get file size
    fseek(fHandle, 0, SEEK_END);
    size = ftell(fHandle);
    rewind(fHandle);

    // Allocate memory to hold file contents
    buf = (char*) malloc(size);
    FAIL_IF_ARGS(!buf, goto closeFile, "unable to allocate %lu bytes", size);

    // Read file contents into buffer
    left = size;
    while(left)
    {
        size_t r = fread(buf, 1, left, fHandle);
        FAIL_IF_ARGS(!r, goto freeMem, "couldn't read %lu bytes from file",
                left);
        left -= r;
    }

    // Give buffer to HDF5 library. It will free the buffer when it's done
    *fId = H5LTopen_file_image(buf, size, H5LT_FILE_IMAGE_DONT_COPY);
    FAIL_IF(*fId < 0, , "couldn't open file image");

freeMem:
    if(status)
        free(buf);
closeFile:
    fclose(fHandle);
end:
    return status;
}

asynStatus hdf5Driver::openDataset (hid_t gId, dSetInfo_t *dInfo,
        const char *dName, const char *folder)
{
    asynStatus status = asynSuccess;
    const char *functionName = "openDataset";
    int inMemory;
    herr_t err;
    H5L_info_t info;
    hsize_t dims[3] = {0,0,0};

    getIntegerParam(HDF5InMemory, &inMemory);

    err = H5Lget_info(gId, dName, &info, H5P_DEFAULT);
    FAIL_IF_ARGS(err < 0, goto end, "couldn't get info about %s", dName);

    if(!inMemory || info.type != H5L_TYPE_EXTERNAL)
    {
        dInfo->id = H5Dopen2(gId, dName, H5P_DEFAULT);
        FAIL_IF_ARGS(dInfo->id < 0, goto end, "couldn't open dataset %s",dName);
    }
    else
    {
        char buf[info.u.val_size];
        const char *filename, *objpath;

        H5Lget_val(gId, dName, buf, info.u.val_size, H5P_DEFAULT);
        H5Lunpack_elink_val(buf, info.u.val_size, NULL, &filename, &objpath);

        char dSetFilePath[strlen(folder) + strlen(filename) + 1];
        strcpy(dSetFilePath, folder);
        strcat(dSetFilePath, filename);

        hid_t fileId;
        status = openFile(dSetFilePath, &fileId);
        FAIL_IF_ARGS(status, goto end, "couldn't open file %s", dSetFilePath);

        dInfo->id = H5Dopen2(fileId, objpath, H5P_DEFAULT);
        H5Fclose(fileId);
        FAIL_IF_ARGS(dInfo->id < 0, goto end, "couldn't open dataset %s",dName);
    }

    // Read dataset attributes
    err = H5LTget_attribute_int(dInfo->id, ".", "image_nr_low",
            &dInfo->imageNrLow);
    FAIL_IF(err, goto closeDataset, "couldn't read attribute image_nr_low");

    err = H5LTget_attribute_int(dInfo->id, ".", "image_nr_high",
            &dInfo->imageNrHigh);
    FAIL_IF(err, goto closeDataset, "couldn't read attribute image_nr_high");

    // Read dimensions (assume a 3D dataset)
    err = H5LTget_dataset_info(dInfo->id, ".", dims, NULL, NULL);
    FAIL_IF(err, goto closeDataset, "couldn't read dataset info");

    dInfo->nFrames = dims[0];
    dInfo->height  = dims[1];
    dInfo->width   = dims[2];

    // Read type
    status = parseType(dInfo->id, &dInfo->type);
    FAIL_IF(status, , "couldn't parse dataset type");

closeDataset:
    if(status)
        H5Dclose(dInfo->id);
end:
    return status;
}

asynStatus hdf5Driver::loadFile (const char *path)
{
    asynStatus status = asynSuccess;
    const char *functionName = "loadFile";

    hid_t fileId, groupId;
    H5G_info_t groupInfo;
    size_t totalFrames = 0;
    size_t maxWidth = 0, maxHeight = 0;
    herr_t err;
    char *folder = getFolder(path, NULL);

    // Open file
    status = openFile(path, &fileId);
    FAIL_IF_ARGS(status, goto end, "couldn't open file %s", path);

    // Reset some parameters
    setIntegerParam(HDF5DatasetsCount,  0);
    setIntegerParam(HDF5TotalFrames,    0);
    setIntegerParam(HDF5FirstFrame,     0);
    setIntegerParam(HDF5LastFrame,      0);
    setIntegerParam(HDF5CurrentFrame,   0);
    setIntegerParam(ADMaxSizeX,         0);
    setIntegerParam(ADMaxSizeY,         0);
    callParamCallbacks();

    // Get a handle to the '/entry' group
    groupId = H5Gopen2(fileId, "/entry", H5P_DEFAULT);
    FAIL_IF(groupId < 0, goto closeFile, "couldn't open 'entry' group");

    // Need groupInfo to obtain number of links
    err = H5Gget_info(groupId, &groupInfo);
    FAIL_IF(err, goto closeGroup, "couldn't get group info");

    // Deallocate information from previous file
    for(size_t i = 0; i < mDatasetsCount; ++i)
        H5Dclose(mpDatasets[i].id);

    // Allocate memory to store dataset information
    mpDatasets = (struct dSetInfo*) realloc(mpDatasets,
            groupInfo.nlinks*sizeof(*mpDatasets));
    mDatasetsCount = 0;

    // Iterate over '/entry' objects
    for(size_t i = 0; i < groupInfo.nlinks; ++i)
    {
        // Get object name (only interested in the first four characters)
        ssize_t dSetNameLen = H5Lget_name_by_idx(groupId, ".", H5_INDEX_NAME,
                H5_ITER_INC, i, NULL, 0, H5P_DEFAULT) + 1;

        char dSetName[dSetNameLen];
        H5Lget_name_by_idx(groupId, ".", H5_INDEX_NAME, H5_ITER_INC, i,
                dSetName, dSetNameLen, H5P_DEFAULT);

        // If it doesn't start with 'data' it isn't a dataset. Ignore it.
        if(strncmp(dSetName, "data", 4))
            continue;

        // Get a handle to the dataset info structure
        struct dSetInfo *pDSet = &mpDatasets[mDatasetsCount++];

        // Open dataset
        if(openDataset(groupId, pDSet, dSetName, folder))
        {
            // Close previously opened datasets
            for(size_t j = 0; j < mDatasetsCount-1; ++j)
                H5Dclose(pDSet->id);
            mDatasetsCount = 0;
        }

        totalFrames += pDSet->nFrames;
        maxWidth     = std::max(maxWidth,  pDSet->width);
        maxHeight    = std::max(maxHeight, pDSet->height);
    }

    // Update parameters
    setIntegerParam(HDF5DatasetsCount, (int) mDatasetsCount);
    setIntegerParam(HDF5TotalFrames,   (int) totalFrames);

    if(mDatasetsCount > 0)
    {
        int firstFrame = mpDatasets[0].imageNrLow;
        int lastFrame  = mpDatasets[mDatasetsCount-1].imageNrHigh;
        setIntegerParam(HDF5FirstFrame,   firstFrame);
        setIntegerParam(HDF5LastFrame,    lastFrame);
        setIntegerParam(HDF5CurrentFrame, firstFrame);
        setIntegerParam(ADMaxSizeX,       maxWidth);
        setIntegerParam(ADMaxSizeY,       maxHeight);

        int autoLoad;
        getIntegerParam(HDF5AutoLoadFrame, &autoLoad);
        if(autoLoad)
        {
            setIntegerParam(ADImageMode, ADImageSingle);
            setIntegerParam(ADNumImages, 1);
            setIntegerParam(ADAcquire,   1);
            epicsEventSignal(mStartEventId);
        }
    }

    callParamCallbacks();

closeGroup:
    H5Gclose(groupId);
closeFile:
    H5Fclose(fileId);
end:
    free(folder);
    return status;
}

struct hdf5Driver::dSetInfo *hdf5Driver::getDatasetByFrame (int frame)
{
    for(size_t i = 0; i < mDatasetsCount; ++i)
    {
        struct dSetInfo *pDSet = &mpDatasets[i];
        if(frame >= pDSet->imageNrLow && frame <= pDSet->imageNrHigh)
            return pDSet;
    }

    asynPrint(pasynUserSelf, ASYN_TRACE_ERROR,
            "%s:%s couldn't find the dataset that contains frame #%d\n",
            driverName, "getDatasetByFrame", frame);
    return NULL;
}

asynStatus hdf5Driver::getFrameInfo (int frame, size_t *pDims,
        NDDataType_t *pType)
{
    struct dSetInfo *pDSet = getDatasetByFrame(frame);
    if(!pDSet)
        return asynError;

    pDims[0] = pDSet->width;
    pDims[1] = pDSet->height;
    *pType = pDSet->type;
    return asynSuccess;
}

asynStatus hdf5Driver::getFrameData (int frame, void *pData)
{
    const char *functionName = "getFrameData";
    asynStatus status = asynSuccess;

    struct dSetInfo *pDSet;
    hid_t dSpace, dType, mSpace;
    herr_t err;

    pDSet = getDatasetByFrame(frame);
    if(!pDSet)
        return asynError;

    hsize_t offset[3] = {(hsize_t)(frame - pDSet->imageNrLow), 0, 0};
    hsize_t count[3]  = {1, pDSet->height, pDSet->width};

    dSpace = H5Dget_space(pDSet->id);
    FAIL_IF(dSpace < 0, goto end, "couldn't get dataspace");

    dType  = H5Dget_type(pDSet->id);
    FAIL_IF(dType < 0, goto end, "couldn't get dataset type");

    mSpace = H5Screate_simple(3, count, NULL);
    FAIL_IF(mSpace < 0, goto closeType, "couldn't create dataset");

    // Select the hyperslab
    err = H5Sselect_hyperslab(dSpace, H5S_SELECT_SET, offset, NULL, count, NULL);
    FAIL_IF(err, goto closeSpace, "couldn't select hyperslab");

    // And finally read the image
    err = H5Dread(pDSet->id, dType, mSpace, dSpace, H5P_DEFAULT, pData);
    FAIL_IF(err, goto closeSpace, "couldn't read image");

closeSpace:
    H5Sclose(mSpace);
closeType:
    H5Tclose(dType);
end:
    return status;
}

asynStatus hdf5Driver::parseType (hid_t id, NDDataType_t *pNDType)
{
    const char *functionName = "parseType";
    asynStatus status = asynSuccess;

    hid_t type = H5Dget_type(id);
    H5T_class_t dsetTypeClass = H5Tget_class(type);
    H5T_sign_t dsetTypeSign = H5Tget_sign(type);
    size_t dsetTypeSize = H5Tget_size(type);

    if(dsetTypeClass == H5T_INTEGER)
    {
        if(dsetTypeSign == H5T_SGN_NONE)
        {
            switch(dsetTypeSize)
            {
                case 1: *pNDType = NDUInt8;  break;
                case 2: *pNDType = NDUInt16; break;
                case 4: *pNDType = NDUInt32; break;
                default:
                    FAIL_IF_ARGS(true, goto end, "unsigned int (%lu bytes)",
                            dsetTypeSize);
            }
        }
        else if(dsetTypeSign == H5T_SGN_2)
        {
            switch(dsetTypeSize)
            {
                case 1: *pNDType = NDInt8;  break;
                case 2: *pNDType = NDInt16; break;
                case 4: *pNDType = NDInt32; break;
                default:
                    FAIL_IF_ARGS(true, goto end, "signed int (%lu bytes)",
                            dsetTypeSize);
            }
        }
        else
            FAIL_IF(true, goto end, "invalid dataset type sign");
    }
    else if(dsetTypeClass == H5T_FLOAT)
    {
        switch(dsetTypeSize)
        {
            case 4: *pNDType = NDFloat32; break;
            case 8: *pNDType = NDFloat64; break;
            default:
                FAIL_IF_ARGS(true, goto end, "invalid float (%lu bytes)",
                        dsetTypeSize);
        }
    }
    else
    {
        FAIL_IF(true, goto end, "dataset type class not accepted");
    }

end:
    H5Tclose(type);
    return status;
}

extern "C" int hdf5DriverConfig(const char *portName, int maxBuffers,
        size_t maxMemory, int priority, int stackSize)
{
    new hdf5Driver(portName, maxBuffers, maxMemory, priority, stackSize);
    return(asynSuccess);
}

/* Code for iocsh registration */
static const iocshArg hdf5DriverConfigArg0 = {"Port name", iocshArgString};
static const iocshArg hdf5DriverConfigArg1 = {"maxBuffers", iocshArgInt};
static const iocshArg hdf5DriverConfigArg2 = {"maxMemory", iocshArgInt};
static const iocshArg hdf5DriverConfigArg3 = {"priority", iocshArgInt};
static const iocshArg hdf5DriverConfigArg4 = {"stackSize", iocshArgInt};
static const iocshArg * const hdf5DriverConfigArgs[] = {
    &hdf5DriverConfigArg0, &hdf5DriverConfigArg1, &hdf5DriverConfigArg2,
    &hdf5DriverConfigArg3, &hdf5DriverConfigArg4};

static const iocshFuncDef confighdf5Driver = {"hdf5DriverConfig", 5,
        hdf5DriverConfigArgs};

static void confighdf5DriverCallFunc(const iocshArgBuf *args)
{
    hdf5DriverConfig(args[0].sval, args[1].ival, args[2].ival, args[3].ival,
            args[4].ival);
}

static void hdf5DriverRegister(void)
{
    iocshRegister(&confighdf5Driver, confighdf5DriverCallFunc);
}

extern "C" {
    epicsExportRegistrar(hdf5DriverRegister);
}

