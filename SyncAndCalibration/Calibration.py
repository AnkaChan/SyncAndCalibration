import os
import glob
import numpy as np
from io import StringIO
import cv2
import random
import time

class CalibConfig:
    def __init__(self):
        self.numCalibFramesIntrinsics = 120
        self.numCalibFramesEntrinsics = 80
        self.chbSize = [24, 17]
        self.chbSquareSize = 30
        self.imgSize = (4000, 2160)
        self.useExistingIntrinsics = False
        self.predetectStep = 4

class CameraParam:
    def __init__(self):
        if __name__ == '__main__':
            self.M = np.zeros((3,3), dtype=np.float32)
            self.D = np.zeros((5), dtype=np.float32)
            self.R = np.zeros((3,3), dtype=np.float32)
            self.T = np.zeros((3,1), dtype=np.float32)


class ChbPoints:
    def __init__(self):
        self.cornerPoints = None
        self.avaliable = False
        self.name = ""

class CalibSequence:
    def __init__(self):
        self.chbSequence = []

def readCorners(inFile, chbPts):
    f = open(inFile)
    label = f.readline()
    label = label.replace('\n', '')
    chbPts.name = inFile
    if label == "true":
        chbPts.avaliable = True
        chbPts.cornerPoints = np.loadtxt(StringIO(f.read()), dtype=np.float32)
        # print(chbPts.cornerPoints)
    elif label == "false":
        chbPts.avaliable = False
        chbPts.cornerPoints = []
    else:
        print("Wrong file head: " , label, ". in file: ", chbPts)

    return

def calibIntrinsics(calibSequence, numCalibFrames, chbSize = [24, 17], chbSquareSize = 30, imgSize =  (4000, 2160)):
    # prepare object points on checkerboard
    objPoints =  np.zeros((chbSize[0] * chbSize[1], 3), np.float32)
    objPoints[:, :2] = chbSquareSize*np.mgrid[0:chbSize[0], 0:chbSize[1]].T.reshape(-1, 2)

    # pick frames used to calibrate
    objPtsSet = []
    imgPtsSet = []
    avaliableChbFrames = []

    for chb in calibSequence.chbSequence:
        if chb.avaliable:
            imgPtsSet.append(chb.cornerPoints)

    random.shuffle(imgPtsSet)
    if numCalibFrames > len(imgPtsSet):
        numCalibFrames = len(imgPtsSet)
    imgPtsSet = imgPtsSet[0 : numCalibFrames]
    objPtsSet = [objPoints for i in range(numCalibFrames)]
    cameraMatrix = None
    distCoeffs = None
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPtsSet, imgPtsSet, imgSize, cameraMatrix, distCoeffs)

    # print("camera matrix: ", mtx)
    # print("distCoeffs: ", dist)
    print("End intrinsic calibration with ret: ", ret)
    # print("rvecs: ", rvecs)
    # print("tvecs: ", tvecs)

    return mtx, dist

def fileParts(filePath):
    path, f = os.path.split(filePath)
    name, ext = os.path.splitext(f)
    return path, name, ext

def calibExtrinsics(calibSequence1, calibSequence2, numCalibFrames, M1, M2, D1, D2,
                    chbSize = [24, 17], chbSquareSize = 30, imgSize =  (4000, 2160)):
    numTotalFrames = min(len(calibSequence1.chbSequence), len(calibSequence2.chbSequence))

    objPoints = np.zeros((chbSize[0] * chbSize[1], 3), np.float32)
    objPoints[:, :2] = chbSquareSize * np.mgrid[0:chbSize[0], 0:chbSize[1]].T.reshape(-1, 2)

    # imgPtsSet1 = imgPtsSet2 = []
    imgPtsSet1 = []
    imgPtsSet2 = []
    for i in range(numTotalFrames):
        chb1 = calibSequence1.chbSequence[i]
        chb2 = calibSequence2.chbSequence[i]
        _, n1, _ = fileParts(chb1.name)
        _, n2, _ = fileParts(chb2.name)
        assert (n1 == n2)
        if chb1.avaliable and chb2.avaliable:
            imgPtsSet1.append(chb1.cornerPoints)
            imgPtsSet2.append(chb2.cornerPoints)

    if numCalibFrames > len(imgPtsSet1):
        numCalibFrames = len(imgPtsSet1)

    print(numCalibFrames, " frames used to calibrate extrinsics.")

    c = list(zip(imgPtsSet1, imgPtsSet2))

    random.shuffle(c)

    imgPtsSet1, imgPtsSet2 = zip(*c)
    imgPtsSet1 = imgPtsSet1[0 : numCalibFrames]
    imgPtsSet2 = imgPtsSet2[0 : numCalibFrames]

    objPtsSet = [objPoints for i in range(numCalibFrames)]
    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    # flags |= cv2.CALIB_FIX_FOCAL_LENGTH
    flags |= cv2.CALIB_FIX_ASPECT_RATIO
    flags |= cv2.CALIB_ZERO_TANGENT_DIST
    flags |= cv2.CALIB_RATIONAL_MODEL
    # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
    flags |= cv2.CALIB_FIX_K1
    flags |= cv2.CALIB_FIX_K2
    flags |= cv2.CALIB_FIX_K3
    flags |= cv2.CALIB_FIX_K4
    flags |= cv2.CALIB_FIX_K5
    flags |= cv2.CALIB_FIX_K6

    stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    """
       stereoCalibrate(objectPoints, imagePoints1, imagePoints2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
        imageSize[, R[, T[, E[, F[, flags[, criteria]]]]]]) -> retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F
       .
       """

    # print(type(objPtsSet))
    # print(type(objPtsSet[0]))
    # print(objPtsSet[0].shape)
    # print(objPtsSet[0].dtype)
    # print('')
    # print(type(imgPtsSet1))
    # print(type(imgPtsSet1[0]))
    # print(imgPtsSet1[0].shape)
    # print(imgPtsSet1[0].dtype)
    # print('')
    # print(type(imgPtsSet2))
    # print(type(imgPtsSet2[0]))
    # print(imgPtsSet2[0].shape)
    # print(imgPtsSet2[0].dtype)
    # print('')
    # print(type(M1))
    # print(M1.shape)
    # print('')
    # print(type(D1))
    # print(D1.shape)
    # print('')
    # print(type(imgSize))
    # print(type(stereocalib_criteria))
    # print(type(flags))

    ret, M1, D1, M2, D2, R, T, E, F = cv2.stereoCalibrate(
        objPtsSet, imgPtsSet1, imgPtsSet2,
        M1, D1, M2, D2, imgSize,
        criteria=stereocalib_criteria, flags=flags)

    print("End extrinsics calibration with ret: ", ret)
    print("Extrinsic R:\n", R)
    print("Extrinsic T:\n", T)
    # print("M1:\n", M1, "\nM2:\n", M2)
    return M1, D1, M2, D2, R, T, E, F


def calibCameras(inputFolders, calibOrders, outputIntrinsicsPath = [], outputExtrinsicFiles = [],  config = CalibConfig(), batchName = ""):
    # read files
    cameraCalibSequences = []
    for inFolder in inputFolders:
        calibSeq = CalibSequence()
        files = glob.glob(inFolder + "/*.txt")
        files.sort()
        for f in files:
            chbPts = ChbPoints()
            readCorners(f, chbPts)
            calibSeq.chbSequence.append(chbPts)

        cameraCalibSequences.append(calibSeq)

    # print(cameraCalibSequences)

    cameraParameters = []

    if config.useExistingIntrinsics:
        # read intrinsics
        for i in range(len(cameraCalibSequences)):
            fs = cv2.FileStorage(outputIntrinsicsPath[i], cv2.FILE_STORAGE_READ)
            mtx = fs.getNode("M1").mat()
            dist = fs.getNode("D1").mat()
            cParam = CameraParam()
            cParam.M = mtx
            cParam.D = dist
            cameraParameters.append(cParam)
    else:
        # calibration intrinsics
        for i in range(len(cameraCalibSequences)):
            seq = cameraCalibSequences[i]
            mtx, dist = calibIntrinsics(seq, config.numCalibFramesIntrinsics, chbSize = config.chbSize, chbSquareSize = config.chbSquareSize, imgSize=config.imgSize)
            if len(outputIntrinsicsPath) != 0:
                fs = cv2.FileStorage(outputIntrinsicsPath[i], cv2.FILE_STORAGE_WRITE)
                fs.write("M1", mtx)
                fs.write("D1", dist)
                fs.release()

            cParam = CameraParam()
            cParam.M = mtx
            cParam.D = dist
            cameraParameters.append(cParam)


    for i in range(len(calibOrders)):
        calibPair = calibOrders[i]
        camId1 = calibPair[0]
        camId2 = calibPair[1]
        print("Calibrating: ", outputExtrinsicFiles[i])
        M1, D1, M2, D2, R, T, E, F = calibExtrinsics(cameraCalibSequences[camId1], cameraCalibSequences[camId2],config.numCalibFramesEntrinsics,
                        cameraParameters[camId1].M, cameraParameters[camId2].M, cameraParameters[camId1].D, cameraParameters[camId2].D, 
                        chbSize = config.chbSize, chbSquareSize = config.chbSquareSize, imgSize=config.imgSize)
        if len(outputExtrinsicFiles) != 0:
            fs = cv2.FileStorage(outputExtrinsicFiles[i], cv2.FILE_STORAGE_WRITE)
            fs.write("M1", M1)
            fs.write("D1", D1)
            fs.write("M2", M2)
            fs.write("D2", D2)
            fs.write("R", R)
            fs.write("T", T)

def readPairCalib(file):
    cv_file = cv2.FileStorage(file, cv2.FILE_STORAGE_READ)
    # FileNode object back instead of a matrix
    M1 = cv_file.getNode("M1").mat()
    M2 = cv_file.getNode("M2").mat()
    D1 = cv_file.getNode("D1").mat()
    D2 = cv_file.getNode("D2").mat()
    R = cv_file.getNode("R").mat()
    t = cv_file.getNode("T").mat()
    cv_file.release()
    return {"M1": M1, "M2": M2, "D1":D1, "D2":D2, "R": R, "t": t}

def readWorldCalib(file):
    cv_file = cv2.FileStorage(file, cv2.FILE_STORAGE_READ)
    I = cv_file.getNode("intrinsics").mat()
    dist = cv_file.getNode("dist").mat()
    E = cv_file.getNode("extrinsics").mat()

    return {"intrinsics" : I, "dist":dist, "extrinsics":E}

def IEToProjMat(I, E):
    I_P = np.eye(4)
    I_P[0:3, 0:3] = I
    P = np.matmul(I_P, E)
    return P[0:3, :]

def makeToWorldCalib(etrinsicFolder, intrinsicFolder, globalCalibFolder, camNames, calibOrders, referenceCamId = 0, inBatchNum = "", outputBatchNum = "" ):
    extrinsics = []
    intrinsics = []
    dists = []
    extrinsicFiles = []
    globalCalibs = []

    os.makedirs(globalCalibFolder, exist_ok=True)

    for name in camNames:
        intrinsicsFile = intrinsicFolder + "/Intrinsics" + name + ".xml"
        cv_file = cv2.FileStorage(intrinsicsFile, cv2.FILE_STORAGE_READ)
        M1 = cv_file.getNode("M1").mat()
        D1 = cv_file.getNode("D1").mat()
        intrinsics.append(M1)
        dists.append(D1)

    for calibPair in calibOrders:
        if inBatchNum != "":
            extrinFile = etrinsicFolder + "/" + "Extrinsics" + camNames[calibPair[0]] + "_" + camNames[calibPair[1]] + "_" + inBatchNum + ".xml"
        else:
            extrinFile = etrinsicFolder + "/" + "Extrinsics" + camNames[calibPair[0]] + "_" + camNames[calibPair[1]] + ".xml"

        extrinsicFiles.append(extrinFile)
        pairCalib = readPairCalib(extrinFile)

        T = np.eye(4)

        T[0:3, 0:3] = pairCalib['R']
        T[0:3, 3:] = pairCalib['t']
        extrinsics.append(np.asmatrix(T))

    for i in range(len(camNames)):
        if i == referenceCamId:
            E = np.asmatrix(np.eye(4))
        elif i < referenceCamId:
            E = np.asmatrix(np.eye(4))
            for j in range(i, referenceCamId):
                E = E * extrinsics[j].I
        else:
            E = np.asmatrix(np.eye(4))
            for j in range(referenceCamId, i):
                E = extrinsics[j] * E

        globalCalibs.append({"intrinsics":intrinsics[i], "dist":dists[i], "extrinsics":E})

    globalCalibFiles = []
    for i in range(len(camNames)):
        if outputBatchNum != "":
            globalCalibFile = globalCalibFolder + "/P" + camNames[i] + "_" + outputBatchNum + ".xml"
        else:
            globalCalibFile = globalCalibFolder + "/P" + camNames[i] + ".xml"

        fs = cv2.FileStorage(globalCalibFile, cv2.FILE_STORAGE_WRITE)
        fs.write("intrinsics", globalCalibs[i]["intrinsics"])
        fs.write("dist",  globalCalibs[i]["dist"])
        fs.write("extrinsics", globalCalibs[i]["extrinsics"])
        fs.release()
        globalCalibFiles.append(globalCalibFile)
    return globalCalibFiles


if __name__ == "__main__":

    pa = cv2.FileStorage(r"H:\00_WorkingCopy\2019_08_09_AllPoseCapture\CalibGlobal\PA.xml", cv2.FILE_STORAGE_READ)
    pb = cv2.FileStorage(r"H:\00_WorkingCopy\2019_08_09_AllPoseCapture\CalibGlobal\PB.xml", cv2.FILE_STORAGE_READ)

    ea = pa.getNode("extrinsics").mat()
    eb = pb.getNode("extrinsics").mat()

    print(ea)
    print(eb)

    o1 = - np.matmul(np.linalg.inv(ea[0:3, 0:3]), ea[0:3, 3])
    o2 = - np.matmul(np.linalg.inv(eb[0:3, 0:3]), eb[0:3, 3])

    print(o1)
    print(o2)

    print(o1 - o2)
    print(np.linalg.norm(o1 - o2))

    exit()

    # inputFolders = [
    #     r"E:\Mocap\WorkingCopy\2019_03_06_6CamCapture\Transformed\A\Corners",
    #     r"E:\Mocap\WorkingCopy\2019_03_06_6CamCapture\Transformed\B\Corners",
    #     r"E:\Mocap\WorkingCopy\2019_03_06_6CamCapture\Transformed\C\Corners",
    #     r"E:\Mocap\WorkingCopy\2019_03_06_6CamCapture\Transformed\D\Corners",
    #     r"E:\Mocap\WorkingCopy\2019_03_06_6CamCapture\Transformed\E\Corners",
    #     r"E:\Mocap\WorkingCopy\2019_03_06_6CamCapture\Transformed\F\Corners",
    # ]
    #
    # camNames = ["A", "B", "C", "D", "E", "F"]
    #
    # outputIntrinsicFiles = [
    #     r"E:\Mocap\WorkingCopy\2019_03_06_6CamCapture\Transformed\Calib\IntrinsicA.xml",
    #     r"E:\Mocap\WorkingCopy\2019_03_06_6CamCapture\Transformed\Calib\IntrinsicB.xml",
    #     r"E:\Mocap\WorkingCopy\2019_03_06_6CamCapture\Transformed\Calib\IntrinsicC.xml",
    #     r"E:\Mocap\WorkingCopy\2019_03_06_6CamCapture\Transformed\Calib\IntrinsicD.xml",
    #     r"E:\Mocap\WorkingCopy\2019_03_06_6CamCapture\Transformed\Calib\IntrinsicE.xml",
    #     r"E:\Mocap\WorkingCopy\2019_03_06_6CamCapture\Transformed\Calib\IntrinsicF.xml",
    # ]

    # inBatchFolder = r"E:\Mocap\WorkingCopy\2019_03_29_6CameNewLightSet\Transformed"
    # outputExtrinsicFolder = r"E:\Mocap\WorkingCopy\2019_03_29_6CameNewLightSet\Calib\Calib2"
    #
    # config = CalibConfig()
    # config.useExistingIntrinsics = False
    #
    # # inputSeqFolders = glob.glob(inBatchFolder + r"\*\\")
    # # inputFolders = [f + "\\Corners" for f in inputSeqFolders]
    # inputFolders = [
    #         r"E:\Mocap\WorkingCopy\2019_03_29_6CameNewLightSet\Transformed\A001_03082015_C001\Corners",
    #         r"E:\Mocap\WorkingCopy\2019_03_29_6CameNewLightSet\Transformed\B001_03280254_C001\Corners",
    #         r"E:\Mocap\WorkingCopy\2019_03_29_6CameNewLightSet\Transformed\C001_03251035_C001\Corners",
    #         r"E:\Mocap\WorkingCopy\2019_03_29_6CameNewLightSet\Transformed\D001_03050804_C001\Corners",
    #         r"E:\Mocap\WorkingCopy\2019_03_29_6CameNewLightSet\Transformed\E001_03060921_C001\Corners",
    #         r"E:\Mocap\WorkingCopy\2019_03_29_6CameNewLightSet\Transformed\F001_03290207_C001\Corners",
    #     ]
    #
    # camNames = ["A", "B", "C", "D", "E", "F"]
    #
    # outputIntrinsicFiles = [
    #     r"E:\Mocap\WorkingCopy\2019_03_29_6CameNewLightSet\Calib\IntrinsicsA.xml",
    #     r"E:\Mocap\WorkingCopy\2019_03_29_6CameNewLightSet\Calib\IntrinsicsB.xml",
    #     r"E:\Mocap\WorkingCopy\2019_03_29_6CameNewLightSet\Calib\IntrinsicsC.xml",
    #     r"E:\Mocap\WorkingCopy\2019_03_29_6CameNewLightSet\Calib\IntrinsicsD.xml",
    #     r"E:\Mocap\WorkingCopy\2019_03_29_6CameNewLightSet\Calib\IntrinsicsE.xml",
    #     r"E:\Mocap\WorkingCopy\2019_03_29_6CameNewLightSet\Calib\IntrinsicsF.xml",
    # ]
    #
    # calibOrders = [
    #     [0, 1],
    #     [1, 2],
    #     [2, 3],
    #     [3, 4],
    #     [4, 5]
    # ]
    #
    # batchName = "3"
    # inBatchFolder = r"E:\Mocap\WorkingCopy\2019_03_29_6CameNewLightSet\Transformed"
    # outputExtrinsicFolder = r"E:\Mocap\WorkingCopy\2019_03_29_6CameNewLightSet\Calib"
    #
    # config = CalibConfig()
    #
    # # config.chbSize = [8, 11]
    # # config.chbSize = 60
    #
    # config.useExistingIntrinsics = True
    #
    # # inputSeqFolders = glob.glob(inBatchFolder + r"\*\\")
    # # inputFolders = [f + "\\Corners" for f in inputSeqFolders]
    # inputFolders = [
    #     r"F:\WorkingCopy2\2019_04_16_8CamsCapture\Converted\A\Corners",
    #     r"F:\WorkingCopy2\2019_04_16_8CamsCapture\Converted\B\Corners",
    #     r"F:\WorkingCopy2\2019_04_16_8CamsCapture\Converted\B\Corners",
    #     r"F:\WorkingCopy2\2019_04_16_8CamsCapture\Converted\D\Corners",
    #     r"F:\WorkingCopy2\2019_04_16_8CamsCapture\Converted\E\Corners",
    #     r"F:\WorkingCopy2\2019_04_16_8CamsCapture\Converted\F\Corners",
    # ]
    #
    # camNames = ["A", "B", "C", "D", "E", "F"]
    #
    # outputIntrinsicFiles = [
    #     r"F:\WorkingCopy2\2019_04_16_8CamsCapture\Calib\IntrinsicsA.xml",
    #     r"F:\WorkingCopy2\2019_04_16_8CamsCapture\Calib\IntrinsicsB.xml",
    #     r"F:\WorkingCopy2\2019_04_16_8CamsCapture\Calib\IntrinsicsC.xml",
    #     r"F:\WorkingCopy2\2019_04_16_8CamsCapture\Calib\IntrinsicsD.xml",
    #     r"F:\WorkingCopy2\2019_04_16_8CamsCapture\Calib\IntrinsicsE.xml",
    #     r"F:\WorkingCopy2\2019_04_16_8CamsCapture\Calib\IntrinsicsF.xml",
    # ]
    #
    # calibOrders = [
    #     [0, 1],
    #     [1, 2],
    #     [2, 3],
    #     [3, 4],
    #     [4, 5]
    #
    #     # [0, 2],
    #     # [1, 3],
    #     # [2, 4],
    #     # [3, 5],
    # ]



    #
    # camNames = ["A", "C"]

    # outputIntrinsicFiles = [
    #     r"E:\Mocap\WorkingCopy\2019_03_29_6CameNewLightSet\Calib\CalibTest\IntrinsicsA.xml",
    #     # r"E:\Mocap\WorkingCopy\2019_03_29_6CameNewLightSet\Calib\IntrinsicsB.xml",
    #     r"E:\Mocap\WorkingCopy\2019_03_29_6CameNewLightSet\Calib\CalibTest\IntrinsicsC.xml",
    #     # r"E:\Mocap\WorkingCopy\2019_03_29_6CameNewLightSet\Calib\IntrinsicsD.xml",
    #     # r"E:\Mocap\WorkingCopy\2019_03_29_6CameNewLightSet\Calib\IntrinsicsE.xml",
    #     # r"E:\Mocap\WorkingCopy\2019_03_29_6CameNewLightSet\Calib\IntrinsicsF.xml",
    # ]
    #
    # calibOrders = [
    #     [0, 1],
    # ]

    # inputFolders = [
    #
    #     r"E:\Mocap\WorkingCopy\2019_03_06_6CamCapture\Transformed\C001_03171426_C001\Corners",
    #     r"E:\Mocap\WorkingCopy\2019_03_06_6CamCapture\Transformed\D001_02260102_C001\Corners",
    # ]
    #
    # camNames = ["A", "B", "C", "D", "E", "F"]
    #
    # outputIntrinsicFiles = []
    # # outputIntrinsicFiles = [
    # #     r"E:\Mocap\WorkingCopy\2019_03_06_6CamCapture\Transformed\Calib\IntrinsicA.xml",
    # #     r"E:\Mocap\WorkingCopy\2019_03_06_6CamCapture\Transformed\Calib\IntrinsicB.xml",
    # #     r"E:\Mocap\WorkingCopy\2019_03_06_6CamCapture\Transformed\Calib\IntrinsicC.xml",
    # #     r"E:\Mocap\WorkingCopy\2019_03_06_6CamCapture\Transformed\Calib\IntrinsicD.xml",
    # #     r"E:\Mocap\WorkingCopy\2019_03_06_6CamCapture\Transformed\Calib\IntrinsicE.xml",
    # #     r"E:\Mocap\WorkingCopy\2019_03_06_6CamCapture\Transformed\Calib\IntrinsicF.xml",
    # # ]
    #
    # outputExtrinsicFolder = r"E:\Mocap\WorkingCopy\2019_03_06_6CamCapture\Transformed\Calib"
    # calibOrders = [
    #     [0, 1],
    #     # [1, 2],
    #     # [2, 3],
    #     # [3, 4],
    #     # [4, 5]
    # ]
    config = CalibConfig()
    config.chbSquareSize = 60
    config.numCalibFramesEntrinsics = 80
    config.numCalibFramesEntrinsics = 80
    config.chbSize = [11,8]
    batchName = '1'

    camNames = ["A", "B", "C", "D", "E", "F", "G", "H"]

    inputFolders = [
        r"H:\00_WorkingCopy\2019_08_09_AllPoseCapture\Converted\A\Corners",
        r"H:\00_WorkingCopy\2019_08_09_AllPoseCapture\Converted\B\Corners",
        # r"H:\00_WorkingCopy\2019_08_09_AllPoseCapture\Converted\B\Corners",
        # r"H:\00_WorkingCopy\2019_08_09_AllPoseCapture\Converted\D\Corners",
        # r"H:\00_WorkingCopy\2019_08_09_AllPoseCapture\Converted\E\Corners",
        # r"H:\00_WorkingCopy\2019_08_09_AllPoseCapture\Converted\F\Corners",
        # r"H:\00_WorkingCopy\2019_08_09_AllPoseCapture\Converted\G\Corners",
        # r"H:\00_WorkingCopy\2019_08_09_AllPoseCapture\Converted\H\Corners",
    ]
    outputExtrinsicFolder = r"F:\WorkingCopy2\2019_08_09_AllPoseCapture\Calib2"

    outputIntrinsicFiles = [
        r"F:\WorkingCopy2\2019_08_09_AllPoseCapture\Calib2\IntrinsicA.xml",
        r"F:\WorkingCopy2\2019_08_09_AllPoseCapture\Calib2\IntrinsicB.xml",
        # r"F:\WorkingCopy2\2019_08_09_AllPoseCapture\Calib2\IntrinsicC.xml",
        # r"F:\WorkingCopy2\2019_08_09_AllPoseCapture\Calib2\IntrinsicD.xml",
        # r"F:\WorkingCopy2\2019_08_09_AllPoseCapture\Calib2\IntrinsicE.xml",
        # r"F:\WorkingCopy2\2019_08_09_AllPoseCapture\Calib2\IntrinsicF.xml",
    ]

    calibOrders = [
        [0, 1],
        # [1, 2],
        # [2, 3],
        # [3, 4],
        # [4, 5]
    ]

    outputExtrinsicFiles = []

    for calibPair in calibOrders:
        outputExtrinsicFiles.append(outputExtrinsicFolder + "/" + "Extrinsics_" + camNames[calibPair[0]]
                                    + "_" + camNames[calibPair[1]] + "_" + batchName + ".xml")

    for _ in range(1):
        startTime = time.time()

        calibCameras(inputFolders, calibOrders, outputIntrinsicFiles, outputExtrinsicFiles, config=config)

        endTime = time.time()

        print("Time consumption: ", endTime - startTime)
    # chbSize = [24, 17]
    # objPoints = np.zeros((chbSize[0] * chbSize[1], 3), np.float32)
    # objPoints[:, :2] = np.mgrid[0:chbSize[0], 0:chbSize[1]].T.reshape(-1, 2)
    # print(objPoints)