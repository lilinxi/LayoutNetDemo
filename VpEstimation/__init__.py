import CoordsTransfrom
import Projection
import numpy as np
import cv2
import time


def __SolidAngleSegmentation(segmentationShape=(10, 20)):
    """
    dw = sin(phi) * d(theta) * d(phi)
    f(phi_1 -> phi_0) = -(cos(phi_1) - cos(phi_0)) * d(theta)
    按照 cos(phi) * theta 将 Sphere 划分为等立体角（等面积）的网格 Segmentation
    :param segmentationShape:           划分网格的细粒度
    :return:phi_theta2segmentation, segmentation2phi_theta
    """
    # 每个间隔为一个分段
    cosPhiSegmentation = np.linspace(1, -1, segmentationShape[0] + 1)
    phiSegmentation = np.arccos(cosPhiSegmentation)
    thetaSegmentation = np.linspace(0, 2 * np.pi, segmentationShape[1] + 1)

    # phi, theta to segmentation index
    def phi_theta2segmentation(phi, theta):
        phi_index = 0
        theta_index = 0
        for i in range(1, segmentationShape[0]):
            if phiSegmentation[i] > phi:
                break
            else:
                phi_index += 1

        for i in range(1, segmentationShape[1]):
            if thetaSegmentation[i] > theta:
                break
            else:
                theta_index += 1

        return phi_index, theta_index

    # segmentation index to center_phi, center_theta
    def segmentation2phi_theta(phi_index, theta_index):
        cosPhi1 = cosPhiSegmentation[phi_index]
        cosPhi2 = cosPhiSegmentation[phi_index + 1]
        cosPhi = (cosPhi1 + cosPhi2) / 2
        phi = np.arccos(cosPhi)

        theta1 = thetaSegmentation[theta_index]
        theta2 = thetaSegmentation[theta_index + 1]
        theta = (theta1 + theta2) / 2

        return phi, theta

    return phi_theta2segmentation, segmentation2phi_theta


def __GetOrthogonalVps(vps):
    """
    找到两个方向的灭点，计算第三个方向的灭点
    """
    if len(vps) < 2:
        print("vps not enough")
        exit(-1)

    # find orthogonalVpX
    # 遍历所有灭点对，找到平行 err 加垂直 err 最小的作为第一个灭点
    vpXErrs = []
    for i in range(len(vps)):
        errSum = 0
        for j in range(len(vps)):
            err = np.dot(vps[i], vps[j])
            err = min(abs(err), 1 - abs(err))
            errSum += err
        vpXErrs.append(errSum)
    vpXIndex = vpXErrs.index(min(vpXErrs))
    orthogonalVpX = vps[vpXIndex]
    print("Vp-X", orthogonalVpX)

    # find orthogonalVpY
    # 与第一个灭点垂直 err 最小的作为第二个灭点
    vpYErrs = []
    for i in range(len(vps)):
        err = np.dot(vps[i], orthogonalVpX)
        vpYErrs.append(abs(err))
    vpYIndex = vpYErrs.index(min(vpYErrs))
    orthogonalVpY = vps[vpYIndex]
    print("Vp-Y", orthogonalVpY)

    # compute orthogonalVpZ
    # 两个灭点叉乘，计算第三个灭点
    orthogonalVpZ = np.cross(orthogonalVpX, orthogonalVpY)
    print("Vp-Z", orthogonalVpZ)

    return [
        orthogonalVpX,
        orthogonalVpY,
        orthogonalVpZ,
    ]


def HoughVpEstimation(panoImage,
                      panoImageShape=(2000, 1000), projectScale=600, minLineThreshold=1 / 15,
                      segmentationShape=(300, 600), circleSamples=600):
    """
    霍夫变换根据 Segmentation 投票计算灭点
    :param panoImage:           全景图
    :param panoImageShape:      全景图大小
    :param projectScale:        投影图大小（能保持精度的最大倍数 pano -> project 倍数 pi/2）
    :param minLineThreshold:    投影直线阈值
    :param segmentationShape:   划分 Segmentation 大小
    :param circleSamples:       霍夫投票对 Segmentation 的细粒度
    :return:
    """
    # 初始化
    panoImage = cv2.resize(panoImage, panoImageShape, interpolation=cv2.INTER_AREA)
    minLineSquare = (projectScale * minLineThreshold) ** 2
    # 划分 Segmentation
    phi_theta2segmentation, segmentation2phi_theta = __SolidAngleSegmentation(segmentationShape)
    # 存储霍夫投票
    panoImageVpsAll = np.zeros(segmentationShape)
    # 计算投影图
    projectImageAndMappings = Projection.ARoundProjection(panoImage, projectScale)
    for i in range(len(projectImageAndMappings)):
        print("project-line", i)
        projectImage = projectImageAndMappings[i][0]
        mapping = projectImageAndMappings[i][1]
        scalemapping = projectImageAndMappings[i][2]
        # step1. 检测直线
        grayImage = cv2.cvtColor(projectImage, cv2.COLOR_BGR2GRAY)
        fld = cv2.ximgproc.createFastLineDetector()
        lines = fld.detect(grayImage)
        if lines is None:
            continue
        # TODO Debug
        # DrawPanoLine(panoImage, lines, scalemapping, (0, 255, 0))
        # goodLines = []
        for line in lines:
            x0 = line[0][0]
            y0 = line[0][1]
            x1 = line[0][2]
            y1 = line[0][3]
            dx = x1 - x0
            dy = y1 - y0
            # 筛选出足够长的线段
            if dx ** 2 + dy ** 2 < minLineSquare:
                continue
            # 去掉接近垂直的直线，这里不检测垂直的灭点
            if abs(dx) < 0.1:
                continue
            if abs(dy / dx) > 10:
                continue
            # goodLines.append(line)
            # step2. 从投影 xy 坐标映射回全景图 uv 坐标
            u0, v0 = mapping(x0, y0)
            u1, v1 = mapping(x1, y1)
            # step3. 由线段两个端点和球心计算直线所在大圆的 normal
            xa, ya, za = CoordsTransfrom.uv2xyz(u0, v0)
            xb, yb, zb = CoordsTransfrom.uv2xyz(u1, v1)
            a = np.array([xa, ya, za])
            b = np.array([xb, yb, zb])
            normal = np.cross(a, b)
            # step4. normal 垂直的大圆变换回 xy 并存储进 hough 图像
            # 可以 Hough 投票检测到 4 个消失点
            v1, v2 = Projection.BuildCoords(normal)
            # 防止重复投票
            panoImageVpsSelect = np.zeros(segmentationShape)
            for theta in np.linspace(0, 2 * np.pi, circleSamples, endpoint=False):
                p = np.cos(theta) * v1 + np.sin(theta) * v2

                # step5. 抑制线上的投票点，灭点不可能在直线上，p 与 a,b 叉乘，同向的点保留，不同向的点为线上点，或线上点的对称点，不予保留，都与 normal 方向相同或者相反即为同向
                pa = np.cross(p, a)
                pb = np.cross(p, b)

                if np.dot(pa, normal) * np.dot(pb, normal) < 0:
                    continue

                pa = np.cross(p, -a)
                pb = np.cross(p, -b)

                if np.dot(pa, normal) * np.dot(pb, normal) < 0:
                    continue

                phi, theta = CoordsTransfrom.xyz2phi_theta(p[0], p[1], p[2])
                segmentationI, segmentationJ = phi_theta2segmentation(phi, theta)
                panoImageVpsSelect[segmentationI][segmentationJ] = 1

            # 叠加投票
            panoImageVpsAll = panoImageVpsAll + panoImageVpsSelect

        # DrawPanoLine(panoImage, goodLines, scalemapping, (0, 255, 0), sampleRate=1.1)

    # TODO Trick 将 Segmentation 水平划分为 10 部分，防止只检测出最大值的灭点，而忽略极大值的灭点
    splitSize = 10
    step = round(segmentationShape[1] / splitSize)
    splitMax = []
    for i in range(splitSize):
        splitMax.append(max(panoImageVpsAll[:, i * step:(i + 1) * step].flat))

    # TODO Trick 一般会有两个效果最好的灭点，先将其去掉求平均值，大于平均值的可以认为存在灭点的概率很大
    print("splitMax", splitMax)
    splitMax = np.sort(splitMax)
    splitMax[7] = splitMax[0]  # dataset-good-21, 不多去一个就会检测不到正交灭点，多的灭点可以筛出去
    splitMax[8] = splitMax[0]
    splitMax[9] = splitMax[0]
    splitMaxAvg = np.sum(splitMax) / 10
    print("splitMaxAvg", splitMaxAvg)
    for i in range(splitSize):
        maxValue = max(panoImageVpsAll[:, i * step:(i + 1) * step].flat)
        if maxValue > splitMaxAvg:
            print("max", i)
            panoImageVpsAll[:, i * step:(i + 1) * step] = panoImageVpsAll[:, i * step:(i + 1) * step] / maxValue
        else:
            panoImageVpsAll[:, i * step:(i + 1) * step] = panoImageVpsAll[:, i * step:(i + 1) * step] / splitMaxAvg

    # 灭点都被归一化到了 1.0
    vps = np.where((0.99 < panoImageVpsAll))
    print("vps", vps)
    sphereVps = []
    # 取值 Segmentation 的中心作为灭点
    for i in range(len(vps[0])):
        segmentationI = vps[0][i]
        segmentationJ = vps[1][i]
        phi, theta = segmentation2phi_theta(segmentationI, segmentationJ)
        x, y, z = CoordsTransfrom.phi_theta2xyz(phi, theta)
        sphereVps.append(np.array([x, y, z]))
        print("segmentation", phi, theta, "->", x, y, z)

    print("sphereVps", sphereVps)
    orthogonalVps = __GetOrthogonalVps(sphereVps)

    # TODO Debug
    # for vp in orthogonalVps:
    #     x, y = CoordsTransfrom.xyz2xy(vp[0], vp[1], vp[2], panoImage.shape)
    #     cv2.circle(panoImage, (x, y), 20, (0, 0, 0), -1)
    # cv2.imshow("debug", panoImage)
    # cv2.waitKey(0)

    # TODO Debug
    panoImageVpsAll = panoImageVpsAll * 255
    panoImageVpsAll.astype(np.uint8)
    cv2.imwrite("../output/output_pano_line_index_0_canny_debug_" + str(time.time()) + ".jpg", panoImageVpsAll)
    # cv2.imshow("debug", panoImageVpsAll)
    # cv2.waitKey(0)

    return orthogonalVps


def GetLineIndex(line, mapping, orthogonalVps):
    """
    由灭点获取直线的分类（0，1 类为水平直线，2 类 为垂直直线，其他为无法判断）
    """
    x0 = line[0][0]
    y0 = line[0][1]
    x1 = line[0][2]
    y1 = line[0][3]
    u0, v0 = mapping(x0, y0)
    u1, v1 = mapping(x1, y1)
    xa, ya, za = CoordsTransfrom.uv2xyz(u0, v0)
    xb, yb, zb = CoordsTransfrom.uv2xyz(u1, v1)
    a = np.array([xa, ya, za])
    b = np.array([xb, yb, zb])
    normal = np.cross(a, b)

    # 计算直线与三个灭点的误差
    errs = []
    errs.append(abs(np.dot(normal, orthogonalVps[0])))
    errs.append(abs(np.dot(normal, orthogonalVps[1])))
    errs.append(abs(np.dot(normal, orthogonalVps[2])))

    errSort = np.sort(errs)
    # 第二误差仍然很大，说明直线朝向灭点较为可信，否则无法判断
    # Trick, dataset-good-1 房顶测试出来的 2.0 阈值
    if errSort[1] > 2 * errSort[0]:
        lineIndex = errs.index(min(errs))
    else:
        lineIndex = -1  # 无法判断

    # print(errs, lineIndex)
    return lineIndex


def DrawPanoLineWithIndex(panoImage, projectScale=600, sampleRate=1.1):
    """
    在全景图中分类绘制投影直线
    """
    # 估计正交灭点
    orthogonalVps = HoughVpEstimation(panoImage)
    # 输出
    panoLineWithIndex = np.zeros(panoImage.shape, dtype=np.uint8)
    # 计算灭点投影图
    # projectImageAndMappings = Projection.OrthogonalVpsProjection(panoImage, projectScale, orthogonalVps)
    projectImageAndMappings = Projection.ARoundProjection(panoImage, projectScale)
    for i in range(len(projectImageAndMappings)):
        print("project-line", i)
        projectImage = projectImageAndMappings[i][0]
        mapping = projectImageAndMappings[i][1]
        scalemapping = projectImageAndMappings[i][2]
        # step1. 检测直线
        grayImage = cv2.cvtColor(projectImage, cv2.COLOR_BGR2GRAY)
        fld = cv2.ximgproc.createFastLineDetector()
        lines = fld.detect(grayImage)
        if lines is None:
            continue
        for line in lines:
            x0 = line[0][0]
            y0 = line[0][1]
            x1 = line[0][2]
            y1 = line[0][3]
            dx = x1 - x0
            dy = y1 - y0

            lineIndex = GetLineIndex(line, mapping, orthogonalVps)
            if lineIndex == 0:
                bgr = (255, 0, 0)
            elif lineIndex == 1:
                bgr = (0, 255, 0)
            elif lineIndex == 2:
                bgr = (0, 0, 255)
            else:
                continue

            mx0, my0 = scalemapping(x0, y0)
            mx1, my1 = scalemapping(x1, y1)
            samples = round(max(abs(mx0 - mx1), abs(my0 - my1)) * sampleRate)

            for dt in np.linspace(0, 1, samples):
                x = x0 + dt * dx
                y = y0 + dt * dy
                mx, my = scalemapping(x, y)
                cv2.circle(panoLineWithIndex, (mx, my), 1, bgr, -1)

    return panoLineWithIndex


def DrawPanoLine(panoImage, lines, scalemapping, bgr, sampleRate=1.1):
    """
    在全景图中绘制投影直线
    :param panoImage:       全景图
    :param lines:           投影直线
    :param scalemapping:    投影图到全景图的映射（带scale）
    :param bgr:             颜色
    :param sampleRate:      绘制直线的采样率
    """
    for line in lines:
        x0 = line[0][0]
        y0 = line[0][1]
        x1 = line[0][2]
        y1 = line[0][3]
        dx = x1 - x0
        dy = y1 - y0

        mx0, my0 = scalemapping(x0, y0)
        mx1, my1 = scalemapping(x1, y1)
        samples = round(max(abs(mx0 - mx1), abs(my0 - my1)) * sampleRate)

        for dt in np.linspace(0, 1, samples):
            x = x0 + dt * dx
            y = y0 + dt * dy
            mx, my = scalemapping(x, y)
            cv2.circle(panoImage, (mx, my), 1, bgr, -1)
