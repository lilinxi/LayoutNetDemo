import CoordsTransfrom
import numpy as np


def __Projection(panoImage, projectScale, genRay):
    """
    :param panoImage:       全景图
    :param projectScale:    投影图大小
    :param genRay:          光线生成函数，投影图采样 uv -> ray
    :return: projectImage, mapping, scalemapping
    """
    projectImage = np.zeros((projectScale, projectScale, 3), dtype=np.uint8)
    # 循环从投影图像中采样光线
    projectU = 0  # projectU -> projectY
    for du in np.linspace(1, -1, projectScale):  # 列从上到下
        projectV = 0  # projectV -> projectX
        for dv in np.linspace(-1, 1, projectScale):  # 行从左到右
            # 采样光线
            ray = genRay(du, dv)
            ray = ray / np.linalg.norm(ray)
            # 光线打到全景图上的坐标
            x, y = CoordsTransfrom.xyz2xy(ray[0], ray[1], ray[2], panoImage.shape)
            # bgr 写入投影图
            bgr = panoImage[y][x]
            projectImage[projectU][projectV] = bgr
            projectV += 1
        projectU += 1

    def mapping(x, y):  # y->u,x->v
        """
        投影图 xy to 全景图 uv
        """
        pu, pv = CoordsTransfrom.xy2project_uv(x, y, projectScale)
        ray = genRay(pu, pv)
        ray = ray / np.linalg.norm(ray)
        mu, mv = CoordsTransfrom.xyz2uv(ray[0], ray[1], ray[2])
        return mu, mv

    def scalemapping(x, y):  # y->u,x->v
        """
        投影图 xy to 全景图 xy
        """
        mu, mv = mapping(x, y)
        mx, my = CoordsTransfrom.uv2xy(mu, mv, panoImage.shape)
        return mx, my

    return projectImage, mapping, scalemapping


def BuildCoords(normal):
    """
    建立局部坐标系(右手坐标系)
    :param normal:  已经正则化，作为坐标系的x轴
    :return:        坐标系的y轴，z轴
    """
    if normal[1] != 0:
        v1 = np.array([-normal[1], normal[0], 0])
    else:
        v1 = np.array([0, 1, 0])
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(normal, v1)
    return v1, v2


def __RayProjection(panoImage, projectScale, xyz):
    """
    以全景图 xyz 为投影图中心的投影转换
    """
    normal = np.array(xyz)
    v1, v2 = BuildCoords(normal)
    rotateMat = np.array([normal, v1, v2]).T  # 将 ray(x,y,z) 的中心旋转到 (1,0,0) 的旋转矩阵，正反没有关系，只要 mapping 是匹配的就可以
    # for du in np.linspace(1, -1, projectScale):  # 列从上到下
    # for dv in np.linspace(-1, 1, projectScale):  # 第一行
    genRay = lambda u, v: rotateMat.dot(np.array([1, v, u]))
    return __Projection(panoImage, projectScale, genRay)


def ARoundProjection(panoImage, projectScale):
    """
    平行环视一圈，用于检测两个水平灭点
    """
    ret = []
    for xyz in [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        # [0, 0, 1],
        # [0, 0, -1],
    ]:
        print("project-xyz", xyz)
        projectImage, mapping, scalemapping = __RayProjection(panoImage, projectScale, xyz)
        # import cv2
        # cv2.imshow("debug", projectImage)
        # cv2.waitKey(0)
        ret.append([projectImage, mapping, scalemapping])

    return ret


def OrthogonalVpsProjection(panoImage, projectScale, orthogonalVps):
    """
    由正交灭点方向生成六个投影图，用于直线检测，灭点方向的直线检测更精确
    """
    ret = []
    for vp in orthogonalVps:
        print("project-vp", vp)

        projectImage, mapping, scalemapping = __RayProjection(panoImage, projectScale, vp)
        ret.append([projectImage, mapping, scalemapping])
        # import cv2
        # cv2.imshow("debug", projectImage)
        # cv2.waitKey(0)

        projectImage, mapping, scalemapping = __RayProjection(panoImage, projectScale, -vp)
        ret.append([projectImage, mapping, scalemapping])
        # import cv2
        # cv2.imshow("debug", projectImage)
        # cv2.waitKey(0)

    return ret
