import PIL
import numpy as np
import torch
from PIL import Image

try:
    import rasterio

    rasterioIMPORTED = True
except ImportError:
    print("no rasterio support")
    rasterioIMPORTED = False


def getBinaryFrequency(labelset):
    freq = np.zeros(2, dtype=int)
    for im in labelset:
        freq[0] += np.sum((im == 0).astype(int))
        freq[1] += np.sum((im == 1).astype(int))
    return freq


def safeuint8(x):
    x0 = np.zeros(x.shape, dtype=float)
    x255 = np.ones(x.shape, dtype=float) * 255
    x = np.maximum(x0, np.minimum(x.copy(), x255))
    return np.uint8(x)


def symetrie(x, y, i, j, k):
    if i == 1:
        x, y = np.transpose(x, axes=(1, 0, 2)), np.transpose(y, axes=(1, 0))
    if j == 1:
        x, y = np.flip(x, axis=1), np.flip(y, axis=1)
    if k == 1:
        x, y = np.flip(x, axis=1), np.flip(y, axis=1)
    return x.copy(), y.copy()


def normalizehistogram(im):
    if len(im.shape) == 2:
        allvalues = list(im.flatten())
        allvalues = sorted(allvalues)
        n = len(allvalues)
        allvalues = allvalues[0:int(98 * n / 100)]
        allvalues = allvalues[int(2 * n / 100):]

        n = len(allvalues)
        k = n // 255
        pivot = [0] + [allvalues[i] for i in range(0, n, k)]
        assert (len(pivot) >= 255)

        out = np.zeros(im.shape, dtype=int)
        for i in range(1, 255):
            out = np.maximum(out, np.uint8(im > pivot[i]) * i)

        return np.uint8(out)

    else:
        output = im.copy()
        for i in range(im.shape[2]):
            output[:, :, i] = normalizehistogram(im[:, :, i])
        return output


class SegSemDataset:
    def __init__(self, datasetname):
        # metadata
        self.datasetname = datasetname
        self.nbchannel = -1
        self.resolution = -1

        # vt structure
        self.setofcolors = []
        self.colorweights = []

        # path to data
        self.root = ""
        self.pathTOdata = {}

    def metadata(self):
        return (self.datasetname, self.nbchannel, len(self.setofcolors))

    def getnames(self):
        return [name for name in self.pathTOdata]

    def getImageAndLabel(self, name, innumpy=True):
        x, y = self.pathTOdata[name]

        if self.nbchannel == 3:
            image = PIL.Image.open(self.root + "/" + x).convert("RGB").copy()
        else:
            image = PIL.Image.open(self.root + "/" + x).convert("L").copy()
        image = np.asarray(image, dtype=np.uint8)  # warning wh swapping

        label = PIL.Image.open(self.root + "/" + y).convert("RGB").copy()
        label = self.colorvtTOvt(np.asarray(label, dtype=np.uint8))  # warning wh swapping

        if innumpy:
            return image, label
        else:
            if self.nbchannel == 3:
                image = torch.Tensor(np.transpose(image, axes=(2, 0, 1))).unsqueeze(0)
            else:
                image = torch.Tensor(image).unsqueeze(0).unsqueeze(0)
            return image, label

    def getrawrandomtiles(self, nbtiles, tilesize):
        XY = []
        nbtilesperimage = nbtiles // len(self.pathTOdata) + 1

        # crop
        for name in self.pathTOdata:
            image, label = self.getImageAndLabel(name)

            row = np.random.randint(0, image.shape[0] - tilesize - 2, size=nbtilesperimage)
            col = np.random.randint(0, image.shape[1] - tilesize - 2, size=nbtilesperimage)

            for i in range(nbtilesperimage):
                im = image[row[i]:row[i] + tilesize, col[i]:col[i] + tilesize, :].copy()
                mask = label[row[i]:row[i] + tilesize, col[i]:col[i] + tilesize].copy()
                XY.append((im, mask))

        # symetrie
        symetrieflag = np.random.randint(0, 2, size=(len(XY), 3))
        XY = [(symetrie(x, y, symetrieflag[i][0], symetrieflag[i][1], symetrieflag[i][2])) for i, (x, y) in
              enumerate(XY)]
        return XY

    def getrandomtiles(self, nbtiles, tilesize, batchsize):
        XY = self.getrawrandomtiles(nbtiles, tilesize)

        # pytorch
        if self.nbchannel == 3:
            X = torch.stack([torch.Tensor(np.transpose(x, axes=(2, 0, 1))).cpu() for x, y in XY])
        else:
            X = torch.stack([torch.Tensor(x).unsqueeze(0).cpu() for x, y in XY])
        Y = torch.stack([torch.from_numpy(y).long().cpu() for x, y in XY])
        dataset = torch.utils.data.TensorDataset(X, Y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=2)

        return dataloader

    def getCriterionWeight(self):
        if self.colorweights == []:
            freq = self.getfrequency()
            self.colorweights = [1., 1. * freq[0] / freq[1]]
            print("frequency in", self.datasetname, "=", self.colorweights)

        return self.colorweights.copy()

    def getfrequency(self):
        alllabels = []
        for name in self.pathTOdata:
            _, label = self.getImageAndLabel(name)
            alllabels.append(label)
        freq = getBinaryFrequency(alllabels)
        return freq

    def vtTOcolorvt(self, mask):
        maskcolor = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=int)
        for i in range(len(self.setofcolors)):
            for ch in range(3):
                maskcolor[:, :, ch] += ((mask == i).astype(int)) * self.setofcolors[i][ch]
        return safeuint8(maskcolor)

    def colorvtTOvt(self, maskcolor):
        mask = np.zeros((maskcolor.shape[0], maskcolor.shape[1]), dtype=int)
        for i in range(len(self.setofcolors)):
            mask1 = (maskcolor[:, :, 0] == self.setofcolors[i][0]).astype(int)
            mask2 = (maskcolor[:, :, 1] == self.setofcolors[i][1]).astype(int)
            mask3 = (maskcolor[:, :, 2] == self.setofcolors[i][2]).astype(int)
            mask += i * mask1 * mask2 * mask3

        return mask

    def copyTOcache(self, pathTOcache="build", outputresolution=-1, color=True, normalize=False, outputname=""):
        nativeresolution = self.resolution
        if outputresolution < 0:
            outputresolution = nativeresolution
        if outputname == "":
            out = SegSemDataset(self.datasetname)
        else:
            out = SegSemDataset(outputname)

        out.nbchannel = self.nbchannel
        out.setofcolors = self.setofcolors
        out.resolution = outputresolution
        out.colorweights = self.colorweights

        out.root = pathTOcache
        for name in self.pathTOdata:
            x, y = self.pathTOdata[name]

            if color:
                image = PIL.Image.open(self.root + "/" + x).convert("RGB").copy()
            else:
                image = PIL.Image.open(self.root + "/" + x).convert("L").copy()

            label = PIL.Image.open(self.root + "/" + y).convert("RGB").copy()

            if nativeresolution != outputresolution:
                image = image.resize((int(image.size[0] * nativeresolution / outputresolution),
                                      int(image.size[1] * nativeresolution / outputresolution)), PIL.Image.BILINEAR)
                label = label.resize((image.size[0], image.size[1]), PIL.Image.NEAREST)

            label = out.vtTOcolorvt(out.colorvtTOvt(
                np.asarray(label, dtype=np.uint8)))  # very slow but avoid frustrating bug due to label color coding
            label = PIL.Image.fromarray(label)

            if normalize:
                image = np.asarray(image, dtype=np.uint8)
                image = normalizehistogram(image)
                image = PIL.Image.fromarray(image)

            image.save(out.root + "/" + name + "_x.png")
            label.save(out.root + "/" + name + "_y.png")
            out.pathTOdata[name] = (name + "_x.png", name + "_y.png")

        return out


def makeSEMCITY(datasetpath, labelflag="normal", weightflag="surfaceonly", dataflag="all"):
    assert rasterioIMPORTED
    assert (labelflag in ["lod0", "normal"])
    assert (weightflag in ["uniform", "iou", "surfaceonly"])
    assert (dataflag in ["all", "train", "test"])

    data = SegSemDataset("SEMCITY")
    data.nbchannel, data.resolution, data.root = 3, 50, ""

    if labelflag == "normal":
        data.setofcolors = [[255, 255, 255],
                            [38, 38, 38],
                            [238, 118, 33],
                            [34, 139, 34],
                            [0, 222, 137],
                            [255, 0, 0],
                            [0, 0, 238],
                            [160, 30, 230]]
        if weightflag == "surfaceonly":
            data.colorweights = [0, 1, 1, 1, 1, 0, 1, 1]
        else:
            data.colorweights = [1, 1, 1, 1, 1, 1, 1, 1]
    if labelflag == "lod0":
        data.setofcolors = [[255, 255, 255], [238, 118, 33]]
        if weightflag == "iou":
            data.colorweights = []
        else:
            data.colorweights = [1, 1]

    l = ["TLS_P_03", "TLS_P_07", "TLS_P_04", "TLS_P_08"]
    for im in l:
        src = rasterio.open(datasetpath + "/" + im + ".tif")
        image = np.int16(src.read(1))
        output = normalizehistogram(image)
        image8bit = PIL.Image.fromarray(np.stack([output] * 3, axis=-1))
        image8bit.save("../../data/tmp/" + im + ".png")

    if dataflag == "test" or dataflag == "all":
        data.pathTOdata["3"] = ("/tmp/TLS_P_03.png", datasetpath + "/TLS_GT_03.tif")
        data.pathTOdata["7"] = ("/tmp/TLS_P_07.png", datasetpath + "/TLS_GT_07.tif")
    if dataflag == "train" or dataflag == "all":
        data.pathTOdata["4"] = ("/tmp/TLS_P_04.png", datasetpath + "/TLS_GT_04.tif")
        data.pathTOdata["8"] = ("/tmp/TLS_P_08.png", datasetpath + "/TLS_GT_08.tif")

    return data


if __name__ == '__main__':
    data = makeSEMCITY('../../data')
    print(data)
