// Copyright (c) 2023 FlyCV Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

if (typeof Module.FS === 'undefined' && typeof FS !== 'undefined') {
    Module.FS = FS;
}

Module['imread'] = (src) => {
    let img = null;
    let canvas = null;
    let context = null;

    if (typeof src === 'string') {
        img = document.getElementById(src);

        if (img === null) {
            img = document.getElementsByClassName(src);
        }
    } else {
        img = src;
    }

    if (img instanceof HTMLImageElement) {
        canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        context = canvas.getContext('2d', { willReadFrequently: true });
        context.drawImage(img, 0, 0, img.width, img.height);
    } else if (img instanceof HTMLCanvasElement) {
        canvas = img;
        context = canvas.getContext('2d');
    } else {
        throw new Error('Please input the valid canvas or img id or classname');
    }

    let imgData = context.getImageData(0, 0, canvas.width, canvas.height);
    return fcv.matFromImageData(imgData);
};

Module['imshow'] = (targetCanvas, mat) => {
    let canvas = null;

    if (typeof targetCanvas === 'string') {
        canvas = document.getElementById(targetCanvas);

        if (canvas === null) {
            canvas = document.getElementsByClassName(targetCanvas);
        }
    } else {
        canvas = targetCanvas;
    }

    if (!(canvas instanceof HTMLCanvasElement)) {
        throw new Error('Please input the valid canvas element, id or classname.');
    }

    if (!(mat instanceof fcv.Mat)) {
        throw new Error('Please input the valid fcv.Mat instance.');
    }

    let image = new fcv.Mat();
    let need_delete = true;

    if (mat.type() === fcv.FCVImageType.PKG_RGBA_U8) {
        image = mat;
        need_delete = false;
    } else if (mat.type() === fcv.FCVImageType.PKG_RGB_U8) {
        fcv.cvtColor(mat, image, fcv.ColorConvertType.CVT_PA_RGB2PA_RGBA);
    } else if (mat.type() === fcv.FCVImageType.PKG_BGR_U8) {
        fcv.cvtColor(mat, image, fcv.ColorConvertType.CVT_PA_BGR2PA_RGBA);
    } else if (mat.type() === fcv.FCVImageType.GRAY_U8) {
        fcv.cvtColor(mat, image, fcv.ColorConvertType.CVT_GRAY2PA_RGBA);
    } else if (mat.type() === fcv.FCVImageType.PKG_BGRA_U8) {
        fcv.cvtColor(mat, image, fcv.ColorConvertType.CVT_PA_BGRA2PA_RGBA);
    } else if (mat.type() === fcv.FCVImageType.NV12) {
        fcv.cvtColor(mat, image, fcv.ColorConvertType.CVT_NV122PA_RGBA);
    } else if (mat.type() === fcv.FCVImageType.NV21) {
        fcv.cvtColor(mat, image, fcv.ColorConvertType.CVT_NV212PA_RGBA);
    }

    var imgData = new ImageData(new Uint8ClampedArray(image.data()), image.width(), image.height());
    var ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    canvas.width = imgData.width;
    canvas.height = imgData.height;
    ctx.putImageData(imgData, 0, 0);

    if (need_delete) {
        image.delete();
    }
};

Module['matFromImageData'] = (imageData) => {
    let mat = new fcv.Mat(imageData.width, imageData.height,
            fcv.FCVImageType.PKG_RGBA_U8);
    mat.data().set(imageData.data);
    return mat;
};
