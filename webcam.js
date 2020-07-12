class Webcam {
  /**
   * @param {HTMLVideoElement} webcamElement A HTMLVideoElement representing the
   *     webcam feed.
   */
  constructor(webcamElement) {
    this.webcamElement = webcamElement;
    // this.width = 480;
    // this.height = 640;
    this.canvas = document.querySelector("#c");
    this.gl = this.canvas.getContext("webgl");
    this.program = webglUtils.createProgramFromScripts(this.gl, [
      "vertex-shader-2d",
      "fragment-shader-2d",
    ]);
    this.kernal;
    this.setKernal("normal");
    this.positionLocation;
    this.texcoordLocation;
    this.positionBuffer,
      this.texcoordBuffer,
      this.textureSizeLocation,
      this.resolutionLocation,
      this.kernelLocation,
      this.kernelWeightLocation;
  }

  /**
   * Captures a frame from the webcam and normalizes it between -1 and 1.
   * Returns a batched image (1-element batch) of shape [1, w, h, c].
   */

  capture() {
    return tf.tidy(() => {
      // Reads the image as a Tensor from the webcam <video> element.
      // const canvas = document.createElement("canvas");
      const canvas = document.getElementById("c2");
      canvas.width = 224;
      canvas.height = 224;
      const ctx = canvas.getContext("2d");
      const cw = (this.webcamElement.width - canvas.width) / 2;
      const ch = (this.webcamElement.height - canvas.height) / 2;
      // alert(cw);
      // alert(this.webcamElement.width);

      ctx.drawImage(
        this.webcamElement,
        cw,
        ch,
        canvas.width,
        canvas.height,
        0,
        0,
        canvas.width,
        canvas.height
      );
      const webcamImage = tf.browser.fromPixels(canvas);
      const reversedImage = webcamImage.reverse(1);

      // Crop the image so we're using the center square of the rectangular
      // webcam.
      const croppedImage = this.cropImage(reversedImage);

      // Expand the outer most dimension so we have a batch size of 1.
      const batchedImage = croppedImage.expandDims(0);

      // Normalize the image between -1 and 1. The image comes in between 0-255,
      // so we divide by 127 and subtract 1.
      return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
    });
  }

  /**
   * Crops an image tensor so we get a square image with no white space.
   * @param {Tensor4D} img An input image Tensor to crop.
   */

  cropImage(img) {
    const size = Math.min(img.shape[0], img.shape[1]);
    const centerHeight = img.shape[0] / 2;
    const beginHeight = centerHeight - size / 2;
    const centerWidth = img.shape[1] / 2;
    const beginWidth = centerWidth - size / 2;
    return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
  }

  /**
   * Adjusts the video size so we can make a centered square crop without
   * including whitespace.
   * @param {number} width The real width of the video element.
   * @param {number} height The real height of the video element.
   */

  adjustVideoSize(width, height) {
    const aspectRatio = width / height;
    if (width >= height) {
      this.webcamElement.width = aspectRatio * this.webcamElement.height;
    } else if (width < height) {
      this.webcamElement.height = this.webcamElement.width / aspectRatio;
    }
  }

  async setup() {
    return new Promise((resolve, reject) => {
      navigator.getUserMedia =
        navigator.getUserMedia ||
        navigator.webkitGetUserMedia ||
        navigator.mozGetUserMedia ||
        navigator.msGetUserMedia;
      if (navigator.mediaDevices) {
        navigator.mediaDevices
          .getUserMedia({
            audio: false,
            video: {
              // height: window.innerHeight * 1,
              // width: window.innerWidth * 3.5,
              facingMode:
                window.innerWidth > window.innerHeight
                  ? {
                      exact: "environment",
                    }
                  : "user",
            },
          })
          .then(
            (stream) => {
              let settings = stream.getTracks()[0].getSettings();
              this.webcamElement.setAttribute("autoplay", "");
              this.webcamElement.setAttribute("muted", "");
              this.webcamElement.setAttribute("playsinline", "");
              this.webcamElement.srcObject = stream;
              this.canvas.height = settings.height;
              this.webcamElement.play();
              this.render();
              this.animate();
              document.body.onclick = () => this.webcamElement.play();
              this.webcamElement.addEventListener(
                "loadeddata",
                async () => {
                  this.adjustVideoSize(
                    this.webcamElement.videoWidth,
                    this.webcamElement.videoHeight
                  );
                  resolve();
                },
                false
              );
            },
            (error) => {
              alert(error);
              reject(error);
            }
          );
      } else if (navigator.getUserMedia) {
        navigator.getUserMedia({ audio: false, video: true }, function (
          stream
        ) {
          let settings = stream.getTracks()[0].getSettings();
          this.webcamElement.setAttribute("autoplay", "");
          this.webcamElement.setAttribute("muted", "");
          this.webcamElement.setAttribute("playsinline", "");
          this.webcamElement.srcObject = stream;
          this.canvas.height = settings.height;
          this.webcamElement.play();
          this.render();
          this.animate();
          document.body.onclick = () => this.webcamElement.play();
          this.webcamElement.addEventListener(
            "loadeddata",
            async () => {
              this.adjustVideoSize(
                this.webcamElement.videoWidth,
                this.webcamElement.videoHeight
              );
              resolve();
            },
            false
          );
        });
      } else {
        reject();
      }
    });
  }

  animate() {
    this.gl.texImage2D(
      this.gl.TEXTURE_2D,
      0,
      this.gl.RGBA,
      this.gl.RGBA,
      this.gl.UNSIGNED_BYTE,
      this.webcamElement
    );
    window.vid = this.webcamElement;
    this.drawShit({ height: this.height, width: this.width });
    requestAnimationFrame(this.animate.bind(this));
  }
  render() {
    // look up where the vertex data needs to go.

    this.positionLocation = this.gl.getAttribLocation(
      this.program,
      "a_position"
    );
    this.texcoordLocation = this.gl.getAttribLocation(
      this.program,
      "a_texCoord"
    );

    this.textureSizeLocation = this.gl.getUniformLocation(
      this.program,
      "u_textureSize"
    );

    this.kernelLocation = this.gl.getUniformLocation(
      this.program,
      "u_kernel[0]"
    );
    this.kernelWeightLocation = this.gl.getUniformLocation(
      this.program,
      "u_kernelWeight"
    );

    // Create a buffer to put three 2d clip space points in
    this.positionBuffer = this.gl.createBuffer();

    // Bind it to ARRAY_BUFFER (think of it as ARRAY_BUFFER = positionBuffer)
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.positionBuffer);
    // Set a rectangle the same size as the image.
    this.setRectangle(
      this.gl,
      0,
      0,
      this.webcamElement.width,
      this.webcamElement.height
    );

    // provide texture coordinates for the rectangle.
    this.texcoordBuffer = this.gl.createBuffer();
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.texcoordBuffer);
    this.gl.bufferData(
      this.gl.ARRAY_BUFFER,
      new Float32Array([
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
      ]),
      this.gl.STATIC_DRAW
    );

    // Create a texture.
    var texture = this.gl.createTexture();
    this.gl.bindTexture(this.gl.TEXTURE_2D, texture);

    // Set the parameters so we can render any size image.
    this.gl.texParameteri(
      this.gl.TEXTURE_2D,
      this.gl.TEXTURE_WRAP_S,
      this.gl.CLAMP_TO_EDGE
    );
    this.gl.texParameteri(
      this.gl.TEXTURE_2D,
      this.gl.TEXTURE_WRAP_T,
      this.gl.CLAMP_TO_EDGE
    );
    this.gl.texParameteri(
      this.gl.TEXTURE_2D,
      this.gl.TEXTURE_MIN_FILTER,
      this.gl.NEAREST
    );
    this.gl.texParameteri(
      this.gl.TEXTURE_2D,
      this.gl.TEXTURE_MAG_FILTER,
      this.gl.NEAREST
    );

    // Upload the image into the texture.
    this.gl.texImage2D(
      this.gl.TEXTURE_2D,
      0,
      this.gl.RGBA,
      this.gl.RGBA,
      this.gl.UNSIGNED_BYTE,
      this.webcamElement
    );

    // lookup uniforms
    this.resolutionLocation = this.gl.getUniformLocation(
      this.program,
      "u_resolution"
    );
  }
  setKernal(type) {
    var edgeDetectKernel = [-1, -1, -1, -1, 8, -1, -1, -1, -1];
    var ones = [1, 1, 1, 1, 1, 1, 1, 1, 1];
    var normal = [0, 0, 0, 0, 1, 0, 0, 0, 0];
    if (type === "normal") this.kernal = normal;
    if (type === "edge") this.kernal = edgeDetectKernel;
  }
  setRectangle(gl, x, y, width, height) {
    var x1 = x;
    var x2 = x + width;
    var y1 = y;
    var y2 = y + height;
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([x1, y1, x2, y1, x1, y2, x1, y2, x2, y1, x2, y2]),
      gl.STATIC_DRAW
    );
  }

  computeKernelWeight(kernel) {
    var weight = kernel.reduce(function (prev, curr) {
      return prev + curr;
    });
    return weight <= 0 ? 1 : weight;
  }

  drawShit(video) {
    webglUtils.resizeCanvasToDisplaySize(this.gl.canvas);

    // Tell WebGL how to convert from clip space to pixels
    this.gl.viewport(0, 0, this.gl.canvas.width, this.gl.canvas.height);

    // Clear the canvas
    this.gl.clearColor(0, 0, 0, 0);
    this.gl.clear(this.gl.COLOR_BUFFER_BIT);

    // Tell it to use our program (pair of shaders)
    this.gl.useProgram(this.program);

    // Turn on the position attribute
    this.gl.enableVertexAttribArray(this.positionLocation);

    // Bind the position buffer.
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.positionBuffer);

    // Tell the position attribute how to get data out of positionBuffer (ARRAY_BUFFER)
    var size = 2; // 2 components per iteration
    var type = this.gl.FLOAT; // the data is 32bit floats
    var normalize = false; // don't normalize the data
    var stride = 0; // 0 = move forward size * sizeof(type) each iteration to get the next position
    var offset = 0; // start at the beginning of the buffer
    this.gl.vertexAttribPointer(
      this.positionLocation,
      size,
      type,
      normalize,
      stride,
      offset
    );

    // Turn on the texcoord attribute
    this.gl.enableVertexAttribArray(this.texcoordLocation);

    // bind the texcoord buffer.
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.texcoordBuffer);

    // Tell the texcoord attribute how to get data out of texcoordBuffer (ARRAY_BUFFER)
    var size = 2; // 2 components per iteration
    var type = this.gl.FLOAT; // the data is 32bit floats
    var normalize = false; // don't normalize the data
    var stride = 0; // 0 = move forward size * sizeof(type) each iteration to get the next position
    var offset = 0; // start at the beginning of the buffer
    this.gl.vertexAttribPointer(
      this.texcoordLocation,
      size,
      type,
      normalize,
      stride,
      offset
    );

    this.gl.uniform2f(
      this.textureSizeLocation,
      this.webcamElement.width,
      this.webcamElement.height
    );
    // set the resolution
    this.gl.uniform2f(
      this.resolutionLocation,
      this.gl.canvas.width,
      this.gl.canvas.height
    );

    this.gl.uniform1fv(this.kernelLocation, this.kernal);
    this.gl.uniform1f(
      this.kernelWeightLocation,
      this.computeKernelWeight(this.kernal)
    );
    // Draw the rectangle.
    var primitiveType = this.gl.TRIANGLES;
    var offset = 0;
    var count = 6;
    this.gl.drawArrays(primitiveType, offset, count);
  }
}
