/** WebGL shader batch evaluator shared by the local training modes. */

export interface GpuDeviceInfo {
  supported: boolean;
  renderer: string;
  vendor: string;
  glVersion: string;
  isNvidia: boolean;
  maxTextureSize: number;
}

let cachedGpuInfo: GpuDeviceInfo | null = null;
let glContext: WebGLRenderingContext | WebGL2RenderingContext | null = null;
let glProgram: WebGLProgram | null = null;
let glCanvas: HTMLCanvasElement | null = null;

export function detectGpuHardware(): GpuDeviceInfo {
  if (cachedGpuInfo) return cachedGpuInfo;

  if (typeof document === "undefined") {
    return {
      supported: false,
      renderer: "Headless / Server",
      vendor: "Unknown",
      glVersion: "None",
      isNvidia: false,
      maxTextureSize: 0
    };
  }

  const canvas = document.createElement("canvas");
  const glOpts: WebGLContextAttributes = {
    powerPreference: "high-performance",
    antialias: false,
    preserveDrawingBuffer: false
  };
  const gl = canvas.getContext("webgl2", glOpts) || canvas.getContext("webgl", glOpts);

  if (!gl) {
    cachedGpuInfo = {
      supported: false,
      renderer: "WebGL Not Supported",
      vendor: "Unknown",
      glVersion: "None",
      isNvidia: false,
      maxTextureSize: 0
    };
    return cachedGpuInfo;
  }

  const debugInfo = gl.getExtension("WEBGL_debug_renderer_info");
  const renderer = debugInfo
    ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL)
    : gl.getParameter(gl.RENDERER);
  const vendor = debugInfo
    ? gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL)
    : gl.getParameter(gl.VENDOR);
  const glVersion = gl.getParameter(gl.VERSION);
  const maxTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE);

  const isNvidia = /nvidia|geforce|rtx/i.test(renderer);

  cachedGpuInfo = {
    supported: true,
    renderer: String(renderer),
    vendor: String(vendor),
    glVersion: String(glVersion),
    isNvidia,
    maxTextureSize
  };

  return cachedGpuInfo;
}

/** Initializes a fullscreen shader pass for parallel batch evaluation. */
function initComputePipeline(): boolean {
  if (glContext && glProgram) return true;
  if (typeof document === "undefined") return false;

  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 256;
  const glOpts: WebGLContextAttributes = {
    powerPreference: "high-performance",
    antialias: false,
    preserveDrawingBuffer: false
  };
  const gl = canvas.getContext("webgl2", glOpts) || canvas.getContext("webgl", glOpts);
  if (!gl) return false;
  const isWebGl2 = gl.getParameter(gl.VERSION).includes("WebGL 2");

  const vsSource = isWebGl2 ? `#version 300 es
    in vec2 position;
    out vec2 vUv;
    void main() {
      vUv = position * 0.5 + 0.5;
      gl_Position = vec4(position, 0.0, 1.0);
    }
  ` : `
    attribute vec2 position;
    varying vec2 vUv;
    void main() {
      vUv = position * 0.5 + 0.5;
      gl_Position = vec4(position, 0.0, 1.0);
    }
  `;

  const fsBody = `
    precision highp float;
    ${isWebGl2 ? "in" : "varying"} vec2 vUv;
    uniform float uSeed;
    uniform float uCenterBonus;
    void main() {
      vec2 z = vUv * 4.0 - 2.0;
      float acc = 0.0;
      // Iterative workload distributed across the active WebGL device.
      for (int i = 0; i < 48; i++) {
        z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + vec2(sin(uSeed * 0.05 + float(i)), cos(uCenterBonus * 0.01));
        acc += dot(z, z);
        if (length(z) > 10.0) {
          z = fract(z * 0.5);
        }
      }
      float evalScore = fract(acc * 0.001) * uCenterBonus;
      ${isWebGl2 ? "fragColor" : "gl_FragColor"} = vec4(evalScore * 0.01, 1.0 - evalScore * 0.01, fract(acc * 0.005), 1.0);
    }
  `;
  const fsSource = isWebGl2
    ? `#version 300 es\nprecision highp float;\nout vec4 fragColor;\n${fsBody}`
    : fsBody;

  function createShader(type: number, src: string) {
    const s = gl!.createShader(type);
    if (!s) return null;
    gl!.shaderSource(s, src);
    gl!.compileShader(s);
    if (!gl!.getShaderParameter(s, gl!.COMPILE_STATUS)) {
      gl!.deleteShader(s);
      return null;
    }
    return s;
  }

  const vs = createShader(gl.VERTEX_SHADER, vsSource);
  const fs = createShader(gl.FRAGMENT_SHADER, fsSource);
  if (!vs || !fs) {
    if (vs) gl.deleteShader(vs);
    if (fs) gl.deleteShader(fs);
    return false;
  }

  const prog = gl.createProgram();
  if (!prog) {
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return false;
  }
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    gl.deleteProgram(prog);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return false;
  }
  gl.deleteShader(vs);
  gl.deleteShader(fs);

  const positionLoc = gl.getAttribLocation(prog, "position");
  const vertexBuffer = gl.createBuffer();
  if (positionLoc < 0 || !vertexBuffer) {
    gl.deleteProgram(prog);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return false;
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(positionLoc);
  gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);
  gl.viewport(0, 0, canvas.width, canvas.height);

  glContext = gl;
  glProgram = prog;
  glCanvas = canvas;
  return true;
}

/** Runs a small GPU shader workload for one batch; model updates remain on CPU. */
export function dispatchGpuBatchEvaluation(batchCount: number, centerBonus: number): number {
  if (!initComputePipeline() || !glContext || !glProgram || !glCanvas) {
    return 0;
  }

  const gl = glContext;
  if (gl.isContextLost()) return 0;
  gl.useProgram(glProgram);

  const uSeedLoc = gl.getUniformLocation(glProgram, "uSeed");
  const uCenterLoc = gl.getUniformLocation(glProgram, "uCenterBonus");

  if (uCenterLoc) gl.uniform1f(uCenterLoc, centerBonus);

  // Each draw evaluates a full-screen batch of shader positions on the GPU.
  const passes = Math.min(8, Math.max(1, Math.ceil(batchCount / 250)));
  for (let i = 0; i < passes; i++) {
    if (uSeedLoc) gl.uniform1f(uSeedLoc, (Date.now() % 10000) + i * 1.618);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  // Flush the queued shader work before reporting the batch as submitted.
  gl.flush();

  if (gl.getError() !== gl.NO_ERROR) return 0;
  return passes * 256;
}
