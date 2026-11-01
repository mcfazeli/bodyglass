// BodyGlass - front end (quick surfaces + detailed labelled structures). Everything is loaded from the local server only.
import * as THREE from './vendor/three/three.module.js';
import { OrbitControls } from './vendor/three/addons/controls/OrbitControls.js';
import { GLTFLoader } from './vendor/three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from './vendor/three/addons/environments/RoomEnvironment.js';

const $ = (s) => document.querySelector(s);
const params = new URLSearchParams(location.search);
if (params.get('clean') === '1') document.body.classList.add('clean');
const PLANE_COLORS = { axial: 0x5fb6ff, coronal: 0x6ee7a0, sagittal: 0xffb35c };

// ---------- renderer / scene
const canvas = $('#gl');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, preserveDrawingBuffer: params.has('shot') });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;
renderer.localClippingEnabled = true;
renderer.setClearColor(0x000000, 0);

const scene = new THREE.Scene();
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.55;

const camera = new THREE.PerspectiveCamera(32, innerWidth / innerHeight, 1, 5000);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.07;
controls.rotateSpeed = 0.75; controls.zoomSpeed = 0.9; controls.panSpeed = 0.6;
controls.minDistance = 40; controls.maxDistance = 1200;   // close enough to inspect one vertebra
controls.autoRotateSpeed = 0.9;

const key = new THREE.DirectionalLight(0xfff1e0, 2.4);
key.position.set(260, 420, -300); key.castShadow = true;
key.shadow.mapSize.set(1024, 1024);
Object.assign(key.shadow.camera, { left: -180, right: 180, top: 180, bottom: -180, near: 50, far: 1400 });
key.shadow.bias = -0.0008; key.shadow.normalBias = 0.6;
scene.add(key, key.target);
const rim = new THREE.DirectionalLight(0x9cc2ff, 1.1); rim.position.set(-300, 160, 380); scene.add(rim);
const fill = new THREE.HemisphereLight(0xcfe0ff, 0x2a1d1a, 0.35); scene.add(fill);

// RAS (mm) -> three: rotate so Superior is up; child group centres the brain at the origin.
const ras = new THREE.Group(); ras.rotation.x = -Math.PI / 2; scene.add(ras);
const centered = new THREE.Group(); ras.add(centered);

// floor shadow + glow ring for the "game" stage look
const floor = new THREE.Mesh(new THREE.CircleGeometry(260, 64), new THREE.ShadowMaterial({ opacity: 0.32 }));
floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true; scene.add(floor);
const ring = new THREE.Mesh(new THREE.RingGeometry(120, 123, 128),
  new THREE.MeshBasicMaterial({ color: 0x7cc4ff, transparent: true, opacity: 0.16, side: THREE.DoubleSide, blending: THREE.AdditiveBlending, depthWrite: false }));
ring.rotation.x = -Math.PI / 2; scene.add(ring);

// ---------- state
let manifest = null, vol = null, dims = null, origin = null, sp = 1;
let neckPad = null;          // neck exam shown: slices of the brain grid it adds below (null = no neck)
const padZ = (a) => { if (!neckPad) return a; const o = new Uint8Array(dims[0] * dims[1] * dims[2]); o.set(a, neckPad * dims[0] * dims[1]); return o; };
const layers = {};          // id -> {def, mesh, material, visible, opacity, row}
const planes = {};          // name -> {mesh, mat, edge, enabled, pos}
let selected = null, contrast = 1.2, isolateId = null, parcels = null;
let aiSel = null;   // the selected part, for Ask AI (bottom of file)
const structById = {};       // structure id -> {st, item, L}
let labelsOn = params.get('labels') !== '0', regionColors = params.get('lobecolors') !== '0';
const BRAIN_PINK = new THREE.Color('#eaa6a2');

// ---------- materials
function fresnelPatch(mat) {
  mat.userData.uFresnel = { value: 0 };
  mat.onBeforeCompile = (sh) => {
    sh.uniforms.uFresnel = mat.userData.uFresnel;
    sh.fragmentShader = 'uniform float uFresnel;\n' + sh.fragmentShader.replace('#include <opaque_fragment>', `#include <opaque_fragment>
      if (uFresnel > 0.5) {
        float fr = pow(1.0 - abs(dot(normalize(normal), normalize(vViewPosition))), 2.2);
        gl_FragColor.a = clamp(gl_FragColor.a * mix(0.35, 2.6, fr), 0.0, 1.0);
      }`);
  };
  return mat;
}
function makeMaterial(def, hasColors) {
  const c = new THREE.Color(def.material === 'brain' && !def.ownColor ? '#eaa6a2' : def.color);
  const base = { color: c, side: THREE.FrontSide, clipShadows: true };
  let m;
  switch (def.material) {
    case 'skin': m = new THREE.MeshPhysicalMaterial({ ...base, roughness: 0.58, sheen: 0.6, sheenColor: new THREE.Color(0xffc9b0), sheenRoughness: 0.5, side: THREE.DoubleSide }); break;
    case 'bone': m = new THREE.MeshPhysicalMaterial({ ...base, roughness: 0.75, clearcoat: 0.1, side: THREE.DoubleSide }); break;
    case 'brain': m = new THREE.MeshPhysicalMaterial({ ...base, roughness: 0.55, clearcoat: 0.1, clearcoatRoughness: 0.45, sheen: 0.35, sheenColor: new THREE.Color(0xffc4b4), sheenRoughness: 0.6, vertexColors: hasColors, side: THREE.DoubleSide }); break;
    case 'wm': m = new THREE.MeshPhysicalMaterial({ ...base, roughness: 0.5, clearcoat: 0.2, vertexColors: hasColors, side: THREE.DoubleSide }); break;
    case 'deep': m = new THREE.MeshPhysicalMaterial({ ...base, roughness: 0.4, clearcoat: 0.35, clearcoatRoughness: 0.3, sheen: 0.3, sheenColor: c.clone().offsetHSL(0, 0, 0.18), sheenRoughness: 0.5, side: THREE.DoubleSide }); break;
    case 'fluid': m = new THREE.MeshPhysicalMaterial({ ...base, roughness: 0.12, metalness: 0.0, clearcoat: 1, clearcoatRoughness: 0.08, emissive: new THREE.Color(0x0b3a5c), side: THREE.DoubleSide }); break;
    default: m = new THREE.MeshStandardMaterial(base);
  }
  if (def.material === 'fluid') m.userData.fluid = true;
  return fresnelPatch(m);
}
// A layer is a Phase 1 surface (one item) or a Phase 2 structure group (one item per structure).
function itemOpacity(L, it) {
  if (L.hidden) return 0;
  if (!isolateId) return L.opacity;
  if (it.st && it.st.id === isolateId) return 1;
  return (L.def.id === 'skin' || L.def.id === 'skull') ? 0 : Math.min(L.opacity, 0.07);
}
function applyOpacity(L) {
  for (const it of L.items) {
    const m = it.material, o = itemOpacity(L, it), transparent = o < 0.99;
    m.transparent = transparent; m.depthWrite = !transparent; m.opacity = o;
    m.userData.uFresnel.value = transparent ? 1 : 0;
    const isoMe = isolateId && it.st && it.st.id === isolateId;
    it.mesh.visible = !L.hidden && (isoMe || (L.visible && (!it.st || it.st.visible) && o > 0.01));
    it.mesh.castShadow = it.mesh.visible && !transparent;
    m.needsUpdate = true;
  }
  if (L.row) { L.row.querySelector('input[type=range]').value = L.opacity; L.row.classList.toggle('off', !(L.visible && L.opacity > 0.01)); L.row.querySelector('input[type=checkbox]').checked = L.visible; }
}
const applyAll = () => Object.values(layers).forEach(applyOpacity);

// ---------- slice planes (sample the 3D texture in RAS space)
const sliceVS = `
  uniform mat4 uToRas; varying vec3 vRas;
  #include <clipping_planes_pars_vertex>
  void main(){ vRas = (uToRas * vec4(position,1.0)).xyz; vec4 mvPosition = modelViewMatrix*vec4(position,1.0); gl_Position = projectionMatrix*mvPosition; }`;
const sliceFS = `
  precision highp sampler3D;
  uniform sampler3D uVol; uniform vec3 uOrigin; uniform vec3 uDims; uniform float uSp; uniform float uContrast; uniform vec3 uTint; uniform vec3 uCutPos; uniform vec3 uCutOn; uniform vec3 uCutSign; uniform int uAxis;
  uniform sampler3D uLab; uniform sampler2D uPal; uniform float uLabOn; uniform float uLabOp;
  uniform sampler3D uSeq; uniform sampler3D uZone; uniform float uZoneOn; uniform sampler2D uCmap; uniform float uSeqOn; uniform float uSeqMode; uniform float uSeqOp;
  varying vec3 vRas;
  void main(){
    vec3 tc = ((vRas - uOrigin)/uSp + 0.5)/uDims;
    if (any(lessThan(tc, vec3(0.0))) || any(greaterThan(tc, vec3(1.0)))) discard;
    for (int a = 0; a < 3; a++) { if (a != uAxis && uCutOn[a] > 0.5 && (vRas[a] - uCutPos[a]) * uCutSign[a] < 0.0) discard; }
    float v = texture(uVol, tc).r;
    if (v < 0.035) discard;                       // let air be see-through
    v = clamp((v - 0.5) * uContrast + 0.5 - (uContrast-1.0)*0.18, 0.0, 1.0);
    vec3 col = mix(vec3(v), vec3(v)*uTint, 0.12);
    if (uSeqOn > 0.5) {   // another scan (FLAIR, SWI, ...) aligned to the T1; 0 = not covered by that scan
      float s = texture(uSeq, tc).r;
      if (s < 0.5 / 255.0) col *= 0.3;
      else if (uSeqMode < 0.5) { if (s < 7.0 / 255.0) discard; /* air in that scan */ float g = clamp((s - 0.5) * uContrast + 0.5 - (uContrast-1.0)*0.18, 0.0, 1.0); col = vec3(g); }
      else { vec4 c = texture(uCmap, vec2(s, 0.5)); float z = uZoneOn > 0.5 ? step(0.5, texture(uZone, tc).r * 255.0) : 1.0; col = mix(col, c.rgb, uSeqOp * c.a * z); }
    }
    if (uLabOn > 0.5) {  // FastSurfer structure colours, nearest-neighbour label lookup
      float li = floor(texture(uLab, tc).r * 255.0 + 0.5);
      if (li > 0.5) { vec4 pc = texture(uPal, vec2((li + 0.5) / 256.0, 0.5)); col = mix(col, pc.rgb, uLabOp * pc.a); }
    }
    gl_FragColor = vec4(col, 1.0);  // display-referred grey values, like a PACS viewer
  }`;
function buildPlanes() {
  const size = [dims[0] * sp, dims[1] * sp, dims[2] * sp];
  const ctr = [0, 1, 2].map((i) => origin[i] + size[i] / 2);
  const tex = new THREE.Data3DTexture(vol, dims[0], dims[1], dims[2]);
  tex.format = THREE.RedFormat; tex.type = THREE.UnsignedByteType;
  tex.minFilter = tex.magFilter = THREE.LinearFilter; tex.unpackAlignment = 1; tex.needsUpdate = true;
  const defs = {
    axial: { w: size[0], h: size[1], axis: 2, rot: [0, 0, 0] },
    coronal: { w: size[0], h: size[2], axis: 1, rot: [Math.PI / 2, 0, 0] },
    sagittal: { w: size[1], h: size[2], axis: 0, rot: [Math.PI / 2, Math.PI / 2, 0] },
  };
  for (const [name, d] of Object.entries(defs)) {
    const mat = new THREE.ShaderMaterial({
      vertexShader: sliceVS, fragmentShader: sliceFS, side: THREE.DoubleSide,
      uniforms: { uVol: { value: tex }, uOrigin: { value: new THREE.Vector3(...origin) }, uDims: { value: new THREE.Vector3(...dims) },
        uSp: { value: sp }, uContrast: { value: contrast }, uToRas: { value: new THREE.Matrix4() }, uTint: { value: new THREE.Color(PLANE_COLORS[name]) }, uCutPos: { value: new THREE.Vector3() }, uCutOn: { value: new THREE.Vector3() }, uCutSign: { value: new THREE.Vector3(1, 1, 1) }, uAxis: { value: d.axis },
        uLab: { value: tex }, uPal: { value: null }, uLabOn: { value: 0 }, uLabOp: { value: overlayOpacity },
        uSeq: { value: tex }, uZone: { value: tex }, uZoneOn: { value: 0 }, uCmap: { value: cmapTex('hot') }, uSeqOn: { value: 0 }, uSeqMode: { value: 0 }, uSeqOp: { value: seqOpacity } },
    });
    const geo = new THREE.PlaneGeometry(d.w, d.h);
    const mesh = new THREE.Mesh(geo, mat);
    mesh.rotation.set(...d.rot, 'XYZ');
    mesh.position.set(...ctr);
    const edge = new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color: PLANE_COLORS[name], transparent: true, opacity: 0.4 }));
    mesh.add(edge);
    mesh.userData.plane = name; mesh.renderOrder = 0;
    centered.add(mesh);
    const brainC = manifest.brainCenter[d.axis];
    const lo = origin[d.axis], hi = origin[d.axis] + size[d.axis];
    planes[name] = { mesh, mat, edge, axis: d.axis, lo, hi, ctr, enabled: false, pos: brainC };
    const slider = document.querySelector(`[data-pos=${name}]`);
    Object.assign(slider, { min: lo, max: hi, step: 0.5, value: brainC });
    slider.oninput = () => setPlanePos(name, +slider.value);
    document.querySelector(`[data-plane=${name}]`).onchange = (e) => enablePlane(name, e.target.checked);
    setPlanePos(name, brainC); enablePlane(name, false);
  }
}
function setPlanePos(name, v) {
  const P = planes[name]; P.pos = v;
  const p = P.ctr.slice(); p[P.axis] = v; P.mesh.position.set(...p);
  P.mesh.updateMatrix(); P.mat.uniforms.uToRas.value.copy(P.mesh.matrix);
  document.querySelector(`[data-pos=${name}]`).value = v;
  if ($('#cutaway').checked) updateClipping();
  else for (const Q of Object.values(planes)) Q.mat.uniforms.uCutPos.value.setComponent(P.axis, v);
  drawInsets();
}
function enablePlane(name, on) {
  planes[name].enabled = on; planes[name].mesh.visible = on;
  document.querySelector(`[data-plane=${name}]`).checked = on;
  updateClipping();
}
// Cutaway: remove the part of every layer that lies beyond all active planes (camera-facing octant).
// default removes the right-front-top octant; ?cutdir=back removes the left-back-top octant (no face in view)
const CUT_BACK = params.get('cutdir') === 'back';
const CUT_DIR = { axial: [0, 0, -1], coronal: [0, CUT_BACK ? 1 : -1, 0], sagittal: [CUT_BACK ? 1 : -1, 0, 0] };
const CUT_SIGN = [CUT_BACK ? -1 : 1, CUT_BACK ? -1 : 1, 1];
let clipPlanes = [];
function updateClipping() {
  const on = $('#cutaway').checked;
  centered.updateMatrixWorld(true);
  clipPlanes = [];
  if (on) for (const [name, P] of Object.entries(planes)) {
    if (!P.enabled) continue;
    const n = new THREE.Vector3(...CUT_DIR[name]);
    const pt = new THREE.Vector3(0, 0, 0); pt.setComponent(P.axis, P.pos);
    clipPlanes.push(new THREE.Plane().setFromNormalAndCoplanarPoint(n, pt).applyMatrix4(centered.matrixWorld));
  }
  const cutOn = new THREE.Vector3(), cutPos = new THREE.Vector3();
  for (const P of Object.values(planes)) { cutOn.setComponent(P.axis, on && P.enabled ? 1 : 0); cutPos.setComponent(P.axis, P.pos); }
  for (const P of Object.values(planes)) { P.mat.uniforms.uCutOn.value.copy(cutOn); P.mat.uniforms.uCutPos.value.copy(cutPos); P.mat.uniforms.uCutSign.value.set(...CUT_SIGN); P.edge.visible = !on; }
  const keepDeep = $('#keepDeep').checked; // deep structures stay whole and poke through the cut
  for (const L of Object.values(layers)) for (const it of L.items) {
    const keep = keepDeep && it.st && (it.st.group === 'deep' || it.st.group === 'ventricles');
    it.material.clippingPlanes = keep ? [] : clipPlanes; it.material.clipIntersection = true; it.material.needsUpdate = true;
  }
}
const isClipped = (p, obj) => clipPlanes.length > 0 && !(obj && obj.material && obj.material.clippingPlanes && obj.material.clippingPlanes.length === 0) && clipPlanes.every((pl) => pl.distanceToPoint(p) < 0);

// ---------- 2D inset slice views (radiological convention: patient right on screen left)
// Grey slices and label overlays are cached per (view, index); redraws are batched per animation frame.
let labelVol = null, labelInfo = [], overlayOn = false, overlayOpacity = 0.5;
let seqKind = null, seqMode = 'replace', seqOpacity = 0.6; const seqData = {};  // other scans
const cmp = { open: false, plane: 'axial', mode: 'side', split: 0.5 };
const sliceCache = new Map(), labCache = new Map(), CACHE_MAX = 96;
function cacheGet(cache, key, make) {
  let v = cache.get(key);
  if (v) { cache.delete(key); cache.set(key, v); return v; }  // LRU touch
  v = make(); cache.set(key, v);
  if (cache.size > CACHE_MAX) cache.delete(cache.keys().next().value);
  return v;
}
function sliceIdx(n) { return Math.max(0, Math.min(dims[planes[n].axis] - 1, Math.round((planes[n].pos - origin[planes[n].axis]) / sp))); }
function viewGeom(name) {
  const [nx, ny, nz] = dims;
  // returns width/height and a voxel-offset function for 2D pixel (c, r) in that view, plus slice index
  if (name === 'axial') { const k = sliceIdx('axial'); return { w: nx, h: ny, k, n: nz, off: (c, r) => k * nx * ny + (ny - 1 - r) * nx + (nx - 1 - c) }; }
  if (name === 'coronal') { const j = sliceIdx('coronal'); return { w: nx, h: nz, k: j, n: ny, off: (c, r) => (nz - 1 - r) * nx * ny + j * nx + (nx - 1 - c) }; }
  const i = sliceIdx('sagittal'); return { w: ny, h: nz, k: i, n: nx, off: (c, r) => (nz - 1 - r) * nx * ny + c * nx + i };
}
function greyLut() { const lut = new Uint8ClampedArray(256); for (let g = 0; g < 256; g++) lut[g] = 255 * Math.min(1, Math.max(0, (g / 255 - 0.5) * contrast + 0.5 - (contrast - 1) * 0.18)); return lut; }
function greyImage(name, V) {
  return cacheGet(sliceCache, name + ':' + V.k, () => {
    const img = new ImageData(V.w, V.h), d = img.data, lut = greyLut();
    for (let r = 0, o = 0; r < V.h; r++) for (let c = 0; c < V.w; c++, o += 4) { const v = lut[vol[V.off(c, r)]]; d[o] = d[o + 1] = d[o + 2] = v; d[o + 3] = 255; }
    return img;
  });
}
// slice of another scan: 'replace' = its own grey picture, 'blend' = colour map over the T1 (cached per mode/opacity)
function seqImage(name, V, kind, mode) {
  const S = seqData[kind]; if (!S || !S.vol) return greyImage(name, V);
  return cacheGet(sliceCache, `${kind}:${mode}:${mode === 'blend' ? seqOpacity : ''}:${name}:${V.k}`, () => {
    const img = new ImageData(V.w, V.h), d = img.data, lut = greyLut(), sv = S.vol, cm = CMAPS[S.meta.cmap] || CMAPS.grey;
    for (let r = 0, o = 0; r < V.h; r++) for (let c = 0; c < V.w; c++, o += 4) {
      const off = V.off(c, r), t = vol[off] < 9 ? 0 : lut[vol[off]], s = sv[off];
      let R, G, B;
      if (!s) R = G = B = t * 0.3;
      else if (mode !== 'blend') R = G = B = s < 7 ? 0 : lut[s];
      else { const a = (zoneVol && !zoneVol[off] ? 0 : seqOpacity * cm[s * 4 + 3] / 255); R = t + (cm[s * 4] - t) * a; G = t + (cm[s * 4 + 1] - t) * a; B = t + (cm[s * 4 + 2] - t) * a; }
      d[o] = R; d[o + 1] = G; d[o + 2] = B; d[o + 3] = 255;
    }
    return img;
  });
}
// colour maps (RGBA, alpha = how strongly the colour shows over the T1 in Blend mode)
const CMAPS = {};
(function makeCmaps() {
  const ramp = (stops) => { const a = new Uint8Array(1024); for (let i = 0; i < 256; i++) { const t = i / 255; let k = 0; while (k < stops.length - 2 && t > stops[k + 1][0]) k++;
    const [t0, ...c0] = stops[k], [t1, ...c1] = stops[k + 1], f = Math.min(1, Math.max(0, (t - t0) / (t1 - t0 || 1)));
    for (let ch = 0; ch < 4; ch++) a[i * 4 + ch] = Math.round(c0[ch] + (c1[ch] - c0[ch]) * f); } a.set([0, 0, 0, 0], 0); return a; };
  CMAPS.grey = ramp([[0, 0, 0, 0, 255], [1, 255, 255, 255, 255]]);
  CMAPS.hot = ramp([[0, 0, 0, 0, 0], [0.35, 120, 10, 0, 40], [0.65, 255, 110, 0, 200], [1, 255, 245, 160, 255]]);          // bright = colourful
  CMAPS.cool = ramp([[0, 0, 0, 0, 0], [0.3, 0, 40, 120, 30], [0.7, 30, 160, 255, 190], [1, 190, 245, 255, 255]]);
  CMAPS.ink = ramp([[0, 70, 0, 120, 255], [0.35, 120, 40, 200, 200], [0.6, 120, 60, 200, 0], [1, 255, 255, 255, 0]]);       // dark (veins) = purple
  CMAPS.adc = ramp([[0, 20, 0, 60, 255], [0.15, 40, 60, 200, 220], [0.3, 30, 190, 120, 160], [0.45, 240, 220, 60, 160], [1, 255, 90, 40, 220]]);
})();
const cmapTexCache = {};
function cmapTex(name) {
  if (!cmapTexCache[name]) { const t = new THREE.DataTexture(CMAPS[name] || CMAPS.grey, 256, 1); t.minFilter = t.magFilter = THREE.LinearFilter; t.needsUpdate = true; cmapTexCache[name] = t; }
  return cmapTexCache[name];
}
function labelCanvas(name, V) {
  return cacheGet(labCache, name + ':' + V.k, () => {
    const cv = document.createElement('canvas'); cv.width = V.w; cv.height = V.h;
    const img = new ImageData(V.w, V.h), d = img.data;
    for (let r = 0, o = 0; r < V.h; r++) for (let c = 0; c < V.w; c++, o += 4) {
      const L = labelInfo[labelVol[V.off(c, r)]]; if (!L) continue;
      d[o] = L.rgb[0]; d[o + 1] = L.rgb[1]; d[o + 2] = L.rgb[2]; d[o + 3] = L.a;
    }
    cv.getContext('2d').putImageData(img, 0, 0); return cv;
  });
}
const PLANE_HEX = (n) => '#' + new THREE.Color(PLANE_COLORS[n]).getHexString();
let insetsQueued = false;
function drawInsets() { if (!insetsQueued) { insetsQueued = true; requestAnimationFrame(() => { insetsQueued = false; drawInsetsNow(); }); } }
function drawInsetsNow() {
  if (!vol || !planes.axial || !planes.coronal || !planes.sagittal) return;
  const [nx, ny, nz] = dims;
  const i = sliceIdx('sagittal'), j = sliceIdx('coronal'), k = sliceIdx('axial');
  const cross = { axial: [nx - 1 - i, ny - 1 - j], coronal: [nx - 1 - i, nz - 1 - k], sagittal: [j, nz - 1 - k] };
  for (const name of ['axial', 'coronal', 'sagittal']) {
    const V = viewGeom(name), cv = document.getElementById('c-' + name);
    const s = Math.max(V.w, V.h);
    if (cv.width !== s) { cv.width = s; cv.height = s; }
    const ctx = cv.getContext('2d');
    ctx.fillStyle = '#000'; ctx.fillRect(0, 0, s, s);
    const ox = (s - V.w) >> 1, oy = (s - V.h) >> 1;
    ctx.putImageData(seqKind ? seqImage(name, V, seqKind, seqMode) : greyImage(name, V), ox, oy);
    if (overlayOn && labelVol) { ctx.globalAlpha = overlayOpacity; ctx.drawImage(labelCanvas(name, V), ox, oy); ctx.globalAlpha = 1; }
    const horizPlane = { axial: 'sagittal', coronal: 'sagittal', sagittal: 'coronal' }[name];
    const vertPlane = ['axial', 'coronal', 'sagittal'].find((n) => n !== name && n !== horizPlane);
    ctx.lineWidth = Math.max(1, s / 200);
    ctx.strokeStyle = PLANE_HEX(horizPlane) + 'aa';
    ctx.beginPath(); ctx.moveTo(ox + cross[name][0] + 0.5, oy); ctx.lineTo(ox + cross[name][0] + 0.5, oy + V.h); ctx.stroke();
    ctx.strokeStyle = PLANE_HEX(vertPlane) + 'aa';
    ctx.beginPath(); ctx.moveTo(ox, oy + cross[name][1] + 0.5); ctx.lineTo(ox + V.w, oy + cross[name][1] + 0.5); ctx.stroke();
    drawMeasInset(name, ctx, ox, oy, s);
    ctx.fillStyle = '#cfd8e6'; ctx.font = `${Math.round(s / 14)}px sans-serif`;
    if (name !== 'sagittal') { ctx.fillText('R', s * 0.04, s * 0.95); ctx.fillText('L', s * 0.9, s * 0.95); }
    else { ctx.fillText('P', s * 0.04, s * 0.95); ctx.fillText('A', s * 0.9, s * 0.95); }
    const cap = document.getElementById('cap-' + name);
    if (cap) cap.textContent = `${seqKind ? seqData[seqKind].meta.name + ' · ' : ''}${name[0].toUpperCase() + name.slice(1)} ${V.k + 1} / ${V.n}`;
    cv._geom = { V, ox, oy, s };
  }
  for (const n of ['axial', 'coronal', 'sagittal']) { const el = document.querySelector(`[data-num=${n}]`); if (el) el.textContent = String(sliceIdx(n) + 1); }
  if (cmp.open) drawCompare();
}
// pointer handling: click = move the other two planes to that point; drag up/down or wheel = scrub this view
const SCRUB_PX = 1.5;      // CSS pixels of drag per slice
const WHEEL_PX = 24;       // wheel delta (pixels) per slice; trackpads send many small deltas
function insetVoxel(cv, e) {
  const G = cv._geom; if (!G) return null;
  const rect = cv.getBoundingClientRect();
  const px = Math.floor((e.clientX - rect.left) / rect.width * G.s - G.ox), py = Math.floor((e.clientY - rect.top) / rect.height * G.s - G.oy);
  if (px < 0 || py < 0 || px >= G.V.w || py >= G.V.h) return null;
  return { px, py, off: G.V.off(px, py) };
}
function stepSlice(name, delta) {
  const P = planes[name]; if (!P || !delta) return;
  const idx = Math.max(0, Math.min(dims[P.axis] - 1, sliceIdx(name) + delta));
  setPlanePos(name, origin[P.axis] + idx * sp);
}
const tip = document.createElement('div'); tip.id = 'insetTip'; document.body.appendChild(tip);
function showTip(e, text) { if (!text) { tip.style.display = 'none'; return; } tip.textContent = text; tip.style.display = 'block'; tip.style.left = (e.clientX + 12) + 'px'; tip.style.top = (e.clientY - 28) + 'px'; }
function labelNameAt(cv, e) {
  if (!labelVol) return '';
  const v = insetVoxel(cv, e); if (!v) return '';
  const L = labelInfo[labelVol[v.off]]; return L ? L.name : '';
}
function seqValueAt(cv, e, kind = seqKind) {   // ADC is a measurement: show the number under the pointer
  const S = kind && seqData[kind]; if (!S || !S.vol || kind !== 'adc') return '';
  const v = insetVoxel(cv, e); if (!v || !S.vol[v.off]) return '';
  return `ADC ≈ ${Math.round((S.vol[v.off] - 1) / 254 * S.meta.range[1]).toLocaleString()} ×10⁻⁶ mm²/s`;
}
function setupInsetInput(names = ['axial', 'coronal', 'sagittal'], canvasFor = (n) => document.getElementById('c-' + n)) {
  for (const name0 of names) {
    const cv = canvasFor(name0);
    const nameOf = () => (typeof name0 === 'function' ? name0() : name0);
    let drag = null, wheelAcc = 0;
    cv.addEventListener('pointerdown', (e) => { drag = { y0: e.clientY, x0: e.clientX, acc: 0, moved: false }; cv.setPointerCapture(e.pointerId); e.preventDefault(); });
    cv.addEventListener('pointermove', (e) => {
      if (drag) {
        const dy = drag.y0 - e.clientY;
        if (!drag.moved && Math.hypot(e.clientX - drag.x0, dy) > 4) drag.moved = true;
        if (drag.moved) {
          const want = Math.trunc(dy / SCRUB_PX);
          if (want !== drag.acc) { stepSlice(nameOf(), want - drag.acc); drag.acc = want; }
          showTip(e, `${nameOf()[0].toUpperCase() + nameOf().slice(1)} slice ${sliceIdx(nameOf()) + 1} / ${dims[planes[nameOf()].axis]}`);
        }
        return;
      }
      showTip(e, [seqValueAt(cv, e), overlayOn ? labelNameAt(cv, e) : ''].filter(Boolean).join(' · '));
    });
    cv.addEventListener('pointerup', (e) => {
      const d = drag; drag = null; showTip(e, '');
      if (!d || d.moved) return;
      const v = insetVoxel(cv, e); if (!v) return;
      if (meas.on) { addMeasPoint(insetToRas(nameOf(), v.px, v.py)); return; }
      const [nx, ny, nz] = dims, toPos = (axis, voxel) => origin[axis] + voxel * sp;
      if (nameOf() === 'axial') { setPlanePos('sagittal', toPos(0, nx - 1 - v.px)); setPlanePos('coronal', toPos(1, ny - 1 - v.py)); }
      if (nameOf() === 'coronal') { setPlanePos('sagittal', toPos(0, nx - 1 - v.px)); setPlanePos('axial', toPos(2, nz - 1 - v.py)); }
      if (nameOf() === 'sagittal') { setPlanePos('coronal', toPos(1, v.px)); setPlanePos('axial', toPos(2, nz - 1 - v.py)); }
      const L = labelVol && labelInfo[labelVol[v.off]];
      if (L) { selectStruct(L.id); if (overlayOn) showTip(e, L.name); }
    });
    cv.addEventListener('pointerleave', () => { if (!drag) tip.style.display = 'none'; });
    cv.addEventListener('wheel', (e) => {
      e.preventDefault();
      wheelAcc += e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1);
      const steps = Math.trunc(wheelAcc / WHEEL_PX);
      if (steps) { wheelAcc -= steps * WHEEL_PX; stepSlice(nameOf(), -steps); }
    }, { passive: false });
  }
}
setupInsetInput();

// Background noise (common on some scanners) would show as speckle on the slice planes: set dark voxels
// that are connected to the edge of the volume to 0 (flood fill), so the shader's air test removes them.
function clearAir(v, [nx, ny, nz], thr = 26) {
  const N = v.length, seen = new Uint8Array(N), q = new Uint32Array(N); let h = 0, t = 0;
  const push = (o) => { if (!seen[o] && v[o] < thr) { seen[o] = 1; q[t++] = o; } };
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) { push(z * nx * ny + y * nx); push(z * nx * ny + y * nx + nx - 1); }
  for (let z = 0; z < nz; z++) for (let x = 0; x < nx; x++) { push(z * nx * ny + x); push(z * nx * ny + (ny - 1) * nx + x); }
  for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) { push(y * nx + x); push((nz - 1) * nx * ny + y * nx + x); }
  const sxy = nx * ny;
  while (h < t) {
    const o = q[h++], x = o % nx, y = ((o / nx) | 0) % ny, z = (o / sxy) | 0;
    if (x > 0) push(o - 1); if (x < nx - 1) push(o + 1);
    if (y > 0) push(o - nx); if (y < ny - 1) push(o + nx);
    if (z > 0) push(o - sxy); if (z < nz - 1) push(o + sxy);
  }
  for (let i = 0; i < t; i++) v[q[i]] = 0;
}

// ---------- UI: layers
const PEEL = [ // opacity per layer for each peel level
  { skin: 1, skull: 0, brain: 1, white_matter: 1, ventricles: 1 },
  { skin: 0.25, skull: 0.0, brain: 1, white_matter: 1, ventricles: 1 },
  { skin: 0.14, skull: 0.2, brain: 1, white_matter: 1, ventricles: 1 },
  { skin: 0.08, skull: 0, brain: 0.18, white_matter: 1, ventricles: 1 },
  { skin: 0.06, skull: 0, brain: 0.1, white_matter: 0.16, ventricles: 1 },
];
const ALL2 = { 'g:cortex': 1, 'g:white_matter': 1, 'g:deep': 1, 'g:ventricles': 1, 'g:hindbrain': 1, 'g:spine': 1 };
const PEEL2 = [ // Phase 2: structure groups instead of the single brain / WM / ventricle surfaces
  { skin: 1, skull: 0, ...ALL2 },
  { skin: 0.25, skull: 0, ...ALL2 },
  { skin: 0.14, skull: 0.2, ...ALL2 },
  { skin: 0.08, skull: 0, ...ALL2, 'g:cortex': 0.14 },
  { skin: 0.03, skull: 0, ...ALL2, 'g:cortex': 0.06, 'g:white_matter': 0.08, 'g:hindbrain': 0.25 },
];
const hasStructs = () => !!(manifest && manifest.structures && manifest.structures.length);
function setPeel(level) {
  $('#peel').value = level;
  for (const [id, o] of Object.entries((hasStructs() ? PEEL2 : PEEL)[level])) if (layers[id]) { layers[id].opacity = o; layers[id].visible = o > 0; applyOpacity(layers[id]); }
}
function buildLayerUI() {
  const box = $('#layers'); box.innerHTML = '';
  for (const L of Object.values(layers)) {
    if (L.hidden) continue;
    const row = document.createElement('div'); row.className = 'layer';
    row.innerHTML = `<input type="checkbox" checked><span class="dot" style="background:${L.def.color}"></span><span class="lname">${L.def.name}</span><input type="range" min="0" max="1" step="0.01">`;
    row.querySelector('input[type=checkbox]').onchange = (e) => { L.visible = e.target.checked; if (L.visible && L.opacity < 0.05) L.opacity = 1; applyOpacity(L); };
    row.querySelector('input[type=range]').oninput = (e) => { L.opacity = +e.target.value; L.visible = L.opacity > 0.01; applyOpacity(L); };
    row.querySelector('.lname').onclick = () => { if (!L.def.group) select(L.def.id); };
    L.row = row; box.appendChild(row);
    if (L.def.veins) applyOpacity(L);   // off by default: sync its checkbox and slider
  }
  $('#peel').oninput = (e) => setPeel(+e.target.value);
}
// ---------- structure-label overlay on slices (2D + 3D planes)
let labTex = null, palTex = null;
async function loadLabelVolume(base) {
  labelVol = null; labelInfo = []; sliceCache.clear(); labCache.clear();
  if (labTex) { labTex.dispose(); labTex = null; }
  const LV = manifest.labelVolume;
  if (LV && hasStructs()) {
    labelVol = new Uint8Array(await (await fetch(base + (neckPad != null ? manifest.neck.labelFile : LV.file))).arrayBuffer());
    const pal = new Uint8Array(256 * 4);
    const add = (code, sid) => {
      const S = structById[sid]; if (!S) return;
      const c = new THREE.Color(S.st.color), a = S.st.group === 'white_matter' ? 90 : 255;  // WM is huge: keep it subtle
      labelInfo[code] = { id: sid, name: S.st.name, rgb: [c.r * 255, c.g * 255, c.b * 255].map(Math.round), a };
      pal.set([...labelInfo[code].rgb, a], code * 4);
    };
    LV.index.forEach((sid, i) => add(i + 1, sid));
    if (neckPad != null) for (const [code, sid] of Object.entries(manifest.neck.labelIndex)) add(+code, sid);  // neck labels are 200+
    labTex = new THREE.Data3DTexture(labelVol, dims[0], dims[1], dims[2]);
    labTex.format = THREE.RedFormat; labTex.type = THREE.UnsignedByteType;
    labTex.minFilter = labTex.magFilter = THREE.NearestFilter; labTex.unpackAlignment = 1; labTex.needsUpdate = true;
    if (!palTex) { palTex = new THREE.DataTexture(pal, 256, 1); } else palTex.image.data.set(pal);
    palTex.minFilter = palTex.magFilter = THREE.NearestFilter; palTex.needsUpdate = true;
  }
  updateOverlayUI();
}
function updateOverlayUI() {
  const ready = !!labelVol, keepDeep = $('#keepDeep').checked;
  const usable = ready && !keepDeep;
  $('#overlayRow').classList.toggle('disabled', !usable);
  $('#labelOverlay').disabled = !usable;
  $('#labelOpacity').disabled = !usable || !$('#labelOverlay').checked;
  $('#overlayNote').textContent = !ready
    ? (manifest && manifest.structures ? 'Preparing slice labels…' : 'Available once structure labelling has finished.')
    : keepDeep ? 'Untick "Keep deep structures whole" to colour the slices.' : '';
  overlayOn = usable && $('#labelOverlay').checked;
  for (const P of Object.values(planes)) {
    const u = P.mat.uniforms;
    u.uLabOn.value = overlayOn ? 1 : 0; u.uLabOp.value = overlayOpacity;
    if (labTex) { u.uLab.value = labTex; u.uPal.value = palTex; }
  }
  drawInsets();
}
$('#labelOverlay').onchange = updateOverlayUI;
$('#labelOpacity').oninput = (e) => { overlayOpacity = +e.target.value; $('#labelOpacityVal').textContent = Math.round(overlayOpacity * 100) + '%'; updateOverlayUI(); };

// plain-language reasons for the series list. Every dicom_scan exclusion / penalty label needs a sentence (tests/test_series_words.py)
const PLAIN = {
  'not MR': 'Not an MRI picture (for example a CT or a report page).',
  'SWI / susceptibility': 'A vein-sensitive scan (SWI). Shown on the slices, too grainy for a 3D surface.',
  'phase image': 'A raw phase picture. It looks like noise to the eye; only used to help other scans.',
  'diffusion / ADC': 'A water-movement scan (diffusion). Low resolution; shown on the slices.',
  'FLAIR': 'A fluid-dark scan (FLAIR). Good for slices, but its surfaces are blurrier than a T1.',
  'T2 / PD-weighted': 'A water-bright scan (T2 / PD). Shown on the slices; a T1 gives cleaner surfaces.',
  'localizer': 'A quick positioning scan taken at the start. Too few, too coarse pictures.',
  'echo-planar (fMRI/DWI)': 'A very fast, distorted scan type (used for brain activity or diffusion).',
  '2D acquisition': 'Taken as separate thick slices, so the 3D surface would look stepped.',
  'derived/reformatted (MPR) image': 'A copy recalculated from another series, slightly blurrier than the original.',
  'MP2RAGE inversion image (low contrast)': 'A helper image of a two-part scan; it has little contrast.',
  'quantitative T1 map': 'A map of numbers calculated from another scan, not a normal picture.',
  'too few slices for a 3D model': 'Too few pictures to cover the whole head.',
  'thick slices': 'Thick slices: the 3D surface would look stepped.',
  'multi-frame file without slice positions': 'A multi-frame file that does not say where each slice sits, so it cannot be stacked into 3D.',
};
const plain = (r) => PLAIN[Object.keys(PLAIN).find((k) => r.replace(/^[-+\d.]+ |^excluded: /g, '').startsWith(k))];
const ROLE = { model: ['3D model', 'ok'], neck: ['Neck', 'ok'], slices: ['Slices', 'info'], veins: ['Slices + veins', 'info'], none: ['Not used', 'off'] };
function seriesRole(s, ch) {
  if (s.uid === ch.uid) return 'model';
  const k = s.kind || (sequences().find((q) => q.uid === s.uid) || {}).kind;
  if (k === 'swi' || k === 'mip') return 'veins';
  return k && k !== 'swi_phase' ? 'slices' : 'none';
}
const KIND_PLAIN = { flair: 'A fluid-dark scan (FLAIR)', t2: 'A water-bright scan (T2)', tse: 'A sharp partial T2 scan', swi: 'A vein-sensitive scan (SWI)',
  mip: 'A vein projection (SWI mIP)', swi_mag: 'The plain picture of the vein scan', dwi: 'A water-movement scan (diffusion)', adc: 'A water-movement map (ADC)' };
function seriesWhy(s, ch, role) {
  if (role === 'model') return 'The sharpest full-head T1 scan in this exam, so the 3D surfaces come from it.';
  const k = s.kind || (sequences().find((q) => q.uid === s.uid) || {}).kind;
  if ((role === 'slices' || role === 'veins') && KIND_PLAIN[k]) return `${KIND_PLAIN[k]}: lined up with the 3D model and shown on the slices${role === 'veins' ? ', and used to trace the veins' : ''}.`;
  if (s.excluded) return plain(s.excluded) || s.excluded;
  const neg = s.reasons.filter((r) => r[0] === '-').map(plain).filter(Boolean);
  return neg[0] || (role === 'slices' ? 'Lined up with the 3D model and shown on the slices.' : 'Usable, but the picked scan is sharper.');
}
// shortcut: a rebuild from another series makes a second model of the same person (case id = series); the old one stays in My body
const canUseFor3D = (s, ch) => !s.excluded && s.uid !== ch.uid && s.nFiles >= 60;
function buildSeriesUI() {
  const ch = manifest.series.chosen, all = manifest.series.all;
  $('#seriesInfo').innerHTML = `3D model built from <b>${ch.description || 'the T1 scan'}</b>` +
     `<br><span class="small">${all.length} series found in this exam · everything ran on this computer</span>`;
  const rows = all.map((s) => ({ s, role: seriesRole(s, ch) })).sort((a, b) => 'model neck veins slices none'.indexOf(a.role) - 'model neck veins slices none'.indexOf(b.role));
  const N = manifest.neck;
  $('#seriesPlain').innerHTML = rows.map(({ s, role }) => `<div class="series ${role}">
     <div class="sname"><span class="role ${ROLE[role][1]}">${ROLE[role][0]}</span> ${s.description || '(no name)'}</div>
     <div class="why">${seriesWhy(s, ch, role)}${canUseFor3D(s, ch) ? ` <button class="link use3d" data-uid="${s.uid}">Use for 3D instead</button>` : ''}</div></div>`).join('') +
     (N && N.series ? `<div class="series neck"><div class="sname"><span class="role ok">Neck</span> ${N.series.description || 'neck exam'}</div>
       <div class="why">${N.status === 'ok' ? 'Separate neck exam, lined up with the brain scan by matching vertebrae.' : 'Separate neck exam; it could not be lined up reliably, so it is not shown.'}</div></div>` : '');
  document.querySelectorAll('#seriesPlain .use3d').forEach((b) => (b.onclick = () => useFor3D(b.dataset.uid)));
  // Technical details: the original score arithmetic, unchanged
  $('#seriesTech').innerHTML = (manifest.brainExtraction ? `<div class="small">Brain found with <b>${manifest.brainExtraction.method === 'synthstrip' ? 'SynthStrip (AI, on this computer)' : 'quick estimate'}</b>` +
       (manifest.volumesMl ? ` · brain tissue ≈ ${Math.round(manifest.volumesMl.brain_tissue).toLocaleString()} mL (rough)` : '') + `</div>` : '') +
     (manifest.phase2 ? `<div class="small">Structures labelled with <b>FastSurfer (AI, on this computer)</b> in ${Math.round(manifest.phase2.timing.total)} s</div>` : '') +
     `<div class="small">Chosen: ${ch.nFiles} images · ${ch.voxel.filter(Boolean).join('×')} mm · ${ch.acquisition || '?'} · built in ${manifest.timing.total} s · picked by DICOM metadata score</div>` +
     all.map((s) => `<div class="series ${s.uid === ch.uid ? 'chosen' : ''}">
     <div class="sname">${s.uid === ch.uid ? '✓ ' : ''}${s.description || '(no description)'} <span class="muted">· ${s.nFiles} img · score ${s.score > -500 ? s.score : 'excluded'}</span></div>
     <div class="why">${s.excluded ? 'Excluded: ' + s.excluded : (s.uid === ch.uid ? s.reasons : [...s.reasons].sort((a, b) => (a[0] === '-') ? -1 : (b[0] === '-') ? 1 : 0)).slice(0, 4).join(', ')}</div></div>`).join('');
}
async function useFor3D(uid) {
  const s = manifest.series.all.find((x) => x.uid === uid);
  if (!confirm(`Build a new 3D model from “${s.description}”?\nThe current model stays in My body.`)) return;
  const r = await fetch('api/rebuild', { method: 'POST', body: JSON.stringify({ caseId: currentCase, seriesUid: uid }) }).then((x) => x.json());
  if (r.error) return toast(r.error, 'err');
  toast('Building a new 3D model…'); pollProcessing();
}
// models built before the mouth fix: offer the quick skin-only rebuild once (nothing else is redone)
function offerReskin() {
  const key = `bg:reskin:${currentCase}`;
  const force = params.get('reskin') === '1';   // screenshots
  if (!force && (params.has('shot') || !layers.skin || (manifest.skinVersion || 1) >= 2 || localStorage.getItem(key) || currentCase.startsWith('example'))) return;
  localStorage.setItem(key, 'offered');
  const t = toast('This model was made before the mouth-area fix. <button class="link" id="reskinGo">Smooth the mouth area now</button> (skin only, about 20 s)', '', force ? 600000 : 15000);
  t.querySelector('#reskinGo').onclick = async () => {
    t.remove(); const busy = toast('Updating the skin surface…', '', 60000);
    const r = await fetch('api/reskin', { method: 'POST', body: JSON.stringify({ caseId: currentCase }) }).then((x) => x.json()).catch(() => ({ error: 'BodyGlass is not running' }));
    busy.remove();
    if (r.error) return toast(r.error, 'err');
    await loadCase(currentCase, true); toast('Skin updated: the mouth area is closed');
  };
}
// transient banner for actions the user started; nothing stays on screen afterwards
function toast(msg, kind = '', ms = 4500) {
  const t = document.createElement('div'); t.className = `toast glass ${kind}`; t.innerHTML = msg;
  $('#toasts').append(t); setTimeout(() => { t.classList.add('out'); setTimeout(() => t.remove(), 400); }, ms);
  return t;
}


// ---------- other scans (FLAIR, T2, SWI, DWI, ADC): aligned to the T1 on this computer, shown on the slices
const SEQ_TEXT = {
  flair: ['FLAIR is a T2-type scan with the signal from fluid switched off. The fluid in and around the brain (CSF) turns dark, so extra water inside the tissue stands out as brighter patches. It is often used to look at the white matter. Some bright areas are normal, for example along the edges of the ventricles.', 'Blend: bright = colour'],
  t2: ['On a T2 scan water is bright: the fluid around the brain and in the ventricles looks white, and white matter is darker than grey matter, the reverse of T1.', 'Blend: bright = colour'],
  tse: ['A T2 turbo-spin-echo scan: often thin, very sharp slices that cover only part of the head (areas without data stay dim). Commonly used for a close look at small structures such as the hippocampus.', 'Blend: bright = colour'],
  swi: ['Susceptibility-weighted imaging (SWI) is very sensitive to iron and to the oxygen-poor blood in veins, which show up as dark lines and dots. Some deep grey-matter nuclei look darker because they normally contain iron.', 'Blend: dark = purple'],
  mip: ['A minimum-intensity projection (mIP) stacks several SWI slices into one thick slab, keeping the darkest value. Veins can then be followed as continuous dark lines. Because each picture is a slab, it is less exact about depth than a single slice.', 'Blend: dark = purple'],
  swi_mag: ['The plain magnitude picture of the SWI scan (a gradient-echo image) before the vein-enhancing processing. It helps the computer line up the SWI with the T1.', ''],
  dwi: ['Diffusion-weighted imaging (here b = 1000) measures how freely water molecules move. Where movement is restricted the signal is bright. These scans are fast and low-resolution, and can look slightly stretched near the air-filled sinuses.', 'Blend: bright = colour'],
  adc: ['The ADC map is calculated from the diffusion scans and gives a number for how freely water moves: fluid is bright (moves freely), tissue is darker. Hover a slice picture to read the value. Doctors read DWI and ADC together.', 'Blend: blue low → yellow / red high'],
};
const sequences = () => (manifest && manifest.sequences) || [];
const seqPending = () => Object.keys((manifest && manifest.sequencesFound) || {}).filter((k) => !sequences().some((q) => q.kind === k) && k !== 'swi_phase');
let seqTex = null, zoneVol = null, zoneTex = null;
async function ensureZone() {
  if (zoneVol || !manifest.brainZone || noVol) return;
  zoneVol = padZ(new Uint8Array(await (await fetch(`${caseBase()}${manifest.brainZone}`)).arrayBuffer()));
  zoneTex = new THREE.Data3DTexture(zoneVol, dims[0], dims[1], dims[2]);
  zoneTex.format = THREE.RedFormat; zoneTex.type = THREE.UnsignedByteType; zoneTex.minFilter = zoneTex.magFilter = THREE.NearestFilter; zoneTex.unpackAlignment = 1; zoneTex.needsUpdate = true;
  sliceCache.clear();
}
async function ensureSeq(kind) {
  const meta = sequences().find((q) => q.kind === kind); if (!meta) return null;
  const S = seqData[kind] || (seqData[kind] = { meta });
  S.meta = meta;
  if (!S.vol) {
    S.loading = S.loading || fetch(`${caseBase()}${meta.file}`).then((r) => r.arrayBuffer()).then((b) => { S.vol = padZ(new Uint8Array(b)); });
    $('#scanNote').textContent = `Loading ${meta.name}…`;
    await S.loading; $('#scanNote').textContent = '';
  }
  return S;
}
async function setSeq(kind, opts = {}) {
  if (kind && kind !== 'none') { if (!(await ensureSeq(kind))) kind = null; else await ensureZone(); } else kind = null;
  seqKind = kind;
  if (seqTex) { seqTex.dispose(); seqTex = null; }
  if (kind) {
    const S = seqData[kind];
    seqTex = new THREE.Data3DTexture(S.vol, dims[0], dims[1], dims[2]);
    seqTex.format = THREE.RedFormat; seqTex.type = THREE.UnsignedByteType;
    seqTex.minFilter = seqTex.magFilter = THREE.LinearFilter; seqTex.unpackAlignment = 1; seqTex.needsUpdate = true;
  }
  applySeqUniforms(); buildScansUI();
  if (kind && !opts.quiet) showSeqCard(kind);
  if (!kind && $('#info').dataset.seq) hideInfo();
  if (cmp.open && !kind) closeCompare();
  drawInsets();
}
function applySeqUniforms() {
  for (const P of Object.values(planes)) {
    const u = P.mat.uniforms;
    u.uSeqOn.value = seqKind ? 1 : 0; u.uSeqMode.value = seqMode === 'blend' ? 1 : 0; u.uSeqOp.value = seqOpacity;
    if (seqTex) { u.uSeq.value = seqTex; u.uCmap.value = cmapTex(seqData[seqKind].meta.cmap); }
    if (zoneTex) { u.uZone.value = zoneTex; u.uZoneOn.value = 1; } else u.uZoneOn.value = 0;
  }
}
function alignText(m) {
  const a = m.align || {};
  const how = a.rejected && /centre of mass/.test(a.method) ? 'Only roughly lined up (centred on the T1): automatic fine-tuning failed, so it may be off by several millimetres.'
    : a.rejected ? 'Lined up using the scanner\u2019s own coordinates (automatic fine-tuning did not improve it).'
    : /same transform/.test(a.method || '') ? `Lined up together with ${a.method.replace('same transform as ', '')} (same acquisition).`
    : `Lined up with the T1 automatically on this computer (rigid fit: moved ${a.shiftMm} mm, turned ${a.rotationDeg}°).`;
  return `${how} From series “${m.description}”, ${m.voxelMm.map((x) => +x.toFixed(1)).join('×')} mm voxels${m.coveragePct < 50 ? ', partial coverage' : ''}.`;
}
function showSeqCard(kind) {
  const m = seqData[kind].meta, [txt, legend] = SEQ_TEXT[kind] || ['', ''];
  $('#info').dataset.seq = kind;
  $('#infoSwatch').style.background = { hot: 'linear-gradient(90deg,#780a00,#ff6e00,#fff5a0)', cool: 'linear-gradient(90deg,#002878,#1ea0ff,#bef5ff)', ink: 'linear-gradient(90deg,#460078,#7828c8,#ddd)', adc: 'linear-gradient(90deg,#28106e,#28be78,#f0dc3c,#ff5a28)' }[m.cmap] || '#bbb';
  $('#infoTitle').textContent = m.name; $('#infoBadge').textContent = 'other scan'; $('#infoBadge').classList.remove('hidden');
  $('#infoText').textContent = txt;
  $('#infoParcel').innerHTML = seqMode === 'blend' && legend ? `<b>Colours:</b> ${legend.replace('Blend: ', '')}.` : '';
  $('#infoExtra').textContent = alignText(m) + ' Educational only, not for diagnosis.';
  $('#infoIsolate').classList.add('hidden');
  $('#info').classList.remove('hidden');
}
function buildScansUI() {
  const sec = $('#scansSection'), list = sequences(), pend = seqPending();
  sec.classList.toggle('hidden', !list.length && !pend.length);
  if (!list.length && !pend.length) return;
  const NAMES = { flair: 'FLAIR', t2: 'T2', tse: 'T2 TSE', swi: 'SWI', mip: 'SWI mIP', swi_mag: 'SWI mag', dwi: 'DWI', adc: 'ADC' };
  const chips = [['none', 'T1'], ...['flair', 't2', 'tse', 'swi', 'mip', 'swi_mag', 'dwi', 'adc'].filter((k) => list.some((q) => q.kind === k) || pend.includes(k)).map((k) => [k, NAMES[k]])];
  $('#scanChips').innerHTML = chips.map(([k, n]) => { const ready = k === 'none' || list.some((q) => q.kind === k);
    return `<button class="chip ${(seqKind || 'none') === k ? 'on' : ''}" data-k="${k}" ${ready ? '' : 'disabled title="Still lining up with the T1…"'}>${n}</button>`; }).join('');
  $('#scanChips').querySelectorAll('button').forEach((b) => (b.onclick = () => setSeq(b.dataset.k)));
  const showOpts = !!seqKind;
  $('#scanOpts').classList.toggle('hidden', !showOpts);
  document.querySelectorAll('#seqMode button').forEach((b) => b.classList.toggle('on', b.dataset.m === seqMode));
  $('#seqOpRow').classList.toggle('hidden', seqMode !== 'blend');
  if (!$('#scanNote').textContent || /Lining up|ready/.test($('#scanNote').textContent))
    $('#scanNote').textContent = pend.length ? `Lining up ${pend.length} more scan${pend.length > 1 ? 's' : ''} with the T1… ${seqProgress}` : '';
}
let seqProgress = '';
document.querySelectorAll('#seqMode button').forEach((b) => (b.onclick = () => { seqMode = b.dataset.m; applySeqUniforms(); buildScansUI(); if (seqKind) showSeqCard(seqKind); drawInsets(); }));
$('#seqOpacity').oninput = (e) => { seqOpacity = +e.target.value; $('#seqOpacityVal').textContent = Math.round(seqOpacity * 100) + '%'; applySeqUniforms(); drawInsets(); };
$('#btnCompare').onclick = () => openCompare();
function cycleSeq(dir = 1) {
  const ks = ['none', ...sequences().map((q) => q.kind)]; const i = ks.indexOf(seqKind || 'none');
  setSeq(ks[(i + dir + ks.length) % ks.length]);
}
async function pollSequences() {
  if (!currentCase || !manifest) return;
  const st = (await (await fetch('api/state')).json()).sequences || {};
  if (!seqPending().length && (manifest.veins || !sequences().some((q) => q.kind === 'swi' || q.kind === 'mip'))) return;
  const active = st.status === 'running' && st.caseId === currentCase;
  seqProgress = active ? `${Math.round(st.pct)}%` : '';
  const m = await (await fetch(`${caseBase()}manifest.json`)).json();
  if ((m.sequences || []).length !== sequences().length) { manifest.sequences = m.sequences; }
  if (m.veins && !manifest.veins) { manifest.veins = m.veins; addVeinsLive(); }
  buildScansUI();
  if (active || st.status === 'idle') setTimeout(pollSequences, 2500);
  else if (seqPending().length || (!manifest.veins && st.status === 'running')) { $('#scanNote').textContent = st.status === 'error' ? 'Some scans could not be lined up.' : ''; }
}

// ---------- compare: T1 next to (or swiped against) the other scan, same slice, scrubbing stays in sync
function openCompare(plane) {
  if (!seqKind) { const first = sequences()[0]; if (!first) return; setSeq(first.kind, { quiet: true }).then(() => openCompare(plane)); return; }
  cmp.open = true; if (plane) cmp.plane = plane;
  $('#compare').classList.remove('hidden'); buildCompareUI(); drawCompare();
}
function closeCompare() { cmp.open = false; $('#compare').classList.add('hidden'); }
function buildCompareUI() {
  document.querySelectorAll('#cmpPlane button').forEach((b) => b.classList.toggle('on', b.dataset.p === cmp.plane));
  document.querySelectorAll('#cmpMode button').forEach((b) => b.classList.toggle('on', b.dataset.m === cmp.mode));
  $('#cmpSeq').innerHTML = sequences().map((q) => `<button class="chip ${q.kind === seqKind ? 'on' : ''}" data-k="${q.kind}">${q.name}</button>`).join('');
  $('#cmpSeq').querySelectorAll('button').forEach((b) => (b.onclick = () => setSeq(b.dataset.k, { quiet: true }).then(() => { buildCompareUI(); drawCompare(); })));
  $('#compare').classList.toggle('swipe', cmp.mode === 'swipe');
  const m = seqData[seqKind].meta;
  $('#cmpRightLbl').textContent = m.name; $('#cmpRightLblSwipe').textContent = m.name + ' →'; $('#cmpText').textContent = (SEQ_TEXT[seqKind] || [''])[0];
  $('#cmpAlign').textContent = alignText(m);
}
function drawCompare() {
  if (!cmp.open || !seqKind || !vol) return;
  const name = cmp.plane, V = viewGeom(name), s = Math.max(V.w, V.h), ox = (s - V.w) >> 1, oy = (s - V.h) >> 1;
  const mode = seqMode === 'blend' ? 'blend' : 'replace';
  const draw = (cv, img) => { if (cv.width !== s) { cv.width = s; cv.height = s; } const c = cv.getContext('2d'); c.fillStyle = '#000'; c.fillRect(0, 0, s, s); c.putImageData(img, ox, oy); cv._geom = { V, ox, oy, s }; return c; };
  const L = $('#cmpL'), R = $('#cmpR');
  if (cmp.mode === 'side') { draw(L, greyImage(name, V)); draw(R, seqImage(name, V, seqKind, mode)); }
  else {
    const c = draw(L, greyImage(name, V));
    const tmp = cmp._tmp || (cmp._tmp = document.createElement('canvas')); tmp.width = V.w; tmp.height = V.h;
    tmp.getContext('2d').putImageData(seqImage(name, V, seqKind, mode), 0, 0);
    const x = Math.round(s * cmp.split);
    c.save(); c.beginPath(); c.rect(x, 0, s - x, s); c.clip(); c.drawImage(tmp, ox, oy); c.restore();
    c.fillStyle = '#7cc4ff'; c.fillRect(x - 1, 0, 2, s);
    c.beginPath(); c.arc(x, s / 2, s / 40, 0, 7); c.fill();
  }
  $('#cmpCap').textContent = `${name[0].toUpperCase() + name.slice(1)} slice ${V.k + 1} / ${V.n}`;
  const lbl = (c, txt) => { const g = c.getContext('2d'); g.fillStyle = '#cfd8e6'; g.font = `${Math.round(s / 26)}px sans-serif`; g.fillText(txt, s * 0.03, s * 0.97); };
  lbl(L, name === 'sagittal' ? 'P' : 'R');
}
function setupCompareInput() {
  for (const cv of [$('#cmpL'), $('#cmpR')]) {
    let drag = null, wheelAcc = 0;
    cv.addEventListener('pointerdown', (e) => { drag = { x0: e.clientX, y0: e.clientY, acc: 0, how: null }; cv.setPointerCapture(e.pointerId); e.preventDefault(); });
    cv.addEventListener('pointermove', (e) => {
      if (!drag) { showTip(e, seqValueAt(cv, e)); return; }
      const dx = e.clientX - drag.x0, dy = drag.y0 - e.clientY;
      if (!drag.how && Math.hypot(dx, dy) > 4) drag.how = cmp.mode === 'swipe' && Math.abs(dx) > Math.abs(dy) ? 'swipe' : 'scrub';
      if (drag.how === 'swipe') { const r = cv.getBoundingClientRect(); cmp.split = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)); drawCompare(); }
      else if (drag.how === 'scrub') { const want = Math.trunc(dy / SCRUB_PX); if (want !== drag.acc) { stepSlice(cmp.plane, want - drag.acc); drag.acc = want; } }
    });
    cv.addEventListener('pointerup', (e) => {
      const d = drag; drag = null;
      if (d && !d.how && cmp.mode === 'swipe') { const r = cv.getBoundingClientRect(); cmp.split = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)); drawCompare(); }
    });
    cv.addEventListener('pointerleave', () => (tip.style.display = 'none'));
    cv.addEventListener('wheel', (e) => { e.preventDefault(); wheelAcc += e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1);
      const st = Math.trunc(wheelAcc / WHEEL_PX); if (st) { wheelAcc -= st * WHEEL_PX; stepSlice(cmp.plane, -st); } }, { passive: false });
  }
  document.querySelectorAll('#cmpPlane button').forEach((b) => (b.onclick = () => { cmp.plane = b.dataset.p; buildCompareUI(); drawCompare(); }));
  document.querySelectorAll('#cmpMode button').forEach((b) => (b.onclick = () => { cmp.mode = b.dataset.m; buildCompareUI(); drawCompare(); }));
  $('#cmpClose').onclick = closeCompare;
}
setupCompareInput();


// ---------- vein tree: traced from the aligned SWI scan, off by default
const VEIN_TEXT = 'Veins traced automatically from the SWI scan: oxygen-poor blood in veins is dark on SWI, and a "vesselness" filter follows those dark tubes. Only the part of the head the SWI covered can show veins, and small or faint veins are missed while some dark non-vein spots can be included. A rough educational picture, not a medical vessel study.';
async function loadVeins(base) {
  const V = manifest.veins; if (!V || !V.file || layers.veins) return;
  const g = await loader.loadAsync(base + V.file);
  const mesh = firstMesh(g);
  mesh.geometry.computeVertexNormals();
  const material = fresnelPatch(new THREE.MeshPhysicalMaterial({ color: new THREE.Color('#4d6bff'), roughness: 0.35, clearcoat: 0.6, clearcoatRoughness: 0.25,
    emissive: new THREE.Color('#0a1550'), side: THREE.DoubleSide, clipShadows: true }));
  material.userData.fluid = true;   // keep its own emissive glow (no selection pulse)
  const m = new THREE.Mesh(mesh.geometry, material); m.userData.layerId = 'veins'; m.renderOrder = 4;
  centered.add(m);
  layers.veins = { def: { id: 'veins', name: 'Veins (from SWI)', color: '#4d6bff', description: VEIN_TEXT, veins: true }, items: [{ mesh: m, material }],
    visible: params.get('veins') === '1', opacity: 1 };
  applyOpacity(layers.veins);
}
async function addVeinsLive() {   // the vein tree finished after the case was opened
  if (!manifest.veins || !manifest.veins.file || layers.veins) return;
  await loadVeins(caseBase()); buildLayerUI(); updateClipping();
  toast('✓ Vein tree ready (Layers → Veins)');
}
function toggleVeins(on) { const L = layers.veins; if (!L) return; L.visible = on === undefined ? !L.visible : on; if (L.visible && L.opacity < 0.05) L.opacity = 1; applyOpacity(L); }

// ---------- measuring: two clicks, in 3D (on any surface or slice plane) or on the slice pictures
const meas = { on: false, pts: [], group: new THREE.Group() };
meas.group.renderOrder = 20; centered.add(meas.group);
const measLbl = document.createElement('div'); measLbl.id = 'measLbl'; document.body.appendChild(measLbl);
function setMeasure(on) {
  meas.on = on === undefined ? !meas.on : on;
  $('#mMeasure').classList.toggle('on', meas.on); document.body.classList.toggle('measuring', meas.on);
  if (!meas.on) { meas.pts = []; redrawMeasure(); }
  $('#measBar').classList.toggle('hidden', !meas.on);
  updateMeasBar();
}
function updateMeasBar() {
  const n = meas.pts.length;
  $('#measText').innerHTML = n === 0 ? 'Click a first point, on the 3D model or a slice picture'
    : n === 1 ? 'Now click a second point'
    : `Distance <b>${meas.pts[0].distanceTo(meas.pts[1]).toFixed(1)} mm</b> <span class="muted">· click again to start a new measurement</span>`;
}
function addMeasPoint(ras) {
  if (meas.pts.length >= 2) meas.pts = [];
  meas.pts.push(ras.clone()); redrawMeasure(); updateMeasBar();
}
const measMat = new THREE.MeshBasicMaterial({ color: 0xffe066, depthTest: false, transparent: true });
const measLineMat = new THREE.LineBasicMaterial({ color: 0xffe066, depthTest: false, transparent: true });
function redrawMeasure() {
  while (meas.group.children.length) { const c = meas.group.children.pop(); c.geometry && c.geometry.dispose(); }
  for (const p of meas.pts) { const d = new THREE.Mesh(new THREE.SphereGeometry(1.6, 16, 12), measMat); d.position.copy(p); d.renderOrder = 20; meas.group.add(d); }
  if (meas.pts.length === 2) { const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints(meas.pts), measLineMat); l.renderOrder = 20; meas.group.add(l); }
  drawInsets();
}
function updateMeasLabel() {
  if (meas.pts.length !== 2) { measLbl.style.display = 'none'; return; }
  const mid = centered.localToWorld(meas.pts[0].clone().add(meas.pts[1]).multiplyScalar(0.5)).project(camera);
  if (mid.z > 1) { measLbl.style.display = 'none'; return; }
  measLbl.textContent = meas.pts[0].distanceTo(meas.pts[1]).toFixed(1) + ' mm';
  measLbl.style.display = 'block';
  measLbl.style.transform = `translate(${Math.round((mid.x + 1) / 2 * innerWidth + 10)}px, ${Math.round((1 - mid.y) / 2 * innerHeight - 12)}px)`;
}
// slice picture pixel <-> RAS millimetres (radiological display: patient right on screen left)
function insetToRas(name, px, py) {
  const [nx, ny, nz] = dims, k = sliceIdx(name);
  const ijk = name === 'axial' ? [nx - 1 - px, ny - 1 - py, k] : name === 'coronal' ? [nx - 1 - px, k, nz - 1 - py] : [k, px, nz - 1 - py];
  return new THREE.Vector3(...ijk.map((v, a) => origin[a] + v * sp));
}
function rasToInset(name, p) {
  const [nx, ny, nz] = dims, [i, j, k] = [0, 1, 2].map((a) => (p.getComponent(a) - origin[a]) / sp);
  if (name === 'axial') return { c: nx - 1 - i, r: ny - 1 - j, d: k - sliceIdx(name) };
  if (name === 'coronal') return { c: nx - 1 - i, r: nz - 1 - k, d: j - sliceIdx(name) };
  return { c: j, r: nz - 1 - k, d: i - sliceIdx(name) };
}
function drawMeasInset(name, ctx, ox, oy, s) {
  if (!meas.pts.length) return;
  const q = meas.pts.map((p) => rasToInset(name, p)), near = q.map((x) => Math.abs(x.d) <= 1.5);
  ctx.save(); ctx.strokeStyle = '#ffe066'; ctx.fillStyle = '#ffe066'; ctx.lineWidth = Math.max(1, s / 160);
  if (q.length === 2) { ctx.globalAlpha = near[0] && near[1] ? 1 : 0.45; ctx.setLineDash(near[0] && near[1] ? [] : [s / 60, s / 60]);
    ctx.beginPath(); ctx.moveTo(ox + q[0].c + 0.5, oy + q[0].r + 0.5); ctx.lineTo(ox + q[1].c + 0.5, oy + q[1].r + 0.5); ctx.stroke(); }
  ctx.setLineDash([]);
  q.forEach((x, n) => { ctx.globalAlpha = near[n] ? 1 : 0.45; ctx.beginPath(); ctx.arc(ox + x.c + 0.5, oy + x.r + 0.5, s / 90, 0, 7); ctx.fill(); });
  ctx.restore();
}
$('#measClear').onclick = () => { meas.pts = []; redrawMeasure(); updateMeasBar(); };
$('#measDone').onclick = () => setMeasure(false);


// ---------- small helpers shared by the tour, export and example views
const vis = (id, o) => { const L = layers[id]; if (!L) return; L.opacity = o; L.visible = o > 0.01; applyOpacity(L); };
const firstStruct = (...keys) => { for (const k of keys) { const st = (manifest.structures || []).find((x) => x.key === k); if (st) return st.id; } return null; };
let serverMac = false;

// ---------- guided tour: smooth camera arcs + short plain-language cards (optional offline Mac voice)
const tour = { on: false, i: 0, steps: [], speak: false };
function tourSteps() {
  const S = hasStructs(), st = [];
  const cortex = (o) => (S ? vis('g:cortex', o) : vis('brain', o));
  st.push({ t: 'Welcome', x: 'This short tour shows the main parts of the head and brain in your scan. Use Next and Back, or the arrow keys. You can rotate and zoom at any time.',
    go() { setPeel(0); vis('skull', 0); setView('backleft', false, { dur: 1800 }); } });
  if (layers.skull) st.push({ t: 'Skin and skull', x: 'The outer layers are made see-through so you can look inside. The skull shape is a rough estimate: bone gives little signal on MRI.',
    go() { setPeel(2); setView('backright', false, { dur: 1800 }); } });
  st.push({ t: 'The brain surface', x: S ? 'The folded outer layer is the cortex (grey matter). The colours mark the four lobes on each side: frontal, parietal, temporal and occipital.' : 'The folded outer layer is the cortex, the grey matter where most thinking happens.',
    go() { vis('skin', 0); vis('skull', 0); cortex(1); if (S) setRegionColors(true); setView('left', false, { dur: 2000 }); } });
  const fr = S && firstStruct('frontal'); if (fr) st.push({ t: 'Frontal lobe', x: 'Behind the forehead. It helps with planning, decisions, movement and speech.', go() { selectStruct(fr); setView([-0.7, 0.35, -0.75], false, { dur: 1800 }); } });
  const hip = S && firstStruct('hippocampus'); if (hip) st.push({ t: 'Deep inside: the hippocampus', x: 'Tucked inside each temporal lobe, the seahorse-shaped hippocampus helps form new memories. The cortex is made see-through here.',
    go() { cortex(0.08); vis('g:white_matter', 0.1); selectStruct(hip); setView([-1, 0.25, 0.1], false, { dur: 2000, radius: 0.75 }); } });
  const th = S && firstStruct('thalamus'); if (th) st.push({ t: 'The thalamus and its neighbours', x: 'Near the centre, the thalamus relays signals between the body and the cortex. Around it sit the caudate, putamen and other deep grey-matter nuclei.',
    go() { cortex(0.06); vis('g:white_matter', 0.08); selectStruct(th); setView('3qhigh', false, { dur: 2000, radius: 0.75 }); } });
  st.push({ t: 'Fluid spaces', x: 'The ventricles are connected chambers filled with clear fluid (CSF) that cushions the brain. Their size differs a lot between healthy people.',
    go() { cortex(0.06); vis('g:white_matter', 0.08); vis('white_matter', 0.08); const v = S && firstStruct('lateral_ventricle'); if (v) selectStruct(v); else select('ventricles'); setView('top', false, { dur: 1800, radius: 0.8 }); } });
  const cb = S && firstStruct('cerebellum'); if (cb) st.push({ t: 'Cerebellum and brainstem', x: 'At the back and bottom, the cerebellum helps with balance and smooth movement. The brainstem connects the brain to the spinal cord.',
    go() { cortex(0.25); vis('g:white_matter', 0.2); selectStruct(cb); setView('backright', false, { dur: 1800 }); } });
  st.push({ t: 'Slices', x: 'An MRI is a stack of thin pictures. The plane shows one of them inside the 3D model; the small pictures bottom right show the three directions. Drag up and down on them to move through the slices.',
    go() { hideInfo(); cortex(1); vis('g:white_matter', 1); enablePlane('axial', true); $('#cutaway').checked = true; updateClipping(); setView('3qhigh', false, { dur: 1800 }); } });
  if (sequences().length) st.push({ t: 'Other scan types', x: 'Your folder has other MRI types too. ' + ((SEQ_TEXT[sequences()[0].kind] || [''])[0]) + ' Pick them under Scans.',
    go() { setSeq(sequences()[0].kind, { quiet: true }); setView('3qhigh', false, { dur: 1200 }); } });
  if (layers.veins) st.push({ t: 'Veins', x: 'These veins were traced from the SWI scan, where oxygen-poor blood looks dark. Only the part of the head the SWI covered is shown.',
    go() { setSeq(null, { quiet: true }); enablePlane('axial', false); $('#cutaway').checked = false; updateClipping(); cortex(0.1); vis('g:white_matter', 0.1); toggleVeins(true); setView('backleft', false, { dur: 2000 }); } });
  st.push({ t: 'That’s the tour', x: 'Explore on your own: click anything to read about it, double-click to orbit around it, search under Explore, or use Tools to measure, export a 3D print or compare with an example brain. Press ⌘K to find anything, ? for all shortcuts. Educational only, not for diagnosis.',
    go() { if (layers.veins) toggleVeins(false); setSeq(null, { quiet: true }); enablePlane('axial', false); $('#cutaway').checked = false; updateClipping(); hideInfo(); setPeel(1); setView('3q', false, { dur: 2000 }); } });
  return st;
}
function startTour() { if (!manifest) return; closeAllModals(); tour.steps = tourSteps(); tour.on = true; tour.i = 0; controls.autoRotate = false; $('#tour').classList.remove('hidden'); showTourStep(); }
function showTourStep() {
  const T = tour.steps[tour.i]; T.go();
  $('#tourStep').textContent = `${tour.i + 1} / ${tour.steps.length}`; $('#tourTitle').textContent = T.t; $('#tourText').textContent = T.x;
  $('#tourBack').disabled = tour.i === 0; $('#tourNext').textContent = tour.i === tour.steps.length - 1 ? 'Finish' : 'Next →';
  $('#tourDots').innerHTML = tour.steps.map((_, n) => `<i class="${n === tour.i ? 'on' : ''}"></i>`).join('');
  if (tour.speak) fetch('api/say', { method: 'POST', body: JSON.stringify({ text: T.t + '. ' + T.x }) });
}
function tourGo(d) { const n = tour.i + d; if (n < 0) return; if (n >= tour.steps.length) return endTour(); tour.i = n; showTourStep(); }
function endTour() { if (!tour.on) return; tour.on = false; $('#tour').classList.add('hidden'); if (tour.speak) fetch('api/say', { method: 'POST', body: '{"text":""}' }); }
$('#tourNext').onclick = () => tourGo(1); $('#tourBack').onclick = () => tourGo(-1); $('#tourExit').onclick = endTour;
$('#tourVoice').onclick = () => { tour.speak = !tour.speak; $('#tourVoice').classList.toggle('on', tour.speak); if (tour.speak) showTourStep(); else fetch('api/say', { method: 'POST', body: '{"text":""}' }); };

// ---------- export for 3D printing (STL / 3MF, life size in mm, always watertight)
function exportChoices() {
  const out = [];
  for (const L of Object.values(layers)) {
    if (L.hidden) continue;
    out.push({ id: L.def.id, name: L.def.name, on: L.visible && L.opacity > 0.5 && L.def.id !== 'skin' && L.def.id !== 'skull' });
  }
  return out;
}
function openExport() {
  if (!manifest) return; closeAllModals();
  const sel = selected && selected.items[0] && selected.items[0].st ? selected.items[0].st : null;
  const ch = exportChoices();
  $('#expList').innerHTML = (sel ? `<label class="check"><input type="radio" name="expwhat" value="sel" checked> Only the selected structure: <b>${sel.name}</b></label><label class="check"><input type="radio" name="expwhat" value="layers"> These layers:</label>` : '') +
    `<div id="expLayers" class="${sel ? 'indent' : ''}">` + ch.map((c) => `<label class="check"><input type="checkbox" value="${c.id}" ${c.on ? 'checked' : ''}> ${c.name}</label>`).join('') + '</div>';
  $('#expMsg').textContent = ''; $('#expGo').disabled = false; $('#export').classList.remove('hidden');
}
async function doExport() {
  const what = document.querySelector('input[name=expwhat]:checked');
  const sel = selected && selected.items[0] && selected.items[0].st;
  const ids = what && what.value === 'sel' ? [sel.id] : [...document.querySelectorAll('#expLayers input:checked')].map((x) => x.value);
  if (!ids.length) { $('#expMsg').textContent = 'Tick at least one layer.'; return; }
  const fmt = document.querySelector('input[name=expfmt]:checked').value;
  $('#expGo').disabled = true; $('#expMsg').textContent = 'Building a solid, printable model on this computer… (5–30 s)';
  try {
    const r = await fetch('api/export', { method: 'POST', body: JSON.stringify({ caseId: currentCase, ids, format: fmt }) });
    if (!r.ok) throw new Error((await r.json()).error || r.statusText);
    const info = JSON.parse(r.headers.get('X-Export-Info') || '{}'), blob = await r.blob();
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
    a.download = `brain-model-${ids.length === 1 ? ids[0].replace(/[^a-z0-9_]/gi, '') : 'layers'}.${fmt}`; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 4000);
    const [x, y, z] = info.sizeMm || [];
    $('#expMsg').innerHTML = `Saved <b>${a.download}</b> (${(blob.size / 1e6).toFixed(1)} MB) to your Downloads folder. Life size: <b>${x} × ${y} × ${z} mm</b>, ` +
      `${info.bodies} solid part${info.bodies > 1 ? 's' : ''}, ${info.watertight ? 'watertight' : 'check in your slicer'}.`;
  } catch (e) { $('#expMsg').textContent = 'Export failed: ' + e.message; }
  $('#expGo').disabled = false;
}
$('#expGo').onclick = doExport; $('#expClose').onclick = () => $('#export').classList.add('hidden');

// ---------- compare with a public example brain (built once at install; nothing downloaded at runtime)
const ex = { info: null, mf: null, on: false, mode: 'side', group: null, items: [], loaded: false };
async function loadExample() {
  if (ex.loaded) return true;
  ex.info = await (await fetch('api/example')).json();
  if (!ex.info.available) return false;
  const base = `example/cases/${ex.info.caseId}/`;
  ex.mf = await (await fetch(base + 'manifest.json')).json();
  const outer = new THREE.Group(); outer.rotation.x = -Math.PI / 2; scene.add(outer);
  const inner = new THREE.Group(); inner.position.set(...ex.mf.brainCenter.map((v) => -v)); outer.add(inner);
  ex.group = outer;
  await Promise.all((ex.mf.structures || []).map(async (st) => {
    const g = await loader.loadAsync(base + st.file); const mesh = firstMesh(g);
    mesh.geometry.computeVertexNormals(); toShading(mesh.geometry);
    const matName = st.key === 'cerebellum' ? 'brain' : GROUP_MAT[st.group];
    const solid = makeMaterial({ material: matName, color: st.color, ownColor: true }, !!mesh.geometry.attributes.color);
    const ghost = new THREE.MeshBasicMaterial({ color: 0xffb35c, transparent: true, opacity: 0.16, depthWrite: false, wireframe: false, side: THREE.DoubleSide });
    const m = new THREE.Mesh(mesh.geometry, solid); m.userData.exStruct = st; inner.add(m);
    ex.items.push({ mesh: m, st, solid, ghost });
  }));
  ex.loaded = true; return true;
}
async function openExample(mode) {
  if (!manifest || !hasStructs()) { toast('The comparison needs the detailed structures (wait for them to finish).'); return; }
  closeAllModals(); endTour();
  const busy = toast('Loading the example brain…', '', 60000);
  const okEx = await loadExample(); busy.remove();
  if (!okEx) { toast('The example brain is not installed. Run ./install.sh again to add it.', 'err'); return; }
  ex.on = true; ex.mode = mode || ex.mode; ex.group.visible = true;
  vis('skin', 0); vis('skull', 0);
  $('#exBar').classList.remove('hidden'); applyExampleMode(true);
}
function applyExampleMode(moveCam) {
  document.querySelectorAll('#exMode button').forEach((b) => b.classList.toggle('on', b.dataset.m === ex.mode));
  const side = ex.mode === 'side';
  // side by side: example to the viewer's right when looking from behind (+x world), centred together
  const bb = new THREE.Box3(); for (const S of Object.values(structById)) bb.expandByObject(S.item.mesh);
  const w = bb.isEmpty() ? 150 : bb.getSize(new THREE.Vector3()).x;
  ex.group.position.set(side ? w + 25 : 0, 0, 0);
  for (const it of ex.items) it.mesh.material = side ? it.solid : it.ghost;
  ex.items.forEach((it) => (it.mesh.renderOrder = side ? 0 : 6));
  if (moveCam) setView([0.15, 0.45, 1], false, { dur: 1600, radius: side ? 1.75 : 1.1, target: new THREE.Vector3(side ? (w + 25) / 2 : 0, viewTarget.y, 0) });
  $('#exTagYou').style.display = $('#exTagEx').style.display = side ? '' : 'none';
  $('#exNote').textContent = side ? 'Same scale (millimetres). Layers and peel apply to both.' : 'Orange ghost = example brain, centred on your brain (no size matching).';
}
function updateExample() {
  if (!ex.on) return;
  for (const it of ex.items) {   // mirror the user's layer visibility / opacity
    const L = layers['g:' + it.st.group], S = Object.values(structById).find((q) => q.st.key === it.st.key && q.st.hemi === it.st.hemi);
    const o = L ? itemOpacity(L, S ? S.item : { st: null }) : 1, show = L && L.visible && o > 0.01 && (!S || S.st.visible);
    it.mesh.visible = !!show;
    if (ex.mode === 'side') { it.solid.opacity = o; it.solid.transparent = o < 0.99; it.solid.depthWrite = o >= 0.99; it.solid.userData.uFresnel.value = o < 0.99 ? 1 : 0; it.solid.color.copy(regionColors || !(it.st.group === 'cortex' || it.st.key === 'cerebellum') ? new THREE.Color(it.st.color) : BRAIN_PINK); }
  }
  const tag = (el, obj, dx) => { const b = new THREE.Box3().setFromObject(obj), c = b.getCenter(new THREE.Vector3()); c.y = b.max.y + 8; c.x += dx; const p = c.project(camera);
    el.style.transform = `translate(${Math.round((p.x + 1) / 2 * innerWidth)}px, ${Math.round((1 - p.y) / 2 * innerHeight)}px)`; };
  if (ex.mode === 'side') {   // tags above the visible brains only (hidden skin would push them up)
    const you = new THREE.Group(); const box = new THREE.Box3();
    for (const S of Object.values(structById)) if (S.item.mesh.visible) box.expandByObject(S.item.mesh);
    if (!box.isEmpty()) { const c = box.getCenter(new THREE.Vector3()); c.y = box.max.y + 8; const p = c.project(camera);
      $('#exTagYou').style.transform = `translate(${Math.round((p.x + 1) / 2 * innerWidth)}px, ${Math.round((1 - p.y) / 2 * innerHeight)}px)`; }
    tag($('#exTagEx'), ex.group, 0);
  }
}
function closeExample() {
  if (!ex.on) return; ex.on = false; ex.group.visible = false; $('#exBar').classList.add('hidden'); $('#exTable').classList.add('hidden');
  setPeel(+$('#peel').value); setView('3q', false, { dur: 1400 });
}
function volumeTable() {
  const pick = (mf) => { const o = {}; for (const st of mf.structures || []) { const k = st.key; o[k] = o[k] || { name: st.hemi ? st.name.replace(/^(Left|Right) /, '') : st.name, L: 0, R: 0, T: 0, group: st.group };
    o[k].T += st.volume_ml; if (st.hemi === 'L') o[k].L += st.volume_ml; if (st.hemi === 'R') o[k].R += st.volume_ml; } return o; };
  const A = pick(manifest), B = pick(ex.mf);
  const tot = (o) => Object.values(o).filter(inBrain).reduce((a, r) => a + r.T, 0);
  const tA = tot(A), tB = tot(B);
  const rows = Object.keys(A).filter((k) => B[k]).map((k) => `<tr><td>${A[k].name[0].toUpperCase() + A[k].name.slice(1)}</td><td>${fmtMl(A[k].T)}</td><td>${fmtMl(B[k].T)}</td>` +
    `<td class="muted">${(100 * A[k].T / tA).toFixed(A[k].T / tA < 0.01 ? 2 : 1)} %</td><td class="muted">${(100 * B[k].T / tB).toFixed(B[k].T / tB < 0.01 ? 2 : 1)} %</td></tr>`).join('');
  $('#exRows').innerHTML = rows + `<tr class="sum"><td>Labelled brain (without ventricles)</td><td>${fmtMl(tA)}</td><td>${fmtMl(tB)}</td><td></td><td></td></tr>`;
  $('#exCredit').innerHTML = `Example: ${ex.info.name}. Licence ${ex.info.license}. ${ex.info.credit}`;
  $('#exTable').classList.remove('hidden');
}
document.querySelectorAll('#exMode button').forEach((b) => (b.onclick = () => { ex.mode = b.dataset.m; applyExampleMode(true); }));
$('#exVolumes').onclick = volumeTable; $('#exClose').onclick = closeExample; $('#exTableClose').onclick = () => $('#exTable').classList.add('hidden');

// ---------- neck exam (second folder): vertebrae, discs and cord placed on the brain scan by matching C1-C3
let neckAvail = false;
const neckLevels = (N) => `${N.levels[0]}–${N.levels[N.levels.length - 1]}`;
function neckText(short) {
  const N = manifest && manifest.neck; if (!N) return '';
  if (N.status !== 'ok') return `The neck exam “${N.series.description}” could not be lined up reliably with the brain scan, so it is not shown. ${N.reason} ` +
    'This can happen when the brain scan does not reach far enough down the neck, or when the two scans are not of the same person.';
  return short ? `Lined up at ${neckLevels(N)} (shapes agree within ${N.rmsMm} mm). Lower in the neck it can be off by several mm.`
    : `From the neck exam “${N.series.description}”, lined up with the brain scan by matching the ${neckLevels(N)} vertebrae seen in both (they agree within ${N.rmsMm} mm). ` +
      'Lower in the neck the match can be off by several millimetres, because the head and neck are usually held a little differently in two separate scans.';
}
function showNeckCard() {
  const N = manifest && manifest.neck; if (!N) return;
  delete $('#info').dataset.seq; $('#infoIsolate').classList.add('hidden'); $('#infoParcel').innerHTML = '';
  $('#infoSwatch').style.background = N.status === 'ok' ? GROUP_DOT.spine : '#e0a050';
  $('#infoTitle').textContent = N.status === 'ok' ? 'Neck exam added' : 'Neck exam not shown';
  $('#infoBadge').textContent = N.status === 'ok' ? 'lined up' : 'not lined up'; $('#infoBadge').classList.remove('hidden');
  $('#infoText').textContent = neckText();
  $('#infoExtra').textContent = N.status === 'ok' ? 'Find the vertebrae, discs and spinal cord under Layers → Spine & neck (or search “C5”); the slices now continue into the neck. Educational only, not for diagnosis.' : 'The brain model is unchanged. Educational only, not for diagnosis.';
  $('#info').classList.remove('hidden');
}
async function addNeck(cid = currentCase) {   // library → neck exam → "Replace with another folder"
  const j = await (await fetch('api/neck', { method: 'POST', body: JSON.stringify({ caseId: cid }) })).json();
  if (j.error) return toast(j.error, 'err');
  if (j.folder) { closeLibrary(); toast('Adding the neck exam…'); pollNeck(); }
}
async function pollNeck() {
  const N = (await (await fetch('api/state')).json()).neck || {};
  if (N.status === 'running') { liveStatus('neck', `Neck exam: ${N.msg}… ${Math.round(N.pct)}%`); setTimeout(pollNeck, 1500); return; }
  liveStatus('neck');
  if (N.status === 'done' && N.caseId === currentCase) { await loadCase(currentCase, true); showNeckCard(); toast(manifest.neck && manifest.neck.status === 'ok' ? '✓ Neck added and lined up' : 'The neck exam could not be lined up (see the card)'); }
  else if (N.status === 'error') toast(`${N.msg} (${N.error})`, 'err', 9000);
}

// ---------- Add scan (sorted automatically) and the "My body" library
async function addScan(cid) {
  $('#addMenu').classList.add('hidden');
  const busy = toast('Choose a folder in the window that opened…', '', 120000);
  const j = await fetch('api/add', { method: 'POST', body: JSON.stringify({ caseId: cid }) }).then((r) => r.json()).catch(() => ({ error: 'BodyGlass is not running' }));
  busy.remove();
  if (j.error) return toast(j.error, 'err', 9000);
  if (!j.folder) return;
  if (j.route === 'neck') { toast(`Adding “${j.neck}” as the neck exam…`); return pollNeck(); }
  toast(j.route === 'both' ? `Found a head scan and a neck scan. Building your 3D model first, then the neck…` : 'Building your 3D model…');
  firstBuild = true; pollProcessing();
  if (j.route === 'both') setTimeout(function waitNeck() { fetch('api/state').then((r) => r.json()).then((s) => (s.neck.status === 'running' ? pollNeck() : s.status === 'running' || s.phase2.status === 'running' ? setTimeout(waitNeck, 3000) : null)); }, 3000);
}
$('#btnAdd').onclick = () => addScan(currentCase);
$('#aToMe').onclick = () => addScan(currentCase);
$('#aNew').onclick = () => addScan(null);
$('#btnAddMore').onclick = (e) => { e.stopPropagation(); $('#addMenu').classList.toggle('hidden'); };
addEventListener('click', (e) => { if (!e.target.closest('#addMenu')) $('#addMenu').classList.add('hidden'); });
const personName = () => (String(currentCase).startsWith('example:') ? 'Example volunteer' : (manifest && manifest.person && manifest.person.name) || 'My body');
function updatePerson() {
  const n = manifest ? 1 + (manifest.neck ? 1 : 0) : 0;
  $('#btnPerson').textContent = manifest ? `${personName()} · ${n} exam${n > 1 ? 's' : ''} ▾` : 'My body ▾';
  document.querySelectorAll('.pname').forEach((e) => (e.textContent = personName()));
}
const fmtDate = (d) => (d && /^\d{8}$/.test(d) ? new Date(+d.slice(0, 4), +d.slice(4, 6) - 1, +d.slice(6, 8)).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' }) : '');
async function openLibrary() {
  const [cases, st] = await Promise.all([fetch('api/cases').then((r) => r.json()), fetch('api/state').then((r) => r.json())]);
  $('#libPath').textContent = st.casesRoot || '';
  $('#libReveal').classList.toggle('hidden', !st.mac);
  $('#libList').innerHTML = cases.length ? '' : '<p class="muted">No saved models yet. Use ＋ Add scan.</p>';
  for (const c of cases.sort((a, b) => String(b.createdAt || '').localeCompare(String(a.createdAt || '')))) {
    const el = document.createElement('div'); el.className = 'libcard' + (c.caseId === currentCase ? ' open' : '');
    el.innerHTML = `<div class="librow"><input class="libname" value="${(c.name || '').replace(/"/g, '&quot;')}" placeholder="${c.caseId === currentCase ? 'Me' : 'Unnamed'} (click to rename)" maxlength="80">
        <span class="libbtns">${c.caseId === currentCase ? '<button class="ghost libClose" title="Close this model">✕ Close</button>' : '<button class="libOpen">Open</button>'}
        ${st.mac ? '<button class="ghost libFinder" title="Show the saved model in Finder">Show in Finder</button>' : ''}<button class="ghost libDel" title="Delete this 3D model">Delete</button></span></div>
      ${c.exams.map((e) => `<div class="libexam"><span class="role ${e.ok === false ? 'off' : 'ok'}">${e.kind}</span> ${e.description || ''} <span class="muted">${fmtDate(e.date) || ''}</span>
        ${e.kind === 'Neck MRI' && c.caseId === currentCase && neckAvail ? '<button class="link libNeck">Replace with another folder…</button>' : ''}</div>`).join('')}
      ${!c.exams.some((e) => e.kind === 'Neck MRI') && c.caseId === currentCase && c.structures ? '<div class="libexam muted">No neck exam yet: use ＋ Add scan with a neck MRI folder.</div>' : ''}
      <div class="small muted">Saved ${c.createdAt ? new Date(c.createdAt).toLocaleString() : ''}</div>`;
    const post = (p, body) => fetch(p, { method: 'POST', body: JSON.stringify({ caseId: c.caseId, ...body }) }).then((r) => r.json());
    el.querySelector('.libname').onchange = async (e) => { await post('api/case/rename', { name: e.target.value }); if (c.caseId === currentCase) { manifest.person = { name: e.target.value.trim() }; updatePerson(); } toast('Renamed'); };
    el.querySelector('.libOpen')?.addEventListener('click', async () => { closeLibrary(); currentCase = c.caseId; history.replaceState(null, '', '?case=' + c.caseId); await loadCase(c.caseId); if (!hasStructs()) pollPhase2(); });
    el.querySelector('.libClose')?.addEventListener('click', () => { closeLibrary(); closeCase(); });
    el.querySelector('.libFinder')?.addEventListener('click', () => post('api/reveal'));
    el.querySelector('.libNeck')?.addEventListener('click', () => addNeck(c.caseId));
    el.querySelector('.libDel').onclick = async () => {
      if (!confirm(`Delete the 3D model “${c.name || c.description}”?\nYour scan folder is not touched; you can rebuild it with ＋ Add scan.`)) return;
      const r = await post('api/case/delete'); if (r.error) return toast(r.error, 'err');
      if (c.caseId === currentCase) closeCase();
      toast('Deleted'); openLibrary();
    };
    $('#libList').append(el);
  }
  $('#library').classList.remove('hidden');
}
function closeLibrary() { $('#library').classList.add('hidden'); }
$('#btnPerson').onclick = () => ($('#library').classList.contains('hidden') ? openLibrary() : closeLibrary());
$('#libClose').onclick = closeLibrary;
$('#libReveal').onclick = () => fetch('api/reveal', { method: 'POST', body: '{}' });
function closeCase() {
  clearScene(); manifest = null; currentCase = null; history.replaceState(null, '', location.pathname); updatePerson(); hideInfo();
  showWelcome();
}
function showWelcome() { setLoading(false); $('#welcome').classList.remove('hidden'); }
function hideWelcome() { $('#welcome').classList.add('hidden'); }
$('#wAdd').onclick = () => addScan(null);
$('#wExample').onclick = () => openExampleVolunteer();
$('#aExample').onclick = () => { $('#addMenu').classList.add('hidden'); openExampleVolunteer(); };
async function openExampleVolunteer() {
  const info = await (await fetch('api/example')).json();
  if (!info.available) return toast('The example volunteer is not installed. Run ./install.sh again to add it.', 'err');
  hideWelcome(); currentCase = 'example:' + info.caseId; history.replaceState(null, '', '?case=' + currentCase);
  await loadCase(currentCase);
  toast(`Example volunteer: ${info.credit} ${info.license}.`, '', 8000);
  maybeAutoTour();
}
// step list while the first model is built (driven by the pipeline's progress %)
const STEPS = [[2, 'Reading the scan folder'], [10, 'Loading the pictures'], [20, 'Lining them up on a 1 mm grid'], [25, 'Finding the brain'], [35, 'Outlining skin, skull and brain'], [45, 'Building the 3D surfaces'], [100, 'Opening your model']];
function showSteps(pct) {
  const ol = $('#steps'); ol.classList.remove('hidden');
  ol.innerHTML = STEPS.map(([p, t], i) => { const next = (STEPS[i + 1] || [101])[0]; return `<li class="${pct >= next ? 'done' : pct >= p ? 'now' : ''}">${t}</li>`; }).join('');
}
// guided tour starts by itself once, on the first model ever opened (skip stays visible)
function maybeAutoTour() {
  if (params.has('shot') || localStorage.getItem('bg:toured') || !manifest) return;
  localStorage.setItem('bg:toured', '1'); setTimeout(() => !tour.on && startTour(), 1200);
}
// ⌘K / Ctrl-K palette: every command and body part in one search box
function paletteItems(q) {
  const cmds = [['＋ Add scan', () => addScan(currentCase), 'A'], ['My body (saved models)', openLibrary, ''], ['Ask AI about your scan', openChat, ''], ['Manage AI models', openAiModels, ''], ['Guided tour', startTour, 'G'], ['Keyboard shortcuts', () => toggleHelp(true), '?'],
    ['Measure a distance', () => setMeasure(), 'M'], ['Export for 3D printing', openExport, 'E'], ['Compare with an example brain', () => openExample('side'), 'X'],
    ['Fit everything', fitAll, '0'], ['Reset view', resetView, 'R'], ['Turntable on/off', () => $('#btnAuto').click(), 'T'], ['Labels on/off', () => $('#lblToggle').click(), 'L'],
    ['Cut away at the slices', () => $('#cutaway').click(), ''], ['Colour structures on the slices', () => $('#labelOverlay').click(), 'O'],
    ...(layers['g:spine'] ? [['Frame: Head', () => frameRegion('head'), ''], ['Frame: Neck', () => frameRegion('neck'), ''], ['Frame: Head + neck', () => frameRegion('both'), '']] : []),
    ...['axial', 'coronal', 'sagittal'].map((p) => [`${p[0].toUpperCase() + p.slice(1)} slice on/off`, () => document.querySelector(`input[data-plane=${p}]`).click(), '']),
    ...PRESETS.map(([n, l]) => [`View: ${n}`, () => { setPeel(l); syncParts(); }, String(l + 1)]),
    ...['Explore', 'Layers', 'Scans', 'About'].map((t) => [`Panel: ${t}`, () => setTab(t.toLowerCase(), true), '']),
    ['Two-finger swipe zooms (setting)', () => $('#wheelZoom').click(), '']];
  const ql = q.trim().toLowerCase();
  const c = cmds.filter(([n]) => !ql || n.toLowerCase().includes(ql)).map(([n, f, k]) => ({ n, f, k }));
  const parts = manifest ? searchParts(q, 6).map((x) => ({ n: x.name, k: x.sub, f: () => goToPart(x) })) : [];
  return [...parts, ...c].slice(0, 12);
}
let palSel = 0, palRes = [];
function renderPalette() {
  palRes = paletteItems($('#palIn').value); palSel = Math.min(palSel, Math.max(0, palRes.length - 1));
  $('#palList').innerHTML = palRes.map((x, i) => `<div class="pi ${i === palSel ? 'on' : ''}" data-i="${i}"><span>${x.n}</span><kbd>${x.k || ''}</kbd></div>`).join('') || '<div class="small muted">Nothing found</div>';
  $('#palList').querySelectorAll('.pi').forEach((el) => (el.onclick = () => runPalette(+el.dataset.i)));
}
function openPalette() { $('#palette').classList.remove('hidden'); $('#palIn').value = ''; palSel = 0; renderPalette(); $('#palIn').focus(); }
function closePalette() { $('#palette').classList.add('hidden'); $('#palIn').blur(); }
function runPalette(i) { const x = palRes[i]; closePalette(); if (x) x.f(); }
$('#palIn').oninput = () => { palSel = 0; renderPalette(); };
$('#palIn').onkeydown = (e) => {
  e.stopPropagation();
  if (e.key === 'ArrowDown' || e.key === 'ArrowUp') { palSel = (palSel + (e.key === 'ArrowDown' ? 1 : -1) + palRes.length) % Math.max(1, palRes.length); renderPalette(); e.preventDefault(); }
  else if (e.key === 'Enter') runPalette(palSel);
  else if (e.key === 'Escape') closePalette();
};
addEventListener('keydown', (e) => { if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); e.stopImmediatePropagation(); $('#palette').classList.contains('hidden') ? openPalette() : closePalette(); } }, true);
addEventListener('click', (e) => { if (!e.target.closest('#palette')) closePalette(); });
// live progress only while work runs; removed when it stops (then a short toast says how it ended)
const live = {};
function liveStatus(key, text) {
  if (!text) { if (live[key]) { live[key].remove(); delete live[key]; } return; }
  if (!live[key]) { live[key] = document.createElement('div'); live[key].className = 'toast glass live'; $('#toasts').append(live[key]); }
  live[key].innerHTML = text;
}

// ---------- Tools menu, modals, collapsible panel sections
function closeAllModals() { closeLibrary(); $('#export').classList.add('hidden'); $('#exTable').classList.add('hidden'); toggleHelp(false); if (cmp.open) closeCompare(); $('#toolsMenu').classList.add('hidden'); }
$('#btnTools').onclick = (e) => { e.stopPropagation(); $('#toolsMenu').classList.toggle('hidden'); };
addEventListener('click', (e) => { if (!e.target.closest('#toolsMenu')) $('#toolsMenu').classList.add('hidden'); });
$('#mMeasure').onclick = () => { $('#toolsMenu').classList.add('hidden'); setMeasure(); };
$('#mExport').onclick = () => { $('#toolsMenu').classList.add('hidden'); openExport(); };
$('#mExample').onclick = () => { $('#toolsMenu').classList.add('hidden'); ex.on ? closeExample() : openExample('side'); };
$('#btnTour').onclick = () => (tour.on ? endTour() : startTour());
document.querySelectorAll('#panel section > h3').forEach((h) => {
  h.classList.add('fold'); const sec = h.parentElement, key = 'fold:' + (sec.id || h.textContent.trim().split(' ')[0]);
  if (localStorage.getItem(key) === '1') sec.classList.add('folded');
  h.onclick = (e) => { if (e.target.closest('.hint')) return; sec.classList.toggle('folded'); localStorage.setItem(key, sec.classList.contains('folded') ? '1' : '0'); };
});

// ---------- left rail (Explore / Layers / Scans / About): one panel tab at a time; click the open tab to hide the panel
function setTab(t, open) {
  const P = $('#panel'), same = !open && P.dataset.tab === t && !P.classList.contains('closed');
  P.dataset.tab = t; P.classList.toggle('closed', same);
  document.querySelectorAll('#rail button').forEach((b) => b.classList.toggle('on', !same && b.dataset.tab === t));
  localStorage.setItem('bg:tab', same ? '' : t);
}
document.querySelectorAll('#rail button').forEach((b) => (b.onclick = () => setTab(b.dataset.tab)));
{ const t = localStorage.getItem('bg:tab'); $('#panel').dataset.tab = t || 'explore'; if (t === '') $('#panel').classList.add('closed'); else document.querySelector(`#rail button[data-tab=${t || 'explore'}]`).classList.add('on'); }
$('#aboutKeys').onclick = () => toggleHelp(true);
$('#aboutTour').onclick = () => startTour();
$('#sliceMore').onclick = (e) => { e.stopPropagation(); $('#sliceOpts').classList.toggle('hidden'); };
addEventListener('click', (e) => { if (!e.target.closest('#sliceOpts') && !e.target.closest('#sliceMore')) $('#sliceOpts').classList.add('hidden'); });
// Explore: search (shared with the ⌘K palette), view presets (= peel levels), parts with eye toggles
function searchParts(q, max = 8) {
  q = q.trim().toLowerCase(); if (!q) return [];
  const all = [...Object.values(structById).map((S) => ({ id: S.st.id, name: S.st.name, sub: (manifest.structureGroups[S.st.group] || {}).name || '', struct: true })),
    ...Object.values(layers).filter((L) => !L.hidden && !L.def.group).map((L) => ({ id: L.def.id, name: L.def.name, sub: 'layer' }))];
  const score = (n) => { n = n.toLowerCase(); return n.startsWith(q) ? 0 : n.split(/[\s(-]+/).some((w) => w.startsWith(q)) ? 1 : n.includes(q) ? 2 : 9; };
  return all.map((x) => [score(x.name), x]).filter(([k]) => k < 9).sort((a, b) => a[0] - b[0] || a[1].name.length - b[1].name.length).slice(0, max).map(([, x]) => x);
}
function goToPart(x) {
  if (x.struct) { const S = structById[x.id]; if (!S.st.visible) { S.st.visible = true; applyOpacity(S.L); } selectStruct(x.id); focusOn([S.item], S.st.name); }
  else { const L = layers[x.id]; if (!L.visible) { L.visible = true; if (L.opacity < 0.05) L.opacity = 1; applyOpacity(L); } select(x.id); focusOn(L.items, L.def.name); }
}
$('#search').oninput = (e) => {
  const res = searchParts(e.target.value);
  $('#searchResults').innerHTML = res.map((x, i) => `<div class="sr" data-i="${i}">${x.name}<small>${x.sub}</small></div>`).join('') || (e.target.value.trim() ? '<div class="small muted">Nothing found</div>' : '');
  $('#searchResults').querySelectorAll('.sr').forEach((el) => (el.onclick = () => goToPart(res[+el.dataset.i])));
};
$('#search').onkeydown = (e) => { if (e.key === 'Enter') { const r = searchParts(e.target.value)[0]; if (r) goToPart(r); } e.stopPropagation(); };
const PRESETS = [['Skin', 0], ['See-through', 1], ['Skull', 2], ['Brain', 3], ['Deep inside', 4]];
function buildExplore() {
  $('#presets').innerHTML = PRESETS.map(([n, l]) => `<button class="chip" data-l="${l}">${n}</button>`).join('') + (layers['g:spine'] ? '<button class="chip" data-l="spine">Spine</button>' : '');
  $('#presets').querySelectorAll('button').forEach((b) => (b.onclick = () => {
    if (b.dataset.l === 'spine') { setPeel(1); vis('skin', 0); frameRegion('neck'); } else setPeel(+b.dataset.l);
    syncParts();
  }));
  $('#partList').innerHTML = '';
  for (const L of Object.values(layers)) {
    if (L.hidden) continue;
    const el = document.createElement('div'); el.className = 'pr';
    el.innerHTML = `<span class="dot" style="background:${L.def.color}"></span>${L.def.name}<button class="eye" title="Show / hide">👁</button>`;
    el.querySelector('.eye').onclick = (e) => { e.stopPropagation(); L.row.querySelector('input[type=checkbox]').click(); syncParts(); };
    el.onclick = () => (L.def.group ? focusOn(L.items, L.def.name) : goToPart({ id: L.def.id }));
    L.part = el; $('#partList').append(el);
  }
  syncParts();
}
function syncParts() { for (const L of Object.values(layers)) if (L.part) L.part.classList.toggle('off', !L.visible); }
$('#infoFocus').onclick = () => focusSelected();
$('#infoHide').onclick = () => {
  if (!selected) return;
  const st = selected.items[0].st;
  if (st) { st.visible = false; applyOpacity(structById[st.id].L); }
  else { const L = Object.values(layers).find((x) => x.items === selected.items); if (L) L.row.querySelector('input[type=checkbox]').click(); }
  hideInfo(); syncParts();
};

// ---------- picking + info card
const ray = new THREE.Raycaster(); const ndc = new THREE.Vector2(); let down = null;
canvas.addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY]; if (e.button === 2 || e.shiftKey) setFocusTag(null); controls.autoRotate = false; $('#btnAuto').classList.remove('on'); });
function pickAt(e) {
  ndc.set(e.clientX / innerWidth * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  const targets = [...Object.values(layers).flatMap((L) => L.items.filter((it) => it.mesh.visible && it.material.opacity > 0.15).map((it) => it.mesh)),
    ...Object.values(planes).filter((P) => P.enabled).map((P) => P.mesh)];
  return ray.intersectObjects(targets, true).find((h) => (h.object.userData.plane ? planeDrawnAt(h) : !isClipped(h.point, h.object))); // skip cut-away plane parts
}
canvas.addEventListener('pointerup', (e) => {
  if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) return;
  if (e.button === 2 || e.shiftKey) setFocusTag(null);
  const hit = pickAt(e);
  if (meas.on) { if (hit) addMeasPoint(worldToRas(hit.point)); return; }
  if (!hit) return hideInfo();
  if (hit.object.userData.plane) return showSliceInfo(hit);
  let o = hit.object; while (o && !o.userData.layerId) o = o.parent;
  select(o.userData.structId || o.userData.layerId, hit.point);
});
function worldToRas(p) { return centered.worldToLocal(p.clone()); }
function parcelAt(point) {
  if (!parcels || !point) return null;
  const r = worldToRas(point), [nx, ny, nz] = dims;
  const i = Math.round((r.x - origin[0]) / sp), j = Math.round((r.y - origin[1]) / sp), k = Math.round((r.z - origin[2]) / sp);
  if (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz) return null;
  const p = parcels[(k * ny + j) * nx + i];
  const e = p && manifest.parcels.index[String(p)];
  if (!e) return null;
  const info = manifest.parcels.info[e[0]] || [e[0], ''];
  return { name: info[0], text: info[1], hemi: e[1] };
}
const inBrain = (o) => o.group !== 'ventricles' && o.group !== 'spine';
function fmtMl(v) { return v < 10 ? v.toFixed(1) : Math.round(v).toLocaleString(); }
function selectStruct(id, point) {
  const S = structById[id]; if (!S) return;
  delete $('#info').dataset.seq;
  const st = S.st;
  selected = { items: [S.item], t: performance.now() };
  aiSel = { name: st.name, volumeMl: st.volume_ml }; aiCtxChanged();
  for (const x of Object.values(layers)) x.row && (x.row.style.background = '');
  document.querySelectorAll('#structList .srow').forEach((r) => r.classList.toggle('sel', r.dataset.key === st.key));
  document.querySelectorAll('#structList .hemi b').forEach((b) => b.classList.toggle('sel', b.dataset.id === id));
  $('#infoSwatch').style.background = st.color;
  $('#infoTitle').textContent = st.name;
  $('#infoBadge').textContent = 'automatic label'; $('#infoBadge').classList.remove('hidden');
  $('#infoText').textContent = manifest.structureInfo[st.key] || '';
  const gm = st.group === 'cortex';
  const pair = st.hemi ? manifest.structures.find((o) => o.key === st.key && o.id !== st.id) : null;
  const total = manifest.structures.filter(inBrain).reduce((a, o) => a + (o.volume_ml || 0), 0);
  $('#infoExtra').innerHTML = `<span class="vol"><b>${fmtMl(st.volume_ml)} mL</b> ${gm ? 'grey matter' : 'volume'}` +
    (pair ? ` · other side ${fmtMl(pair.volume_ml)} mL` : '') + (total > 0 && inBrain(st) ? ` · ${(100 * st.volume_ml / total).toFixed(st.volume_ml / total < 0.01 ? 2 : 1)} % of the labelled brain` : '') +
    `</span><br>Computer estimate from the labels (1 mL = 1 cm³); healthy sizes vary a lot between people` + (st.group === 'spine' ? `.<br>${neckText()}` : ', and small left–right differences are normal.');
  const p = gm ? parcelAt(point) : null;
  $('#infoParcel').innerHTML = p ? `<b>Under the pointer:</b> ${p.name}. ${p.text}` : (gm ? '<span class="muted">Click on the surface to name the fold (gyrus) under the pointer.</span>' : '');
  $('#infoIsolate').classList.remove('hidden');
  $('#infoIsolate').textContent = isolateId === id ? 'Show all' : 'Isolate';
  $('#info').classList.remove('hidden');
}
$('#infoIsolate').onclick = () => {
  if (!selected || !selected.items[0].st) return;
  const id = selected.items[0].st.id;
  isolateId = isolateId === id ? null : id; applyAll();
  $('#infoIsolate').textContent = isolateId ? 'Show all' : 'Isolate';
};
function select(id, point) {
  if (structById[id]) return selectStruct(id, point);
  const L = layers[id]; if (!L) return;
  delete $('#info').dataset.seq;
  selected = { items: L.items, t: performance.now() };
  aiSel = { name: L.def.name, volumeMl: L.def.veins ? null : L.def.volume_ml }; aiCtxChanged();
  $('#infoParcel').innerHTML = ''; $('#infoIsolate').classList.add('hidden'); $('#infoBadge').textContent = 'quick estimate';
  document.querySelectorAll('#structList .srow, #structList .hemi b').forEach((r) => r.classList.remove('sel'));
  for (const x of Object.values(layers)) x.row && (x.row.style.background = x === L ? 'rgba(124,196,255,.12)' : '');
  $('#infoSwatch').style.background = L.def.color;
  $('#infoTitle').textContent = L.def.name;
  $('#infoBadge').classList.toggle('hidden', !L.def.approx);
  $('#infoText').textContent = L.def.description;
  let extra = L.def.veins ? `Traced from “${manifest.veins.sourceName}” in ${manifest.veins.seconds} s on this computer; the SWI covered about ${Math.round(manifest.veins.coveragePct)} % of the head grid.`
    : `Estimated volume: ${Math.round(L.def.volume_ml).toLocaleString()} mL (rough computer estimate).`;
  if (point && (id === 'brain' || id === 'white_matter')) {
    const r = worldToRas(point), dx = r.x - manifest.brainCenter[0];
    extra = (Math.abs(dx) < 4 ? 'Near the midline, between the two hemispheres. ' : `You clicked the ${dx > 0 ? 'right' : 'left'} hemisphere. `) +
      (hasStructs() ? '' : 'Named regions (lobes, hippocampus…) appear once the detailed structures are ready.');
  }
  if (id === 'skin') extra += ' Teeth and air give no MRI signal, so the surface is smoothed across the mouth and nostrils.';
  $('#infoExtra').textContent = extra;
  $('#info').classList.remove('hidden');
}
function showSliceInfo(hit) {
  const r = worldToRas(hit.point), name = hit.object.userData.plane;
  delete $('#info').dataset.seq;
  $('#infoSwatch').style.background = '#' + new THREE.Color(PLANE_COLORS[name]).getHexString();
  $('#infoTitle').textContent = name[0].toUpperCase() + name.slice(1) + ' slice';
  $('#infoParcel').innerHTML = ''; $('#infoIsolate').classList.add('hidden');
  $('#infoBadge').classList.add('hidden');
  $('#infoText').textContent = { axial: 'A horizontal cut, as if looking up from the feet.', coronal: 'A vertical cut from ear to ear, as if facing the person.', sagittal: 'A vertical cut from front to back, seen from the side.' }[name] +
    (seqKind ? ` Showing the ${seqData[seqKind].meta.name} scan here. ` + (SEQ_TEXT[seqKind] || [''])[0] : ' On a T1 scan, fat and white matter look bright, grey matter is grey, and fluid (CSF) is dark.');
  $('#infoExtra').textContent = `Position (RAS): ${r.x.toFixed(0)}, ${r.y.toFixed(0)}, ${r.z.toFixed(0)} mm`;
  $('#info').classList.remove('hidden');
}
function hideInfo() {
  $('#info').classList.add('hidden'); selected = null; aiSel = null; aiCtxChanged(); delete $('#info').dataset.seq;
  for (const x of Object.values(layers)) x.row && (x.row.style.background = '');
  document.querySelectorAll('#structList .srow, #structList .hemi b').forEach((r) => r.classList.remove('sel'));
  if (isolateId) { isolateId = null; applyAll(); }
}
$('#infoClose').onclick = hideInfo;

// ---------- camera helpers
const VIEWS = { '3q': [0.75, 0.32, -1.0], '3qhigh': [0.85, 0.85, -1.0], backleft: [-0.85, 0.8, 1.0], backright: [0.8, 0.35, 1.0], front: [0, 0.12, -1], left: [-1, 0.1, 0], right: [1, 0.12, -0.15], top: [0.01, 1, -0.2], back: [0.5, 0.3, 1] };
let camAnim = null, viewRadius = 300; const viewTarget = new THREE.Vector3(0, 10, 0);
// opts: { radius: multiplier, target: Vector3 (world), dur: ms }. Animated moves follow an arc around the
// target (direction slerp + radius ease), not a straight line through the head.
function setView(name, instant, opts = {}) {
  const tgt = opts.target ? opts.target.clone() : viewTarget.clone();
  const dir = new THREE.Vector3(...(Array.isArray(name) ? name : VIEWS[name] || VIEWS['3q'])).normalize();
  const r = opts.dist || viewRadius * (opts.radius || 1);
  if (instant) { camera.position.copy(dir.multiplyScalar(r).add(tgt)); controls.target.copy(tgt); controls.update(); return; }
  const d0 = camera.position.clone().sub(controls.target);
  camAnim = { tFrom: controls.target.clone(), tTo: tgt, r0: d0.length(), r1: r, q: new THREE.Quaternion().setFromUnitVectors(d0.clone().normalize(), dir),
    d0: d0.normalize(), t0: performance.now(), dur: opts.dur || 900 };
}

// ---------- camera: focus (orbit around a part), region framing, fit all, trackpad pan
// distance at which a sphere of radius r fills the view (smaller of the vertical / horizontal field of view)
function fitDist(r, fovDeg, aspect, margin = 1.2) { const v = fovDeg * Math.PI / 360, h = Math.atan(Math.tan(v) * aspect); return margin * r / Math.sin(Math.min(v, h)); }
const meshBox = (meshes) => { scene.updateMatrixWorld(true); const b = new THREE.Box3(); for (const m of meshes) b.expandByObject(m); return b; };
function frameBox(box, label) {
  if (box.isEmpty()) return;
  const sph = box.getBoundingSphere(new THREE.Sphere()), dir = camera.position.clone().sub(controls.target).normalize();
  setView(dir.toArray(), false, { target: sph.center, dist: Math.max(controls.minDistance, fitDist(sph.radius, camera.fov, camera.aspect)), dur: 700 });
  setFocusTag(label);
}
function setFocusTag(label) { const t = $('#focusTag'); t.classList.toggle('hidden', !label); if (label) t.innerHTML = `◎ Orbiting around <b>${label}</b> <span class="muted">· Esc to reset</span>`; }
const allMeshes = (pred = () => true) => Object.values(layers).flatMap((L) => L.items.filter((it) => pred(L, it)).map((it) => it.mesh));
const spineMeshes = () => (layers['g:spine'] ? layers['g:spine'].items.map((it) => it.mesh) : []);
const headMeshes = () => (layers.skin || layers.brain ? [(layers.skin || layers.brain).items[0].mesh] : []);
const REGIONS = { head: () => meshBox(headMeshes()), neck: () => meshBox(spineMeshes()), both: () => meshBox([...headMeshes(), ...spineMeshes()]) };
function frameRegion(r) { document.querySelectorAll('#regionBar button').forEach((b) => b.classList.toggle('on', b.dataset.r === r)); frameBox(REGIONS[r](), null); }
function fitAll() { frameBox(meshBox(allMeshes((L, it) => it.mesh.visible)), null); }
function focusOn(items, label) { frameBox(meshBox(items.map((it) => it.mesh || it)), label); }
function updateRegionBar() { $('#regionBar').classList.toggle('hidden', !spineMeshes().length); document.querySelectorAll('#regionBar button').forEach((b) => b.classList.remove('on')); }
document.querySelectorAll('#regionBar button').forEach((b) => (b.onclick = () => frameRegion(b.dataset.r)));
// Mac trackpad: two-finger swipe pans (like Maps), pinch (ctrlKey) zooms. Mouse wheels keep zooming.
// shortcut: trackpad vs mouse is guessed from the event (Chrome/Safari wheelDelta); 'Scroll zooms' in ⌘K switches it off
let wheelPans = localStorage.getItem('bg:wheelzoom') !== '1';
$('#wheelZoom').checked = !wheelPans;
$('#wheelZoom').onchange = (e) => { wheelPans = !e.target.checked; localStorage.setItem('bg:wheelzoom', e.target.checked ? '1' : '0'); };
addEventListener('wheel', (e) => {
  if (e.target !== canvas || e.ctrlKey || !wheelPans) return;
  const trackpad = e.wheelDeltaY ? e.wheelDeltaY === -3 * e.deltaY || e.deltaX !== 0 : e.deltaMode === 0;
  if (!trackpad) return;
  e.preventDefault(); e.stopImmediatePropagation();
  controls._pan(-e.deltaX, -e.deltaY); controls.update(); setFocusTag(null); hideHint();
}, { capture: true, passive: false });
// Pinch must never zoom the page. Chrome/Firefox send it as ctrlKey wheels (OrbitControls zooms those over the
// canvas; elsewhere, e.g. over panels or labels, we only block the page zoom). Safari sends gesture* events instead.
addEventListener('wheel', (e) => { if (e.ctrlKey) e.preventDefault(); }, { passive: false });
let pinchScale = 1;
addEventListener('gesturestart', (e) => { e.preventDefault(); pinchScale = e.scale; }, { passive: false });
addEventListener('gesturechange', (e) => {
  e.preventDefault();
  const r = Math.min(1.25, Math.max(0.8, e.scale / pinchScale)); pinchScale = e.scale;   // clamp one step
  if (e.target === canvas) { controls._dollyOut(r); controls.update(); setFocusTag(null); hideHint(); }   // fingers apart = closer
}, { passive: false });
addEventListener('gestureend', (e) => e.preventDefault(), { passive: false });
canvas.addEventListener('dblclick', (e) => {
  const hit = pickAt(e); if (!hit) return;
  if (hit.object.userData.plane) { frameBox(new THREE.Box3().setFromCenterAndSize(hit.point, new THREE.Vector3(60, 60, 60)), null); return; }
  const S = structById[hit.object.userData.structId];
  if (S) { selectStruct(S.st.id, hit.point); focusOn([S.item], S.st.name); return; }
  let o = hit.object; while (o && !o.userData.layerId) o = o.parent;
  const L = o && layers[o.userData.layerId]; if (L) { select(L.def.id, hit.point); focusOn(L.items, L.def.name); }
});
function focusSelected() { if (!selected) return fitAll(); const st = selected.items[0].st; focusOn(selected.items, st ? st.name : ($('#infoTitle').textContent || null)); }
function resetView() { setFocusTag(null); updateRegionBar(); setView('3q'); }
// first-run gesture hint: hides after the first move, never shown again (? lists everything)
function hideHint() { const h = $('#hint'); if (!h.classList.contains('hidden')) { h.classList.add('hidden'); localStorage.setItem('bg:hint', '1'); } }
controls.addEventListener('start', () => setTimeout(hideHint, 1500));
$('#hintClose').onclick = hideHint;

// ---------- loading
const loader = new GLTFLoader();
const firstMesh = (gltf) => { let m = null; gltf.scene.traverse((o) => { if (o.isMesh && !m) m = o; }); return m; };
function setLoading(on, msg, pct) {
  $('#loading').classList.toggle('hidden', !on);
  if (msg) $('#loadingMsg').textContent = msg;
  if (pct != null) $('#loadingBar').style.width = pct + '%';
}
const GROUP_MAT = { cortex: 'brain', white_matter: 'wm', deep: 'deep', ventricles: 'fluid', hindbrain: 'deep', spine: 'bone' };
const GROUP_DOT = { cortex: 'linear-gradient(135deg,#e8938f,#e6c07b 50%,#86a9e3)', white_matter: '#f1ece2', deep: 'linear-gradient(135deg,#4fb0a5,#e28fb7 50%,#f2d16b)', ventricles: '#5fb6e8', hindbrain: '#d98fa6', spine: 'linear-gradient(135deg,#eadfc8,#7fb7d6 50%,#f2cf5b)' };
function toShading(geo) { // Phase 1 sulcal-depth vertex colours -> grey shading, so lobe colours tint it
  const c = geo.attributes.color; if (!c) return false;
  const lum = new Float32Array(c.count); let mx = 1e-6;
  for (let i = 0; i < c.count; i++) { lum[i] = 0.3 * c.getX(i) + 0.59 * c.getY(i) + 0.11 * c.getZ(i); mx = Math.max(mx, lum[i]); }
  for (let i = 0; i < c.count; i++) { const v = Math.min(1, 0.12 + lum[i] / mx); c.setXYZ(i, v, v, v); }
  c.needsUpdate = true; return true;
}
function setRegionColors(on) {
  regionColors = on; $('#lobeColors').checked = on;
  for (const S of Object.values(structById)) if (S.st.group === 'cortex' || S.st.key === 'cerebellum') S.item.material.color.copy(on ? S.item.baseColor : BRAIN_PINK);
}
async function loadStructures(base) {
  const groups = Object.entries(manifest.structureGroups).sort((a, b) => a[1].order - b[1].order);
  for (const [gid, g] of groups) if (manifest.structures.some((s) => s.group === gid)) layers['g:' + gid] = { def: { id: 'g:' + gid, name: g.name, color: GROUP_DOT[gid], group: gid }, items: [], visible: true, opacity: 1 };
  let n = 0;
  await Promise.all(manifest.structures.map(async (st) => {
    const g = await loader.loadAsync(base + st.file);
    const mesh = firstMesh(g);
    mesh.geometry.computeVertexNormals();
    const shaded = toShading(mesh.geometry);
    const matName = st.key === 'cerebellum' ? 'brain' : GROUP_MAT[st.group];
    const material = makeMaterial({ material: matName, color: st.color, ownColor: true }, shaded);
    if (matName === 'fluid') material.emissive = new THREE.Color(st.color).multiplyScalar(0.12);
    const m = new THREE.Mesh(mesh.geometry, material);
    m.userData.layerId = 'g:' + st.group; m.userData.structId = st.id; m.receiveShadow = true;
    m.renderOrder = st.group === 'cortex' ? 1 : 0;
    centered.add(m);
    st.visible = true;
    // label anchor: the surface vertex nearest the centroid (a C-shaped ventricle's centroid is outside it)
    const pos = mesh.geometry.attributes.position, c0 = new THREE.Vector3(...st.centroid), v = new THREE.Vector3();
    let bestD = Infinity; const anchor = c0.clone();
    for (let i = 0; i < pos.count; i += 3) { v.fromBufferAttribute(pos, i); const d = v.distanceToSquared(c0); if (d < bestD) { bestD = d; anchor.copy(v); } }
    const item = { mesh: m, material, st, baseColor: new THREE.Color(st.color), anchor };
    structById[st.id] = { st, item, L: layers['g:' + st.group] };
    setLoading(true, `Loading ${st.name}…`, 50 + 45 * (++n / manifest.structures.length));
  }));
  for (const st of manifest.structures) layers['g:' + st.group].items.push(structById[st.id].item); // manifest order
  for (const id of ['brain', 'white_matter', 'ventricles']) if (layers[id]) layers[id].hidden = true;
  parcels = manifest.parcels ? padZ(new Uint8Array(await (await fetch(base + manifest.parcels.file)).arrayBuffer())) : null;
  setRegionColors(regionColors);
}
function buildStructUI() {
  const sec = $('#structSection');
  sec.classList.toggle('hidden', !hasStructs());
  if (!hasStructs()) return;
  const box = $('#structList'); box.innerHTML = '';
  const groups = Object.entries(manifest.structureGroups).sort((a, b) => a[1].order - b[1].order);
  for (const [gid, g] of groups) {
    const keys = [...new Set(manifest.structures.filter((s) => s.group === gid).map((s) => s.key))];
    if (!keys.length) continue;
    const h = document.createElement('div'); h.className = 'sgroup'; h.textContent = g.name; box.appendChild(h);
    if (gid === 'spine') { const n = document.createElement('div'); n.className = 'small muted'; n.textContent = neckText(true); box.appendChild(n); }
    for (const key of keys) {
      const sts = manifest.structures.filter((s) => s.key === key);
      const row = document.createElement('div'); row.className = 'srow'; row.dataset.key = key;
      const nm = sts[0].hemi ? sts[0].name.replace(/^(Left|Right) /, '') : sts[0].name;
      row.innerHTML = `<input type="checkbox" checked><span class="dot" style="background:${sts[0].color}"></span><span class="sname">${nm[0].toUpperCase() + nm.slice(1)}</span><span class="hemi">${sts.filter((s) => s.hemi).map((s) => `<b data-id="${s.id}" title="${s.name}">${s.hemi}</b>`).join('')}</span>`;
      row.querySelector('input').onchange = (e) => { for (const s of sts) s.visible = e.target.checked; applyOpacity(layers['g:' + gid]); };
      row.querySelector('.sname').onclick = () => selectStruct(sts[0].id);
      row.querySelectorAll('.hemi b').forEach((b) => (b.onclick = () => selectStruct(b.dataset.id)));
      box.appendChild(row);
    }
  }
}
// ---------- floating 3D labels (nearest side of each pair, outermost visible layer only)
const labelBox = $('#labels'); const labelEls = {};
const NO_AUTO_LABEL = new Set(['cingulate', 'insula', 'cerebral_wm']);
function planeDrawnAt(hit) { // mirrors the octant discard in sliceFS
  if (!$('#cutaway').checked) return true;
  const r = worldToRas(hit.point), P0 = planes[hit.object.userData.plane];
  for (const P of Object.values(planes)) {
    if (P === P0 || !P.enabled) continue;
    if ((r.getComponent(P.axis) - P.pos) * CUT_SIGN[P.axis] < 0) return false;
  }
  return true;
}
const occRay = new THREE.Raycaster(); const occluded = new Set(); let lastOcc = -1e9;
function updateLabels(now) {
  const want = [];
  if (labelsOn && hasStructs() && !document.body.classList.contains('nolabels')) {
    const op = (g) => (layers['g:' + g] && layers['g:' + g].visible ? layers['g:' + g].opacity : 0);
    const eligible = { cortex: true, hindbrain: true, spine: true, white_matter: op('cortex') < 0.5, deep: op('cortex') < 0.5 && op('white_matter') < 0.5 };
    eligible.ventricles = eligible.deep;
    const best = {};
    const selId = selected && selected.items[0] && selected.items[0].st ? selected.items[0].st.id : null;
    for (const S of Object.values(structById)) {
      const it = S.item, st = S.st;
      if (!it.mesh.visible) continue;
      const isSel = st.id === selId;
      if (!isSel && (isolateId || NO_AUTO_LABEL.has(st.key) || !eligible[st.group] || it.material.opacity < 0.5)) continue;
      const w = centered.localToWorld(it.anchor.clone());
      const d = w.distanceTo(camera.position) - (isSel ? 1e4 : 0);
      if (!best[st.key] || d < best[st.key].d) best[st.key] = { d, w, st };
    }
    if (now - lastOcc > 350) { // occlusion test, throttled: hide labels of structures hidden behind others
      lastOcc = now; occluded.clear();
      const blockers = Object.values(structById).map((S) => S.item).filter((it) => it.mesh.visible && it.material.opacity >= 0.5).map((it) => it.mesh);
      for (const b of Object.values(best)) {
        const dir = b.w.clone().sub(camera.position); const dist = dir.length();
        occRay.set(camera.position, dir.normalize()); occRay.far = dist;
        const h = occRay.intersectObjects([...blockers, ...Object.values(planes).filter((P) => P.enabled).map((P) => P.mesh)], false).find((x) => (x.object.userData.plane ? planeDrawnAt(x) : !isClipped(x.point, x.object)));
        if (isClipped(b.w, structById[b.st.id] && structById[b.st.id].item.mesh)) { occluded.add(b.st.key); continue; }
        if (h && h.object.userData.plane && h.distance < dist - 2) { occluded.add(b.st.key); continue; }
        // plane hits have no structId: reading it used to throw and stop the render loop (rotate/zoom froze)
        if (h && !h.object.userData.plane && h.distance < dist - 2 && h.object.userData.structId !== b.st.id && structById[h.object.userData.structId].st.key !== b.st.key) occluded.add(b.st.key);
      }
    }
    for (const b of Object.values(best)) {
      if (occluded.has(b.st.key) && b.d > -1e3) continue;
      const p = b.w.clone().project(camera);
      if (p.z > 1 || Math.abs(p.x) > 1.1 || Math.abs(p.y) > 1.1) continue;
      want.push({ id: b.st.key, x: (p.x + 1) / 2 * innerWidth, y: (1 - p.y) / 2 * innerHeight, st: b.st });
    }
  }
  want.sort((a, b) => a.y - b.y);
  const placed = [];
  for (const L of want) { // nudge down to avoid overlaps
    for (const P of placed) if (Math.abs(P.y - L.y) < 19 && Math.abs(P.x - L.x) < 120) L.y = P.y + 19;
    placed.push(L);
  }
  const keep = new Set();
  for (const L of placed) {
    let el = labelEls[L.id];
    if (!el) {
      el = labelEls[L.id] = document.createElement('div'); el.className = 'lbl';
      const nm = L.st.hemi ? L.st.name.replace(/^(Left|Right) /, '') : L.st.name;
      el.innerHTML = `<i style="background:${L.st.color}"></i>${nm[0].toUpperCase() + nm.slice(1)}`;
      el.onclick = () => selectStruct(L.st.id);
      labelBox.appendChild(el);
    }
    el.style.transform = `translate(${Math.round(L.x)}px, ${Math.round(L.y)}px)`;
    el.style.display = '';
    keep.add(L.id);
  }
  for (const [k, el] of Object.entries(labelEls)) if (!keep.has(k)) el.style.display = 'none';
}
// the example volunteer (read-only, installed with the app) opens like a saved model: case id 'example:<id>'
const caseBase = (cid = currentCase) => (String(cid).startsWith('example:') ? `example/cases/${cid.slice(8)}/` : `cases/${cid}/`);
let noVol = false;
let keepSeq = null;
function clearScene(keepCam) {
  for (const L of Object.values(layers)) for (const it of L.items) centered.remove(it.mesh);
  for (const P of Object.values(planes)) centered.remove(P.mesh);
  for (const k in layers) delete layers[k]; for (const k in planes) delete planes[k];
  for (const k in structById) delete structById[k];
  for (const k in labelEls) { labelEls[k].remove(); delete labelEls[k]; }
  isolateId = null; selected = null; parcels = null; meas.pts = []; redrawMeasure(); endTour(); closeExample();
  keepSeq = keepCam ? seqKind : null; seqKind = null; if (seqTex) { seqTex.dispose(); seqTex = null; }
  if (!keepCam) { for (const k in seqData) delete seqData[k]; }
  zoneVol = null; if (zoneTex) { zoneTex.dispose(); zoneTex = null; }
}
async function loadCase(cid, keepCam) {
  setLoading(true, 'Loading 3D model…', 5); hideWelcome();
  clearScene(keepCam);
  const base = caseBase(cid);
  manifest = await (await fetch(base + 'manifest.json')).json();
  const N = manifest.neck && manifest.neck.status === 'ok' && manifest.structures ? manifest.neck : null, oldPad = neckPad;
  neckPad = N ? N.pad : null;
  if (N) { manifest.structures = [...manifest.structures, ...N.structures]; manifest.structureGroups = { ...manifest.structureGroups, spine: N.group }; manifest.structureInfo = { ...manifest.structureInfo, ...N.structureInfo }; }
  if (keepCam && neckPad !== oldPad) { for (const k in seqData) delete seqData[k]; }   // other scans must be re-padded
  sp = manifest.volume.spacing[0];
  dims = [...manifest.volume.dims]; origin = [...manifest.volume.origin];
  dims[2] += neckPad || 0; origin[2] -= (neckPad || 0) * sp;   // slices continue below the brain scan into the neck
  const vr = await fetch(base + (N ? N.volumeFile : manifest.volume.file)); noVol = !vr.ok;   // the example volunteer ships without picture slices
  vol = noVol ? new Uint8Array(dims[0] * dims[1] * dims[2]) : new Uint8Array(await vr.arrayBuffer());
  document.body.classList.toggle('novol', noVol);
  clearAir(vol, dims);
  centered.position.set(...manifest.brainCenter.map((v) => -v));
  let n = 0;
  await Promise.all(manifest.layers.map(async (def) => {
    const g = await loader.loadAsync(base + def.file);
    const mesh = firstMesh(g);
    mesh.geometry.computeVertexNormals();
    const material = makeMaterial(def, !!mesh.geometry.attributes.color);
    const m = new THREE.Mesh(mesh.geometry, material);
    m.userData.layerId = def.id; m.receiveShadow = true;
    m.renderOrder = { skin: 3, skull: 2, brain: 1 }[def.id] || 0;
    centered.add(m);
    layers[def.id] = { def, items: [{ mesh: m, material }], visible: true, opacity: 1 };
    setLoading(true, `Loading ${def.name}…`, 10 + 40 * (++n / manifest.layers.length));
  }));
  // keep panel order = manifest order
  const ordered = {}; for (const d of manifest.layers) ordered[d.id] = layers[d.id];
  for (const k in layers) delete layers[k]; Object.assign(layers, ordered);
  if (hasStructs()) await loadStructures(base);
  await loadVeins(base);
  buildLayerUI(); buildSeriesUI(); buildStructUI(); buildPlanes();
  await loadLabelVolume(base);
  buildScansUI(); if (keepSeq) await setSeq(keepSeq, { quiet: true });
  if (seqPending().length) setTimeout(pollSequences, 1500);
  for (const L of Object.values(layers)) if (L.hidden) applyOpacity(L);
  // stage placement: floor just under the head
  scene.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject((layers.skin || layers.brain).items[0].mesh);
  for (const it of (layers['g:spine'] || { items: [] }).items) box.expandByObject(it.mesh);   // the neck reaches below the head
  floor.position.y = box.min.y - 1; ring.position.y = box.min.y - 0.5;
  const sz = box.getSize(new THREE.Vector3()), bc = box.getCenter(new THREE.Vector3());
  floor.position.x = ring.position.x = bc.x; floor.position.z = ring.position.z = bc.z;
  ring.scale.setScalar(Math.max(sz.x, sz.z) * 0.62 / 120);
  viewTarget.set(bc.x * 0.5, bc.y * 0.4 + 6, bc.z * 0.5);
  viewRadius = box.getBoundingSphere(new THREE.Sphere()).radius * 2.55;
  setPeel(keepCam ? +$('#peel').value : +(params.get('peel') ?? 1));
  if (!keepCam) { setView(params.get('cam') || '3q', true); setFocusTag(null); }
  updateRegionBar(); updatePerson(); buildExplore();
  offerReskin();
  setLoading(false);
}
let currentCase = null;
async function pollPhase2() {
  const s = await (await fetch('api/state')).json();
  const P = s.phase2 || {};
  if (P.status === 'running') { liveStatus('p2', `Adding detailed structures on this computer… ${Math.round(P.pct)}%`); setTimeout(pollPhase2, 2000); return; }
  liveStatus('p2');
  if (P.status === 'done' && currentCase && !hasStructs()) { await loadCase(currentCase, true); toast('✓ Detailed structures ready'); }
  else if (P.status === 'error') toast(P.msg || 'Adding the detailed structures failed; the basic model still works', 'err', 9000);
}
async function pollProcessing() {
  const s = await (await fetch('api/state')).json();
  if (s.status === 'running') { hideWelcome(); setLoading(true, s.msg, s.pct); showSteps(s.pct); setTimeout(pollProcessing, 600); }
  else if (s.status === 'done' && s.caseId) {
    const fresh = s.caseId !== currentCase || !manifest;
    history.replaceState(null, '', '?case=' + s.caseId); currentCase = s.caseId; await loadCase(s.caseId);
    $('#steps').classList.add('hidden');
    if (firstBuild && fresh) { firstBuild = false; toast('✓ Your 3D model is ready. Detailed structures follow in the background.'); maybeAutoTour(); }
    if (!hasStructs()) pollPhase2();
  }
  else if (s.status === 'error' && firstBuild) { firstBuild = false; setLoading(false); toast(s.error, 'err', 10000); if (!manifest) showWelcome(); }
  return s;
}
let firstBuild = false;
$('#btnReset').onclick = resetView;
$('#btnAuto').onclick = (e) => { controls.autoRotate = !controls.autoRotate; e.target.classList.toggle('on', controls.autoRotate); };
$('#cutaway').onchange = updateClipping;
$('#keepDeep').onchange = () => { updateClipping(); updateOverlayUI(); };
$('#lblToggle').checked = labelsOn; $('#lblToggle').onchange = (e) => { labelsOn = e.target.checked; };
$('#lobeColors').onchange = (e) => setRegionColors(e.target.checked);
$('#contrast').oninput = (e) => { if (cmp.open) setTimeout(drawCompare); contrast = +e.target.value; sliceCache.clear(); for (const P of Object.values(planes)) P.mat.uniforms.uContrast.value = contrast; drawInsets(); };

// ---------- help / cheat sheet ("?" key or the ? button)
function toggleHelp(on) { const h = $('#help'); h.classList.toggle('hidden', on === undefined ? !h.classList.contains('hidden') : !on); }
$('#btnHelp').onclick = () => toggleHelp();
$('#helpClose').onclick = () => toggleHelp(false);
addEventListener('keydown', (e) => {
  if (e.target.matches('input, textarea')) return;
  if (e.key === '?') toggleHelp();
  else if (tour.on && (e.key === 'ArrowRight' || e.key === 'ArrowLeft')) tourGo(e.key === 'ArrowRight' ? 1 : -1);
  else if (tour.on && e.key === 'Escape') endTour();
  else if (e.key === 'g' || e.key === 'G') (tour.on ? endTour() : startTour());
  else if (e.key === 'e' || e.key === 'E') openExport();
  else if (e.key === 'x' || e.key === 'X') (ex.on ? closeExample() : openExample('side'));
  else if (ex.on && e.key === 'Escape') closeExample();
  else if ((e.key === 'm' || e.key === 'M')) setMeasure();
  else if ((e.key === 'v' || e.key === 'V') && layers.veins) toggleVeins();
  else if (e.key === 'Escape' && meas.on) setMeasure(false);
  else if (e.key === 'Escape' && !$('#export').classList.contains('hidden')) $('#export').classList.add('hidden');
  else if (e.key === 'Escape') { if (cmp.open) closeCompare(); else if (!$('#focusTag').classList.contains('hidden')) resetView(); else { toggleHelp(false); hideInfo(); } }
  else if ((e.key === 's' || e.key === 'S') && sequences().length) cycleSeq(e.shiftKey ? -1 : 1);
  else if ((e.key === 'c' || e.key === 'C') && sequences().length) (cmp.open ? closeCompare() : openCompare());
  else if (e.key === 'r' || e.key === 'R') resetView();
  else if (e.key === 'f' || e.key === 'F') focusSelected();
  else if (e.key === '0') fitAll();
  else if (e.key === 't' || e.key === 'T') $('#btnAuto').click();
  else if (e.key === 'l' || e.key === 'L') { const c = $('#lblToggle'); c.checked = !c.checked; labelsOn = c.checked; }
  else if (e.key === 'o' || e.key === 'O') { const c = $('#labelOverlay'); if (!c.disabled) { c.checked = !c.checked; updateOverlayUI(); } }
  else if (/^[1-5]$/.test(e.key)) setPeel(+e.key - 1);
  else if (['ArrowUp', 'ArrowDown', 'PageUp', 'PageDown'].includes(e.key)) {
    const n = cmp.open ? cmp.plane : activeInset || 'axial', st = (e.key.startsWith('Page') ? 10 : 1) * (e.key.endsWith('Up') ? 1 : -1);
    stepSlice(n, st); e.preventDefault();
  }
});
let activeInset = null;
for (const n of ['axial', 'coronal', 'sagittal']) document.getElementById('c-' + n).addEventListener('pointerenter', () => (activeInset = n));

addEventListener('resize', () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });

// ---------- loop
function tick(now) {
  requestAnimationFrame(tick); // schedule first: an exception in one frame must not freeze rotate/zoom for good
  if (camAnim) {
    const A = camAnim, t = Math.min(1, (now - A.t0) / A.dur), e = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    controls.target.lerpVectors(A.tFrom, A.tTo, e);
    const q = new THREE.Quaternion().slerp(A.q, e);   // identity -> full rotation
    camera.position.copy(A.d0.clone().applyQuaternion(q).multiplyScalar(A.r0 + (A.r1 - A.r0) * e).add(controls.target));
    if (t >= 1) camAnim = null;
  }
  for (const L of Object.values(layers)) for (const it of L.items) {
    const m = it.material; if (!m.emissive || m.userData.fluid) continue;
    if (selected && selected.items.includes(it)) { const k = 0.5 + 0.5 * Math.sin((now - selected.t) / 260); m.emissive.setRGB(0.06 * k, 0.09 * k, 0.14 * k); }
    else m.emissive.setRGB(0, 0, 0);
  }
  updateLabels(now); updateMeasLabel(); updateExample();
  controls.update();
  renderer.render(scene, camera);
}
requestAnimationFrame(tick);
// ---------- Ask AI (off until the user picks Private or ChatGPT; the page only talks to the local server)
const LS = { get: (k) => localStorage.getItem('bg.' + k), set: (k, v) => localStorage.setItem('bg.' + k, v), del: (k) => localStorage.removeItem('bg.' + k) };
const escH = (t) => String(t).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const postJ = (p, body) => fetch(p, { method: 'POST', body: JSON.stringify(body || {}) }).then((r) => r.json());
const ai = { state: null, chat: null, extra: [], removed: new Set(), ctrl: null, poll: null, cgModels: null, lastReq: null };
const fmtMin = (s) => (s == null ? '' : s < 90 ? 'about a minute left' : `about ${Math.round(s / 60)} min left`);
function measText() { return meas.pts.length === 2 ? `${meas.pts[0].distanceTo(meas.pts[1]).toFixed(1)} mm` : null; }
function onScreen() { return Object.values(layers).filter((L) => !L.hidden && L.visible && L.items.some((it) => it.mesh.visible && it.material.opacity > 0.15)).map((L) => L.def.name); }
// what the AI will see: chips from the screen (removable) + what the user added (pictures, report)
function autoChips() {
  const out = [];
  if (aiSel) out.push({ key: 'part:' + aiSel.name, icon: '◉', label: aiSel.name + (aiSel.volumeMl ? ` · ${fmtMl(aiSel.volumeMl)} mL` : ''), part: aiSel });
  const m = measText(); if (m) out.push({ key: 'meas:' + m, icon: '📏', label: 'Distance ' + m, meas: `Distance ${m} (between two points I clicked)` });
  const v = manifest ? onScreen() : []; if (v.length) out.push({ key: 'view:' + v.join(), icon: '👁', label: v.length > 2 ? `On screen: ${v.length} groups of parts` : 'On screen: ' + v.join(', '), title: v.join(', '), view: v });
  return out.filter((c) => !ai.removed.has(c.key));
}
const chips = () => [...autoChips(), ...ai.extra];
function aiCtxChanged() { if (!$('#chat').classList.contains('hidden')) { renderChips(); renderSuggest(); } }
function renderChips() {
  $('#chips').innerHTML = '';
  for (const c of chips()) {
    const el = document.createElement('span'); el.className = 'chip ctxchip' + (c.warn ? ' warn' : ''); el.title = c.title || c.label;
    el.innerHTML = (c.image ? `<img src="${c.image}" alt="">` : `<span>${c.icon}</span>`) + `<span>${escH(c.label)}${c.warn ? ' <span class="warn">⚠ ' + escH(c.warn) + '</span>' : ''}</span><button class="rm" title="Don't send this">×</button>`;
    el.querySelector('.rm').onclick = () => { if (ai.extra.includes(c)) ai.extra.splice(ai.extra.indexOf(c), 1); else ai.removed.add(c.key); renderChips(); };
    $('#chips').appendChild(el);
  }
  const cloud = curBackend() === 'chatgpt', n = chips().length;
  $('#sentLine').innerHTML = cloud ? `☁ Sent to OpenAI${n ? ' with the items above' : ''} · <button class="link" id="seeExact">See exact text</button>`
    : `🔒 Private: everything stays on this Mac` + swapHint();
  $('#seeExact')?.addEventListener('click', showExact);
  $('#getVision')?.addEventListener('click', () => runFix('image_model'));
}
function swapHint() {   // say before sending, not after, when a picture goes to another model
  const m = activeLocal(); if (!chips().some((c) => c.image) || !m || m.vision) return '';
  const v = localReady().find((x) => x.vision);
  return v ? `<br>🖼 ${escH(m.name)} reads text only, so ${escH(v.name)} answers this one.` : `<br>🖼 ${escH(m.name)} reads text only. <button class="link" id="getVision">Add the picture-reading model</button>`;
}
function aiCtx() {
  const ctx = {}, cs = chips();
  const parts = cs.filter((c) => c.part).map((c) => c.part); if (parts.length) ctx.structures = parts;
  const ms = cs.filter((c) => c.meas).map((c) => c.meas); if (ms.length) ctx.measurements = ms;
  const v = cs.find((c) => c.view); if (v) ctx.view = v.view;
  const r = cs.find((c) => c.report); if (r) ctx.findings = r.report;
  return { ctx, image: (cs.find((c) => c.image) || {}).image || null };
}
// models: 'local:<id>' or 'chatgpt'; capability icons in the switcher
const curModel = () => LS.get('aiModel') || '';
const curBackend = () => (curModel().startsWith('chatgpt') ? 'chatgpt' : 'local');
const localReady = () => (ai.state ? ai.state.models.filter((m) => m.usable) : []);
const cgReady = () => ai.state && ai.state.chatgpt.signedIn && ai.state.chatgpt.sharing;
function activeLocal() { const r = localReady(); return r.find((m) => 'local:' + m.id === curModel()) || r[0]; }
function modelLabel() {
  if (curBackend() === 'chatgpt') return '☁ ChatGPT' + (ai.cgModels && LS.get('cgModel') ? ' · ' + (ai.cgModels.find((m) => m.slug === LS.get('cgModel')) || {}).name : '');
  const m = activeLocal(); return m ? `🔒 ${m.name}` : '🔒 Private';
}
async function aiLoad() { ai.state = await fetch('api/ai/state').then((r) => r.json()); return ai.state; }
async function openChat(focus = true) {
  $('#chat').classList.remove('hidden'); document.body.classList.add('chatopen');
  await aiLoad(); aiRender(); if (focus && !$('#composer').classList.contains('hidden')) $('#chatQ').focus();
}
function closeChat() { $('#chat').classList.add('hidden'); document.body.classList.remove('chatopen'); $('#modelMenu').classList.add('hidden'); }
function aiRender() {
  const st = ai.state; if (!st) return;
  if (curBackend() === 'chatgpt' && !cgReady()) LS.del('aiModel');
  if (curBackend() === 'local' && !localReady().length && cgReady()) LS.set('aiModel', 'chatgpt');
  const ready = localReady().length || cgReady(), consent = curBackend() !== 'chatgpt' || LS.get('cgConsent') === '1';
  $('#aiWelcome').classList.toggle('hidden', !!ready);
  $('#cgConsent').classList.toggle('hidden', !ready || consent);
  $('#chatBody').classList.toggle('hidden', !ready);   // the chat stays visible while ChatGPT asks for consent
  $('#composer').classList.toggle('hidden', !ready || !consent || !$('#reportBox').classList.contains('hidden'));
  for (const s of ['#modelBtn', '#chatHist', '#chatNew']) $(s).classList.toggle('hidden', !ready);
  $('#modelBtn').textContent = modelLabel() + ' ▾'; $('#modelBtn').classList.toggle('cloud', curBackend() === 'chatgpt');
  if (!ready) renderWelcome(); else { renderLog(); renderChips(); renderSuggest(); }
  renderAskButton();
}
function renderAskButton() {
  const s = ai.state && ai.state.setup, busy = s && ['runtime', 'starting', 'download', 'convert'].includes(s.step);
  $('#btnChat').innerHTML = '✦ Ask AI' + (busy ? `<span class="pct">${s.step === 'download' ? s.pct + '%' : '…'}</span>` : '');
  clearTimeout(ai.poll);
  if (busy || (ai.state && ai.state.chatgpt.pending)) ai.poll = setTimeout(async () => { const was = localReady().length; await aiLoad(); if (!was && localReady().length) { toast('Private AI is ready. Ask anything about your scan.'); if (!LS.get('aiModel')) LS.set('aiModel', 'local:' + localReady()[0].id); } aiRender(); }, 1500);
}
function renderWelcome() {
  const s = ai.state.setup, p = s.plan, cg = ai.state.chatgpt;
  const what = p.vision ? 'reads text and pictures' : 'reads text (pictures need a Mac with Apple silicon)';
  $('#tlDesc').innerHTML = `Free, and nothing you ask leaves this Mac. Downloads <b>${escH(p.name)}</b> once (${p.downloadGB} GB${p.runtime === 'mlx' ? `, kept as ${p.diskGB} GB` : ''}), picked for this Mac's ${p.ramGB} GB of memory. It ${what}.`;
  const bar = (pct, msg) => `<div class="prog"><i style="width:${pct}%"></i></div><div class="small">${msg}</div><div class="small muted">You can close this panel or keep exploring; setup carries on.</div>`;
  let prog = '', btn = '<button id="btnSetup">Set up</button>';
  if (s.step === 'starting' || s.step === 'runtime') { prog = bar(2, 'Adding the AI engine to BodyGlass…'); btn = ''; }
  else if (s.step === 'download') { prog = bar(s.pct, `Downloading… ${s.doneGB} of ${s.totalGB} GB${s.etaS ? ' · ' + fmtMin(s.etaS) : ''}`); btn = '<button id="btnPause" class="ghost">Pause</button>'; }
  else if (s.step === 'convert') { prog = bar(100, `Preparing the model on this Mac${s.etaS ? ' · ' + fmtMin(s.etaS) : '…'}`); btn = ''; }
  else if (s.step === 'paused') { prog = bar(s.pct, `Paused at ${s.doneGB} of ${s.totalGB} GB`); btn = '<button id="btnSetup">Resume</button>'; }
  else if (s.step === 'error') { prog = `<div class="small err">${escH(s.error || 'Setup stopped.')}</div>`; btn = s.fix === 'community' ? `<button id="btnCommunity">Use the smaller copy (${p.prebuiltGB} GB)</button>` : '<button id="btnSetup">Try again</button>'; }
  $('#tlProg').innerHTML = prog; $('#tileLocal .tilebtns').innerHTML = btn;
  $('#btnSetup')?.addEventListener('click', async () => { await postJ('api/ai/setup'); await aiLoad(); aiRender(); });
  $('#btnCommunity')?.addEventListener('click', async () => { await postJ('api/ai/setup', { variant: 'prebuilt' }); await aiLoad(); aiRender(); });
  $('#btnPause')?.addEventListener('click', async () => { await postJ('api/ai/models/pause', { id: p.id }); setTimeout(async () => { await aiLoad(); aiRender(); }, 600); });
  $('#tcMsg').innerHTML = cg.pending ? '<span class="muted">Finish signing in in your browser…</span>' : cg.signedIn && !cg.sharing ? '<span class="err">You signed in but didn\'t allow BodyGlass to use your plan. Sign in again and allow it.</span>' : cg.error && ai.cgTried ? `<span class="err">${escH(cg.error.startsWith('Keychain') ? 'BodyGlass could not use the macOS Keychain to keep your sign-in. ' + cg.error : cg.error)}</span>` : '';
  $('#btnCgIn').textContent = cg.pending ? 'Waiting for the browser…' : 'Sign in with ChatGPT';
}
async function cgSignIn() { ai.cgTried = true; const r = await postJ('api/ai/signin'); if (r.error) toast(r.error, 'err'); LS.set('aiModel', 'chatgpt'); await aiLoad(); aiRender(); }
$('#btnCgIn').onclick = cgSignIn;
$('#cgOk').onclick = () => { LS.set('cgConsent', '1'); aiRender(); $('#chatQ').focus(); };
$('#cgNo').onclick = () => { LS.set('aiModel', localReady().length ? 'local:' + localReady()[0].id : ''); aiRender(); if (!localReady().length) { $('#aiWelcome').classList.remove('hidden'); renderWelcome(); } };
// model switcher
async function openModelMenu() {
  const st = await aiLoad(), cur = curModel(), menu = $('#modelMenu'); menu.innerHTML = '';
  const item = (key, title, sub, on, fn) => { const b = document.createElement('button'); b.className = 'mi' + (on ? ' on' : ''); b.innerHTML = `<span>${title}<small>${sub}</small></span>`; b.onclick = fn; menu.appendChild(b); };
  const pick = (k) => () => { LS.set('aiModel', k); menu.classList.add('hidden'); aiRender(); };
  for (const m of localReady()) item(m.id, `🔒 ${escH(m.name)}`, (m.vision ? '📝 text · 🖼 pictures' : '📝 text only') + (m.medical ? ' · medical-tuned' : '') + ' · private, offline', curBackend() === 'local' && activeLocal() === m, pick('local:' + m.id));
  if (cgReady()) {
    if (!ai.cgModels) { try { ai.cgModels = (await fetch('api/ai/chatgpt-models').then((r) => r.json())).models || []; } catch (e) { ai.cgModels = []; } }
    for (const m of ai.cgModels.slice(0, 4)) item(m.slug, `☁ ChatGPT · ${escH(m.name)}`, '📝 text · 🖼 pictures · cloud (OpenAI)', curBackend() === 'chatgpt' && (LS.get('cgModel') || ai.cgModels[0].slug) === m.slug, () => { LS.set('cgModel', m.slug); pick('chatgpt')(); });
  } else item('cg', '☁ Use my ChatGPT…', 'Sign in with your ChatGPT plan · cloud', false, () => { menu.classList.add('hidden'); cgSignIn(); });
  if (!localReady().length) item('setup', '🔒 Set up the private model…', 'Free, offline after a one-time download', false, () => { menu.classList.add('hidden'); LS.del('aiModel'); $('#aiWelcome').classList.remove('hidden'); renderWelcome(); });
  menu.appendChild(document.createElement('hr'));
  item('manage', 'Manage models…', 'Other models, 8-bit, delete', false, () => { menu.classList.add('hidden'); openAiModels(); });
  if (st.chatgpt.signedIn) item('out', 'Sign out of ChatGPT', escH(st.chatgpt.email || ''), false, async () => { menu.classList.add('hidden'); const r = await postJ('api/ai/signout'); ai.cgModels = null; toast(r.revoked ? 'Signed out of ChatGPT' : 'Signed out here; OpenAI did not confirm. You can also disconnect BodyGlass in ChatGPT Settings.'); await aiLoad(); aiRender(); });
  menu.classList.remove('hidden');
}
$('#modelBtn').onclick = (e) => { e.stopPropagation(); $('#modelMenu').classList.contains('hidden') ? openModelMenu() : $('#modelMenu').classList.add('hidden'); };
document.addEventListener('click', (e) => { if (!e.target.closest('#modelMenu, #modelBtn')) $('#modelMenu').classList.add('hidden'); if (!e.target.closest('#attachMenu, #attachBtn')) $('#attachMenu').classList.add('hidden'); });
// chats, saved per person on this computer (localStorage of this local page; nothing leaves the Mac)
// shortcut: localStorage is per browser + port; move to files in Application Support if people switch browsers
const chatKey = () => 'chats.' + (currentCase || 'none');
const allChats = () => { try { return JSON.parse(LS.get(chatKey()) || '[]'); } catch (e) { return []; } };
function saveChat() { if (!ai.chat || !ai.chat.msgs.length) return; const all = allChats().filter((c) => c.id !== ai.chat.id); all.unshift(ai.chat); LS.set(chatKey(), JSON.stringify(all.slice(0, 50))); }
function newChat() { ai.chat = null; ai.extra = []; $('#histList').classList.add('hidden'); aiRender(); $('#chatQ').focus(); }
$('#chatNew').onclick = newChat;
$('#chatHist').onclick = () => {
  const h = $('#histList'); if (!h.classList.contains('hidden')) return h.classList.add('hidden');
  const all = allChats();
  h.innerHTML = all.length ? '' : '<p class="small muted">No saved chats for this person yet.</p>';
  for (const c of all) {
    const r = document.createElement('div'); r.className = 'hrow';
    r.innerHTML = `<span class="t">${escH(c.title)}</span><small>${escH(c.model || '')} · ${new Date(c.at).toLocaleDateString()}</small><button class="ghost icon" title="Delete this chat">🗑</button>`;
    r.onclick = (e) => { if (e.target.closest('button')) { LS.set(chatKey(), JSON.stringify(allChats().filter((x) => x.id !== c.id))); if (ai.chat && ai.chat.id === c.id) ai.chat = null; h.classList.add('hidden'); $('#chatHist').click(); aiRender(); return; } ai.chat = c; h.classList.add('hidden'); aiRender(); };
    h.appendChild(r);
  }
  h.classList.remove('hidden');
};
// answers: thinking kept apart, part names become links into the 3D view
function partTerms() {
  const t = new Map();
  for (const S of Object.values(structById)) {
    const n = S.st.name.toLowerCase(), base = n.replace(/^(left|right) /, '');
    t.set(n, [S.st.id]); if (base !== n) t.set(base, [...(t.get(base) || []), S.st.id]);
  }
  const all = (w) => Object.values(structById).filter((S) => S.st.name.toLowerCase().includes(w)).map((S) => S.st.id);
  const alias = [['hippocampi', 'hippocampus'], ['hippocampal', 'hippocampus'], ['thalami', 'thalamus'], ['amygdalae', 'amygdala'], ['cerebellar', 'cerebellum']];
  for (const [word, one] of alias) if (t.has(one)) t.set(word, t.get(one));   // plurals and adjectives reports use
  for (const w of ['ventricle', 'white matter']) { const ids = all(w); if (ids.length && !t.has(w)) t.set(w, ids); }
  return t;
}
function linkParts(html) {
  const t = partTerms(); if (!t.size) return html;
  const re = new RegExp(`\\b(${[...t.keys()].sort((a, b) => b.length - a.length).map((k) => k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})(es|s)?\\b`, 'gi');
  return html.replace(re, (m, k) => `<a class="part" data-ids="${t.get(k.toLowerCase()).join(',')}">${m}</a>`);
}
function mentioned(text) { const out = new Map(); linkParts(escH(text)).replace(/data-ids="([^"]+)">([^<]+)/g, (m, ids, w) => out.set(ids, w)); return [...out]; }
function focusParts(ids) {
  const items = ids.map((id) => structById[id]).filter(Boolean); if (!items.length) return;
  for (const S of items) if (!S.st.visible) { S.st.visible = true; applyOpacity(S.L); }
  if (items.length === 1 || new Set(items.map((S) => S.st.name.replace(/^(Left|Right) /, ''))).size === 1) {
    selectStruct(items[0].st.id); return focusOn(items.map((S) => S.item), items.length > 1 ? items[0].st.name.replace(/^(Left|Right) /, '') : items[0].st.name);
  }
  hideInfo(); focusOn(items.map((S) => S.item), `${items.length} parts from your report`);   // several parts: frame them all
}
$('#chatLog').addEventListener('click', (e) => { const a = e.target.closest('a.part'); if (a) focusParts(a.dataset.ids.split(',')); });
function splitThink(raw) {
  const i = raw.indexOf('</think>');
  if (i >= 0) return [raw.slice(i + 8).trim(), raw.slice(0, i).replace('<think>', '').trim(), false];
  if (raw.trimStart().startsWith('<think>')) return ['', raw.replace('<think>', '').trim(), true];
  return [raw, '', false];
}
const fmtAnswer = (t) => linkParts(escH(t)).replace(/\*\*(.+?)\*\*/g, '<b>$1</b>').replace(/(^|[\s(])\*([^*\n]+)\*/g, '$1<i>$2</i>')
  .replace(/^#{1,4} (.+)$/gm, '<b>$1</b>').replace(/^\s*[-*] /gm, '• ');
const FIX = { setup: 'Set up the private model', image_model: 'Add the picture-reading model', lighter: 'Use the lighter model', signin: 'Sign in to ChatGPT again', retry: 'Try again', usage: 'See ChatGPT usage', community: 'Use the smaller copy' };
function msgEl(m, live) {
  const el = document.createElement('div'); el.className = 'msg ' + (m.role === 'user' ? 'u' : 'a') + (m.error ? ' err' : '');
  if (m.role === 'user') {
    el.innerHTML = escH(m.text) + (m.ctx ? `<span class="ctx">${escH(m.ctx)}</span>` : '') + (m.thumb ? `<img class="thumb" src="${m.thumb}" alt="">` : '') +
      (m.mentions && m.mentions.length ? `<span class="ctx">In your report: ${m.mentions.map(([ids, w]) => `<a class="part" data-ids="${ids}">${escH(w)}</a>`).join(', ')}</span>` : '');
    return el;
  }
  if (m.error) { el.innerHTML = `${escH(m.text)}${m.fix && FIX[m.fix] ? `<div class="fix"><button class="small">${FIX[m.fix]}</button></div>` : ''}`; el.querySelector('.fix button')?.addEventListener('click', () => runFix(m.fix)); return el; }
  const [ans, think, thinking] = splitThink(m.text);
  el.innerHTML = `<span class="by">${escH(m.model || 'AI')}${m.note ? ' · ' + escH(m.note) : ''}</span>` +
    (think ? `<details><summary>${thinking ? 'Thinking it through…' : 'How it reasoned'}</summary>${escH(think)}</details>` : '') +
    `<span class="${live ? 'caret' : ''}">${fmtAnswer(ans)}</span>` + (m.stopped ? ' <span class="muted small">(stopped)</span>' : '') + (m.caveat ? `<div class="caveat">⚠ ${escH(m.caveat)}</div>` : '') +
    (!live ? '<div class="tools"><button title="Copy the answer">Copy</button></div>' : '');
  el.querySelector('.tools button')?.addEventListener('click', () => { navigator.clipboard.writeText(ans).then(() => toast('Copied'), () => toast('Copy not allowed here', 'err')); });
  return el;
}
function renderLog() {
  const log = $('#chatLog'); log.innerHTML = '';
  for (const m of (ai.chat ? ai.chat.msgs : [])) log.appendChild(msgEl(m));
  $('#chatBody').scrollTop = 1e9;
}
function renderSuggest() {
  const box = $('#suggest'); box.innerHTML = '';
  if (ai.chat && ai.chat.msgs.length) return;
  const n = aiSel && aiSel.name, nl = n && n[0].toLowerCase() + n.slice(1);
  const qs = n ? [`What does the ${nl} do?`, ...(aiSel.volumeMl ? [`How is the size of the ${nl} measured?`] : []), `What can change the size of the ${nl} over a lifetime?`]
    : ['What am I looking at in this 3D view?', 'What are the main parts of the brain and what do they do?', 'How does an MRI scan see inside the body?'];
  box.innerHTML = `<div class="hint">${n ? `Ask about the ${escH(nl)}` : 'Try asking'}</div>`;
  for (const q of qs) { const b = document.createElement('button'); b.className = 'sq'; b.textContent = q; b.onclick = () => send(q); box.appendChild(b); }
  const r = document.createElement('button'); r.className = 'sq'; r.textContent = '📄 Explain my radiology report in plain words…'; r.onclick = openReport; box.appendChild(r);
}
// sending, streamed; Stop aborts the request and the server stops generating
async function send(text, opts = {}) {
  text = (text ?? $('#chatQ').value).trim(); if (!text || ai.ctrl) return;
  if (curBackend() === 'chatgpt' && LS.get('cgConsent') !== '1') return aiRender();
  const { ctx, image } = aiCtx(), cs = chips();
  if (!ai.chat) ai.chat = { id: Date.now().toString(36), title: text.slice(0, 60), at: Date.now(), model: modelLabel().replace(/^[🔒☁] /, ''), msgs: [] };
  const history = ai.chat.msgs.filter((m) => !m.error).map((m) => ({ role: m.role, text: m.role === 'user' ? m.text : splitThink(m.text)[0] }));
  const req = { backend: curBackend(), model: curBackend() === 'chatgpt' ? LS.get('cgModel') : (activeLocal() || {}).id, ctx, question: text, history, image };
  ai.lastReq = req;
  const um = { role: 'user', text, ctx: cs.filter((c) => !c.image).map((c) => c.label).join(' · '), thumb: image, mentions: opts.mentions };
  ai.chat.msgs.push(um); $('#chatQ').value = ''; autoGrow();
  ai.extra = ai.extra.filter((c) => !c.image && !c.report);   // a picture / report goes with one message
  renderLog(); renderChips(); renderSuggest();
  const am = { role: 'assistant', text: '', model: '' }; ai.chat.msgs.push(am);
  const el = msgEl(am, true); $('#chatLog').appendChild(el); $('#chatBody').scrollTop = 1e9;
  el.querySelector('.caret').insertAdjacentHTML('beforebegin', `<span class="muted small">${req.backend === 'chatgpt' ? 'Asking ChatGPT…' : 'Thinking on this Mac… (the first answer loads the model)'}</span>`);
  ai.ctrl = new AbortController(); setSending(true);
  let buf = '', cur = el;
  try {
    const r = await fetch('api/ai/chat', { method: 'POST', body: JSON.stringify(req), signal: ai.ctrl.signal });
    if (!r.ok) { const j = await r.json().catch(() => ({})); throw Object.assign(new Error(j.error || 'The AI could not answer.'), { fix: j.fix || 'retry' }); }
    const rd = r.body.getReader(), dec = new TextDecoder();
    for (;;) {
      const { value, done } = await rd.read(); if (done) break;
      buf += dec.decode(value, { stream: true });
      let i; while ((i = buf.indexOf('\n')) >= 0) {
        const ev = JSON.parse(buf.slice(0, i)); buf = buf.slice(i + 1);
        if (ev.error) throw Object.assign(new Error(ev.error), { fix: ev.fix });
        if (ev.model) { am.model = ev.model; am.note = ev.note; ai.chat.model = ev.model; }
        if (ev.t) am.text += ev.t;
        if (ev.caveat) am.caveat = ev.caveat;
      }
      const n = msgEl(am, true); cur.replaceWith(n); cur = n; $('#chatBody').scrollTop = 1e9;
    }
  } catch (e) {
    if (e.name === 'AbortError') am.stopped = true;
    else if (!am.text) Object.assign(am, { error: true, text: e.message === 'Failed to fetch' ? 'BodyGlass lost contact with its own server. Is the BodyGlass window still open?' : e.message, fix: e.fix || 'retry' });
    else Object.assign(am, { stopped: true, note: (am.note ? am.note + ' · ' : '') + e.message });
  } finally {
    ai.ctrl = null; setSending(false);
    const n = msgEl(am); cur.replaceWith(n); $('#chatBody').scrollTop = 1e9;
    saveChat();
  }
}
function setSending(on) { const b = $('#sendBtn'); b.classList.toggle('stop', on); b.textContent = on ? '■' : '↑'; b.title = on ? 'Stop' : 'Send (Return)'; }
$('#sendBtn').onclick = () => (ai.ctrl ? ai.ctrl.abort() : send());
function autoGrow() { const q = $('#chatQ'); q.style.height = 'auto'; q.style.height = Math.min(120, q.scrollHeight) + 'px'; }
$('#chatQ').addEventListener('input', autoGrow);
$('#chatQ').addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); } });
$('#chatQ').addEventListener('paste', (e) => {   // a pasted report becomes a report chip, ready to explain
  const t = e.clipboardData.getData('text'); if (t.length < 300) return;
  e.preventDefault(); addReport(t); if (!$('#chatQ').value.trim()) { $('#chatQ').value = 'Explain my report in plain words.'; autoGrow(); }
});
async function runFix(fix) {
  if (fix === 'retry' && ai.lastReq) { const q = ai.lastReq.question; ai.chat.msgs.splice(-2, 2); renderLog(); return send(q); }
  if (fix === 'signin') return cgSignIn();
  if (fix === 'usage') return window.open(ai.state.chatgpt.usageUrl, '_blank', 'noopener');
  if (fix === 'setup') { LS.del('aiModel'); $('#aiWelcome').classList.remove('hidden'); $('#chatBody').classList.add('hidden'); $('#composer').classList.add('hidden'); return renderWelcome(); }
  if (fix === 'lighter') { const l = localReady().find((m) => m.id === 'qwen3.5-4b'); if (l) { LS.set('aiModel', 'local:qwen3.5-4b'); toast('Switched to Qwen3.5-4B'); return aiRender(); } return openAiModels(); }
  if (fix === 'image_model' && ai.state.setup.plan.runtime !== 'mlx') return toast('Reading pictures on this computer needs a Mac with Apple silicon. ChatGPT (menu at the top) can read them.', '', 7000);
  if (fix === 'image_model') { const id = ai.state.setup.plan.vision ? ai.state.setup.plan.id : 'qwen3.5-4b'; await postJ('api/ai/models/download', { id, variant: '4' }); toast('Downloading the picture-reading model. Progress: Manage models.'); return openAiModels(); }
}
function showExact() { postJ('api/ai/preview', { ...aiCtx(), question: $('#chatQ').value || '(your question)', history: [] }).then((r) => { $('#exactText').textContent = r.preview || r.error; $('#exactBox').classList.remove('hidden'); }); }
$('#exactClose').onclick = () => $('#exactBox').classList.add('hidden');
// pictures and reports
function toPng(src, warn, label) {
  const img = new Image();
  img.onload = () => {
    const k = Math.min(1, 1024 / Math.max(img.width, img.height)), c = document.createElement('canvas');
    c.width = Math.round(img.width * k); c.height = Math.round(img.height * k);
    c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
    ai.extra = ai.extra.filter((x) => !x.image);   // one picture per message
    ai.extra.push({ key: 'img', icon: '🖼', label, image: c.toDataURL('image/png'), warn });
    renderChips(); $('#chatQ').focus();
  };
  img.onerror = () => toast('That file is not a picture BodyGlass can read', 'err');
  img.src = src;
}
const skinShown = () => layers.skin && layers.skin.items.some((it) => it.mesh.visible && it.material.opacity > 0.05);
function attach(a) {
  $('#attachMenu').classList.add('hidden');
  if (['axial', 'coronal', 'sagittal'].includes(a)) {
    if (document.body.classList.contains('novol')) return toast('This model has no slice pictures', 'err');
    return toPng($('#c-' + a).toDataURL('image/png'), a === 'sagittal' ? 'may show your face' : null, `${a[0].toUpperCase() + a.slice(1)} slice`);
  }
  if (a === 'snap') { renderer.render(scene, camera); return toPng(canvas.toDataURL('image/png'), skinShown() ? 'skin is on: may show your face' : null, '3D view'); }
  if (a === 'file') return $('#imgFile').click();
  if (a === 'report') return openReport();
}
$('#attachBtn').onclick = (e) => { e.stopPropagation(); $('#attachMenu').classList.toggle('hidden'); };
document.querySelectorAll('#attachMenu .mi').forEach((b) => (b.onclick = () => attach(b.dataset.a)));
function readFile(f) {
  if (f.type.startsWith('image/')) { const r = new FileReader(); r.onload = () => toPng(r.result, null, f.name.slice(0, 30)); r.readAsDataURL(f); return; }
  if (f.type.startsWith('text/') || /\.txt$/i.test(f.name)) { f.text().then((t) => { openReport(); $('#reportText').value = t.slice(0, 12000); }); return; }
  toast('Drop a picture (PNG or JPEG) or a text file with your report', 'err');
}
$('#imgFile').onchange = (e) => { if (e.target.files[0]) readFile(e.target.files[0]); e.target.value = ''; };
$('#chat').addEventListener('dragover', (e) => { e.preventDefault(); $('#chat').classList.add('drop'); });
$('#chat').addEventListener('dragleave', (e) => { if (!e.relatedTarget || !$('#chat').contains(e.relatedTarget)) $('#chat').classList.remove('drop'); });
$('#chat').addEventListener('drop', (e) => { e.preventDefault(); $('#chat').classList.remove('drop'); if (e.dataTransfer.files[0]) readFile(e.dataTransfer.files[0]); else { const t = e.dataTransfer.getData('text'); if (t) { openReport(); $('#reportText').value = t; } } });
function openReport() { $('#reportBox').classList.remove('hidden'); $('#composer').classList.add('hidden'); $('#reportText').focus(); }
function closeReport() { $('#reportBox').classList.add('hidden'); aiRender(); }
function addReport(t) { ai.extra = ai.extra.filter((x) => !x.report); ai.extra.push({ key: 'report', icon: '📄', label: `My report (${t.trim().split(/\s+/).length} words)`, report: t.trim() }); renderChips(); }
$('#reportCancel').onclick = closeReport;
$('#reportGo').onclick = () => {
  const t = $('#reportText').value.trim(); if (!t) return $('#reportText').focus();
  addReport(t); $('#reportText').value = ''; closeReport();
  const ms = mentioned(t); if (ms.length) focusParts(ms.flatMap(([ids]) => ids.split(',')));
  send('Explain my report in plain words: what each finding means, and which parts of my body it is about.', { mentions: ms });
};
// Manage models (secondary screen)
function gb(x) { return x == null ? '?' : `${x} GB`; }
async function openAiModels() {
  $('#aiModels').classList.remove('hidden');
  const st = ai.state = await aiLoad();
  $('#aimPath').textContent = st.modelsDir;
  $('#aimList').innerHTML = '';
  let busy = false;
  for (const m of st.models) {
    if (m.runtime === 'llama_cpp' && st.setup.plan.runtime === 'mlx' && m.state.status !== 'ready') continue;   // the Intel fallback only where MLX can't run
    const s = m.state, el = document.createElement('div'); el.className = 'mcard';
    const pct = s.total ? Math.floor(100 * (s.done || 0) / s.total) : 0;
    busy ||= s.status === 'downloading' || s.status === 'converting';
    el.innerHTML = `<h4>${escH(m.name)} <span class="small muted">${m.vision ? '📝 text · 🖼 pictures' : '📝 text only'}${m.medical ? ' · medical-tuned' : ''}</span></h4>
      <div>${escH(m.purpose)}</div>
      <div class="facts">Download ${gb(m.downloadGB)} · keeps ${m.bits.map((b) => `${gb(m.diskGB[b])}${m.bits.length > 1 ? ` (${b}-bit)` : ''}`).join(' or ')} · uses ${m.bits.map((b) => gb(m.ramGB[b])).join(' / ')} of memory while answering<br>
      License: ${escH(m.license)} · <span title="${escH(m.revision)}">${escH(m.repo)} @ ${escH(m.revision.slice(0, 7))}</span></div>
      ${['downloading', 'paused', 'converting'].includes(s.status) ? `<div class="prog"><i style="width:${pct}%"></i></div><div class="small muted">${escH(s.status === 'converting' ? 'Preparing on this Mac' : s.msg || '')} · ${pct}%${s.etaS ? ' · ' + fmtMin(s.etaS) : ''}</div>` : ''}
      ${s.status === 'ready' ? `<div class="small"><span class="role ok">On this Mac</span> ${s.variant === 'prebuilt' ? 'community 4-bit copy' : m.runtime === 'llama_cpp' ? 'GGUF' : `${s.variant}-bit`}</div>` : ''}
      ${s.status === 'error' ? `<div class="small err">${escH(s.error || 'Failed')}</div>` : ''}<div class="mbtns"></div>`;
    const btns = el.querySelector('.mbtns'), add = (label, fn, ghost) => { const b = document.createElement('button'); b.textContent = label; b.className = ghost ? 'ghost small' : 'small'; b.onclick = fn; btns.appendChild(b); };
    const act = async (p, body) => { const r = await postJ(p, { id: m.id, ...body }); if (r.error) toast(r.error, 'err'); openAiModels(); };
    if (s.status === 'downloading') add('Pause', () => act('api/ai/models/pause'), true);
    else if (s.status === 'paused') add('Resume', () => act('api/ai/models/download', { variant: s.variant || '4' }));
    else if (s.status !== 'ready' && s.status !== 'converting') {
      add(`Download`, () => act('api/ai/models/download', { variant: '4' }));
      if (m.bits.includes(8)) add('Download 8-bit (sharper, more memory)', () => act('api/ai/models/download', { variant: '8' }), true);
      if (m.prebuilt) add(`Smaller: community 4-bit copy (${gb(m.prebuilt.downloadGB)})`, () => act('api/ai/models/download', { variant: 'prebuilt' }), true);
    }
    if (['ready', 'paused', 'error'].includes(s.status)) add('Delete', () => confirm(`Delete ${m.name} from this Mac?`) && act('api/ai/models/delete'), true);
    $('#aimList').appendChild(el);
  }
  clearTimeout(openAiModels.t);
  if (busy) openAiModels.t = setTimeout(() => !$('#aiModels').classList.contains('hidden') && openAiModels(), 1500);
}
$('#aimClose').onclick = async () => { $('#aiModels').classList.add('hidden'); clearTimeout(openAiModels.t); if (!$('#chat').classList.contains('hidden')) { await aiLoad(); aiRender(); } };
$('#aimCheck').onclick = async () => {
  $('#aimCheckMsg').textContent = 'Checking Hugging Face…';
  const r = await postJ('api/ai/models/check');
  $('#aimCheckMsg').textContent = r.error ? r.error : r.changed.length ? `Newer versions pinned: ${r.changed.join(', ')}. Download again to use them.` : 'You have the newest versions.';
  openAiModels();
};
$('#btnChat').onclick = () => ($('#chat').classList.contains('hidden') ? openChat() : closeChat());
$('#chatClose').onclick = closeChat;
$('#infoAsk').onclick = () => { openChat(); };
aiLoad().then(renderAskButton).catch(() => {});   // shows setup progress on the button after a restart

window.__bv = { ex, tour, layers, planes, THREE, structById, centered, camera, controls, seqData, cmp };

// ---------- boot (+ deterministic states for screenshots)
fetch('api/state').then((r) => r.json()).then((st) => { serverMac = !!st.mac; $('#tourVoice').classList.toggle('hidden', !serverMac); neckAvail = !!st.neckAvailable; if (st.neck && st.neck.status === 'running') pollNeck(); });
(async () => {
  let cid = params.get('case');
  const st = cid ? { status: 'skip' } : await pollProcessing();   // an explicit ?case= opens that cached case
  if (!cid && st.status !== 'running') {
    const cases = await (await fetch('api/cases')).json();
    cid = cases.length ? cases[cases.length - 1].caseId : null;
  }
  if (st.status === 'running') return;
  if (!cid) { setLoading(false); showWelcome(); return; }
  currentCase = cid; await loadCase(cid);
  if (!hasStructs()) pollPhase2();
  for (const p of (params.get('planes') || '').split(',').filter(Boolean)) enablePlane(p, true);
  if (params.get('cut') === '1') { $('#cutaway').checked = true; updateClipping(); }
  for (const kv of (params.get('pos') || '').split(',').filter(Boolean)) { const [n, v] = kv.split(':'); setPlanePos(n, manifest.brainCenter[planes[n].axis] + +v); }
  for (const kv of (params.get('op') || '').split(',').filter(Boolean)) { const [id, o] = kv.split(':'); if (layers[id]) { layers[id].opacity = +o; layers[id].visible = +o > 0; applyOpacity(layers[id]); } }
  if (params.get('details') === '1') $('#seriesDetails').open = true;
  for (const g of (params.get('hide') || '').split(',').filter(Boolean)) { const S = structById[g]; if (S) { S.st.visible = false; applyOpacity(S.L); } }
  if (params.get('select')) {
    let pt = null;
    if (params.get('pick')) { const [x, y, z] = params.get('pick').split(':').map(Number); pt = centered.localToWorld(new THREE.Vector3(x, y, z)); }
    select(params.get('select'), pt);
  }
  if (params.get('isolate') === '1') $('#infoIsolate').click();
  if (params.get('keepdeep') === '0') { $('#keepDeep').checked = false; updateClipping(); }
  if (params.get('lop')) { $('#labelOpacity').value = params.get('lop'); $('#labelOpacity').oninput({ target: $('#labelOpacity') }); }
  if (params.get('overlay') === '1') { $('#labelOverlay').checked = true; }
  updateOverlayUI();
  if (params.get('seqmode')) seqMode = params.get('seqmode');
  if (params.get('sop')) { $('#seqOpacity').value = params.get('sop'); $('#seqOpacity').oninput({ target: $('#seqOpacity') }); }
  if (params.get('seq')) await setSeq(params.get('seq'), { quiet: params.get('seqcard') === '0' });
  if (params.get('compare')) { cmp.mode = params.get('cmpmode') || 'side'; cmp.split = +(params.get('swipe') || 0.5); openCompare(params.get('compare')); }
  if (params.get('help') === '1') toggleHelp(true);
  if (params.get('tour')) { startTour(); for (let n = 1; n < +params.get('tour'); n++) tourGo(1); if (camAnim) camAnim.dur = 1; }
  if (params.get('example')) { await openExample(params.get('example')); if (camAnim) camAnim.dur = 1; if (params.get('volumes') === '1') volumeTable(); }
  if (params.get('export') === '1') openExport();
  if (params.get('menu') === '1') $('#toolsMenu').classList.remove('hidden');
  if (params.get('library') === '1') { neckAvail = !!(await (await fetch('api/state')).json()).neckAvailable; await openLibrary(); }
  if (params.get('neckcard') === '1') showNeckCard();
  if (params.get('measure')) { setMeasure(true); const v = params.get('measure').split(',').map(Number); for (let q = 0; q + 2 < v.length; q += 3) addMeasPoint(new THREE.Vector3(v[q], v[q + 1], v[q + 2]).add(new THREE.Vector3(...manifest.brainCenter))); }
  if (params.get('zoom')) { viewRadius /= +params.get('zoom'); setView(params.get('cam') || '3q', true); }
  if (params.get('auto') === '1') { controls.autoRotate = true; $('#btnAuto').classList.add('on'); }
  if (!params.has('shot') && localStorage.getItem('bg:hint') !== '1') $('#hint').classList.remove('hidden');
  if (params.get('region')) { frameRegion(params.get('region')); if (camAnim) camAnim.dur = 1; }
  if (params.get('focus')) { const S = structById[params.get('focus')]; if (S) { selectStruct(S.st.id); focusOn([S.item], S.st.name); if (camAnim) camAnim.dur = 1; } }
  if (params.get('hint') === '1') $('#hint').classList.remove('hidden');
  if (params.get('tab')) setTab(params.get('tab'), true);
  if (params.get('q')) { $('#search').value = params.get('q'); $('#search').oninput({ target: $('#search') }); }
  if (params.get('palette')) { openPalette(); $('#palIn').value = params.get('palette'); renderPalette(); }
  if (params.get('slicemore') === '1') $('#sliceOpts').classList.remove('hidden');
  let frames = 0; const wait = () => (++frames > 20 ? (window.__ready = true) : requestAnimationFrame(wait)); requestAnimationFrame(wait);
})();
