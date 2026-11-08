"""Browser chat state without WebGL/models: execute the real app.js handlers in Node.

Like test_camera.py, this uses only Node and small DOM/stream doubles. All case,
report and image values below are synthetic; no server or model is contacted.
"""
import json
import re
from pathlib import Path
import shutil
import subprocess

import pytest

APP = Path(__file__).resolve().parents[1] / "bodyglass" / "web" / "app.js"
pytestmark = pytest.mark.skipif(not shutil.which("node"), reason="node not installed")

HARNESS = r"""
const assert = require('node:assert/strict');
const { randomUUID } = require('node:crypto');
const crypto = { randomUUID };
class Element {
  constructor() {
    this.children = []; this.value = ''; this.textContent = ''; this.style = {}; this.dataset = {};
    this.listeners = {}; this.classes = new Set(['hidden']); this.scrollHeight = 20;
    this.classList = { add: (...xs) => xs.forEach(x => this.classes.add(x)),
      remove: (...xs) => xs.forEach(x => this.classes.delete(x)), contains: x => this.classes.has(x),
      toggle: (x, on = !this.classes.has(x)) => on ? this.classes.add(x) : this.classes.delete(x) };
  }
  set innerHTML(v) { this.textContent = v; this.children = []; }
  get innerHTML() { return this.textContent; }
  appendChild(x) { this.children.push(x); return x; }
  addEventListener(k, fn) { this.listeners[k] = fn; }
  querySelector() { return new Element(); }
  replaceWith(x) { replaced.push(x); }
  insertAdjacentHTML() {}
  focus() {}
  click() { this.onclick?.({ target: this }); }
  closest() { return null; }
  getContext() { return { drawImage() {}, clearRect: () => { this.cleared = true; } }; }
  toDataURL() { return 'data:image/png;base64,synthetic'; }
}
const nodes = new Map(), replaced = [], images = [], readers = [], requests = [], notices = [];
const $ = id => { if (!nodes.has(id)) nodes.set(id, new Element()); return nodes.get(id); };
const document = { createElement: () => new Element(), querySelectorAll: () => [], addEventListener() {}, body: new Element() };
const window = {}, navigator = {};
const localStorage = { data: new Map(), getItem(k) { return this.data.get(k) ?? null; },
  setItem(k, v) { this.data.set(k, String(v)); }, removeItem(k) { this.data.delete(k); } };
let currentCase = null, aiSel = null, manifest = null, caseVersion = 0, renderedCase = null;
const layers = {}, structById = {}, meas = { pts: [] }, fmtMl = x => x;
const toast = (...x) => notices.push(x);
class Image { constructor() { images.push(this); this.width = this.height = 1; } }
class FileReader { constructor() { readers.push(this); } readAsDataURL() {} }
function deferred() { let resolve, reject; const promise = new Promise((ok, no) => { resolve = ok; reject = no; }); return { promise, resolve, reject }; }
function fetch(url, opts) {
  const d = deferred(); requests.push({ url, body: JSON.parse(opts.body), signal: opts.signal, ...d }); return d.promise;
}
const tick = async () => { for (let n = 0; n < 12; n++) await Promise.resolve(); };
function stream(i) {
  let pending = null, queue = [];
  const reader = { read() { if (queue.length) return Promise.resolve(queue.shift()); pending = deferred(); return pending.promise; }, cancel() { return Promise.resolve(); } };
  requests[i].resolve({ ok: true, body: { getReader: () => reader } });
  const push = value => { if (pending) { pending.resolve(value); pending = null; } else queue.push(value); };
  return { emit: ev => push({ value: new TextEncoder().encode(JSON.stringify(ev) + '\n'), done: false }),
    end: () => push({ done: true }), fail: e => { pending.reject(e); pending = null; } };
}
"""

# Keep real state transitions, storage, sending, attachment and history handlers.
# Rendering itself is covered by the existing Playwright UI tests.
RENDER = r"""
msgEl = m => { const el = new Element(); el.message = { ...m }; return el; };
renderLog = () => { $('#chatLog').children = (ai.chat?.msgs || []).map(m => msgEl(m)); };
renderSuggest = () => {};
aiRender = () => { renderLog(); renderChips(); };
const clickRow = (i, remove = false) => $('#histList').children[i].onclick({ target: { closest: () => remove ? {} : null } });
const history = () => { $('#histList').classList.add('hidden'); $('#chatHist').click(); };
const saved = id => JSON.parse(LS.get(chatKey(id)) || '[]');
async function answer(q, text = 'synthetic answer') {
  const i = requests.length, job = send(q), s = stream(i); await tick();
  s.emit({ model: 'fake', t: text }); await tick(); s.end(); await job;
}
"""


def run_js(test, functions=(), setup=""):
    source = APP.read_text()
    extra = "\n".join(re.search(r"(?:async )?function " + name + r"\(.*?\n}", source, re.S).group(0) for name in functions)
    for name in functions:
        setup = re.sub(r"function " + name + r"\([^)]*\) \{[^\n]*?}", "", setup)
    guard = re.search(r"const caseGuard = .*\n", source).group(0)
    source = source[source.index("const LS ="):source.index("// Manage models (secondary screen)")]
    script = HARNESS + setup + extra + guard + source + RENDER + "\n(async () => {\n" + test + "\n})().catch(e => { console.error(e); process.exitCode = 1; });"
    result = subprocess.run(["node", "-e", script], text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


def test_case_switch_clears_sensitive_state_and_partitions_requests_and_storage():
    run_js(r"""
changeCase('A'); await answer('A question', 'A answer');
const a = ai.chat;
ai.extra = [{ image: 'A picture' }, { report: 'A report' }]; ai.removed.add('A selection');
aiSel = { name: 'A part' }; $('#chatQ').value = 'A draft'; $('#reportText').value = 'A report draft';
$('#exactText').textContent = 'A preview'; $('#histList').innerHTML = 'A history';
changeCase('B');
assert.equal(ai.chat, null); assert.deepEqual(ai.extra, []); assert.equal(ai.removed.size, 0);
assert.equal(aiSel, null); assert.equal(ai.lastReq, null);
for (const id of ['#chatQ', '#reportText', '#imgFile']) assert.equal($(id).value, '');
assert.equal($('#exactText').textContent, ''); assert.equal($('#histList').textContent, '');
for (const id of ['#reportBox', '#exactBox', '#histList']) assert($(id).classList.contains('hidden'));
await answer('B question', 'B answer');
assert.deepEqual(requests[1].body.history, []); assert.deepEqual(requests[1].body.ctx, {}); assert.equal(requests[1].body.image, null);
assert(!JSON.stringify(saved('B')).includes('A question')); assert.equal(saved('A')[0].msgs[0].text, 'A question');
saveChat('B', a); assert.equal(saved('B').length, 1); // explicit ownership mismatch is refused
changeCase('A'); history(); clickRow(0); assert.equal(ai.chat.caseId, 'A');
await answer('A follow-up'); assert.deepEqual(requests[2].body.history.map(m => m.text), ['A question', 'A answer']);
""")


@pytest.mark.parametrize("late", ["fetch", "token", "error"])
def test_late_response_cannot_render_save_or_unlock_new_case(late):
    run_js("const late = " + json.dumps(late) + r""";
changeCase('A'); const old = send('A question'); let a;
if (late !== 'fetch') { a = stream(0); await tick(); a.emit({ t: 'A partial' }); await tick(); }
const oldChat = ai.chat; changeCase('B'); assert(requests[0].signal.aborted);
const fresh = send('B question'), b = stream(1); await tick();
const ctrl = ai.ctrl, renders = replaced.length;
if (late === 'fetch') { a = stream(0); }
else if (late === 'token') { a.emit({ model: 'A stale model', t: 'A stale secret' }); }
else { a.fail(new Error('A stale error')); }
await old;
assert.equal(ai.ctrl, ctrl); assert($('#sendBtn').classList.contains('stop'));
assert.equal(replaced.length, renders); assert.equal(ai.chat.msgs[1].text, '');
assert(!JSON.stringify(saved('B')).includes('A ')); assert(saved('A')[0].msgs[1].stopped);
assert(!JSON.stringify(oldChat).includes('stale'));
b.emit({ model: 'B model', t: 'B answer' }); await tick(); b.end(); await fresh;
assert.equal(saved('B')[0].model, 'B model'); assert.equal(ai.ctrl, null);
""")


def test_new_chat_and_history_switch_stop_stream_without_erasing_other_chats():
    run_js(r"""
changeCase('A'); await answer('first'); const firstId = ai.chat.id;
newChat(); const old = send('second'), s = stream(1); await tick();
const secondId = ai.chat.id; history(); clickRow(1);
assert.equal(ai.chat.id, firstId); assert(requests[1].signal.aborted);
s.emit({ t: 'late second' }); await old;
assert.equal(ai.chat.id, firstId); assert.equal(saved('A').find(c => c.id === secondId).msgs[1].text, '');
newChat(); assert.equal(ai.lastReq, null); await runFix('retry'); assert.equal(requests.length, 2);
const third = send('third'), t = stream(2); await tick(); const thirdId = ai.chat.id;
newChat(); assert(requests[2].signal.aborted); t.end(); await third;
assert.equal(ai.chat, null); assert.equal(saved('A').find(c => c.id === thirdId).msgs[1].stopped, true);
""")


def test_deleting_active_chat_during_stream_cannot_resurrect_it():
    run_js(r"""
changeCase('A'); const job = send('delete me'), s = stream(0); await tick();
history(); clickRow(0, true); assert.equal(ai.chat, null); assert.equal(saved('A').length, 0);
s.emit({ t: 'late deleted answer' }); await job;
assert.equal(saved('A').length, 0); assert.equal(ai.ctrl, null);
""")


def test_async_images_reports_file_picker_and_preview_are_invalidated_even_on_return():
    run_js(r"""
changeCase('A'); toPng('synthetic', null, 'A picture');
readFile({ type: 'image/png', name: 'A.png' });
const report = deferred(); readFile({ type: 'text/plain', name: 'A.txt', text: () => report.promise });
showExact(); attach('file'); const oldEpoch = ai.epoch;
changeCase('B'); changeCase('A'); assert.notEqual(ai.epoch, oldEpoch);
images[0].onload(); readers[0].result = 'A image'; readers[0].onload(); report.resolve('A report');
requests[0].resolve({ json: async () => ({ preview: 'A preview' }) });
$('#imgFile').onchange({ target: { files: [{ type: 'image/png', name: 'A picker.png' }], value: '' } });
await tick();
assert.deepEqual(ai.extra, []); assert.equal(images.length, 1); assert.equal(readers.length, 1);
assert.equal($('#reportText').value, ''); assert($('#reportBox').classList.contains('hidden'));
assert.equal($('#exactText').textContent, ''); assert($('#exactBox').classList.contains('hidden'));
// Current-case attachments still work.
toPng('synthetic', null, 'current picture'); images[1].onload();
assert.equal(ai.extra[0].label, 'current picture');
readFile({ type: 'text/plain', name: 'current.txt', text: async () => 'current report' }); await tick();
assert.equal($('#reportText').value, 'current report');
""")


def test_close_example_legacy_history_and_same_case_refresh():
    run_js(r"""
changeCase('A'); await answer('A'); const chat = ai.chat, epoch = ai.epoch;
changeCase('A'); assert.equal(ai.chat, chat); assert.equal(ai.epoch, epoch);
changeCase(null); assert.equal(ai.chat, null); await answer('no case');
changeCase('example:sample'); assert.equal(ai.chat, null); await answer('example');
assert.equal(saved(null)[0].msgs[0].text, 'no case'); assert.equal(saved('example:sample')[0].msgs[0].text, 'example');
LS.set(chatKey('legacy'), JSON.stringify([{ id: 'old', title: 'legacy', msgs: [{ role: 'user', text: 'legacy question' }] },
  { id: 'wrong', caseId: 'A', title: 'wrong owner', msgs: [{ role: 'user', text: 'A question' }] }]));
changeCase('legacy'); history(); assert.equal($('#histList').children.length, 1); clickRow(0);
assert.equal(ai.chat.caseId, 'legacy'); await answer('legacy follow-up');
assert.equal(requests[3].body.history[0].text, 'legacy question');
""")


def test_loading_blocks_old_view_context_and_snapshot_sending():
    run_js(r"""
changeCase('A'); aiSel = { name: 'old part' }; ai.loading = true;
assert.deepEqual(autoChips(), []); await send('too early'); attach('axial'); attach('snap'); showExact();
assert.equal(requests.length, 0); assert.equal(images.length, 0);
ai.loading = false; await answer('ready'); assert.equal(requests.length, 1);
""")


def test_all_case_entry_points_use_the_reset_boundary():
    source = APP.read_text()
    assert "currentCase = caseId; ai.caseId = caseId" in source
    assert "changeCase(cid);" in source[source.index("async function loadCase("):source.index("let currentCase =")]
    assert "changeCase(null);" in source[source.index("function closeCase("):source.index("function showWelcome(")]
    # Future entry points must not silently bypass the lifecycle hook.
    assignments = re.findall(r"(?<![=!])\bcurrentCase\s*=(?!=)\s*([^;]+);", source)
    assert assignments == ["null, caseVersion = 0, renderedCase = null", "caseId"]


def test_storage_quota_failure_never_blocks_reset_or_leaves_send_stuck():
    run_js(r"""
changeCase('A'); const old = send('A question'), s = stream(0); await tick();
ai.extra.push({ report: 'A report' }); $('#chatQ').value = 'A draft';
localStorage.setItem = () => { throw new Error('QuotaExceededError'); };
changeCase('B');
assert(requests[0].signal.aborted); assert.equal(ai.chat, null); assert.equal(ai.lastReq, null);
assert.deepEqual(ai.extra, []); assert.equal($('#chatQ').value, '');
s.emit({ t: 'late A' }); await old;
await answer('B question'); assert.equal(ai.ctrl, null); assert.equal(ai.request, null);
assert.equal(requests[1].body.question, 'B question'); assert.deepEqual(requests[1].body.history, []);
assert(!JSON.stringify(requests[1].body).includes('A report')); assert(notices.length > 0);
""")


LOADER = r"""
let neckPad = null, sp, dims, origin, vol, noVol = false, keepSeq = null, viewRadius;
let seqKind = null, seqTex = null, zoneVol = null, zoneTex = null, labelVol = null, labelInfo = [], labTex = null;
let parcels = null, isolateId = null, selected = null;
const seqData = {}, planes = {}, labelEls = {}, sliceCache = new Map(), labCache = new Map();
const params = new URLSearchParams(), noop = () => {}, hasStructs = () => !!manifest.structures?.length;
const caseBase = (cid = currentCase) => `cases/${cid}/`, padZ = x => x, sequences = () => manifest.sequences || [];
const centered = { position: { set: noop }, add: noop, remove: noop };
const scene = { updateMatrixWorld: noop }, floor = { position: {} }, ring = { position: {}, scale: { setScalar: noop } }, viewTarget = { set: noop };
const THREE = { Mesh: class { constructor(geometry, material) { this.geometry = geometry; this.material = material; this.userData = {}; this.visible = true; } },
  Vector3: class {}, Sphere: class {}, Box3: class { constructor() { this.min = { y: 0 }; } setFromObject() { return this; }
    getSize() { return { x: 1, y: 1, z: 1 }; } getCenter() { return { x: 0, y: 0, z: 0 }; } getBoundingSphere() { return { radius: 1 }; } } };
const firstMesh = x => x, makeMaterial = () => ({ opacity: 1 }), clearAir = noop;
let loader = { loadAsync: async () => ({ geometry: { attributes: {}, computeVertexNormals: noop } }) };
function clearScene() { for (const k of Object.keys(layers)) delete layers[k]; renderedCase = null; }
function loadStructures() {} function loadVeins() {} function loadLabelVolume() {} function setSeq() {}
function buildLayerUI() {} function buildSeriesUI() {} function buildStructUI() {} function buildPlanes() {}
function buildScansUI() {} function setPeel() {} function setView() {} function setFocusTag() {}
function updateRegionBar() {} function updatePerson() {} function buildExplore() {} function offerReskin() {}
function setLoading() {} function hideWelcome() {} function drawInsetsNow() {} function seqPending() { return []; }
function redrawMeasure() {} function endTour() {} function closeExample() {} function updateOverlayUI() {}
function applySeqUniforms() {} function drawInsets() {} function closeCompare() {} function showSeqCard() {}
const cmp = { open: false }, GROUP_DOT = {}, GROUP_MAT = {};
const model = id => ({ caseId: id, volume: { spacing: [1], dims: [1, 1, 1], origin: [0, 0, 0], file: 'vol' },
  brainCenter: [0, 0, 0], layers: [{ id: 'brain', name: id + ' brain', file: 'mesh' }] });
const mesh = id => ({ geometry: { owner: id, attributes: {}, computeVertexNormals: noop } });
"""


@pytest.mark.parametrize("stage", ["manifest", "volume_headers", "volume_bytes", "mesh"])
def test_overlapping_model_load_cannot_replace_current_case_context(stage):
    run_js("const stage = " + json.dumps(stage) + r""";
const delayed = deferred();
fetch = async url => {
  const id = url.split('/')[1];
  if (url.endsWith('manifest.json')) return { json: () => id === 'A' && stage === 'manifest' ? delayed.promise : Promise.resolve(model(id)) };
  if (id === 'A' && stage === 'volume_headers') return delayed.promise;
  return { ok: true, arrayBuffer: () => id === 'A' && stage === 'volume_bytes' ? delayed.promise : Promise.resolve(new Uint8Array([id === 'A' ? 1 : 2]).buffer) };
};
loader.loadAsync = async url => url.includes('/A/') && stage === 'mesh' ? delayed.promise : mesh(url.split('/')[1]);
const a = loadCase('A'); await tick(); await loadCase('B');
assert.equal(renderedCase, 'B'); assert.equal(ai.loading, false);
delayed.resolve(stage === 'manifest' ? model('A') : stage === 'mesh' ? mesh('A') :
  stage === 'volume_headers' ? { ok: true, arrayBuffer: async () => new Uint8Array([1]).buffer } : new Uint8Array([1]).buffer);
await a;
assert.equal(currentCase, 'B'); assert.equal(manifest.caseId, 'B'); assert.equal(vol[0], 2);
assert.equal(layers.brain.items[0].mesh.geometry.owner, 'B'); assert.equal(renderedCase, 'B');
assert.deepEqual(aiCtx().ctx.view, ['B brain']);
""", functions=("loadCase",), setup=LOADER)


@pytest.mark.parametrize("helper", ["loadVeins", "loadStructures", "loadLabelVolume", "ensureZone", "ensureSeq", "setSeq"])
def test_late_model_helper_cannot_mutate_after_case_switch(helper):
    run_js("const helper = " + json.dumps(helper) + r""";
changeCase('A'); manifest = model('A');
manifest.veins = { file: 'veins' }; manifest.labelVolume = { file: 'labels' }; manifest.brainZone = 'zone';
manifest.structureGroups = { cortex: { name: 'A cortex', order: 0 } }; manifest.structures = [{ id: 'A-structure', group: 'cortex', file: 'structure' }];
manifest.sequences = [{ kind: 't2', name: 'A T2', file: 'sequence' }];
const delayed = deferred(); loader.loadAsync = () => delayed.promise;
fetch = async () => ({ arrayBuffer: () => delayed.promise });
const job = eval(helper)(helper === 'ensureSeq' || helper === 'setSeq' ? 't2' : 'cases/A/'); await tick();
changeCase('B'); manifest = model('B'); renderedCase = 'B'; ai.loading = false;
for (const k of Object.keys(layers)) delete layers[k]; for (const k of Object.keys(seqData)) delete seqData[k];
zoneVol = labelVol = null; seqKind = null;
delayed.resolve(helper === 'loadVeins' || helper === 'loadStructures' ? mesh('A') : new Uint8Array([1]).buffer);
await job;
assert.deepEqual(layers, {}); assert.deepEqual(structById, {}); assert.equal(zoneVol, null);
assert.equal(labelVol, null); assert.equal(seqKind, null); assert.deepEqual(seqData, {});
assert.equal(renderedCase, 'B'); assert.equal(manifest.caseId, 'B');
""", functions=(helper,) if helper != "setSeq" else ("setSeq", "ensureSeq", "ensureZone"), setup=LOADER)


def test_close_clears_canvases_and_cannot_attach_previous_slices_or_view():
    run_js(r"""
changeCase('A'); renderedCase = 'A'; manifest = model('A'); vol = new Uint8Array([1]);
for (const name of ['axial', 'coronal', 'sagittal']) $('#c-' + name).cleared = false;
clearScene(); manifest = null; changeCase(null); ai.loading = false;
assert.equal(vol, null); assert.equal(renderedCase, null);
for (const name of ['axial', 'coronal', 'sagittal']) { assert($('#c-' + name).cleared); attach(name); }
attach('snap'); assert.equal(images.length, 0); assert.deepEqual(aiCtx().ctx, {});
""", functions=("clearScene",), setup=LOADER)
