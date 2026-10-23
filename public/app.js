// Drishyam UI entry: wires the controls, runs a job (live or ?mock=1) and presents its events at reading speed.
import { $, $$, h, num, dataUrl } from './js/dom.js';
import { MOCK, streamJob, postReview, postKill, getJSON, health } from './js/api.js';
import { prepareImage } from './js/prep.js';
import { Viewport, pointOf, tileBoxes } from './js/viewport.js';
import { Budget, Trace, renderStepDetail, isGovernor, plannerFell, explain, kindOf, pad, px, label, words, reviewText, reviewKind } from './js/panels.js';
import { Checks, Findings, ruleOf } from './js/findings.js';
import { countUp, popup, reduced, replay, sound } from './js/fx.js';
import { initResults } from './js/results.js';

const PACE = 3500;  // ms each decision stays on screen, with what its look found: the camera moves every few seconds
const HOLD = 5000;  // the decision that finds the first critical damage stays this long: the hand-off to a person
const GUIDE = 'bridge_03.jpg';  // exposed rebar: the default run ends with critical findings for a human
const GUIDE_REAL = 'real_02.jpg';  // our own wall photo the v1 engine took a look at: its guided run replays the recorded job
const GUIDE_PACE_MS = 1000;  // asks the server to wait this long between looks (pace_ms); its timings exclude the wait
const KILL_WAIT_S = 8;  // a KILL pressed before the server's first record waits this long for it, then cancels the upload
const NOTE = 'Press Run and the agent’s reasoning appears here, one decision at a time, in plain words.';
// ?mock=1 replays public/fixtures/job.ndjson (its step times, its accurate/fast models and its injected drills say so);
// the e2e test pins this date to the fixture's, so a re-recorded fixture must update it.
const RECORDED = 'a saved job, recorded on the classical engine on 2026-10-17 with all three failure drills switched on';
const FIXTURE = 'fixtures/mock.jpg';  // the photo that saved job ran on: the only one ?mock=1 may replay it over
// the wall replays (public/samples/real_0N.ndjson): video/live/walls/*/real.ndjson, from video/record.py --walls
const WALLS = 'Recorded live on this site on 2026-10-23 (the v1 engine on our own photo)';

const el = {
  pill: $('#job-pill'), text: $('#job-text'), count: $('#job-steps'), kill: $('#kill'), resume: $('#resume'), sound: $('#sound'),
  look: $('#look'), runBar: $('#run-bar'), tryKill: $('#try-kill'), skip: $('#skip'), guided: $('#guided'), guidedReal: $('#guided-real'), recorded: $('#recorded'),
  file: $('#file'), drop: $('#drop'), prep: $('#prep'), gallery: $('#gallery'), galleryEmpty: $('#gallery-empty'),
  slider: $('#replay'), sliderOut: $('#replay-out'), play: $('#play'), heat: $('#heat'), tiles: $('#tiles'), tilesNote: $('#tiles-note'),
  detail: $('#step-detail'), done: $('#done'), recap: $('#recap'), narrator: $('#narrator'), narratorSub: $('#narrator-sub'),
  camImg: $('#cam-img'), pip: $('#pip'), camCap: $('#cam-cap'), stages: $('#stages'), say1: $('#say-1'), say4: $('#say-4'),
  states: [1, 2, 3, 4].map((i) => $(`#state-${i}`)),
  tbSheet: $('#tb-sheet'), tbScale: $('#tb-scale'), tbChecked: $('#tb-checked'),
};

const S = {
  image: null, sample: null, ready: null, ctl: null, mode: 'idle', opts: {}, job: null, overview: null, steps: [], done: null,
  token: null, order: null, ui: null, timer: 0, rev: 0, events: [], planners: new Map(), arrived: null, note: null, critSeen: false, killTimer: 0, killPending: false,
  found: [], sent: [], crit: null, slow: false,  // what the step on screen found and sent to a person, its critical NOTE; replayed slowly?
};
// The presentation queue: stream events wait here and are shown one decision at a time. A step's findings and
// review requests arrive with it and join it on screen, so they never wait behind its pause; so does the overview's
// drawing (the heatmap), while its NOTE waits for the photo line's read time.
const Q = { items: [], timer: 0, skip: false };
const JOINS = new Set(['finding', 'review_request', 'overview']);

const viewport = new Viewport($('#stage'), (k) => goto(k));
const budget = new Budget($('#budget'));
const trace = new Trace($('#trace'), $('#trace-empty'), (n) => goto(S.steps.findIndex((s) => s.step === n) + 1));
const findings = new Findings($('#findings'), $('#findings-empty'), review, showFinding, levelsOf);
const checks = new Checks($('#checks'), check, (a) => showBox(a.box));

const GOVERNOR_TEXT = {
  'p95>SLA': (s) => `The segmenter got slow, so the governor switched to the ${s.model ?? 'fast'} model.`,
  'warp-spike': () => 'Cutting out views got slow, so the governor shrank the view.',
  'usd≥80%': () => '80 % of the $ cap is spent, so it now double-checks critical damage only.',
  'cap-hit': () => 'A budget cap was hit, so it stopped and reported what it did not reach.',
  'kill-switch': (s) => (s.step > 1 ? 'KILL: it halted before the next look; its state is kept.'
    : 'The kill switch was already set: it stopped before its first look, so there is nothing to resume.'),
  'planner-fallback': () => 'The planner failed, so the built-in rule planner took over.',
};
const STOPPED = {
  'stop-rule': 'the next look was worth less than its cost', exhausted: 'every candidate region was looked at',
  'cap-hit': 'budget spent', 'kill-switch': 'you pressed KILL', 'planner:stop': 'the planner proposed stopping',
};
// a kill-switch halt before any look can only be the operator-wide switch: the server reads a page's KILL after a look
const stopWords = (cause, looks) => (cause === 'kill-switch' && !looks ? 'the kill switch was already set'
  : cause === 'exhausted' && !looks ? 'no candidate region was proposed' : STOPPED[cause] ?? cause);

// ---- the four steps: Waiting / Running / Done, each with a sentence in plain words -------------------
const pendingCount = () => (S.order?.findings ?? []).filter((f) => f.critical && f.state === 'pending_review').length;
/** What the work order marks not assessed (its not_assessed areas), in words; '' when nothing is. */
function gapText() {
  const na = (S.order?.not_assessed ?? []).filter((a) => !a.resolution);  // a person checked the others by eye
  if (!na.length) return '';
  if (!S.overview?.candidates?.length && na.some((a) => a.reason === 'no candidate region was proposed')) {
    return 'no candidate region was proposed, so the whole photo is not assessed';
  }
  const skipped = na.filter((a) => a.region).length;
  const doubt = na.filter((a) => !a.region && / under the floor /.test(a.reason ?? '')).length;  // agent.py: a held critical mask
  const unread = na.length - skipped - doubt;
  return [skipped ? `${skipped} candidate region${skipped === 1 ? ' was' : 's were'} never looked at` : '',
    doubt ? `${doubt} possible critical defect${doubt === 1 ? ' was' : 's were'} too uncertain to draft and need${doubt === 1 ? 's' : ''} a person to check by eye` : '',
    unread ? `${unread} area${unread === 1 ? '' : 's'} could not be read` : ''].filter(Boolean).join(' and ');
}
/** What a person checked by eye (the work order's resolved not_assessed areas), in words; '' when nothing was. */
function eyeText() {
  const done = (S.order?.not_assessed ?? []).filter((a) => a.resolution);
  if (!done.length) return '';
  const found = done.filter((a) => a.resolution.decision === 'defect-found').length;
  return `a person checked ${done.length} area${done.length === 1 ? '' : 's'} the agent could not assess by eye `
    + (found ? `(${found} defect${found === 1 ? '' : 's'} found by eye, recorded in the work order)` : 'and found nothing');
}
/** A ready work order with nothing critical whose looks saw under half of the salient area closely, in words; '' otherwise. */
function partText() {
  const looked = S.done?.coverage_looked;
  const share = looked ?? S.done?.coverage;  // a recording older than coverage_looked: its coverage counts the survey look
  if (S.order?.status !== 'ready' || S.order.findings.some((f) => f.critical) || !(share < 0.5)) return '';
  return `${Math.round(share * 100)} % of the salient area was ${looked == null ? 'seen at some level' : 'looked at closely'} `
    + 'and nothing critical was found there; the rest was not looked at closely';
}

function syncSteps() {
  const run = ['running', 'killed', 'stopped'].includes(S.mode) ? S.mode : S.done ? 'done' : 'waiting';
  const pending = pendingCount();
  const gap = S.done ? gapText() : '';
  const part = S.done ? partText() : '';
  const eye = S.done ? eyeText() : '';
  const states = [S.image ? 'done' : 'waiting', run, run, S.done && !pending && !gap ? 'done' : 'waiting'];
  states.forEach((state, i) => {
    el.states[i].dataset.state = state;
    el.states[i].textContent = { waiting: 'Waiting', running: 'Running', done: 'Done', killed: 'Killed', stopped: 'Stopped' }[state];
  });
  const critical = (S.order?.findings ?? []).some((f) => f.critical);
  el.say4.textContent = !S.done
    ? (S.mode === 'running' ? 'Findings appear here as the agent makes them. Anything critical waits for your sign-off when the run ends.'
      : 'Nothing to sign yet. When a run ends, anything critical waits here for a person. The agent cannot approve it.')
    : pending ? `Waiting for your sign-off (${pending}): type your name on each card marked NEEDS A HUMAN, then Approve or Reject. The agent cannot do this for you.`
      : gap ? `${!S.overview?.candidates?.length ? 'Not ready' : `${critical ? 'Signed' : 'Nothing critical in what it looked at'}, but not ready`}: `
        + `${gap}. The work order marks this as not assessed, so a person should look there. It does not say the wall is clean. `
        + 'Check each area by eye below: type your name, then Nothing found or Defect found.'
      : eye ? `${critical ? 'Signed and checked' : 'Checked'} by eye: ready to send. ${eye[0].toUpperCase()}${eye.slice(1)}. Download the signed work order below.`
      : critical ? 'Signed: ready to send. Download the signed work order below.'
        : part ? `Ready: ${part}. It does not say the wall is clean.`
          : 'Nothing critical was found, so nothing needs your sign-off. The work order is ready.';
}

// ---- taskbar / mode -----------------------------------------------------------------------------------
function doneKey() {
  const d = S.done;
  if (d?.status === 'halted') return d.cause === 'cap-hit' ? 'budget' : 'halted';
  if (S.order?.status === 'incomplete') return 'incomplete';  // done, but part of the photo is not assessed yet
  if (d?.status === 'awaiting_review' && S.order?.status !== 'blocked') return 'signed';
  if (S.order?.status === 'ready' && S.order.approvals?.length) return 'signed';  // checked by eye: a person signed
  return d?.status ?? 'completed';
}
const PILL = {
  idle: 'ready', running: 'running', killed: 'killed', stopped: 'stopped', error: 'error',
  awaiting_review: 'needs sign-off', signed: 'signed', completed: 'done', halted: 'halted', budget: 'budget spent',
  incomplete: 'needs a look',
};

function setMode(mode, text) {
  const onBar = [el.kill, el.skip].includes(document.activeElement);  // hiding them would drop keyboard focus on the page body
  S.mode = mode;
  $('#taskbar').dataset.mode = mode;
  const key = mode === 'done' ? doneKey() : mode;
  el.pill.className = `pill pill--${key}`;
  el.pill.textContent = PILL[key] ?? key;
  el.text.textContent = text ?? ({
    idle: MOCK ? `Recorded run: ${RECORDED}; the live engine today is v1 with v1-policy2. Press Run.`
      : S.image ? 'Photo ready. Press Run, or Watch it inspect a bridge for a guided run.' : 'Pick a photo, then press Run.',
    running: 'Sending the photo to the agent…',
  }[mode] ?? '');
  const running = mode === 'running';
  el.kill.hidden = !running || Boolean(S.arrived);  // after the server's done event a KILL can stop nothing
  el.skip.hidden = !running || Q.skip;
  el.runBar.hidden = el.tryKill.hidden = running;
  el.resume.hidden = !(['killed', 'stopped'].includes(mode) || (mode === 'error' && S.steps.length > 0)
    || (mode === 'done' && S.done?.cause === 'kill-switch' && S.done.summary?.steps > 0));  // a halt before any look has nothing to resume
  el.look.disabled = el.runBar.disabled = el.tryKill.disabled = running || !S.image;
  el.guided.disabled = el.guidedReal.disabled = running;
  if (!running) el.count.textContent = '';
  if (['killed', 'stopped'].includes(mode) || (mode === 'error' && !S.events.length)) narrate(el.text.textContent);  // the NOTE says how the run ended
  if (onBar && !running) (el.resume.hidden ? el.look : el.resume).focus({ preventScroll: true });
  showCam(viewport.cursor);  // its caption says whether a job is on its way
  syncSteps();
}

function narrate(text, sub = '') {
  el.narrator.textContent = text;
  el.narratorSub.textContent = sub;
}

// A saved job replayed (?mock=1, or a real photo's recorded trace): no server, and the page says so.
const recorded = () => MOCK || Boolean(S.opts.trace);
// Seconds from a fixed cost table (a recorded trace says so in its job record) are modelled, not measured.
const modelled = () => S.job?.cost_model?.kind === 'fixed';
const agentTime = () => (modelled() ? 'modelled agent time' : 'agent time');

// The server's pause between looks: as the job record echoes it, else as this page asked for it (pace_ms).
const paceMs = () => S.job?.pace_ms ?? S.opts.pace ?? 0;

const heatmap = () => Boolean(dataUrl('image/png', S.overview?.heatmap_png_b64));  // a recording may have its images removed

const overviewLevel = () => {
  const i = S.job?.levels?.findIndex(([w, hh]) => w * hh === S.overview?.pixels_used) ?? -1;
  return i >= 0 ? i : null;
};

function photoText() {
  const pv = S.job.privacy ?? {};
  return `Photo received: ${S.job.width} × ${S.job.height} px. Before any look, faces and people are blurred (${pv.faces_blurred ?? 0} faces and ${pv.people_blurred ?? 0} people here).`;
}

function overviewText() {
  const n = S.overview.candidates?.length ?? 0;
  const top = (S.job?.levels?.length ?? 1) - 1;
  return `It shrank the photo into a pyramid (L0 is full size, each level up halves it, down to L${top}), `
    + `mapped where damage looks likely${heatmap() ? ' (the heatmap)' : ''} and marked ${n} region${n === 1 ? '' : 's'} worth a closer look (dashed boxes).`;
}

/** The camera panel: the newest look's crop as the agent saw it, or the stages before the first look. */
function showCam(k) {
  const look = S.steps.slice(0, k).reverse().find((s) => s.crop_jpeg_b64);
  const url = look ? dataUrl('image/jpeg', look.crop_jpeg_b64) : null;
  el.camImg.hidden = el.pip.hidden = !url;
  el.stages.hidden = Boolean(url);
  if (look && !url) {  // a recording with its images removed (the wall replays)
    el.camCap.textContent = `step ${look.step}: this recording keeps no image of its looks`;
    return;
  }
  if (!url) {
    el.camCap.textContent = S.mode === 'running'
      ? (S.job ? 'Before the first look: the stages above light up as the server prepares the photo'
        : 'Sending the photo: the first look follows the server’s preparation')
      : !S.job ? 'No look yet. Press Run.' : S.steps.some((s) => s.crop_jpeg_b64) ? 'Before the first look' : 'Stopped before any look';
    return;
  }
  el.camImg.src = el.pip.src = url;
  el.camImg.alt = `What the agent saw at step ${look.step}`;
  const L = look.decision.args.level;
  const ov = overviewLevel();
  const zoom = num(ov) ? 2 ** (ov - L) : null;
  el.camCap.textContent = [`step ${look.step}`, `L${L} · 1:${2 ** L}`,
    zoom > 1 ? `${zoom}× closer than the overview` : zoom === 1 ? 'the overview’s scale' : zoom ? `${1 / zoom}× wider than the overview` : null,
    look !== S.steps[k - 1] ? 'the last look before this step' : null].filter(Boolean).join(' · ');
}

function stage(name, text) {
  const li = [...el.stages.children].find((x) => x.dataset.stage === name);
  if (!li) return;  // a stage this page does not draw
  li.classList.add('is-on');
  if (text) li.textContent = text;
}

// ---- stepping / replay --------------------------------------------------------------------------------
function goto(k, animate = false) {
  k = Math.max(0, Math.min(k, S.steps.length));
  const step = S.steps[k - 1];
  viewport.setCursor(k, animate);
  el.slider.value = k;
  el.sliderOut.textContent = `step ${k} / ${S.steps.length}`;
  trace.mark(step?.step);
  el.tbSheet.textContent = `${k} of ${S.steps.length}`;
  const level = S.steps.slice(0, k).reverse().find((s) => num(s.decision?.args?.level))?.decision.args.level;  // a halt keeps the last look's scale
  el.tbScale.textContent = num(level) ? `L${level} · 1:${2 ** level}` : 'overview';
  const planner = step && S.planners.get(step.step);
  renderStepDetail(el.detail, step, planner, Boolean(S.job));
  budget.show(step
    ? { pixels: step.pixels_frac ?? 0, usd: step.usd_used ?? 0, seconds: step.seconds_used ?? 0 }
    : { pixels: S.overview?.pixels_frac ?? 0, usd: 0, seconds: 0 });
  budget.setOps(step);
  showCam(k);
  if (step) {  // the NOTE keeps to the main sentence; the rest goes on its secondary line
    const [main, ...more] = explain(step, planner);
    narrate(main, more.join(' '));
  }
  else if (S.overview) narrate(overviewText());
}

function stopPlay() {
  clearInterval(S.timer);
  S.timer = 0;
  el.play.textContent = 'Play';
}

function togglePlay() {
  if (S.timer) return stopPlay();
  if (viewport.cursor >= S.steps.length) goto(0);
  el.play.textContent = 'Pause';
  S.timer = setInterval(() => (viewport.cursor >= S.steps.length ? stopPlay() : goto(viewport.cursor + 1, true)), 1200);  // at least the 950 ms draw
}

function clearJob() {
  stopPlay();
  Object.assign(S, { job: null, overview: null, done: null, token: null, order: null, ui: null, rev: 0, planners: new Map(), critSeen: false, found: [], sent: [], crit: null });
  el.tbChecked.textContent = 'human reviewer';
  viewport.reset(null);
  S.steps = viewport.steps;
  trace.clear();
  findings.clear();
  checks.clear();
  el.slider.max = 0;
  el.slider.disabled = el.play.disabled = true;
  el.tiles.checked = false;
  el.tiles.disabled = true;
  el.tilesNote.hidden = true;
  el.recap.hidden = true;
  for (const li of el.stages.children) {
    li.classList.remove('is-on');
    li.textContent = li.dataset.text;
  }
  el.done.replaceChildren(h('p', { class: 'note' }, 'No job has finished yet. Its status, coverage and the signed work order appear here.'));
  narrate(NOTE);
  goto(0);
}

function levelsOf(region) {
  const levels = S.steps.filter((s) => s.region === region && num(s.decision?.args?.level)).map((s) => `L${s.decision.args.level}`);
  return levels.length ? `looked at ${levels.filter((l, i) => l !== levels[i - 1]).join(' → ')}` : '';
}

/** Which earlier measurement caused this step: the look it follows up, or the overview. */
function causeFrom(s) {
  if (['escalate', 'verify'].includes(s.decision?.action) && s.cause !== 'replay' && !String(s.cause).startsWith('planner:')) {
    const prev = S.steps.slice(0, -1).reverse().find((p) => p.region === s.region && p.decision?.action !== 'halt');
    return prev ? pad(prev.step) : null;
  }
  return s.cause === 'survey' || /^(?:.*; )?argmax/.test(s.rationale ?? '') ? 'the overview' : null;
}

function scrollToViewer() {
  bars();  // the job bar has just changed height (running), and the scroll margin is computed from it now
  $('.win--view').scrollIntoView({ behavior: reduced() ? 'auto' : 'smooth', block: 'start' });
}

function showBox(box) {
  viewport.highlight(box);
  scrollToViewer();
}

function showFinding(f) {
  const i = S.steps.findIndex((s) => s.step === f.step);
  if (i >= 0) goto(i + 1);
  showBox(f.box);
}

// ---- stream events, presented one decision at a time ---------------------------------------------------
/** Present one event; returns how long it stays on screen before the next one (ms). */
function handle(ev) {
  switch (ev.type) {
    case 'job': {  // the last job is cleared (run(), or a Resume's first record), and any `stage` record before this one has lit its stage
      S.job = ev;
      viewport.reset(ev);
      S.steps = viewport.steps;
      budget.setCaps(ev.budget);
      goto(0);
      const pv = ev.privacy ?? {};
      const took = (ms, what) => (num(ms) ? ` · ${what} ${Math.round(ms)} ms` : '');  // prep_ms: the server's {read, decode, privacy}
      stage('photo', `photo received · ${ev.width} × ${ev.height} px${took(ev.prep_ms?.decode, 'decoded in')}`);
      stage('privacy', `privacy blur · ${pv.faces_blurred ?? 0} faces, ${pv.people_blurred ?? 0} people${took(ev.prep_ms?.privacy, 'in')}`);
      showCam(0);
      narrate(photoText());
      if (!S.arrived) {
        el.text.textContent = S.opts.killHint ? `${recorded() ? 'Recorded run' : 'Live'}: each look is shown the moment it arrives. Press KILL in this bar while it is looking.`
          : S.opts.trace ? `${ev.cost_model?.kind === 'fixed' ? 'Recorded run of the v1 engine on our own photo; its seconds and $ are modelled, not measured'
            : `${WALLS}, measured costs`}. Replayed from its saved trace (no server), at reading speed${heatmap() ? '' : '; the recording keeps no images, so no heatmap or crops'}.`
          : MOCK ? `Recorded run: replaying ${RECORDED} (no server), at reading speed.`
            : paceMs() > 0 ? `Paced for viewing: the agent waits ${paceMs() / 1000} s between looks; timings exclude the wait.`
              : 'Running live on the server; each decision is shown at reading speed.';
      }
      return 900;
    }
    case 'overview': {
      S.overview = ev;
      viewport.setOverview(ev);
      const n = ev.candidates?.length ?? 0;
      stage('pyramid', `pyramid L0–L${(S.job?.levels?.length ?? 1) - 1}`);
      stage('saliency', 'saliency map (the heatmap)');
      stage('regions', `${n} candidate region${n === 1 ? '' : 's'}`);
      goto(0);
      narrate(photoText());  // the photo line keeps its read time; the overview's NOTE follows in the queue
      Q.items.unshift({ type: 'overview-note' });
      return 0;
    }
    case 'overview-note':  // no step has arrived yet: the viewer is on the overview
      narrate(overviewText());
      return 2400;
    case 'stage': {  // the server's preparation, lit as each part finishes
      const li = [...el.stages.children].find((x) => x.dataset.stage === ev.name);
      stage(ev.name, li && num(ev.ms) ? `${li.dataset.text} · ${Math.round(ev.ms)} ms` : null);
      return 0;
    }
    case 'step':
      return addStep(ev);
    case 'finding':
      return addFinding(ev.finding);
    case 'review_request': {
      const reason = String(ev.reason);
      const plain = reviewText(reason);
      const kind = reviewKind(ev);
      findings.addReason(ev.finding_id, reason, plain);
      if (kind !== 'critical') {  // that one is said by the finding's own line
        const f = findings.cards.get(ev.finding_id)?.finding;
        const decides = / a person decides\b/.test(plain);
        const ask = f && !f.critical ? 'so it is flagged for a person to look at (a draft: no sign-off)' : 'so a person must check it';
        trace.addNote('review', S.steps.at(-1)?.step ?? 0, ev.finding_id, `${f ? label(f.cls) : 'Finding'} ${ev.finding_id} → human`,
          decides ? plain : `${plain}, ${ask}`, () => findings.focus(ev.finding_id), reason, kind.replace(/[-_]/g, ' '));
        S.sent.push(`${f ? label(f.cls) : 'Finding'} ${ev.finding_id} goes to a person: ${plain}.`);
        sayStep();
      }
      return 0;
    }
    case 'planner':
      S.planners.set(ev.step, ev);
      trace.addPlanner(ev);
      return 0;
    case 'done':
      finish(ev);
      return 0;
    case 'error':
      setMode('error', ev.message ?? 'The job failed.');
      return 0;
    case 'end':  // the stream closed without a done record
      if (S.mode === 'running') setMode('killed', `The stream ended at step ${S.steps.length} before the job finished. Resume to continue.`);
      return 0;
    default:
      return 0;
  }
}

/** The newest step's NOTE with what its look found (its findings arrive with it): one narrated line per decision. */
function sayStep() {
  const step = S.steps.at(-1);
  if (!step || viewport.cursor !== S.steps.length) return;  // the viewer has picked another step
  const [look, ...more] = explain(step, S.planners.get(step.step));
  const found = S.found.length ? `It found ${S.found.join(', ')}.` : '';
  narrate(S.crit ?? look, [found, ...S.sent, S.crit ? look : '', ...more].filter(Boolean).join(' '));
}

function addStep(step) {
  const follow = viewport.cursor === S.steps.length;
  Object.assign(S, { found: [], sent: [], crit: null });
  S.steps.push(step);
  trace.addStep(step, causeFrom(step));
  findings.refresh();
  el.slider.max = S.steps.length;
  el.slider.disabled = el.play.disabled = false;
  el.count.textContent = `step ${step.step}`;
  if (follow) goto(S.steps.length, !Q.skip);
  else el.sliderOut.textContent = `step ${viewport.cursor} / ${S.steps.length}`;
  // a halt record may carry the governor's last override; a failed planner consultation is in planner_status
  for (const cause of [step.governor?.cause, step.cause, plannerFell(step) && 'planner-fallback']) {
    if (isGovernor(cause)) popup(++S.rev, cause, GOVERNOR_TEXT[cause](step));
  }
  sound.scratch();
  return step.decision?.action === 'halt' ? 1500 : PACE;
}

function addFinding(f) {
  const before = findings.cards.get(f.id)?.finding;
  // A re-measure of a known finding updates its card only, unless a finer look measured it past the critical rule.
  if (!findings.add(f) && !(f.critical && before && !before.critical)) return 0;
  const at = S.steps.at(-1)?.step ?? f.step;
  trace.addNote('finding', at, f.id,
    `${label(f.cls)} · ${px(f.measure?.area_px)} px²${f.critical ? ` → human sign-off requested${before ? ' (re-measured: now critical)' : ''}` : ' · draft, no sign-off needed'}`,
    f.critical ? ruleOf(f) : null, () => findings.focus(f.id));
  S.found.push(`${words(f.cls)} (${f.critical ? 'critical' : 'a draft'})`);
  const first = f.critical && !S.critSeen;
  if (first) {
    S.critSeen = true;
    S.crit = `Critical: ${words(f.cls)}, found at step ${at}. The agent cannot approve this; only you can, in step 4 once the run ends.`;
  }
  sayStep();
  return first ? HOLD : 0;
}

function pump() {
  while (Q.items.length && (!Q.timer || JOINS.has(Q.items[0].type))) {
    const wait = handle(Q.items.shift());
    if (wait && !Q.skip) {
      clearTimeout(Q.timer);  // a joining event may lengthen the pause of the step it joins
      Q.timer = setTimeout(() => {
        Q.timer = 0;
        pump();
      }, wait);
    }
  }
}

/** Show everything that has arrived at once, and anything still to come as it arrives. */
function skip() {
  Q.skip = true;
  clearTimeout(Q.timer);
  Q.timer = 0;
  if (document.activeElement === el.skip) el.kill.focus({ preventScroll: true });
  el.skip.hidden = true;
  pump();
}

const paced = () => (paceMs() > 0 ? ' (not counting the paced waits)' : '');

/** " · server total …" from the job record's prep_ms {read, decode, privacy} (ms) and the agent's seconds; '' without it. */
function serverTotal(seconds) {
  const p = S.job?.prep_ms;
  if (!p) return '';
  const ms = ['read', 'decode', 'privacy'].reduce((t, k) => t + (num(p[k]) ? p[k] : 0), 0);
  return ` · server total ${(seconds + ms / 1000).toFixed(2)} s (reading the upload, decode, privacy blur and the agent)`;
}

function arrive(ev) {
  // a Resume keeps the killed run (and Resume) until the server's first record: a refused re-send loses nothing
  if (S.opts.replay && !S.events.length && !['end', 'error'].includes(ev.type)) clearJob();
  if (ev.type === 'done') {
    S.arrived = ev;
    if (document.activeElement === el.kill) el.skip.focus({ preventScroll: true });
    el.kill.hidden = true;  // the job is over on the server: KILL could only claim a stop that did not happen
    if (Q.timer || Q.items.length) {
      S.slow = true;
      const n = S.events.filter((e) => e.type === 'step').length;
      el.text.textContent = `Replay: ${recorded() ? 'the recorded agent' : 'the agent'} took ${ev.summary.seconds} s${modelled() ? ' (modelled)' : ''} for its ${n} decisions`  // a phone shows only the start
        + `${paced()}; we are showing it slowly so you can read it${serverTotal(ev.summary.seconds)}. Skip to end shows the result now.`;
    }
  }
  if (!['end', 'error'].includes(ev.type)) S.events.push(ev);
  Q.items.push(ev);
  if (ev.type === 'error') skip();  // say what went wrong now, after what had arrived
  else pump();
  if (S.killPending && ev.type === 'job') {  // KILL was pressed before this record: send it now that the job is known
    S.killPending = false;
    sendKill(ev);
  }
}

// ---- work order / done state ---------------------------------------------------------------------------
function tile(label) {
  const v = h('b', { class: 'tile__v' }, '0');
  return [h('div', { class: 'tile' }, v, h('span', { class: 'tile__k' }, label)), v];
}

function save(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = h('a', { href: url, download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function tokenExp(token) {
  try {
    return JSON.parse(atob(String(token).split('.')[0].replace(/-/g, '+').replace(/_/g, '/'))).exp ?? null;
  } catch {
    return null;
  }
}

function buildDone() {
  const { summary: sm, coverage_looked: looked, unvisited, cause } = S.done;
  const coverage = looked ?? S.done.coverage;  // a recording older than coverage_looked: its coverage counts the survey look
  const pill = h('span', { class: 'pill' });
  const orderState = h('span', { class: 'note' });
  const specs = [
    ['looks', sm.steps, {}], ['of the pixels', sm.pixels_frac * 100, { decimals: 1, suffix: ' %' }],
    ['$ modelled', sm.usd, { decimals: 6, prefix: '$' }], [`seconds of ${agentTime()}`, sm.seconds, { decimals: 1 }],
    ['findings', sm.findings, {}], ['critical, waiting for you', sm.critical_pending, {}],
  ].map(([label, value, opts]) => ({ value, opts, pair: tile(label) }));
  const cover = h('div', { class: 'meter__fill' });
  const coverNum = h('b', { class: 'meter__val' }, '0 %');
  const privacy = S.job?.privacy;
  const order = h('button', { class: 'btn', type: 'button' }, 'Download work_order.json');
  order.addEventListener('click', () => save('work_order.json',
    JSON.stringify({ work_order: S.order, token: S.token, alg: 'HMAC-SHA256', exp: tokenExp(S.token) }, null, 2), 'application/json'));
  const log = h('button', { class: 'btn', type: 'button' }, 'Download trace.ndjson');
  log.addEventListener('click', () => save('trace.ndjson', `${S.events.map((e) => JSON.stringify(e)).join('\n')}\n`, 'application/x-ndjson'));
  const box = (id) => S.overview?.candidates?.find((c) => c.id === id)?.box;
  el.done.replaceChildren(h('div', { class: 'stack' },
    h('div', { class: 'done__head' }, pill, h('span', { class: 'note' }, `stopped because ${stopWords(cause, sm.steps)} (${cause})`)),
    orderState,
    h('div', { class: 'tiles' }, specs.map((s) => s.pair[0])),
    h('div', { class: 'meter' },
      h('div', { class: 'meter__head' }, h('span', {}, looked == null ? 'salient area seen at any level (incl. survey)' : 'salient area looked at closely (the survey look does not count)'), h('span', {}, coverNum)),
      h('div', { class: 'meter__track', role: 'progressbar', 'aria-label': looked == null ? 'Saliency-weighted coverage at any level' : 'Saliency-weighted share looked at closely', 'aria-valuemin': '0', 'aria-valuemax': '100', 'aria-valuenow': Math.round(coverage * 100) }, cover)),
    h('div', {}, h('h4', { class: 'label' }, 'not zoomed into'),
      unvisited?.length ? h('p', { class: 'chips' }, unvisited.map((r) => {
        const b = h('button', { class: 'chip chip--look chip--btn', type: 'button' }, `${r}: check it yourself`);
        b.addEventListener('click', () => showBox(box(r)));
        return b;
      })) : h('p', { class: 'note' }, S.overview?.candidates?.length ? 'none: every candidate region was looked at'
        : 'none proposed: no region stood out, so check the whole photo yourself')),
    privacy ? h('p', { class: 'note' }, `privacy: ${privacy.faces_blurred ?? 0} faces and ${privacy.people_blurred ?? 0} people blurred`) : null,
    h('div', { class: 'done__dl' }, order, log)));
  S.ui = { pill, orderState, pending: specs[5].pair[1] };
  specs.forEach((s) => countUp(s.pair[1], s.value, s.opts));
  requestAnimationFrame(() => { cover.style.width = `${coverage * 100}%`; });
  countUp(coverNum, coverage * 100, { decimals: 0, suffix: ' %' });
  updateDone();
}

function updateDone() {
  const pending = pendingCount();
  const key = doneKey();
  S.ui.pill.className = `pill pill--stamp pill--${key}`;
  if (S.ui.pill.textContent !== PILL[key]) {
    S.ui.pill.textContent = PILL[key];
    replay(S.ui.pill, 'is-stamped');
  }
  const signers = [...new Set((S.order.approvals ?? []).map((a) => a.reviewer).filter(Boolean))];
  el.tbChecked.textContent = signers.length ? signers.join(', ') : 'human reviewer';
  const critical = S.order.findings.some((f) => f.critical);
  const gap = gapText();
  const part = partText();
  const eye = eyeText();
  S.ui.orderState.textContent = pending
    ? `Waiting for your sign-off (${pending}): work order ${S.order.id} is ${S.order.status} until a person approves or rejects each critical finding.`
    : gap ? `Work order ${S.order.id} is ${S.order.status}: ${gap}. A person should look there.`
    : eye ? `Work order ${S.order.id} is ${S.order.status}: ${eye}.`
    : critical ? `Signed: ready to send. Work order ${S.order.id} is ${S.order.status}.`
      : part ? `Work order ${S.order.id} is ${S.order.status}: ${part}.`
        : `Work order ${S.order.id} is ${S.order.status}: nothing critical needed a sign-off.`;
  countUp(S.ui.pending, pending);
  setMode('done', el.text.textContent);
}

/** "What you just saw": the run in four honest numbers, and two ways to try it harder. */
function recap() {
  const kinds = { vision: 0, kill: 0, governor: 0, planner: 0, replay: 0 };
  for (const s of S.steps) kinds[kindOf(s.cause)] += 1;
  const sm = S.done.summary;
  const crit = S.order.findings.filter((f) => f.critical).length;
  const gap = gapText();
  const why = [[kinds.vision, 'from what OpenCV measured in the photo'], [kinds.kill, 'from the kill switch'], [kinds.governor, 'from the governor’s timing and budget checks'],
    [kinds.planner, 'from the planner drill (the rule planner’s choice stood)'], [kinds.replay, 'replayed from before KILL']]
    .filter(([n]) => n).map(([n, t]) => `${n} ${t}`);
  const breakIt = h('button', { class: 'btn btn--amber btn--sm', type: 'button' }, 'Now break it');
  breakIt.addEventListener('click', () => run({ pace: GUIDE_PACE_MS, chaos: 'planner_malformed,p95_spike' }));
  const tryKill = h('button', { class: 'btn btn--red btn--sm', type: 'button' }, 'Try KILL');
  tryKill.addEventListener('click', () => run({ pace: GUIDE_PACE_MS, chaos: 'planner_timeout', killHint: true }));
  const sign = h('button', { class: 'btn btn--amber btn--sm', type: 'button', id: 'sign-off' }, 'Sign them off');
  sign.addEventListener('click', () => findings.focus(S.order.findings.find((f) => f.critical && f.state === 'pending_review')?.id, true));
  // A decision is a step record; a look is one that moved the camera (the work order's "looks"); the halt is the decision to stop.
  const looks = S.steps.filter((s) => s.decision?.action !== 'halt').length;
  const halt = S.steps.find((s) => s.decision?.action === 'halt');
  const stopped = !halt ? '' : halt.cause !== 'kill-switch' ? ' and the decision to stop'
    : looks ? ' and the stop KILL asked for' : ' and the stop the kill switch made';
  // a real photo of ours has no ground truth: say what the agent flagged, beside what we see by eye, and score nothing
  const samples = $$('.thumb', el.gallery).map((b) => b.sample);
  const real = S.opts.trace ? samples.find((x) => x.file === S.sample && x.synthetic === false) : null;
  const flagged = S.order.findings.map((f) => `${words(f.cls)} (${f.critical ? 'critical' : 'a draft'})`);
  const other = real && samples.find((x) => x.synthetic === false && x.trace && x.file !== real.file);
  const otherBtn = other && h('button', { class: 'btn btn--amber btn--sm', type: 'button' }, 'Our other wall photo');
  const whole = S.steps.find((x) => x.cause === 'fallback-l2');  // the one look a photo with no candidate gets
  otherBtn?.addEventListener('click', () => guidedReal(other.file));
  el.recap.replaceChildren(
    h('h4', { class: 'recap__t' }, 'What you just saw'),
    h('ul', { class: 'recap__list' },
      real ? h('li', { class: 'recap__real' }, h('b', {}, 'REAL PHOTO (ours)'), ', no ground truth, so nothing here is scored. '
        + `On the v1 engine’s recorded run (the dacl10k-trained segmenter) the agent flagged ${flagged.length ? flagged.join(', ') : 'nothing'}${S.overview?.candidates?.length ? ''
          : whole ? `: its overview proposed no region, so it took one look at the whole photo${whole.quality?.ok === false ? `, which the quality gate refused (${whole.quality.reason})` : ''}`
            : ': it saw no region worth a closer look'}. `
        + `${real.note} Anything we see by eye that it did not flag, it missed.`) : null,
      h('li', {}, h('b', {}, String(S.steps.length)), ` decision${S.steps.length === 1 ? '' : 's'} (${looks ? looks : 'no'} look${looks === 1 ? '' : 's'}${stopped}), `
        + `${S.steps.length === 1 ? 'with its' : 'every one with a'} recorded cause: ${why.join(', ')}.`),
      h('li', {}, h('b', {}, String(crit)), ` critical finding${crit === 1 ? ' was' : 's were'} sent to a person: the agent is not allowed to approve ${crit === 1 ? 'it' : 'them'}.`),
      S.done.coverage_looked == null
        ? h('li', {}, h('b', {}, `${Math.round(S.done.coverage * 100)} %`), ' of the salient area was seen at some level (the first survey look counts).')
        : h('li', {}, h('b', {}, `${Math.round(S.done.coverage_looked * 100)} %`), ' of the salient area was looked at closely (the first survey look does not count).'),
      gap ? h('li', {}, h('b', {}, 'Not assessed'), `: ${gap}. The work order marks this for a person to look at; it does not say the wall is clean.`) : null,
      h('li', {}, h('b', {}, `$${sm.usd.toFixed(6)}`), ` modelled for ${sm.seconds} s of ${agentTime()} (${modelled() ? 'seconds from a fixed cost table' : 'wall time'} × $0.339/h, not billed).`)),
    h('div', { class: 'recap__next' },
      pendingCount() ? h('p', {}, sign, ' takes you to the first card that needs a person: type your name, then Approve or Reject.') : null,
      otherBtn ? h('p', {}, otherBtn, ' replays the v1 engine’s recorded run on it, also with no ground truth.') : null,
      MOCK ? null : h('p', {}, breakIt, ' reruns it with a planner that sends a shell command and a segmenter slowed on purpose.'),
      h('p', {}, tryKill, MOCK ? ' replays the same saved job, each look shown as it arrives, so you can halt it with KILL in the bar above.'
        : ' starts a run shown live, whose planner stalls 2 s at step 3, so you can halt it with KILL in the bar above.')));
  el.recap.hidden = false;
}

function finish(ev) {
  clearTimeout(S.killTimer);
  S.done = ev;
  S.token = ev.token;
  S.order = ev.work_order;
  const sm = ev.summary;
  const n = sm.critical_pending;
  setMode('done', S.note ?? (ev.cause === 'kill-switch'
    ? (sm.steps ? `Halted by KILL at step ${sm.steps}: the trace shows kill-switch and the state is kept. Resume carries on.`
      : 'Halted before its first look: the kill switch was already set, so there is no look to resume. Run starts it again.')
    : S.slow && !Q.skip ? `The agent took ${sm.seconds} s${modelled() ? ' (modelled)' : ''}${paced()}; we replayed it slowly so you can read it${serverTotal(sm.seconds)}.`
      : `Done in ${sm.seconds} s of ${agentTime()}: ${sm.steps} looks, ${(sm.pixels_frac * 100).toFixed(1)} % of the pixels${serverTotal(sm.seconds)}.`));
  const gap = gapText();
  const part = partText();
  narrate(n ? `${n} critical finding${n === 1 ? ' needs' : 's need'} you. Sign ${n === 1 ? 'it' : 'them'} off in step 4: the agent cannot.`
    : gap ? `Not assessed: ${gap}. A person should look there; this is not a clean result.`
      : part ? `Ready: ${part}.`
        : 'Nothing critical was found, so nothing needs your sign-off.',
  `It stopped because ${stopWords(ev.cause, sm.steps)}${S.steps.length ? `: step ${S.steps.at(-1).step} says why` : ''}.`);
  findings.setReady();
  findings.refresh();
  checks.show(S.order);
  el.tiles.disabled = false;
  buildDone();
  recap();
  replay(el.tbSheet, 'is-stamped');  // the sheet number stamps in: this run is on file
  sound.stamp();
  if (S.opts.guided) {  // end on what the run showed, in view; its Sign them off button hands over to the human
    el.recap.scrollIntoView({ block: 'start', behavior: reduced() ? 'auto' : 'smooth' });
    $('#sign-off')?.focus({ preventScroll: true });
  }
}

async function review(id, decision, reviewer) {
  const res = await postReview({ token: S.token, finding_id: id, decision, reviewer }, recorded());
  S.token = res.token;
  S.order = res.work_order;
  findings.sync(S.order);
  updateDone();
}

/** A person checked not_assessed entry `index` by eye: nothing-found or defect-found, with an optional note. */
async function check(index, decision, reviewer, note) {
  const res = await postReview({ token: S.token, not_assessed: index, decision, reviewer, ...(note ? { note } : {}) }, recorded());
  S.token = res.token;
  S.order = res.work_order;
  checks.sync(S.order);
  updateDone();
}

// ---- running a job -----------------------------------------------------------------------------------
function params(opts) {
  const p = {
    pixel_cap: +(Number($('#cap-px').value) / 100).toFixed(2),
    usd_cap: $('input[name=usd]:checked').value,
    seconds_cap: $('#cap-sec').value,
  };
  const chaos = opts.chaos ?? $$('input[name=chaos]:checked').map((i) => i.value).join(',');
  if (chaos) p.chaos = chaos;
  if (opts.pace) p.pace_ms = opts.pace;
  return p;
}

/** opts: { replay, pace (ms the server waits between looks), chaos, guided, killHint }. */
async function run(opts = {}) {
  if (!S.image || S.mode === 'running') return;
  if (MOCK && !opts.trace && S.sample !== FIXTURE) {  // the saved job is never drawn over another photo
    el.text.replaceChildren('The recorded run replays one saved job; ', h('a', { href: `${location.pathname}#demo` }, 'open the live demo'), ' to run your photo.');
    narrate(el.text.textContent);
    return;
  }
  S.opts = opts;
  S.ctl = new AbortController();
  clearTimeout(S.killTimer);
  Object.assign(S, { events: [], arrived: null, note: null, slow: false, killPending: false });
  clearTimeout(Q.timer);
  Object.assign(Q, { items: [], timer: 0, skip: Boolean(opts.killHint) });  // a KILL drill is shown live: never behind the job it asks to stop
  const active = document.activeElement;  // all hidden or disabled while running
  const refocus = [el.look, el.runBar, el.tryKill, el.resume, el.guided, el.guidedReal].includes(active) || el.recap.contains(active);
  if (!opts.replay) clearJob();  // the last run goes at once: the page answers the click before the server does
  setMode('running', recorded() ? 'Recorded run: replaying a saved job (no server).'
    : opts.replay ? 'Resuming: re-sending the photo and replaying the moves so far…' : undefined);
  narrate(recorded() ? 'Loading the recorded job…' : `Sending the photo (${(S.image.blob.size / 1e6).toFixed(1)} MB). On the server it is decoded, `
    + 'and faces and people are blurred, before the agent’s first look.');
  if (refocus) el.skip.focus({ preventScroll: true });  // the safe next key press: show the result now
  scrollToViewer();
  const t0 = performance.now();
  const wait = setInterval(() => {  // until the server's first record, the wait counts up in the job bar
    const waiting = S.mode === 'running' && !S.events.length;
    if (waiting) el.count.textContent = `${((performance.now() - t0) / 1000).toFixed(1)} s`;
    else {
      clearInterval(wait);
      if (S.mode === 'running' && !S.steps.length) el.count.textContent = '';
    }
  }, 100);
  try {
    for await (const ev of streamJob({ blob: S.image.blob, params: params(opts), replay: opts.replay, trace: opts.trace, signal: S.ctl.signal })) arrive(ev);
    arrive({ type: 'end' });
  } catch (err) {
    if (err.name === 'AbortError') return;
    const offline = err instanceof TypeError;  // fetch could not reach the server at all
    el.recorded.hidden = MOCK || !offline;
    arrive({ type: 'error', message: offline ? 'Could not reach the API. Check your connection, or watch a recorded run instead.'
      : err.status === 429 ? err.message : `The job failed: ${err.message}` });
  }
}

/** Abort the stream here, show what arrived, and say so. `unconfirmed`: the server never confirmed a KILL, so this is the
    page stopping its view of the job, not a kill (behind a Lambda Function URL the job may run on to its cap). */
function stop(why, unconfirmed = false) {
  S.ctl.abort();
  skip();
  const n = S.steps.length;
  if (!n) {  // no look had been shown: there is no state to keep and nothing to resume
    setMode('idle', `Stopped before the agent’s first look${why}. No look had arrived, so the page keeps nothing; Run starts it again.`);
    narrate(el.text.textContent);
    return;
  }
  if (unconfirmed) {
    setMode('stopped', `The page stopped showing the job at step ${n}${why}; the server may still be running it, up to its time cap. `
      + `What arrived is kept: Resume re-sends the photo and replays your ${n} moves.`);
    return;
  }
  setMode('killed', `Killed at step ${n}${why} — state preserved. Resume re-sends the photo and replays your ${n} moves.`);
}

/** KILL pressed, the halt not here yet: the pill says so at once, not "running", while the records the server sends before
    it reads the KILL (it reads one only after a step record) still arrive. The halt, or stop(), sets the final state. */
function stopping(text) {
  el.pill.className = 'pill pill--stopping';
  el.pill.textContent = 'stopping';
  el.text.textContent = text;
}

/** POST /api/kill for `job` (its first record): the server halts before its next look and sends the kill-switch record. */
function sendKill(job) {
  postKill(job.job, job.kill_token);
  skip();
  stopping('KILL sent: stopping before the agent’s next look…');
  clearTimeout(S.killTimer);
  S.killTimer = setTimeout(() => {
    if (S.mode === 'running' && !S.arrived) stop(': the server did not confirm KILL within 6 s', true);
  }, 6000);
}

function kill() {
  if (S.arrived) {  // the server finished before the click: there is nothing left to stop
    S.note = `The job had already finished on the server (the agent took ${S.arrived.summary.seconds} s) before KILL could reach it; here is its result.`;
    return skip();
  }
  if (!S.events.length && !recorded()) {  // the job id and kill token come with the server's first record (a Resume's too)
    if (S.killPending) return;
    S.killPending = true;
    stopping('KILL pressed before the server answered: the page has no job id yet, so KILL is sent the moment it arrives.');
    narrate('KILL queued: it is sent as soon as the server answers.');
    clearTimeout(S.killTimer);
    S.killTimer = setTimeout(() => {  // no answer at all: cancel the upload rather than leave KILL doing nothing
      if (S.mode === 'running' && S.killPending) {
        stop(`: the server had not answered ${KILL_WAIT_S} s after KILL, so the page cancelled the upload (if the server had `
          + 'already started the job, it may run on, unseen, up to its time cap)');
      }
    }, KILL_WAIT_S * 1000);
    return;
  }
  if (recorded()) return stop(S.events.length ? '' : ' (the recording had not started)');  // no server to tell
  sendKill(S.job);
}

function resume() {
  const moves = S.steps.filter((s) => ['gaze', 'escalate', 'verify'].includes(s.decision?.action) && pointOf(s))
    .map((s) => [s.decision.action, s.decision.args.x, s.decision.args.y, s.decision.args.level, s.decision.args.size]
      .filter((v) => v !== undefined));
  run({ pace: S.opts.pace, chaos: S.opts.chaos, trace: S.opts.trace, replay: moves.slice(0, 200) });
}

async function guided() {
  if (S.mode === 'running') return;
  await S.ready;
  const btn = !MOCK && $(`.thumb[data-file="${GUIDE}"]`, el.gallery);
  if (btn && S.sample !== GUIDE) await choose(btn);
  run({ pace: GUIDE_PACE_MS, guided: true });
}

/** The guided run on one of our own wall photos: the v1 engine's recorded job on it is replayed (no server), so the page
    shows exactly the run the docs report. */
async function guidedReal(file = GUIDE_REAL) {
  if (S.mode === 'running') return;
  await S.ready;
  const btn = $(`.thumb[data-file="${file}"]`, el.gallery);
  if (!btn?.sample.trace) return;
  if (S.sample !== file) await choose(btn);
  if (S.sample !== file) return;  // the photo did not load: never replay a job over another photo
  run({ trace: `samples/${btn.sample.trace}`, guided: true });
}

// ---- image picking -----------------------------------------------------------------------------------
// The latest photo asked for wins, not the one whose load finishes last (the startup preselect can still be loading
// when a viewer taps another sample).
let picks = 0;

async function useImage(blob, label, file = null, pick = ++picks) {
  if (S.mode === 'running') {
    el.prep.textContent = 'A job is running. KILL it first to change the photo.';
    return;
  }
  el.prep.textContent = 'Reading the photo…';
  try {
    const p = await prepareImage(blob);
    if (pick !== picks) return;  // another photo was asked for meanwhile
    if (S.image) URL.revokeObjectURL(S.image.url);
    S.image = { blob: p.blob, url: URL.createObjectURL(blob) };
    S.sample = file;
    clearJob();
    viewport.setImage(S.image.url);
    el.prep.textContent = [label, `${p.width}×${p.height}`, `${(p.blob.size / 1e6).toFixed(1)} MB`, p.note].filter(Boolean).join(' · ');
    const note = $$('.thumb', el.gallery).find((b) => b.dataset.file === file)?.sample.note;  // in full: the card clamps it
    el.say1.textContent = `Ready: ${label} (${p.width} × ${p.height} px).${note ? ` ${note}` : ''} Pick another sample, or use your own photo.`;
    for (const b of $$('.thumb', el.gallery)) b.setAttribute('aria-pressed', String(b.dataset.file === file));
    setMode('idle');
  } catch {
    el.prep.textContent = 'Could not read that image. Use a JPEG or PNG.';
  }
}

function pickFile(file) {
  if (!file) return;
  if (!/^image\/(jpeg|png)$/.test(file.type)) {
    el.prep.textContent = 'Only JPEG or PNG files, please.';
    return;
  }
  useImage(file, file.name);
}

async function choose(btn) {
  const s = btn.sample;
  if (S.mode === 'running') { el.prep.textContent = 'A job is running. KILL it first to change the photo.'; return; }
  el.look.disabled = el.runBar.disabled = el.tryKill.disabled = true;  // the old photo must not run while the new one loads
  el.prep.textContent = `Loading ${String(s.title ?? s.file)}…`;
  const pick = ++picks;
  const res = await fetch(`samples/${s.file}`).catch(() => null);
  if (pick !== picks) return;  // another photo was asked for meanwhile
  if (!res?.ok) {
    el.prep.textContent = 'That sample could not be loaded.';
    el.look.disabled = el.runBar.disabled = el.tryKill.disabled = !S.image;
    return;
  }
  const blob = await res.blob();
  if (pick === picks) await useImage(blob, String(s.title ?? s.file), s.file, pick);
}

/** Builds the gallery; resolves to the sample to preselect (the guided one when it is there). */
async function loadSamples() {
  const list = await getJSON('samples/index.json');
  const items = (Array.isArray(list) ? list : []).filter((s) => s && /^[\w.-]+$/.test(s.file ?? ''));
  el.galleryEmpty.hidden = items.length > 0;
  const btns = items.map((s) => {
    const thumb = /^\w[\w./-]*$/.test(s.thumb ?? '') && !s.thumb.includes('..') ? s.thumb : s.file;
    const real = s.synthetic === false;
    if (!/^[\w.-]+\.ndjson$/.test(s.trace ?? '')) delete s.trace;
    const btn = h('button', { class: real ? 'thumb thumb--real' : 'thumb', type: 'button', 'aria-pressed': 'false', 'data-file': s.file },
      h('img', { src: `samples/${thumb}`, alt: '', loading: 'lazy', width: 96, height: 72 }),
      h('span', { class: 'thumb__t' }, String(s.title ?? s.file)),
      s.note ? h('span', { class: 'thumb__n' }, String(s.note)) : null,
      s.synthetic ? h('span', { class: 'tag' }, 'synthetic') : real ? h('span', { class: 'tag tag--real' }, 'real photo') : null);
    btn.sample = s;
    btn.addEventListener('click', () => choose(btn));
    return btn;
  });
  el.gallery.replaceChildren(...btns.map((b) => h('li', {}, b)));
  if (!items.length) el.say1.textContent = 'Drop your own JPEG or PNG to start.';
  el.guidedReal.hidden = !btns.some((b) => b.dataset.file === GUIDE_REAL && b.sample.trace);
  return btns.find((b) => b.dataset.file === GUIDE) ?? btns[0];
}

// ---- wiring ------------------------------------------------------------------------------------------
el.look.addEventListener('click', () => run());
el.runBar.addEventListener('click', () => run());
// the KILL drill, also offered before any run: shown live, and its planner stalls 2 s at step 3 so KILL lands mid-job
el.tryKill.addEventListener('click', () => run({ pace: GUIDE_PACE_MS, chaos: 'planner_timeout', killHint: true }));
el.guided.addEventListener('click', guided);
el.guidedReal.addEventListener('click', () => guidedReal());
el.skip.addEventListener('click', skip);
el.kill.addEventListener('click', kill);
el.resume.addEventListener('click', resume);
el.play.addEventListener('click', togglePlay);
el.heat.addEventListener('change', () => viewport.setHeat(el.heat.checked));
el.tiles.addEventListener('change', () => {
  viewport.setTiles(el.tiles.checked);
  el.tilesNote.hidden = !el.tiles.checked;
  if (S.job) el.tilesNote.textContent = `${tileBoxes(S.job.width, S.job.height).length} tiles of 512 px at full resolution for a full tiling scan, vs ${S.done?.summary.steps ?? S.steps.length} agent looks.`;
});
el.slider.addEventListener('input', () => {
  stopPlay();
  goto(Number(el.slider.value));
});
el.sound.addEventListener('click', () => {
  const on = sound.toggle();
  el.sound.setAttribute('aria-pressed', String(on));
  el.sound.textContent = `Sound: ${on ? 'on' : 'off'}`;
  if (on) sound.click();
});
el.file.addEventListener('change', () => {
  pickFile(el.file.files[0]);
  el.file.value = '';
});
el.drop.addEventListener('dragover', (e) => { e.preventDefault(); el.drop.classList.add('is-over'); });
el.drop.addEventListener('dragleave', () => el.drop.classList.remove('is-over'));
el.drop.addEventListener('drop', (e) => { e.preventDefault(); el.drop.classList.remove('is-over'); pickFile(e.dataTransfer.files[0]); });
for (const type of ['dragover', 'drop']) window.addEventListener(type, (e) => e.preventDefault());
$('#cap-px').addEventListener('input', (e) => { $('#cap-px-out').textContent = `${e.target.value} %`; });
$('#cap-sec').addEventListener('input', (e) => { $('#cap-sec-out').textContent = `${e.target.value} s`; });
for (const li of el.stages.children) li.dataset.text = li.textContent;

// The nav marks the section in the middle of the screen; the phone menu closes once a link is picked.
const NAV = { demo: '#demo', how: '#how', benchmarks: '#benchmarks', failures: '#benchmarks', docs: '#docs' };
const navLinks = $$('.nav__links a, .nav__drop a');
const navSpy = new IntersectionObserver((entries) => {
  for (const e of entries.filter((x) => x.isIntersecting)) {
    for (const a of navLinks) {
      if (a.getAttribute('href') === NAV[e.target.id]) a.setAttribute('aria-current', 'true');
      else a.removeAttribute('aria-current');
    }
  }
}, { rootMargin: '-45% 0px -50% 0px' });
$$('#top, main > section, #docs').forEach((s) => navSpy.observe(s));
$('#nav-menu').addEventListener('click', (e) => { if (e.target.closest('a')) $('#nav-menu').open = false; });

// Phones see only the photo and its numbered balloons (the parts list below carries the callout text).
const artMq = matchMedia('(max-width: 560px)');
const fitArt = () => $('.art__svg').setAttribute('viewBox', artMq.matches ? '0 0 492 372' : '0 0 600 430');
artMq.onchange = fitArt;
fitArt();

// The sticky bars' real heights: scroll padding and focus scrolling keep controls clear of them.
function bars() {
  document.documentElement.style.setProperty('--nav', `${$('#nav').offsetHeight}px`);
  document.documentElement.style.setProperty('--bar', `${$('#taskbar').offsetHeight}px`);
}
new ResizeObserver(bars).observe($('#taskbar'));

const pops = new IntersectionObserver((entries) => {
  let i = 0;
  for (const e of entries) {
    if (!e.isIntersecting) continue;
    e.target.style.setProperty('--d', `${i++ * 90}ms`);
    e.target.classList.add('is-in');
    pops.unobserve(e.target);
  }
}, { threshold: 0.06 });
$$('.pop').forEach((w) => pops.observe(w));
document.documentElement.classList.add('js');  // only now may CSS hide not-yet-scrolled windows (a failed script leaves them visible)

clearJob();
setMode('idle');
S.ready = loadSamples().then((first) => {
  if (!MOCK) return first && choose(first);
  const pick = ++picks;  // asked for now: a tap while it loads wins
  return fetch(FIXTURE).then((r) => r.blob())
    .then((b) => (pick === picks ? useImage(b, 'mock.jpg (recorded job)', FIXTURE, pick) : null));
});
initResults();
health().then((hl) => {
  const arch = hl?.arch ?? hl?.machine;
  const where = [hl?.host && `host ${hl.host}`, arch, hl?.commit && `commit ${String(hl.commit).slice(0, 7)}`].filter(Boolean);
  $('#api-status').textContent = MOCK ? 'Recorded run: no API is called.' : hl?.ok
    ? ['API online', `OpenCV ${hl.opencv}`, hl.build, `segmenter ${hl.segmenter}`, ...where].join(' · ')
    : 'The API is not reachable from this page. Watch a recorded run instead: it replays a saved job.';
  if (hl?.ok && hl.host) $('#live-where').textContent = `Live demo: OpenCV ${hl.opencv}, ${hl.build}, on ${hl.host}${arch ? ` (${arch})` : ''}.`;
  const b = hl?.ok ? hl.cv2_build : null;  // read live from cv2.getBuildInformation() on the server
  if (b) {
    $('#api-build').textContent = [b.hal && `Custom HAL: ${b.hal}`, b.parallel && `parallel framework: ${b.parallel}`,
      b.cpu_baseline && `CPU baseline: ${b.cpu_baseline}`, b.cpu_dispatch && `dispatched: ${b.cpu_dispatch}`,
      Array.isArray(b.dnn_engines) && b.dnn_engines.length && `cv2.dnn engines built in: ${b.dnn_engines.join(', ')}`,
      b.onnx_runtime && `ONNX Runtime: ${b.onnx_runtime}`].filter(Boolean).join(' · ');
    $('#api-build').hidden = false;
  }
  el.recorded.hidden = MOCK || Boolean(hl?.ok);
});
