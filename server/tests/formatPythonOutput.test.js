const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');
const { formatPythonOutput } = require('../services/analysisPipeline');

const sample = JSON.parse(
  fs.readFileSync(path.resolve(__dirname, '../../examples/sample_report.json'), 'utf8')
);

test('formatPythonOutput maps the Python report to the UI schema', () => {
  const out = formatPythonOutput(sample);
  assert.strictEqual(out.hook, sample.hook);
  assert.strictEqual(out.drop_off_risks.length, sample.drop_off_risks.length);
  assert.strictEqual(out.segments.length, sample.segments.length);
  assert.deepStrictEqual(out.improvements, sample.improvements);
  assert.strictEqual(out.source, 'python_model');
  for (const seg of out.segments) {
    assert.match(seg.start_time, /^\d{2}:\d{2}$/);
    assert.ok(seg.risk_score >= 0 && seg.risk_score <= 10);
    assert.ok(seg.engagement_score >= 0 && seg.engagement_score <= 10);
  }
});

test('formatPythonOutput tolerates an empty report', () => {
  const out = formatPythonOutput({});
  assert.deepStrictEqual(out.segments, []);
  assert.deepStrictEqual(out.drop_off_risks, []);
  assert.strictEqual(out.model_status, 'unknown');
});
