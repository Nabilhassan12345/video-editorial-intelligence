const { execFile } = require('child_process');
const path = require('path');
const fs = require('fs');
const os = require('os');

const PROJECT_ROOT = path.resolve(__dirname, '../../');
const DEFAULT_TIMEOUT_MS = 120000;

function resolvePythonBin() {
  if (process.env.PYTHON_BIN) return process.env.PYTHON_BIN;
  const venvPython = path.join(PROJECT_ROOT, '.venv', 'bin', 'python');
  return fs.existsSync(venvPython) ? venvPython : 'python3';
}

/**
 * Runs the Python pipeline (python -m src.cli analyze) on a transcript and
 * resolves with the parsed JSON report produced by src/pipeline/report_generator.py.
 */
function runPythonPipeline(transcript) {
  if (process.env.DISABLE_PYTHON_PIPELINE === 'true') {
    return Promise.reject(new Error('Python pipeline disabled via DISABLE_PYTHON_PIPELINE'));
  }

  const pythonBin = resolvePythonBin();
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'video-editorial-'));
  const transcriptPath = path.join(tempDir, 'transcript.txt');
  const reportBaseName = 'node_analysis';
  fs.writeFileSync(transcriptPath, transcript, 'utf8');

  const args = [
    '-m', 'src.cli', 'analyze',
    '--transcript', transcriptPath,
    '--output-dir', tempDir,
    '--report-name', reportBaseName,
  ];
  const timeout = Number(process.env.PYTHON_PIPELINE_TIMEOUT_MS) || DEFAULT_TIMEOUT_MS;

  return new Promise((resolve, reject) => {
    execFile(pythonBin, args, { cwd: PROJECT_ROOT, timeout, maxBuffer: 10 * 1024 * 1024 }, (error, _stdout, stderr) => {
      try {
        if (error) {
          const detail = (stderr || '').trim().split('\n').slice(-3).join(' | ');
          return reject(new Error('Python pipeline failed: ' + error.message + (detail ? ' :: ' + detail : '')));
        }
        const jsonPath = path.join(tempDir, reportBaseName + '.json');
        if (!fs.existsSync(jsonPath)) {
          return reject(new Error('Python pipeline did not write ' + jsonPath));
        }
        return resolve(JSON.parse(fs.readFileSync(jsonPath, 'utf8')));
      } catch (e) {
        return reject(e);
      } finally {
        fs.rmSync(tempDir, { recursive: true, force: true });
      }
    });
  });
}

module.exports = { runPythonPipeline };
