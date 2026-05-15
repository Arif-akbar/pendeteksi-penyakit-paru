const API_PATH = "/api/predict";
const LOCAL_API_ENDPOINT = "http://127.0.0.1:5000/api/predict";

const fileInput = document.querySelector("#fileInput");
const dropZone = document.querySelector("#dropZone");
const fileName = document.querySelector("#fileName");
const analyzeButton = document.querySelector("#analyzeButton");
const resetButton = document.querySelector("#resetButton");
const xrayPreview = document.querySelector("#xrayPreview");
const boxOverlay = document.querySelector("#boxOverlay");
const targetFrame = document.querySelector("#targetFrame");
const targetStatus = document.querySelector("#targetStatus");
const scanTimer = document.querySelector("#scanTimer");
const findingList = document.querySelector("#findingList");
const findingTemplate = document.querySelector("#findingTemplate");
const reportText = document.querySelector("#reportText");
const confidenceRing = document.querySelector("#confidenceRing");
const confidenceValue = document.querySelector("#confidenceValue");
const fragmentMeter = document.querySelector("#fragmentMeter");
const connectionStatus = document.querySelector("#connectionStatus");
const endpointReadout = document.querySelector("#endpointReadout");
const stateReadout = document.querySelector("#stateReadout");
const commandRibbon = document.querySelector(".command-ribbon");

let selectedFile = null;
let previewUrl = null;
let timerHandle = null;
let scanStart = 0;
let lastDetections = [];

const ringLength = 2 * Math.PI * 48;
confidenceRing.style.strokeDasharray = ringLength;
confidenceRing.style.strokeDashoffset = ringLength;
endpointReadout.textContent = resolveApiEndpoint();

function resolveApiEndpoint() {
  const localHostnames = new Set(["localhost", "127.0.0.1"]);

  if (window.location.protocol === "file:") {
    return LOCAL_API_ENDPOINT;
  }

  if (localHostnames.has(window.location.hostname) && window.location.port !== "5000") {
    return LOCAL_API_ENDPOINT;
  }

  return API_PATH;
}

function setState(label, tone = "normal") {
  stateReadout.textContent = label;
  commandRibbon.classList.toggle("is-error", tone === "error");
}

function setConnection(message, tone = "normal") {
  connectionStatus.textContent = message;
  commandRibbon.classList.toggle("is-error", tone === "error");
}

function startTimer() {
  scanStart = performance.now();
  scanTimer.textContent = "00.00s";
  clearInterval(timerHandle);
  timerHandle = setInterval(() => {
    const elapsed = (performance.now() - scanStart) / 1000;
    scanTimer.textContent = `${elapsed.toFixed(2).padStart(5, "0")}s`;
  }, 60);
}

function stopTimer() {
  clearInterval(timerHandle);
  timerHandle = null;
}

function normalizeConfidence(value) {
  const numeric = Number(value);

  if (!Number.isFinite(numeric)) {
    return 0;
  }

  return Math.max(0, Math.min(100, numeric <= 1 ? numeric * 100 : numeric));
}

function isDicomFile(file) {
  return /\.(dcm|dicom)$/i.test(file.name);
}

function isSupportedFile(file) {
  return file.type.startsWith("image/") || isDicomFile(file);
}

function clearOverlay() {
  boxOverlay.innerHTML = "";
  lastDetections = [];
}

function severityClass(value = "") {
  const severity = String(value).toLowerCase();

  if (severity.includes("parah")) {
    return "severity-severe";
  }

  if (severity.includes("sedang")) {
    return "severity-medium";
  }

  return "severity-mild";
}

function getRenderedImageRect() {
  if (!xrayPreview.naturalWidth || !xrayPreview.naturalHeight) {
    return null;
  }

  const frameRect = targetFrame.getBoundingClientRect();
  const imageBox = xrayPreview.getBoundingClientRect();
  const imageRatio = xrayPreview.naturalWidth / xrayPreview.naturalHeight;
  const boxRatio = imageBox.width / imageBox.height;

  let width = imageBox.width;
  let height = imageBox.height;

  if (boxRatio > imageRatio) {
    width = height * imageRatio;
  } else {
    height = width / imageRatio;
  }

  return {
    left: imageBox.left - frameRect.left + (imageBox.width - width) / 2,
    top: imageBox.top - frameRect.top + (imageBox.height - height) / 2,
    width,
    height,
    naturalWidth: xrayPreview.naturalWidth,
    naturalHeight: xrayPreview.naturalHeight,
  };
}

function renderOverlay(detections = lastDetections) {
  boxOverlay.innerHTML = "";
  lastDetections = detections;

  if (!detections.length) {
    return;
  }

  const imageRect = getRenderedImageRect();
  if (!imageRect) {
    return;
  }

  detections.forEach((detection, index) => {
    if (!Array.isArray(detection.box) || detection.box.length !== 4) {
      return;
    }

    const [x1, y1, x2, y2] = detection.box.map(Number);
    const node = document.createElement("div");
    const label = document.createElement("span");
    const left = imageRect.left + (x1 / imageRect.naturalWidth) * imageRect.width;
    const top = imageRect.top + (y1 / imageRect.naturalHeight) * imageRect.height;
    const width = ((x2 - x1) / imageRect.naturalWidth) * imageRect.width;
    const height = ((y2 - y1) / imageRect.naturalHeight) * imageRect.height;

    node.className = `detection-box ${severityClass(detection.keparahan)}`;
    node.style.left = `${left}px`;
    node.style.top = `${top}px`;
    node.style.width = `${Math.max(18, width)}px`;
    node.style.height = `${Math.max(18, height)}px`;

    label.textContent = `${detection.temuan || `A-${index + 1}`} ${Math.round(normalizeConfidence(detection.confidence))}%`;
    node.appendChild(label);
    boxOverlay.appendChild(node);
  });
}

function setConfidence(percent) {
  const safePercent = Math.max(0, Math.min(100, percent));
  const dashOffset = ringLength - (ringLength * safePercent) / 100;

  confidenceRing.style.strokeDashoffset = dashOffset;
  confidenceValue.textContent = `${Math.round(safePercent)}%`;
  confidenceRing.style.stroke = safePercent >= 75 ? "var(--cyan)" : safePercent >= 45 ? "var(--amber)" : "var(--red)";

  const activeFragments = Math.round((safePercent / 100) * fragmentMeter.children.length);
  [...fragmentMeter.children].forEach((fragment, index) => {
    fragment.classList.toggle("active", index < activeFragments);
  });
}

function setPreview(file) {
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
  }

  selectedFile = file;
  clearOverlay();

  if (file.type.startsWith("image/")) {
    previewUrl = URL.createObjectURL(file);
    xrayPreview.src = previewUrl;
    targetFrame.classList.add("has-image");
  } else {
    previewUrl = null;
    xrayPreview.removeAttribute("src");
    targetFrame.classList.remove("has-image");
  }

  fileName.textContent = `${file.name} / ${(file.size / 1024 / 1024).toFixed(2)} MB`;
  targetStatus.textContent = isDicomFile(file) ? "TARGET: DICOM LOCKED" : "TARGET: IMAGE LOCKED";
  analyzeButton.disabled = false;
  setState("READY");
  setConnection("API LINK: READY");
}

function handleFiles(files) {
  const [file] = files;

  if (!file) {
    return;
  }

  if (!isSupportedFile(file)) {
    setConnection("API LINK: FORMAT REJECTED", "error");
    setState("REJECTED", "error");
    return;
  }

  setPreview(file);
}

function renderFindings(detections = []) {
  findingList.innerHTML = "";

  if (!detections.length) {
    findingList.innerHTML = '<div class="empty-feed">No anomaly vectors resolved.</div>';
    return;
  }

  detections.forEach((detection, index) => {
    const confidence = normalizeConfidence(detection.confidence);
    const node = findingTemplate.content.firstElementChild.cloneNode(true);

    node.querySelector('[data-field="temuan"]').textContent = detection.temuan || `Anomaly ${index + 1}`;
    node.querySelector('[data-field="confidence"]').textContent = `${Math.round(confidence)}%`;
    node.querySelector('[data-field="keparahan"]').textContent = detection.keparahan || "Pending";
    node.querySelector('[data-field="box"]').textContent = Array.isArray(detection.box)
      ? `box: ${detection.box.join(", ")}`
      : "box: --";
    node.querySelector(".data-bar span").style.setProperty("--fill", `${confidence}%`);

    node.classList.add(severityClass(detection.keparahan));

    findingList.appendChild(node);
  });
}

function renderResult(payload) {
  const data = payload?.data || {};
  const detections = Array.isArray(data.deteksi) ? data.deteksi : [];
  const averageConfidence = detections.length
    ? detections.reduce((sum, item) => sum + normalizeConfidence(item.confidence), 0) / detections.length
    : 0;

  renderFindings(detections);
  renderOverlay(detections);
  setConfidence(averageConfidence);
  reportText.textContent = data.laporan || "Analysis complete. No narrative report was returned by the model.";
  targetStatus.textContent = detections.length
    ? `TARGET: ${detections.length} VECTOR${detections.length > 1 ? "S" : ""} RESOLVED`
    : "TARGET: CLEAR FIELD";
}

async function analyzeSelectedImage() {
  if (!selectedFile) {
    return;
  }

  const endpoint = resolveApiEndpoint();
  const formData = new FormData();
  formData.append("file", selectedFile);

  analyzeButton.disabled = true;
  targetFrame.classList.add("processing");
  setState("SCANNING");
  setConnection("API LINK: TRANSMITTING");
  targetStatus.textContent = "TARGET: LASER SCAN ACTIVE";
  reportText.textContent = "Neural detector is processing radiographic texture, bounding anomaly regions, and synthesizing medical observations.";
  renderFindings([]);
  clearOverlay();
  setConfidence(0);
  startTimer();

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      body: formData,
    });

    const payload = await response.json().catch(() => ({}));

    if (!response.ok) {
      throw new Error(payload.error || `API request failed with HTTP ${response.status}`);
    }

    renderResult(payload);
    setState("COMPLETE");
    setConnection("API LINK: SYNCHRONIZED");
  } catch (error) {
    setState("FAULT", "error");
    setConnection("API LINK: FAULT", "error");
    targetStatus.textContent = "TARGET: SCAN INTERRUPTED";
    reportText.textContent = `Diagnostic transmission failed: ${error.message}`;
    renderFindings([]);
    clearOverlay();
    setConfidence(0);
  } finally {
    stopTimer();
    targetFrame.classList.remove("processing");
    analyzeButton.disabled = !selectedFile;
  }
}

function resetSession() {
  selectedFile = null;
  fileInput.value = "";

  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }

  xrayPreview.removeAttribute("src");
  targetFrame.classList.remove("has-image", "processing");
  targetStatus.textContent = "TARGET: UNASSIGNED";
  scanTimer.textContent = "00.00s";
  fileName.textContent = "DICOM/JPG/PNG input ready";
  reportText.textContent = "Upload a lung X-Ray and initiate the diagnostic scan to generate an AI-assisted observation report.";
  analyzeButton.disabled = true;
  renderFindings([]);
  clearOverlay();
  setConfidence(0);
  setState("IDLE");
  setConnection("API LINK: STANDBY");
  stopTimer();
}

fileInput.addEventListener("change", (event) => handleFiles(event.target.files));
analyzeButton.addEventListener("click", analyzeSelectedImage);
resetButton.addEventListener("click", resetSession);
xrayPreview.addEventListener("load", () => renderOverlay());
window.addEventListener("resize", () => renderOverlay());

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("drag-over");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("drag-over");
  });
});

dropZone.addEventListener("drop", (event) => {
  handleFiles(event.dataTransfer.files);
});

window.addEventListener("pointermove", (event) => {
  const x = Math.round((event.clientX / window.innerWidth) * 100);
  const y = Math.round((event.clientY / window.innerHeight) * 100);
  document.documentElement.style.setProperty("--glow-x", `${x}%`);
  document.documentElement.style.setProperty("--glow-y", `${y}%`);
});

resetSession();
