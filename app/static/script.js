const video = document.getElementById("video");
const canvas = document.createElement("canvas");
const ctx = canvas.getContext("2d");
document.body.appendChild(canvas);
canvas.style.position = "absolute";
canvas.style.top = video.offsetTop + "px";
canvas.style.left = video.offsetLeft + "px";

let keypointBuffer = [];
const SEQUENCE_LENGTH = 30;
let sendInterval = 700;

let lastStaticPrediction = "";
let stableStaticPrediction = "";
let staticConsistentCount = 0;
const REQUIRED_STATIC_STABILITY = 3;

let lastDynamicPrediction = "";
let stableDynamicPrediction = "";
let dynamicConsistentCount = 0;
const REQUIRED_DYNAMIC_STABILITY = 3;

const maxHistory = 5;
const historyList = [];
let collectedWord = "";
let wordBuffer = "";

function updateHistory(pred) {
  if (historyList.length >= maxHistory) {
    historyList.shift();
  }
  historyList.push(pred);
  const historyContainer = document.getElementById("history");
  historyContainer.innerHTML = "";
  for (let item of historyList.slice().reverse()) {
    const li = document.createElement("li");
    li.textContent = item;
    historyContainer.appendChild(li);
  }
}

function updateCollectedWord() {
  document.getElementById("collectedWord").innerText = collectedWord;
}

function deleteLastChar() {
  collectedWord = collectedWord.slice(0, -1);
  updateCollectedWord();
}

async function loadMediaPipe() {
  const hands = new Hands({
    locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
  });

  hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
  });

  hands.onResults(onResults);

  const camera = new Camera(video, {
    onFrame: async () => {
      await hands.send({ image: video });
    },
    width: 640,
    height: 480
  });
  camera.start();
}

function onResults(results) {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.save();
  ctx.scale(-1, 1);
  ctx.translate(-canvas.width, 0);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  ctx.restore();

  let keypoints = [];

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    const landmarks = results.multiHandLandmarks[0];
    for (let lm of landmarks) {
      keypoints.push(1 - lm.x, lm.y);
      ctx.beginPath();
      ctx.arc((1 - lm.x) * canvas.width, lm.y * canvas.height, 5, 0, 2 * Math.PI);
      ctx.fillStyle = "lime";
      ctx.fill();
    }
    keypointBuffer.push(keypoints);
    if (keypointBuffer.length > SEQUENCE_LENGTH) {
      keypointBuffer.shift();
    }
  }

  document.getElementById("pointsCount").innerText = keypoints.length;
}

setInterval(() => {
  if (keypointBuffer.length >= SEQUENCE_LENGTH) {
    fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type: "auto", data: keypointBuffer })
    })
      .then(res => res.json())
      .then(data => {
        const prediction = data.prediction;
        const gestureType = data.type || "auto";

        document.getElementById("gestureType").innerText = gestureType;

        if (gestureType === "dynamic") {
          // Стабилизация динамических жестов
          if (prediction === lastDynamicPrediction) {
            dynamicConsistentCount++;
          } else {
            lastDynamicPrediction = prediction;
            dynamicConsistentCount = 1;
          }

          if (dynamicConsistentCount >= REQUIRED_DYNAMIC_STABILITY) {
            if (prediction !== stableDynamicPrediction) {
              stableDynamicPrediction = prediction;
              document.getElementById("result").innerText = `Нәтиже: ${prediction}`;
              updateHistory(prediction);
              collectedWord += prediction;  // добавление динамических букв
              updateCollectedWord();
            }
          }
        } else if (gestureType === "static") {
          // Обработка статических жестов
          if (prediction === lastStaticPrediction) {
            staticConsistentCount++;
          } else {
            lastStaticPrediction = prediction;
            staticConsistentCount = 1;
          }

          if (staticConsistentCount >= REQUIRED_STATIC_STABILITY) {
            if (prediction !== stableStaticPrediction) {
              stableStaticPrediction = prediction;
              document.getElementById("result").innerText = `Нәтиже: ${prediction}`;
              updateHistory(prediction);
              collectedWord += prediction;  // добавление статических букв
              updateCollectedWord();
            }
          }
        }
      });
  }
}, sendInterval);

// === Загрузка MediaPipe
import("https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js").then(() => {
  import("https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js").then(() => {
    loadMediaPipe();
  });
});
